"""MCP server — the pipeline as tools an agent can call directly.

The CLI and this server expose the same pipeline; they differ in who is driving.
Over MCP the agent gets the pipeline as typed tools, and — the part that matters
— the *reading* tools return their content inline rather than writing a file and
hoping something reads it back. `pack_takes` hands back the transcript view,
`lint_edl` hands back the report. That is the difference between an agent that
can reason about a cut and one that has to shell out and parse stdout.

Long-running work (render, transcribe, autocut) is pushed to a worker thread so
a multi-minute render does not stall the server's event loop and block every
other call behind it.

Run it:

    reelforge-mcp

Register it with Claude Code:

    claude mcp add reelforge -- reelforge-mcp

or in `claude_desktop_config.json` / `.mcp.json`:

    { "mcpServers": { "reelforge": { "command": "reelforge-mcp" } } }
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer

from . import __version__
from .captions import PRESETS as CAPTION_PRESETS
from .config import PLATFORMS, QUALITY, get_platform
from .edl import EDL, EDLError
from .grade import list_presets as list_grades
from .transcribe import WHISPER_MODEL, TranscriptionError, load_dotenv, resolve_backend

WORK_DIRNAME = ".reelforge"

server = MCPServer(
    name="reelforge",
    version=__version__,
    instructions=(
        "Edits footage into vertical video (Instagram Reels, TikTok, YouTube Shorts).\n\n"
        "Two routes. For a single take that just needs tightening and verticalising, "
        "call `autocut` then `render` — no transcript required. For multi-take work, "
        "restructuring, or captions, call `transcribe` then `pack_takes`, read the "
        "returned transcript, write an EDL with `write_edl`, then `lint_edl` and "
        "`render`.\n\n"
        "Always confirm the plan with the user before rendering. Always run "
        "`lint_edl` before a final render and report what it says."
    ),
)


def _work(root: Path) -> Path:
    return root / WORK_DIRNAME


def _resolve_dir(directory: str) -> Path:
    p = Path(directory).expanduser().resolve()
    if not p.is_dir():
        raise ValueError(f"not a directory: {p}")
    return p


def _resolve_file(path: str, base: Path | None = None) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute() and base is not None:
        p = base / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    return p


def _err(e: Exception) -> str:
    """Render an exception as something an agent can act on."""
    if isinstance(e, EDLError):
        return "EDL is invalid:\n" + "\n".join(f"  - {p}" for p in e.problems)
    return f"{type(e).__name__}: {e}"


# --- Inspection -------------------------------------------------------------


@server.tool(
    description=(
        "Inspect a media file: dimensions, frame rate, duration, audio, rotation, "
        "HDR. Also reports whether it will need reframing for a given vertical "
        "target, and which reframe mode suits it. Call this before planning an edit."
    )
)
async def probe_media(path: str, platform: str = "reels") -> str:
    from .ffmpeg import probe

    try:
        info = await anyio.to_thread.run_sync(probe, _resolve_file(path))
        target = get_platform(platform)
    except Exception as e:  # noqa: BLE001
        return _err(e)

    w, h = info.display_size
    needs = abs(w / h - target.aspect) > 1e-3
    orientation = "vertical" if h > w else ("square" if w == h else "landscape")

    out: dict[str, Any] = {
        "path": str(info.path),
        "display_size": f"{w}x{h}",
        "orientation": orientation,
        "fps": round(info.fps, 3),
        "duration_s": round(info.duration, 3),
        "has_audio": info.has_audio,
        "codec": info.video_codec,
        "rotation": info.rotation,
        "is_hdr": info.is_hdr,
        "target": f"{target.label} {target.width}x{target.height}",
        "needs_reframe": needs,
    }
    if needs and w / h > target.aspect:
        effective = h * target.aspect
        out["reframe_note"] = (
            f"cropping to {target.aspect_label} yields {effective:.0f}px of width, "
            f"upscaled to {target.width}px"
            if effective < target.width * 0.9
            else f"crops cleanly to {target.aspect_label}"
        )
        out["suggested_mode"] = "track (people) or blur_pad (screen recordings)"
    return json.dumps(out, indent=2)


@server.tool(
    description=(
        "List delivery targets with their canvas, frame rate, duration limits and "
        "UI safe zones, plus the available caption styles, colour grades, reframe "
        "modes and render qualities."
    )
)
async def list_capabilities() -> str:
    platforms = {}
    for key, p in PLATFORMS.items():
        safe = p.safe_at(p.width, p.height)
        platforms[key] = {
            "label": p.label,
            "canvas": f"{p.width}x{p.height}",
            "aspect": p.aspect_label,
            "fps": p.fps,
            "max_duration_s": p.max_duration_s,
            "sweet_spot_s": list(p.sweet_spot_s),
            "safe_zone_px": {
                "top": safe.top, "bottom": safe.bottom,
                "left": safe.left, "right": safe.right,
            },
        }
    return json.dumps({
        "platforms": platforms,
        "caption_styles": sorted(CAPTION_PRESETS),
        "grades": list_grades(),
        "reframe_modes": ["track", "static", "center", "blur_pad", "fit"],
        "qualities": sorted(QUALITY),
    }, indent=2)


@server.tool(
    description=(
        "Check the environment: ffmpeg, face tracking, and which transcription "
        "backend is available. Call this first on a cold start to learn what the "
        "machine can actually do before promising it."
    )
)
async def check_environment(directory: str = ".") -> str:
    from .reframe import vision_available
    from .transcribe import whisper_available

    try:
        root = _resolve_dir(directory)
        load_dotenv(root / ".env")
    except ValueError:
        root = Path.cwd()

    try:
        backend = resolve_backend("auto")
    except TranscriptionError as e:
        backend = f"unavailable — {e}"

    return json.dumps({
        "ffmpeg": shutil.which("ffmpeg") or "MISSING (required)",
        "ffprobe": shutil.which("ffprobe") or "MISSING (required)",
        "npx": shutil.which("npx") or "missing — HyperFrames overlay slots unavailable",
        "face_tracking": vision_available(),
        "local_whisper": whisper_available(),
        "transcription_backend": backend,
        "reelforge_version": __version__,
    }, indent=2)


# --- Editing ----------------------------------------------------------------


@server.tool(
    description=(
        "Build an EDL by trimming dead air, using silence detection on the "
        "waveform. Needs no transcript, no model and no network. This is the fast "
        "route for a single take that just needs tightening and verticalising. "
        "Writes edl.json and returns the ranges it chose plus how much it removed. "
        "Review the ranges before rendering."
    )
)
async def autocut(
    source: str,
    directory: str = ".",
    platform: str = "reels",
    reframe: str = "track",
    grade: str = "none",
    noise_db: float = -32.0,
    min_silence: float = 0.35,
    pad: float = 0.08,
    output: str = "edl.json",
) -> str:
    from .autocut import autocut as run_autocut

    try:
        root = _resolve_dir(directory)
        src = _resolve_file(source, root)
        edl, stats = await anyio.to_thread.run_sync(
            lambda: run_autocut(
                src, platform=platform, reframe=reframe, grade=grade,
                noise_db=noise_db, min_silence=min_silence, pad=pad,
            )
        )
        out_path = root / output
        edl.save(out_path)
    except Exception as e:  # noqa: BLE001
        return _err(e)

    return json.dumps({
        "edl_path": str(out_path),
        "stats": stats,
        "ranges": [
            {"start": r.start, "end": r.end, "duration": round(r.duration, 3)}
            for r in edl.ranges
        ],
        "next": "review the ranges, then call lint_edl and render",
    }, indent=2)


@server.tool(
    description=(
        "Transcribe every media file in the directory to word-level timings, "
        "cached per source. backend 'auto' prefers hosted ElevenLabs Scribe when "
        "ELEVENLABS_API_KEY is set, else local Whisper. Scribe is verbatim and "
        "keeps filler words, which is what you select takes on; Whisper needs no "
        "key but normalises them away. Slow on first run — the local model "
        "downloads, and transcription is CPU-bound."
    )
)
async def transcribe(
    directory: str = ".",
    backend: str = "auto",
    model: str = WHISPER_MODEL,
    force: bool = False,
    speakers: int | None = None,
) -> str:
    from .transcribe import find_media, transcribe_dir

    try:
        root = _resolve_dir(directory)
        load_dotenv(root / ".env")
        media = find_media(root)
        if not media:
            return f"no media files found in {root}"
        results = await anyio.to_thread.run_sync(
            lambda: transcribe_dir(
                media, _work(root) / "transcripts", backend=backend,
                model_size=model, force=force, num_speakers=speakers,
            )
        )
    except Exception as e:  # noqa: BLE001
        return _err(e)

    return json.dumps({
        "backend": results[0].backend if results else backend,
        "transcripts": [
            {"source": r.source.name, "words": r.words, "cached": r.cached}
            for r in results
        ],
        "next": "call pack_takes to read the transcripts as a cut-planning view",
    }, indent=2)


@server.tool(
    description=(
        "Return the packed transcript view for this project: every take as "
        "timestamped phrases, with silences marked as cut candidates. This is the "
        "primary artifact for planning a cut — read it and choose ranges from its "
        "timestamps, which already sit on word boundaries. Returns the content "
        "inline; do not read frames to understand the footage."
    )
)
async def pack_takes(directory: str = ".", intent: str = "") -> str:
    from .pack import pack_project

    try:
        root = _resolve_dir(directory)
        transcripts = _work(root) / "transcripts"
        if not transcripts.exists() or not any(transcripts.glob("*.json")):
            return (
                f"no transcripts in {transcripts}. Call transcribe first, or drop "
                "word-level transcript JSON there as <source>.json."
            )
        text, _ = await anyio.to_thread.run_sync(
            lambda: pack_project(transcripts, root, intent)
        )
        (_work(root) / "takes.md").write_text(text)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return text


@server.tool(
    description=(
        "Validate an EDL and write it to disk. Pass the EDL as a JSON string. "
        "Validation is exhaustive and reports every problem at once, so a rejected "
        "EDL can be fixed in a single pass. See the reelforge README for the schema; "
        "at minimum: sources, ranges (source/start/end), platform."
    )
)
async def write_edl(edl_json: str, directory: str = ".", output: str = "edl.json") -> str:
    try:
        root = _resolve_dir(directory)
        data = json.loads(edl_json)
        edl = EDL.from_dict(data)
        edl.validate(base_dir=root)
        path = root / output
        edl.save(path)
    except json.JSONDecodeError as e:
        return f"edl_json is not valid JSON: {e}"
    except Exception as e:  # noqa: BLE001
        return _err(e)

    return json.dumps({
        "edl_path": str(path),
        "platform": edl.platform,
        "segments": len(edl.ranges),
        "total_duration_s": round(edl.total_duration, 3),
        "next": "call lint_edl, then render",
    }, indent=2)


@server.tool(
    description=(
        "Analyse an EDL before rendering and return a retention report: hook "
        "latency and filler openers, pace, dead air, shot-length monotony, caption "
        "legibility, overlay safe-zone violations, and source upscaling. Every "
        "finding carries a concrete fix. Pass `rendered` to additionally check a "
        "finished file for duration drift and geometry. Run this before every "
        "final render and tell the user what it found."
    )
)
async def lint_edl(
    edl_path: str = "edl.json", directory: str = ".", rendered: str | None = None
) -> str:
    from .retention import lint

    try:
        root = _resolve_dir(directory)
        path = _resolve_file(edl_path, root)
        edl = EDL.load(path)
        rendered_path = _resolve_file(rendered, root) if rendered else None
        report = await anyio.to_thread.run_sync(
            lambda: lint(edl, base_dir=root, work_dir=_work(root), rendered=rendered_path)
        )
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return report.format()


@server.tool(
    description=(
        "Render an EDL to a finished video. Quality 'draft' is for checking cut "
        "points, 'preview' is full resolution and honest about how it will look, "
        "'final' is for posting. Runs the full pipeline: subject-tracked reframing "
        "to vertical, colour grade, overlay compositing, burned captions, music "
        "mixing and loudness normalisation. Can take minutes on long timelines."
    )
)
async def render(
    edl_path: str = "edl.json",
    directory: str = ".",
    output: str = "final.mp4",
    quality: str = "preview",
    platform: str | None = None,
    captions: bool = True,
) -> str:
    from .render import render as run_render

    try:
        root = _resolve_dir(directory)
        path = _resolve_file(edl_path, root)
        edl = EDL.load(path)
        if platform:
            edl.platform = platform
        out = Path(output)
        out = out if out.is_absolute() else root / out
        result = await anyio.to_thread.run_sync(
            lambda: run_render(
                edl, out, quality=quality, base_dir=root, work_dir=_work(root),
                build_captions=captions, verbose=False,
            )
        )
    except Exception as e:  # noqa: BLE001
        return _err(e)

    return json.dumps({
        "output": str(result.output),
        "duration_s": round(result.duration, 3),
        "size": f"{result.width}x{result.height}",
        "segments": result.segments,
        "caption_cues": result.caption_cues,
        "reframe": [
            {
                "mode": p.mode,
                "detector": p.detector,
                "keyframes": len(p.keys),
                "pan_px": round(p.motion_px),
                "notes": p.notes,
            }
            for p in result.reframe_plans
        ],
        "size_mb": round(result.output.stat().st_size / (1024 * 1024), 2),
        "next": "call lint_edl with rendered=<output> to verify the finished file",
    }, indent=2)


# --- Motion graphics --------------------------------------------------------


@server.tool(
    description=(
        "Scaffold a HyperFrames overlay slot sized to the delivery target, with "
        "the platform safe area already inset. Returns the composition path and a "
        "self-contained brief for building it. Spawn one sub-agent per slot, in "
        "parallel. Requires Node.js."
    )
)
async def create_overlay_slot(
    slot_id: str,
    goal: str,
    duration: float = 4.0,
    directory: str = ".",
    platform: str = "reels",
    text: str = "REPLACE ME",
) -> str:
    from .overlays import create_slot, slot_brief

    try:
        root = _resolve_dir(directory)
        target = get_platform(platform)
        slot = await anyio.to_thread.run_sync(
            lambda: create_slot(
                slot_id, _work(root) / "slots", target, duration, placeholder=text
            )
        )
        brief = slot_brief(slot, target, goal)
    except Exception as e:  # noqa: BLE001
        return _err(e)

    return json.dumps({
        "slot_id": slot.slot_id,
        "composition": str(slot.composition),
        "canvas": f"{slot.width}x{slot.height}",
        "duration_s": slot.duration,
        "brief": brief,
        "next": "build the composition, then call render_overlay_slot",
    }, indent=2)


@server.tool(
    description=(
        "Render a built overlay slot to WebM with alpha, ready to composite. "
        "Returns the overlay entry to add to the EDL's overlays array."
    )
)
async def render_overlay_slot(
    slot_id: str,
    start_in_output: float,
    directory: str = ".",
    platform: str = "reels",
    duration: float = 4.0,
    anchor: str = "center",
) -> str:
    from .overlays import create_slot, lint_slot, render_slot

    try:
        root = _resolve_dir(directory)
        target = get_platform(platform)
        slot = create_slot(slot_id, _work(root) / "slots", target, duration)
        lint_output = await anyio.to_thread.run_sync(lambda: lint_slot(slot))
        path = await anyio.to_thread.run_sync(lambda: render_slot(slot))
        overlay = slot.to_overlay(start_in_output, anchor=anchor)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        return _err(e)

    return json.dumps({
        "render": str(path),
        "lint": lint_output.strip()[-500:],
        "overlay_entry": {
            "file": overlay.file,
            "start_in_output": overlay.start_in_output,
            "duration": overlay.duration,
            "anchor": overlay.anchor,
            "alpha": overlay.alpha,
            "ignore_safe_area": overlay.ignore_safe_area,
        },
    }, indent=2)


def main() -> None:
    """Entry point for `reelforge-mcp`."""
    server.run("stdio")


if __name__ == "__main__":
    main()
