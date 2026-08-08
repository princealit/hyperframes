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
from mcp.server.mcpserver import Image, MCPServer

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
        "Three routes.\n"
        "1. Existing footage, one take: `autocut` then `render`. No transcript needed.\n"
        "2. Existing footage, multiple takes or captions wanted: `transcribe`, "
        "`pack_takes`, read the returned transcript, `write_edl`, `lint_edl`, `render`.\n"
        "3. No footage: `generate_asset` for images/video/speech/music, "
        "`still_to_clip` for stills, then assemble as usual.\n"
        "4. A reference likeness saying new words: `avatar_preflight` first, then "
        "`clone_voice`, `speak_as` to check pronunciation, then `talking_head`. "
        "Its output is ordinary footage \u2014 edit it like any other source.\n\n"
        "Use `timeline_view` to LOOK at footage when the transcript cannot settle a "
        "question, and `review_cuts` on a render before showing it to the user.\n\n"
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
    from .transitions import catalog as transition_catalog

    return json.dumps({
        "platforms": platforms,
        "caption_styles": sorted(CAPTION_PRESETS),
        "grades": list_grades(),
        "reframe_modes": ["track", "static", "center", "blur_pad", "fit"],
        "transitions": transition_catalog(),
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


# --- Visual drill-down ------------------------------------------------------


@server.tool(
    description=(
        "LOOK at a time range: returns a PNG of evenly spaced frames with "
        "timestamps burned in, over the waveform for the same range. Use it at "
        "decision points the transcript cannot settle — did the subject stay in "
        "frame, is there a flash at this cut, which take is framed better, did "
        "the reframe hold. It is a drill-down, not a scan: sampling a whole "
        "timeline this way is slow and mostly shows nothing."
    ),
    # Returns image content, which has no meaningful JSON output schema; the
    # SDK derives one from the return annotation otherwise, and Image is not
    # a type pydantic can build a schema for.
    structured_output=False,
)
async def timeline_view(
    source: str,
    start: float,
    end: float,
    directory: str = ".",
    frames: int = 8,
) -> Image:
    from .timeline_view import timeline_view as make_view

    root = _resolve_dir(directory)
    src = _resolve_file(source, root)
    out = _work(root) / "views" / f"{src.stem}_{start:.2f}_{end:.2f}.png"
    view = await anyio.to_thread.run_sync(
        lambda: make_view(src, start, end, out, frames=frames)
    )
    return Image(path=str(view.path))


@server.tool(
    description=(
        "Self-review a rendered file at its cut boundaries. Renders one "
        "filmstrip per seam (plus or minus a window) and returns them, so "
        "flashes, jump cuts and lost subjects are visible rather than assumed. "
        "Run this before showing a render to the user."
    ),
    # Returns image content, which has no meaningful JSON output schema; the
    # SDK derives one from the return annotation otherwise, and Image is not
    # a type pydantic can build a schema for.
    structured_output=False,
)
async def review_cuts(
    rendered: str,
    edl_path: str = "edl.json",
    directory: str = ".",
    window: float = 1.2,
    max_seams: int = 6,
) -> list[Image | str]:
    from .timeline_view import cut_boundaries
    from .timeline_view import timeline_view as make_view

    try:
        root = _resolve_dir(directory)
        video = _resolve_file(rendered, root)
        edl = EDL.load(_resolve_file(edl_path, root))
    except Exception as e:  # noqa: BLE001
        return [_err(e)]

    seams = cut_boundaries(edl, _work(root), window=window)
    if not seams:
        return ["single segment — no seams to review"]

    out: list[Image | str] = []
    shown = seams[:max_seams]
    if len(seams) > len(shown):
        out.append(f"{len(seams)} seams; showing the first {len(shown)}")
    for i, (a, b) in enumerate(shown, start=1):
        path = _work(root) / "views" / f"seam_{i:02d}.png"
        try:
            view = await anyio.to_thread.run_sync(
                lambda a=a, b=b, path=path: make_view(video, a, b, path, frames=6)
            )
        except Exception as e:  # noqa: BLE001
            out.append(f"seam {i} at {a:.2f}s could not be rendered: {e}")
            continue
        out.append(f"seam {i}: {a:.2f}s - {b:.2f}s")
        out.append(Image(path=str(view.path)))
    return out


# --- Generation -------------------------------------------------------------


@server.tool(
    description=(
        "List asset generation providers and whether each is usable right now. "
        "Offline providers ('mock' placeholders, 'espeak' local speech) need no "
        "key and always work; cloud providers need their API key in the "
        "environment. Call this before promising generated footage."
    )
)
async def list_generation_providers(kind: str | None = None) -> str:
    from .generate import available_providers

    return json.dumps(available_providers(kind), indent=2)  # type: ignore[arg-type]


@server.tool(
    description=(
        "Generate an asset — image, video, speech or music — from a prompt. "
        "provider 'auto' uses a cloud provider when its key is present and "
        "falls back to an offline one, so this always produces a file; check "
        "the returned provider to see which ran. Results are cached by request "
        "hash, so repeating an identical prompt costs nothing. For speech the "
        "returned duration is real and is what you should time picture against."
    )
)
async def generate_asset(
    kind: str,
    prompt: str,
    directory: str = ".",
    duration: float = 4.0,
    provider: str = "auto",
    width: int = 1080,
    height: int = 1920,
    options: dict[str, Any] | None = None,
) -> str:
    from .generate import GenRequest, generate

    try:
        root = _resolve_dir(directory)
        load_dotenv(root / ".env")
        req = GenRequest(
            kind=kind,  # type: ignore[arg-type]
            prompt=prompt, duration=duration,
            width=width, height=height, options=options or {},
        )
        asset = await anyio.to_thread.run_sync(
            lambda: generate(req, _work(root), provider=provider)
        )
    except Exception as e:  # noqa: BLE001
        return _err(e)

    return json.dumps({
        "path": str(asset.path),
        "kind": asset.kind,
        "provider": asset.provider,
        "duration_s": round(asset.duration, 3),
        "cached": asset.cached,
        "next": (
            "for an image, call still_to_clip to make it usable on the timeline"
            if asset.kind == "image" else
            "reference this path from the EDL as a source, overlay or music track"
        ),
    }, indent=2)


@server.tool(
    description=(
        "Turn a still image into a clip the timeline can use, with a slow push "
        "applied by default. A motionless still in a feed reads as a loading "
        "error; the drift is what makes it read as a shot."
    )
)
async def still_to_clip(
    image: str,
    duration: float,
    directory: str = ".",
    output: str | None = None,
    zoom: bool = True,
) -> str:
    from .generate import still_to_clip as make_clip

    try:
        root = _resolve_dir(directory)
        src = _resolve_file(image, root)
        out = root / (output or f"{src.stem}_clip.mp4")
        path = await anyio.to_thread.run_sync(
            lambda: make_clip(src, out, duration, zoom=zoom)
        )
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return json.dumps({
        "path": str(path),
        "duration_s": duration,
        "next": "add it to the EDL sources and reference it from a range",
    }, indent=2)


# --- Talking heads ----------------------------------------------------------


@server.tool(
    description=(
        "Test every avatar/lipsync credential and report which actually work: "
        "Fish (voice cloning), sync.so (lipsync), Replicate, OpenRouter, "
        "Together, and whether local files can be exposed as URLs. Call this "
        "FIRST — this stack has several independent providers that all fail "
        "the same way at the point of use, and finding out here costs one call "
        "instead of a half-built pipeline."
    )
)
async def avatar_preflight() -> str:
    from .avatar import preflight

    try:
        root = _resolve_dir(".")
        load_dotenv(root / ".env")
    except Exception:  # noqa: BLE001
        pass
    report = await anyio.to_thread.run_sync(preflight)
    return json.dumps(report, indent=2)


@server.tool(
    description=(
        "Register a voice for cloning from a reference recording. Wants 15-30s "
        "of clean speech — more is not better, and noise or music is worse. "
        "Supply reference_text (what the recording actually says, verbatim): "
        "cloning is in-context, so the transcript tells the model which sounds "
        "map to which graphemes, and quality drops noticeably without it. "
        "Normalises and trims the audio, then saves a reusable voice profile."
    )
)
async def clone_voice(
    name: str,
    reference_audio: str,
    reference_text: str = "",
    directory: str = ".",
) -> str:
    from .avatar import VoiceProfile, prepare_reference_audio

    try:
        root = _resolve_dir(directory)
        src = _resolve_file(reference_audio, root)
        voices = _work(root) / "voices"
        prepared = await anyio.to_thread.run_sync(
            lambda: prepare_reference_audio(src, voices / f"{name}.wav")
        )
        voice = VoiceProfile(
            name=name, reference_audio=prepared, reference_text=reference_text
        )
        path = voice.save(voices / f"{name}.json")
    except Exception as e:  # noqa: BLE001
        return _err(e)

    from .ffmpeg import media_duration

    return json.dumps({
        "voice": name,
        "profile": str(path),
        "reference_seconds": round(media_duration(prepared), 2),
        "has_transcript": bool(reference_text),
        "warning": (
            None if reference_text else
            "no reference_text given — the clone will be measurably worse"
        ),
        "next": "call speak_as to test it, or talking_head to make a video",
    }, indent=2)


@server.tool(
    description=(
        "Speak text in a cloned voice and return the audio path. Language is "
        "inferred from the script itself rather than set as a parameter, so "
        "Persian text in Persian script produces Persian — no language flag and "
        "no transliteration into a neighbouring language. Use this to check "
        "pronunciation before spending anything on video."
    )
)
async def speak_as(
    voice: str,
    text: str,
    directory: str = ".",
    output: str | None = None,
    model: str = "s1",
) -> str:
    from .avatar import VoiceProfile, speak
    from .ffmpeg import media_duration

    try:
        root = _resolve_dir(directory)
        load_dotenv(root / ".env")
        profile = VoiceProfile.load(_work(root) / "voices" / f"{voice}.json")
        out = Path(output) if output else _work(root) / "voices" / f"{voice}_take.wav"
        out = out if out.is_absolute() else root / out
        await anyio.to_thread.run_sync(lambda: speak(text, profile, out, model=model))
    except Exception as e:  # noqa: BLE001
        return _err(e)

    return json.dumps({
        "audio": str(out),
        "duration_s": round(media_duration(out), 2),
        "voice": voice,
    }, indent=2)


@server.tool(
    description=(
        "Turn a reference photo or video plus a script into a video of that "
        "likeness speaking it, in a cloned voice. A video reference keeps the "
        "original body movement; a still is animated first, because lipsyncing "
        "a motionless photo animates a mouth on a mannequin. Speech is "
        "generated before any video work so the driver can be sized to it "
        "rather than truncating the script. Output is an ordinary MP4 that "
        "feeds straight into autocut/render."
    )
)
async def talking_head(
    reference: str,
    script: str,
    voice: str,
    directory: str = ".",
    output: str = "talking.mp4",
    animate_seconds: float = 5.0,
) -> str:
    from .avatar import VoiceProfile
    from .avatar import talking_head as run_talking_head

    try:
        root = _resolve_dir(directory)
        load_dotenv(root / ".env")
        ref = _resolve_file(reference, root)
        profile = VoiceProfile.load(_work(root) / "voices" / f"{voice}.json")
        out = Path(output)
        out = out if out.is_absolute() else root / out
        result = await anyio.to_thread.run_sync(
            lambda: run_talking_head(
                ref, script, profile, out,
                work_dir=_work(root) / "avatars",
                animate_seconds=animate_seconds, verbose=False,
            )
        )
    except Exception as e:  # noqa: BLE001
        return _err(e)

    return json.dumps({
        "output": str(result.output),
        "reference_kind": result.reference_kind,
        "duration_s": round(result.duration, 2),
        "voice": result.voice,
        "steps": result.steps,
        "next": "treat this as source footage — autocut, reframe and render it",
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
