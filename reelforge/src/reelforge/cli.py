"""Command-line interface.

The surface is intentionally small and maps one-to-one onto the pipeline stages,
because an agent driving this needs commands whose effects are obvious and whose
output is parseable. Every command is re-runnable: transcription is cached,
scaffolding is idempotent, and rendering always writes to a named output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__
from .captions import PRESETS as CAPTION_PRESETS
from .config import PLATFORMS, QUALITY, get_platform
from .edl import EDL, EDLError, Range
from .ffmpeg import probe
from .grade import list_presets as list_grades
from .pack import pack_project
from .reframe import vision_available
from .retention import lint as lint_edl
from .transcribe import (
    WHISPER_MODEL,
    TranscriptionError,
    find_media,
    load_dotenv,
    resolve_backend,
    transcribe_dir,
    whisper_available,
)

WORK_DIRNAME = ".reelforge"


def _work(root: Path) -> Path:
    return root / WORK_DIRNAME


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


# --- init -------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.directory).resolve()
    if not root.is_dir():
        return _fail(f"{root} is not a directory")

    work = _work(root)
    (work / "transcripts").mkdir(parents=True, exist_ok=True)
    (work / "slots").mkdir(parents=True, exist_ok=True)

    media = find_media(root)
    if not media:
        print(f"no media found in {root}")
        print("  drop your footage here, then run `reelforge init` again")
        return 0

    print(f"{len(media)} source(s) in {root}\n")
    sources: dict[str, str] = {}
    platform = get_platform(args.platform)
    for path in media:
        try:
            info = probe(path)
        except Exception as e:  # noqa: BLE001
            print(f"  {path.name:<32} unreadable ({e})")
            continue
        w, h = info.display_size
        orientation = "vertical" if h > w else ("square" if w == h else "landscape")
        needs = "" if abs(w / h - platform.aspect) < 1e-3 else "  -> will be reframed"
        print(
            f"  {path.name:<32} {w}x{h} {info.fps:.0f}fps "
            f"{info.duration:6.1f}s  {orientation}{needs}"
        )
        sources[path.stem] = path.name

    project = work / "project.md"
    if not project.exists():
        project.write_text(
            f"# Project\n\n**Target:** {platform.label}\n\n"
            f"**Sources:** {len(sources)}\n\n"
            "## Sessions\n\nAppend one section per session: strategy, decisions, "
            "and anything deferred.\n"
        )

    stub = work / "edl.draft.json"
    if not stub.exists() or args.force:
        # A one-range stub over the first source, purely so the shape of the
        # file is in front of whoever writes the real edit.
        ranges: list[Range] = []
        for name, rel in list(sources.items())[:1]:
            try:
                end = min(5.0, probe(root / rel).duration)
            except Exception:  # noqa: BLE001 — already reported in the listing
                end = 5.0
            ranges.append(Range(source=name, start=0.0, end=end, beat="HOOK"))
        draft = EDL(sources=sources, ranges=ranges, platform=args.platform)
        stub.write_text(json.dumps(draft.to_dict(), indent=2) + "\n")

    print(f"\nproject ready. target: {platform.label}")
    print(f"  draft EDL   {stub.relative_to(root)}")
    print("  next        reelforge transcribe")
    return 0


# --- transcribe -------------------------------------------------------------


def cmd_transcribe(args: argparse.Namespace) -> int:
    root = Path(args.directory).resolve()
    load_dotenv(root / ".env")
    load_dotenv(Path.cwd() / ".env")

    media = find_media(root)
    if not media:
        return _fail(f"no media found in {root}")

    out_dir = _work(root) / "transcripts"
    try:
        chosen = resolve_backend(args.backend)
    except TranscriptionError as e:
        return _fail(str(e))

    detail = f"{chosen} ({args.model})" if chosen == "whisper" else chosen
    print(f"transcribing {len(media)} source(s) with {detail} -> {out_dir}")
    if chosen == "whisper":
        print("  local backend: no upload, no key. Filler words are normalised away,")
        print("  which weakens take selection — use scribe when choosing between takes.")
    try:
        results = transcribe_dir(
            media, out_dir, backend=args.backend, workers=args.workers,
            force=args.force, num_speakers=args.speakers, model_size=args.model,
        )
    except TranscriptionError as e:
        return _fail(str(e))

    for r in results:
        state = "cached" if r.cached else "done"
        print(f"  {r.source.name:<32} {r.words:5d} words  ({state})")
    print("\nnext  reelforge pack")
    return 0


# --- pack -------------------------------------------------------------------


def cmd_pack(args: argparse.Namespace) -> int:
    root = Path(args.directory).resolve()
    transcripts = _work(root) / "transcripts"
    if not transcripts.exists() or not any(transcripts.glob("*.json")):
        return _fail(f"no transcripts in {transcripts} — run `reelforge transcribe` first")

    text, takes = pack_project(transcripts, root, args.intent or "")
    out = Path(args.output) if args.output else _work(root) / "takes.md"
    out.write_text(text)

    words = sum(t.word_count for t in takes)
    size_kb = len(text.encode()) / 1024
    print(f"{len(takes)} take(s), {words} words -> {out}  ({size_kb:.1f} KB)")
    print("\nthis is the agent's reading view; write edl.json from it")
    return 0


# --- lint -------------------------------------------------------------------


def cmd_lint(args: argparse.Namespace) -> int:
    edl_path = Path(args.edl).resolve()
    base = Path(args.directory).resolve() if args.directory else edl_path.parent
    try:
        edl = EDL.load(edl_path)
    except EDLError as e:
        print(str(e), file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        return _fail(str(e))

    rendered = Path(args.rendered).resolve() if args.rendered else None
    report = lint_edl(edl, base_dir=base, work_dir=_work(base), rendered=rendered)
    print(report.format())

    if args.json:
        Path(args.json).write_text(json.dumps({
            "score": report.score,
            "stats": report.stats,
            "findings": [
                {"severity": f.severity, "code": f.code, "message": f.message,
                 "fix": f.fix, "at": f.at}
                for f in report.findings
            ],
        }, indent=2) + "\n")

    if args.strict and report.warnings:
        return 1
    return 0 if report.ok else 1


# --- render -----------------------------------------------------------------


def cmd_render(args: argparse.Namespace) -> int:
    from .render import render

    edl_path = Path(args.edl).resolve()
    base = Path(args.directory).resolve() if args.directory else edl_path.parent
    try:
        edl = EDL.load(edl_path)
    except EDLError as e:
        print(str(e), file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        return _fail(str(e))

    if args.platform:
        edl.platform = args.platform
    out = Path(args.output) if args.output else base / f"{args.quality}.mp4"

    try:
        result = render(
            edl, out, quality=args.quality, base_dir=base,
            work_dir=_work(base), loudnorm=not args.no_loudnorm,
            build_captions=not args.no_captions, verbose=True,
        )
    except Exception as e:  # noqa: BLE001 — surfaced to the shell
        return _fail(str(e))

    if not args.no_lint:
        print()
        report = lint_edl(edl, base_dir=base, work_dir=_work(base), rendered=result.output)
        print(report.format())
    return 0


# --- slot -------------------------------------------------------------------


def cmd_slot(args: argparse.Namespace) -> int:
    from .overlays import HyperFramesError, available, create_slot, render_slot, slot_brief

    root = Path(args.directory).resolve()
    platform = get_platform(args.platform)
    slot = create_slot(
        args.slot_id, _work(root) / "slots", platform, args.duration,
        placeholder=args.text, overwrite=args.force,
    )
    print(f"slot        {slot.slot_id}")
    print(f"composition {slot.composition}")
    print(f"canvas      {slot.width}x{slot.height} @ {platform.fps}fps, {slot.duration:.2f}s")

    if args.brief:
        print("\n--- sub-agent brief ---")
        print(slot_brief(slot, platform, args.brief))

    if args.render:
        if not available():
            return _fail("npx not found — Node.js is required to render HyperFrames slots")
        try:
            path = render_slot(slot)
        except HyperFramesError as e:
            return _fail(str(e))
        print(f"\nrendered    {path}")
        print(f"add to the EDL overlays array:")
        print(json.dumps(slot.to_overlay(0.0).__dict__, indent=2))
    return 0


# --- info -------------------------------------------------------------------


def cmd_autocut(args: argparse.Namespace) -> int:
    from .autocut import autocut

    root = Path(args.directory).resolve()
    source = Path(args.source)
    source = source if source.is_absolute() else root / source
    if not source.exists():
        return _fail(f"source not found: {source}")

    try:
        edl, stats = autocut(
            source, platform=args.platform, noise_db=args.noise,
            min_silence=args.min_silence, pad=args.pad, min_keep=args.min_keep,
            reframe=args.reframe, grade=args.grade, captions=args.captions,
        )
    except ValueError as e:
        return _fail(str(e))

    out = Path(args.output) if args.output else root / "edl.json"
    edl.save(out)
    print(
        f"{stats['source_duration']:.2f}s -> {stats['kept_duration']:.2f}s  "
        f"({stats['removed_s']:.2f}s of dead air removed, {stats['removed_pct']:.0f}%)"
    )
    print(f"{stats['cuts']} span(s) kept from {stats['silences_found']} silence(s)")
    print(f"\nwrote {out}")
    print("  review the ranges, then: reelforge render "
          f"{out.name} -q preview -o preview.mp4")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    from .timeline_view import timeline_view

    root = Path(args.directory).resolve()
    source = Path(args.source)
    source = source if source.is_absolute() else root / source
    if not source.exists():
        return _fail(f"source not found: {source}")

    out = Path(args.output) if args.output else (
        _work(root) / "views" / f"{source.stem}_{args.start:.2f}_{args.end:.2f}.png"
    )
    try:
        view = timeline_view(source, args.start, args.end, out, frames=args.frames)
    except Exception as e:  # noqa: BLE001
        return _fail(str(e))

    print(f"{view.path}  ({view.width}x{view.height}, {view.frames} frames"
          f"{', waveform' if view.has_waveform else ''})")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .generate import GenRequest, available_providers, generate

    root = Path(args.directory).resolve()
    load_dotenv(root / ".env")

    if args.list_providers:
        print(f"{'provider':<12}{'kinds':<26}{'offline':<9}{'usable':<8}requires")
        for p in available_providers():
            print(
                f"{p['name']:<12}{','.join(p['kinds']):<26}"
                f"{str(p['offline']):<9}{str(p['usable']):<8}{p['requires'] or '-'}"
            )
        return 0

    if not args.prompt:
        return _fail("a prompt is required (or pass --list-providers)")

    try:
        asset = generate(
            GenRequest(kind=args.kind, prompt=args.prompt, duration=args.duration),
            _work(root), provider=args.provider,
        )
    except Exception as e:  # noqa: BLE001
        return _fail(str(e))

    state = "cached" if asset.cached else "generated"
    print(f"{asset.path}  ({asset.provider}, {asset.duration:.2f}s, {state})")
    if asset.kind == "image":
        print("  make it usable on the timeline: "
              f"reelforge clip {asset.path} -t 3.0")
    return 0


def cmd_clip(args: argparse.Namespace) -> int:
    from .generate import still_to_clip

    root = Path(args.directory).resolve()
    image = Path(args.image)
    image = image if image.is_absolute() else root / image
    if not image.exists():
        return _fail(f"image not found: {image}")
    out = Path(args.output) if args.output else root / f"{image.stem}_clip.mp4"
    try:
        path = still_to_clip(image, out, args.duration, zoom=not args.no_zoom)
    except Exception as e:  # noqa: BLE001
        return _fail(str(e))
    print(f"{path}  ({args.duration:.2f}s)")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    from .avatar import preflight

    root = Path(args.directory).resolve()
    load_dotenv(root / ".env")
    report = preflight()

    print(f"  {'provider':<20}{'state':<12}detail")
    exit_code = 0
    for name, info in report.items():
        if not info.get("configured"):
            state = "not set"
        elif info.get("ok"):
            state = "ok"
        else:
            state = "FAILING"
            exit_code = 1
        print(f"  {name:<20}{state:<12}{info.get('detail', '')[:64]}")
    return exit_code


def cmd_voice(args: argparse.Namespace) -> int:
    from .avatar import VoiceProfile, prepare_reference_audio
    from .ffmpeg import media_duration

    root = Path(args.directory).resolve()
    src = Path(args.reference)
    src = src if src.is_absolute() else root / src
    if not src.exists():
        return _fail(f"reference audio not found: {src}")

    voices = _work(root) / "voices"
    try:
        prepared = prepare_reference_audio(src, voices / f"{args.name}.wav")
    except Exception as e:  # noqa: BLE001
        return _fail(str(e))

    voice = VoiceProfile(
        name=args.name, reference_audio=prepared, reference_text=args.text or ""
    )
    path = voice.save(voices / f"{args.name}.json")
    print(f"{path}  ({media_duration(prepared):.1f}s reference)")
    if not args.text:
        print("  warning: no --text given. Supply the reference transcript "
              "verbatim — cloning is in-context and is measurably worse without it.")
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    from .avatar import VoiceProfile, speak
    from .ffmpeg import media_duration

    root = Path(args.directory).resolve()
    load_dotenv(root / ".env")
    profile_path = _work(root) / "voices" / f"{args.voice}.json"
    if not profile_path.exists():
        return _fail(f"no voice named {args.voice!r} — create one with `reelforge voice`")

    out = Path(args.output) if args.output else _work(root) / "voices" / f"{args.voice}_take.wav"
    try:
        speak(args.text, VoiceProfile.load(profile_path), out, model=args.model)
    except Exception as e:  # noqa: BLE001
        return _fail(str(e))
    print(f"{out}  ({media_duration(out):.2f}s)")
    return 0


def cmd_talking_head(args: argparse.Namespace) -> int:
    from .avatar import VoiceProfile, talking_head

    root = Path(args.directory).resolve()
    load_dotenv(root / ".env")
    ref = Path(args.reference)
    ref = ref if ref.is_absolute() else root / ref
    profile_path = _work(root) / "voices" / f"{args.voice}.json"
    if not profile_path.exists():
        return _fail(f"no voice named {args.voice!r} — create one with `reelforge voice`")

    script = args.script
    if args.script_file:
        script = Path(args.script_file).read_text().strip()
    if not script:
        return _fail("a script is required (positional, or --script-file)")

    out = Path(args.output) if args.output else root / "talking.mp4"
    try:
        result = talking_head(
            ref, script, VoiceProfile.load(profile_path), out,
            work_dir=_work(root) / "avatars", animate_seconds=args.animate_seconds,
        )
    except Exception as e:  # noqa: BLE001
        return _fail(str(e))

    print(f"\n{result.output}  ({result.duration:.2f}s, from a {result.reference_kind})")
    print("  edit it like any other footage: "
          f"reelforge autocut {result.output.name}")
    return 0


def cmd_platforms(_: argparse.Namespace) -> int:
    print(f"{'key':<12}{'target':<24}{'canvas':<14}{'fps':<6}{'max':<8}sweet spot")
    for key, p in PLATFORMS.items():
        lo, hi = p.sweet_spot_s
        print(
            f"{key:<12}{p.label:<24}{f'{p.width}x{p.height}':<14}{p.fps:<6}"
            f"{f'{p.max_duration_s:.0f}s':<8}{lo:.0f}-{hi:.0f}s"
        )
    print("\ncaption styles:", ", ".join(sorted(CAPTION_PRESETS)))
    print("grades:        ", ", ".join(list_grades()))
    print("reframe modes:  track, static, center, blur_pad, fit")
    print("quality:       ", ", ".join(QUALITY))

    from .transitions import TRANSITIONS

    print("\ntransitions")
    for t in TRANSITIONS.values():
        default = f"{t.default_duration:.2f}s" if t.default_duration else "-"
        print(f"  {t.name:<12}{default:<8}{t.use.split('.')[0]}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    ok = True

    def check(label: str, good: bool, detail: str) -> None:
        nonlocal ok
        print(f"  {'ok  ' if good else 'MISS'}  {label:<22}{detail}")
        ok = ok and good

    print("environment\n")
    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        check(binary, path is not None, path or "not on PATH — required")

    npx = shutil.which("npx")
    print(f"  {'ok  ' if npx else 'warn'}  {'npx':<22}"
          f"{npx or 'not found — HyperFrames overlay slots unavailable'}")

    vision = vision_available()
    print(f"  {'ok  ' if vision else 'warn'}  {'opencv (face track)':<22}"
          f"{'available' if vision else 'missing — falls back to saliency tracking'}")

    import os

    local = whisper_available()
    print(f"  {'ok  ' if local else 'warn'}  {'whisper (local ASR)':<22}"
          f"{'available — no key needed' if local else 'missing — install reelforge[local]'}")

    key = bool(os.environ.get("ELEVENLABS_API_KEY"))
    print(f"  {'ok  ' if key else 'warn'}  {'ELEVENLABS_API_KEY':<22}"
          f"{'set — hosted Scribe available' if key else 'unset — hosted Scribe unavailable'}")

    try:
        backend = resolve_backend("auto")
        print(f"  ok    {'transcription':<22}will use '{backend}'")
    except TranscriptionError:
        ok = False
        print(f"  MISS  {'transcription':<22}no backend available")

    print()
    if not ok:
        print("ffmpeg is required. Install it and re-run.")
        return 1
    print("core pipeline is ready.")
    return 0


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reelforge",
        description="Agentic editor for vertical video.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def with_dir(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sp.add_argument("-d", "--directory", default=".", help="project directory")
        return sp

    sp = with_dir(sub.add_parser("init", help="scan sources and scaffold the project"))
    sp.add_argument("-p", "--platform", default="reels", choices=sorted(PLATFORMS))
    sp.add_argument("-f", "--force", action="store_true", help="overwrite the draft EDL")
    sp.set_defaults(func=cmd_init)

    sp = with_dir(sub.add_parser("transcribe", help="word-level ASR, cached per source"))
    sp.add_argument(
        "-b", "--backend", default="auto", choices=("auto", "scribe", "whisper"),
        help="auto prefers hosted Scribe when a key is set, else local Whisper",
    )
    sp.add_argument(
        "--model", default=WHISPER_MODEL,
        choices=("tiny", "base", "small", "medium", "large-v3"),
        help="local Whisper model size (whisper backend only)",
    )
    sp.add_argument("-w", "--workers", type=int, default=4)
    sp.add_argument("-f", "--force", action="store_true", help="ignore the cache")
    sp.add_argument("--speakers", type=int, default=None, help="known speaker count")
    sp.set_defaults(func=cmd_transcribe)

    sp = with_dir(sub.add_parser("pack", help="transcripts -> takes.md reading view"))
    sp.add_argument("-o", "--output")
    sp.add_argument("--intent", help="one-line brief, embedded in the header")
    sp.set_defaults(func=cmd_pack)

    sp = with_dir(sub.add_parser("lint", help="retention and correctness report"))
    sp.add_argument("edl")
    sp.add_argument("--rendered", help="also check a rendered file")
    sp.add_argument("--json", help="write the report as JSON")
    sp.add_argument("--strict", action="store_true", help="fail on warnings too")
    sp.set_defaults(func=cmd_lint)

    sp = with_dir(sub.add_parser("render", help="render an EDL"))
    sp.add_argument("edl")
    sp.add_argument("-o", "--output")
    sp.add_argument("-q", "--quality", default="preview", choices=sorted(QUALITY))
    sp.add_argument("-p", "--platform", choices=sorted(PLATFORMS), help="override the EDL target")
    sp.add_argument("--no-captions", action="store_true")
    sp.add_argument("--no-loudnorm", action="store_true")
    sp.add_argument("--no-lint", action="store_true", help="skip the post-render report")
    sp.set_defaults(func=cmd_render)

    sp = with_dir(sub.add_parser("slot", help="scaffold or render a HyperFrames overlay"))
    sp.add_argument("slot_id")
    sp.add_argument("-t", "--duration", type=float, default=4.0)
    sp.add_argument("-p", "--platform", default="reels", choices=sorted(PLATFORMS))
    sp.add_argument("--text", default="REPLACE ME", help="placeholder card text")
    sp.add_argument("--brief", help="print a sub-agent brief for this goal")
    sp.add_argument("--render", action="store_true", help="render the slot to WebM")
    sp.add_argument("-f", "--force", action="store_true", help="overwrite an existing slot")
    sp.set_defaults(func=cmd_slot)

    sp = with_dir(sub.add_parser(
        "autocut", help="build an EDL by trimming dead air (no transcript needed)"
    ))
    sp.add_argument("source", help="the clip to cut")
    sp.add_argument("-o", "--output", help="EDL path (default: edl.json)")
    sp.add_argument("-p", "--platform", default="reels", choices=sorted(PLATFORMS))
    sp.add_argument("--noise", type=float, default=-32.0,
                    help="silence threshold in dBFS (default: -32)")
    sp.add_argument("--min-silence", type=float, default=0.35,
                    help="shortest pause treated as a cut (default: 0.35s)")
    sp.add_argument("--pad", type=float, default=0.08,
                    help="silence kept either side of a span (default: 0.08s)")
    sp.add_argument("--min-keep", type=float, default=0.30,
                    help="discard kept spans shorter than this (default: 0.30s)")
    sp.add_argument("--reframe", default="track",
                    choices=("track", "static", "center", "blur_pad", "fit"))
    sp.add_argument("--grade", default="none")
    sp.add_argument("--captions", action="store_true",
                    help="enable captions (needs a transcript)")
    sp.set_defaults(func=cmd_autocut)

    sp = with_dir(sub.add_parser(
        "view", help="filmstrip + waveform PNG for a time range"
    ))
    sp.add_argument("source")
    sp.add_argument("start", type=float)
    sp.add_argument("end", type=float)
    sp.add_argument("-o", "--output")
    sp.add_argument("-n", "--frames", type=int, default=8)
    sp.set_defaults(func=cmd_view)

    sp = with_dir(sub.add_parser(
        "generate", help="generate an image, video, speech or music asset"
    ))
    sp.add_argument("prompt", nargs="?")
    sp.add_argument("-k", "--kind", default="image",
                    choices=("image", "video", "speech", "music"))
    sp.add_argument("-t", "--duration", type=float, default=4.0)
    sp.add_argument("--provider", default="auto",
                    help="auto prefers a credentialled cloud provider, else offline")
    sp.add_argument("--list-providers", action="store_true")
    sp.set_defaults(func=cmd_generate)

    sp = with_dir(sub.add_parser(
        "clip", help="turn a still into a clip with a slow push"
    ))
    sp.add_argument("image")
    sp.add_argument("-t", "--duration", type=float, default=3.0)
    sp.add_argument("-o", "--output")
    sp.add_argument("--no-zoom", action="store_true", help="hold the frame static")
    sp.set_defaults(func=cmd_clip)

    sp = with_dir(sub.add_parser(
        "preflight", help="test every avatar/generation credential"
    ))
    sp.set_defaults(func=cmd_preflight)

    sp = with_dir(sub.add_parser("voice", help="register a voice for cloning"))
    sp.add_argument("name")
    sp.add_argument("reference", help="15-30s of clean speech")
    sp.add_argument("--text", help="what the reference says, verbatim (strongly advised)")
    sp.set_defaults(func=cmd_voice)

    sp = with_dir(sub.add_parser("say", help="speak text in a cloned voice"))
    sp.add_argument("voice")
    sp.add_argument("text")
    sp.add_argument("-o", "--output")
    sp.add_argument("--model", default="s1")
    sp.set_defaults(func=cmd_say)

    sp = with_dir(sub.add_parser(
        "talking-head", help="reference photo/video + script -> them saying it"
    ))
    sp.add_argument("reference")
    sp.add_argument("script", nargs="?", default="")
    sp.add_argument("--script-file")
    sp.add_argument("-v", "--voice", required=True)
    sp.add_argument("-o", "--output")
    sp.add_argument("--animate-seconds", type=float, default=5.0,
                    help="motion generated from a still before looping")
    sp.set_defaults(func=cmd_talking_head)

    sub.add_parser("platforms", help="list targets, styles and presets").set_defaults(
        func=cmd_platforms
    )
    sub.add_parser("doctor", help="check the environment").set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
