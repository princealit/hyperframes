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
    TranscriptionError,
    find_media,
    load_dotenv,
    transcribe_dir,
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
    print(f"transcribing {len(media)} source(s) -> {out_dir}")
    try:
        results = transcribe_dir(
            media, out_dir, workers=args.workers, force=args.force,
            num_speakers=args.speakers,
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

    try:
        import requests  # noqa: F401
        asr = True
    except ImportError:
        asr = False
    print(f"  {'ok  ' if asr else 'warn'}  {'requests (ASR)':<22}"
          f"{'available' if asr else 'missing — install reelforge[transcribe]'}")

    import os
    key = bool(os.environ.get("ELEVENLABS_API_KEY"))
    print(f"  {'ok  ' if key else 'warn'}  {'ELEVENLABS_API_KEY':<22}"
          f"{'set' if key else 'unset — transcription will fail'}")

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
