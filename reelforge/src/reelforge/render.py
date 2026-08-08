"""EDL to finished file.

The pipeline order is not a preference; each step is where it is because moving
it produces a specific, known defect.

    1. extract each range separately  — tone-map, reframe, grade, retime, fade
    2. concat with -c copy            — no re-encode
    3. composite overlays             — PTS-shifted into their output window
    4. burn captions                  — after overlays, never before
    5. mix music                      — ducked under the dialogue
    6. normalise loudness             — two-pass, to the platform target

Why the order matters:

**Per-segment extract, then stream-copy concat.** A single filtergraph over all
ranges re-encodes the whole timeline again for every overlay pass. Extracting
once and concatenating losslessly means each frame is encoded exactly once in
step 1 and once more in step 3, instead of once per pass.

**Captions after overlays.** An overlay composited on top of burned-in captions
covers them. Nothing errors; the captions are simply gone from the region the
overlay occupies, which is usually noticed after upload.

**PTS shift on overlays.** `overlay` with `enable=between(...)` gates *visibility*
but not playback — the overlay stream keeps running from t=0, so without
`setpts=PTS-STARTPTS+T/TB` an overlay revealed at 8s shows its 8-second mark
rather than its first frame.

**Audio fades at extract time.** A splice between two segments is a step
discontinuity in the waveform, audible as a click at every cut.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import captions as cap
from .config import Platform, Quality, get_quality
from .edl import EDL, Overlay, Range
from .ffmpeg import (
    TONEMAP_CHAIN,
    audio_edge_fades,
    even,
    probe,
    require_ffmpeg,
    run,
)
from .grade import resolve_grade
from .reframe import ReframePlan, plan_reframe


@dataclass
class RenderResult:
    output: Path
    duration: float
    width: int
    height: int
    segments: int
    reframe_plans: list[ReframePlan] = field(default_factory=list)
    caption_cues: int = 0
    captions_path: Path | None = None
    srt_path: Path | None = None
    log: list[str] = field(default_factory=list)


def _log(result: RenderResult, message: str, verbose: bool) -> None:
    result.log.append(message)
    if verbose:
        print(message)


# --- Step 1: per-segment extraction -----------------------------------------


def _speed_filters(speed: float) -> tuple[str, str]:
    """Video and audio retime filters for a playback rate.

    `atempo` is only defined on [0.5, 2.0], so larger changes are decomposed
    into a chain of in-range stages.
    """
    if abs(speed - 1.0) < 1e-3:
        return "", ""
    vf = f"setpts={1 / speed:.6f}*PTS"
    stages: list[float] = []
    remaining = speed
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    af = ",".join(f"atempo={s:.6f}" for s in stages)
    return vf, af


def extract_segment(
    source: Path,
    rng: Range,
    reframe: ReframePlan,
    grade_filter: str,
    out_path: Path,
    platform: Platform,
    quality: Quality,
    out_w: int,
    out_h: int,
    fps: int,
    has_audio: bool,
) -> None:
    """Extract one range as a standalone, fully-treated MP4.

    Everything that can be done per-segment is done here, so the concat in step 2
    is a pure stream copy and the composite in step 3 touches each frame once.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = probe(source)

    chain: list[str] = []
    # Tone-map before anything samples pixel values, so the grade operates on
    # Rec.709 rather than on HDR code points.
    if info.is_hdr:
        chain.append(TONEMAP_CHAIN)
    if reframe.filter_chain:
        chain.append(reframe.filter_chain)
    else:
        chain.append(f"scale={out_w}:{out_h}")
    if grade_filter:
        chain.append(grade_filter)

    vf_speed, af_speed = _speed_filters(rng.speed)
    if vf_speed:
        chain.append(vf_speed)
    chain.append(f"fps={fps}")
    chain.append("format=yuv420p")

    af_parts: list[str] = []
    if af_speed:
        af_parts.append(af_speed)
    af_parts.append(audio_edge_fades(rng.duration))

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        # -ss before -i seeks on keyframes and decodes forward, which is both
        # fast and frame-accurate for the trim that follows.
        "-ss", f"{rng.start:.3f}",
        "-i", str(source),
        "-t", f"{rng.source_duration:.3f}",
        "-vf", ",".join(c for c in chain if c),
        "-c:v", "libx264", "-preset", quality.preset, "-crf", str(quality.crf),
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-video_track_timescale", "90000",
    ]
    if has_audio and info.has_audio:
        cmd += ["-af", ",".join(af_parts), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    elif has_audio:
        # A silent track keeps every segment structurally identical, which the
        # concat demuxer requires; a missing stream on one segment desyncs the
        # rest of the timeline.
        cmd += [
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-shortest", "-c:a", "aac", "-b:a", "192k",
        ]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    run(cmd)


# --- Step 2: concat ---------------------------------------------------------


def concat(segments: list[Path], out_path: Path, work_dir: Path) -> None:
    """Join segments without re-encoding."""
    listing = work_dir / "_concat.txt"
    # Single quotes are the concat demuxer's escape; a path containing one would
    # break the listing, so it is escaped in the demuxer's own dialect.
    listing.write_text(
        "".join(f"file '{str(p.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for p in segments)
    )
    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c", "copy", "-movflags", "+faststart", str(out_path),
    ])
    listing.unlink(missing_ok=True)


# --- Step 3+4: overlays and captions ----------------------------------------


def overlay_position(
    ov: Overlay, platform: Platform, width: int, height: int
) -> tuple[str, str]:
    """Resolve an overlay's anchor into ffmpeg `overlay` x/y expressions.

    Expressed against `W,H` (main) and `w,h` (overlay) rather than resolved to
    numbers, so the overlay's own dimensions never need probing and a scaled
    overlay stays correctly anchored.
    """
    if ov.ignore_safe_area:
        x0, y0, bw, bh = 0, 0, width, height
    else:
        x0, y0, bw, bh = platform.safe_at(width, height).content_box(width, height)

    vertical, _, horizontal = ov.anchor.partition("-")
    if horizontal == "left":
        x = f"{x0}"
    elif horizontal == "right":
        x = f"{x0 + bw}-w"
    else:
        x = f"{x0}+({bw}-w)/2"

    if vertical == "top":
        y = f"{y0}"
    elif vertical == "bottom":
        y = f"{y0 + bh}-h"
    else:
        y = f"{y0}+({bh}-h)/2"

    if ov.dx:
        x = f"({x})+({ov.dx})"
    if ov.dy:
        y = f"({y})+({ov.dy})"
    return x, y


def build_composite(
    base: Path,
    overlays: list[Overlay],
    ass_path: Path | None,
    out_path: Path,
    platform: Platform,
    width: int,
    height: int,
    quality: Quality,
    base_dir: Path,
) -> None:
    """Composite overlays onto the base and burn captions on top of the result."""
    if not overlays and ass_path is None:
        shutil.copy2(base, out_path)
        return

    inputs: list[str] = ["-i", str(base)]
    for ov in overlays:
        p = Path(ov.file)
        inputs += ["-i", str(p if p.is_absolute() else (base_dir / p))]

    safe = platform.safe_at(width, height)
    _, _, safe_w, _ = safe.content_box(width, height)

    parts: list[str] = []
    current = "0:v"

    for i, ov in enumerate(overlays, start=1):
        stream = f"{i}:v"
        stage: list[str] = []
        if ov.scale:
            stage.append(f"scale={even(safe_w * ov.scale)}:-2")
        # yuva420p so alpha survives; overlays without one are unaffected.
        if ov.alpha or ov.opacity < 1 or ov.fade_in or ov.fade_out:
            stage.append("format=yuva420p")
        if ov.opacity < 1:
            stage.append(f"colorchannelmixer=aa={ov.opacity:.3f}")
        # Fades run in the overlay's own timebase, before the PTS shift moves
        # it onto the output timeline.
        if ov.fade_in > 0:
            stage.append(f"fade=t=in:st=0:d={ov.fade_in:.3f}:alpha=1")
        if ov.fade_out > 0:
            stage.append(
                f"fade=t=out:st={max(0.0, ov.duration - ov.fade_out):.3f}:"
                f"d={ov.fade_out:.3f}:alpha=1"
            )
        stage.append(f"setpts=PTS-STARTPTS+{ov.start_in_output:.4f}/TB")
        parts.append(f"[{stream}]{','.join(stage)}[ov{i}]")

        x, y = overlay_position(ov, platform, width, height)
        label = f"cmp{i}"
        parts.append(
            f"[{current}][ov{i}]overlay=x={x}:y={y}:"
            f"enable='between(t,{ov.start_in_output:.3f},{ov.end_in_output:.3f})'"
            f":eof_action=pass[{label}]"
        )
        current = label

    if ass_path is not None:
        escaped = str(ass_path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        parts.append(f"[{current}]ass='{escaped}'[outv]")
        current = "outv"
    elif overlays:
        parts.append(f"[{current}]null[outv]")
        current = "outv"

    cmd = [
        "ffmpeg", "-y", "-v", "error", *inputs,
        "-filter_complex", ";".join(parts),
        "-map", f"[{current}]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", quality.preset, "-crf", str(quality.crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", str(out_path),
    ]
    run(cmd)


# --- Step 5: music ----------------------------------------------------------


def mix_music(
    video: Path, music: Path, out_path: Path, gain_db: float, duck: bool = True
) -> None:
    """Mix a bed under the dialogue, optionally ducking it beneath speech.

    Sidechain compression keyed on the dialogue is what separates a bed you stop
    noticing from one that fights the voice: the music drops a few dB whenever
    someone speaks and recovers in the gaps, with no manual automation.
    """
    if duck:
        filt = (
            f"[1:a]volume={gain_db}dB,aloop=loop=-1:size=2e9[bed];"
            "[0:a]asplit=2[voice][key];"
            "[bed][key]sidechaincompress=threshold=0.05:ratio=6:attack=15:release=350[ducked];"
            "[voice][ducked]amix=inputs=2:duration=first:dropout_transition=0,"
            "alimiter=limit=0.95[aout]"
        )
    else:
        filt = (
            f"[1:a]volume={gain_db}dB,aloop=loop=-1:size=2e9[bed];"
            "[0:a][bed]amix=inputs=2:duration=first:dropout_transition=0,"
            "alimiter=limit=0.95[aout]"
        )
    run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video), "-i", str(music),
        "-filter_complex", filt,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", str(out_path),
    ])


# --- Step 6: loudness -------------------------------------------------------


def normalise_loudness(
    src: Path, out_path: Path, platform: Platform, two_pass: bool = True
) -> bool:
    """Bring integrated loudness to the platform target.

    Every major platform normalises uploads to roughly -14 LUFS. Delivering
    louder means their limiter squashes the mix on the way in; delivering
    quieter means the video is simply quieter than everything around it in the
    feed. Two-pass measures first so the correction is linear rather than a
    dynamic-range guess.
    """
    import json as _json
    import subprocess

    base_filter = f"loudnorm=I={platform.lufs}:TP={platform.true_peak}:LRA=11"

    measured: dict[str, str] | None = None
    if two_pass:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(src),
             "-af", base_filter + ":print_format=json", "-vn", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        stderr = proc.stderr or ""
        start, end = stderr.rfind("{"), stderr.rfind("}")
        if start != -1 and end > start:
            try:
                data = _json.loads(stderr[start : end + 1])
                if {"input_i", "input_tp", "input_lra", "input_thresh"} <= data.keys():
                    measured = data
            except _json.JSONDecodeError:
                measured = None

    if measured:
        filt = (
            f"{base_filter}"
            f":measured_I={measured['input_i']}"
            f":measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured.get('target_offset', 0)}"
            f":linear=true"
        )
    else:
        filt = base_filter

    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-c:v", "copy", "-af", filt,
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", str(out_path),
    ])
    return measured is not None


# --- Orchestration ----------------------------------------------------------


def render(
    edl: EDL,
    out_path: str | Path,
    *,
    work_dir: str | Path | None = None,
    quality: str = "final",
    build_captions: bool = True,
    loudnorm: bool = True,
    verbose: bool = True,
    base_dir: Path | None = None,
) -> RenderResult:
    """Execute an EDL end to end."""
    require_ffmpeg()

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = Path(base_dir) if base_dir else Path.cwd()
    # Rooted at the project, not the output. Transcripts and analysis are
    # properties of the footage and must survive rendering to a new location.
    work = Path(work_dir) if work_dir else base / ".reelforge"
    work.mkdir(parents=True, exist_ok=True)

    q = get_quality(quality)
    platform = edl.target
    width = even(platform.width * q.scale)
    height = even(platform.height * q.scale)
    fps = platform.fps

    result = RenderResult(
        output=out_path, duration=edl.total_duration, width=width, height=height,
        segments=len(edl.ranges),
    )
    _log(result, f"target   {platform.label} {width}x{height}@{fps} ({quality})", verbose)
    _log(result, f"timeline {len(edl.ranges)} segment(s), {edl.total_duration:.2f}s", verbose)

    any_audio = any(probe(edl.resolve_source(r.source, base)).has_audio for r in edl.ranges)

    # 1. extract
    seg_dir = work / f"segments_{quality}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []
    for i, rng in enumerate(edl.ranges):
        src = edl.resolve_source(rng.source, base)
        info = probe(src)
        mode = edl.reframe_for(rng)
        plan = plan_reframe(
            src, rng.start, rng.source_duration, width, height,
            mode=mode, detector=edl.detector, src_size=info.display_size,
        )
        result.reframe_plans.append(plan)
        grade_filter = resolve_grade(edl.grade_for(rng))
        seg_path = seg_dir / f"seg_{i:03d}_{rng.source}.mp4"
        _log(
            result,
            f"  [{i:02d}] {rng.source} {rng.start:7.2f}-{rng.end:7.2f} "
            f"({rng.duration:5.2f}s) {rng.beat:<10} {mode}/{plan.detector}",
            verbose,
        )
        extract_segment(
            src, rng, plan, grade_filter, seg_path, platform, q,
            width, height, fps, any_audio,
        )
        segments.append(seg_path)

    # 2. concat
    base_video = work / f"base_{quality}.mp4"
    concat(segments, base_video, work)
    _log(result, f"concat   {len(segments)} segment(s) -> {base_video.name}", verbose)

    # 3/4. captions then composite
    ass_path: Path | None = None
    if build_captions and edl.captions.enabled:
        style = cap.resolve_style(edl.captions)
        cues = _collect_cues(edl, style, work, base)
        if cues:
            ass_path = work / "captions.ass"
            ass_path.write_text(cap.build_ass(cues, style, platform, width, height))
            srt_path = work / "captions.srt"
            srt_path.write_text(cap.build_srt(cues, style))
            result.caption_cues = len(cues)
            result.captions_path = ass_path
            result.srt_path = srt_path
            _log(result, f"captions {len(cues)} cue(s), style '{style.name}'", verbose)
        else:
            _log(result, "captions requested but no transcripts found — skipping", verbose)

    composited = work / f"composite_{quality}.mp4"
    build_composite(
        base_video, edl.overlays, ass_path, composited, platform,
        width, height, q, base,
    )
    if edl.overlays or ass_path:
        _log(result, f"composite {len(edl.overlays)} overlay(s)"
                     f"{', captions burned' if ass_path else ''}", verbose)

    # 5. music
    staged = composited
    if edl.music and any_audio:
        music_path = Path(edl.music)
        music_path = music_path if music_path.is_absolute() else (base / music_path)
        with_music = work / f"music_{quality}.mp4"
        mix_music(staged, music_path, with_music, edl.music_gain_db)
        staged = with_music
        _log(result, f"music    mixed at {edl.music_gain_db}dB with ducking", verbose)

    # 6. loudness
    if loudnorm and any_audio:
        exact = normalise_loudness(staged, out_path, platform, two_pass=(quality == "final"))
        _log(
            result,
            f"loudness normalised to {platform.lufs} LUFS "
            f"({'two-pass' if exact else 'single-pass'})",
            verbose,
        )
    else:
        shutil.copy2(staged, out_path)

    final = probe(out_path)
    result.duration = final.duration
    result.width, result.height = final.width, final.height
    size_mb = out_path.stat().st_size / (1024 * 1024)
    _log(
        result,
        f"done     {out_path} — {final.duration:.2f}s, "
        f"{final.width}x{final.height}, {size_mb:.1f} MB",
        verbose,
    )
    return result


def _collect_cues(
    edl: EDL, style: cap.CaptionStyle, work: Path, base: Path
) -> list[cap.Cue]:
    """Gather word timings per range and map them onto the output timeline."""
    transcripts_dir = work / "transcripts"
    entries: list[tuple[list[cap.Word], float, float, float]] = []
    cache: dict[str, list[cap.Word]] = {}

    for rng, offset in zip(edl.ranges, edl.offsets()):
        if rng.source not in cache:
            path = transcripts_dir / f"{rng.source}.json"
            cache[rng.source] = cap.load_words(path) if path.exists() else []
        words = cache[rng.source]
        if not words:
            continue
        # Retimed ranges need their word timings compressed by the same factor,
        # or captions drift against the audio they describe.
        if abs(rng.speed - 1.0) > 1e-3:
            s = rng.speed
            words = [
                cap.Word(w.text, rng.start + (w.start - rng.start) / s,
                         rng.start + (w.end - rng.start) / s)
                for w in words
            ]
            entries.append((words, rng.start, rng.start + rng.duration, offset))
        else:
            entries.append((words, rng.start, rng.end, offset))

    return cap.build_cues(entries, style) if entries else []
