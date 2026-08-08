"""Filmstrip and waveform for a time range — the visual drill-down.

The transcript answers "what was said and when". It cannot answer "did the
subject leave frame", "is there a flash at this cut", or "which of these two
takes is better framed". For those you have to look.

This renders a range as one PNG: evenly spaced frames tiled left to right with
their timestamps burned in, and the waveform for the same range beneath them on
a shared horizontal axis, so a silence in the audio lines up with the frames it
spans.

**It is a drill-down, not a scan.** Sampling a whole timeline this way is slow
and floods the context with images that mostly show nothing. Use it at decision
points — verifying a cut boundary, choosing between takes, checking that a
reframe held the subject.

Over MCP the PNG is returned as an image, so the agent sees it rather than
reading a path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ffmpeg import even, probe, run

#: Frames in a strip. Enough to read the motion of a range without the
#: individual frames becoming too small to judge.
DEFAULT_FRAMES = 8

#: Total strip width. Wide enough that eight tiles stay legible, small enough
#: that the PNG stays a reasonable thing to hand to a model.
DEFAULT_WIDTH = 1600

#: Waveform height as a fraction of the filmstrip's.
WAVE_RATIO = 0.32

#: Gap between tiles. `tile` inserts this between neighbours but not at the
#: outer edges, so the assembled strip is wider than frames x tile_width and the
#: waveform underneath has to match that real width or `vstack` refuses to run.
TILE_PADDING = 2

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def _font() -> str | None:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


@dataclass
class TimelineView:
    path: Path
    start: float
    end: float
    frames: int
    has_waveform: bool
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def timeline_view(
    source: str | Path,
    start: float,
    end: float,
    out_path: str | Path,
    *,
    frames: int = DEFAULT_FRAMES,
    width: int = DEFAULT_WIDTH,
    labels: bool = True,
) -> TimelineView:
    """Render `start`..`end` of `source` as a filmstrip over its waveform."""
    src = Path(source)
    info = probe(src)
    start = max(0.0, start)
    end = min(info.duration, end) if info.duration else end
    if end <= start:
        raise ValueError(f"empty range: {start:.3f}..{end:.3f}")

    duration = end - start
    frames = max(2, min(frames, 16))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tile_w = even(width / frames)
    src_w, src_h = info.display_size
    tile_h = even(tile_w * src_h / src_w) if src_w else even(tile_w * 9 / 16)
    wave_h = even(tile_h * WAVE_RATIO)
    strip_w = frames * tile_w + (frames - 1) * TILE_PADDING

    # Sample at the midpoint of each slice rather than the leading edge, so the
    # first tile is not the frame the previous cut ended on.
    step = duration / frames
    select = "+".join(
        f"eq(n\\,{max(0, int(round((step * (i + 0.5)) * (info.fps or 30))))})"
        for i in range(frames)
    )

    # Built as one chain and given its output pad last — a filter appended after
    # the pad is a syntax error, not a no-op.
    chain = (
        f"[0:v]select='{select}',scale={tile_w}:{tile_h},"
        f"tile={frames}x1:padding={TILE_PADDING}:color=0x101014"
    )
    font = _font()
    if labels and font:
        # One label per tile, drawn onto the assembled strip so the x positions
        # are simple multiples of the tile width plus its padding.
        draws = [
            f"drawtext=fontfile={font}:text='{start + step * (i + 0.5):.2f}s'"
            f":fontcolor=white:fontsize={max(12, tile_h // 14)}"
            f":x={i * (tile_w + TILE_PADDING) + 8}:y=6"
            f":box=1:boxcolor=0x000000AA:boxborderw=4"
            for i in range(frames)
        ]
        chain += "," + ",".join(draws)
    chain += "[strip]"

    parts = [chain]
    strip_label = "strip"
    has_audio = info.has_audio
    if has_audio:
        parts.append(
            f"[0:a]showwavespic=s={strip_w}x{wave_h}:colors=0x5AC8FA"
            f":split_channels=0[wave]"
        )
        parts.append(f"[{strip_label}][wave]vstack=inputs=2[out]")
        final = "out"
    else:
        final = strip_label

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
        "-filter_complex", ";".join(parts),
        "-map", f"[{final}]", "-frames:v", "1", str(out),
    ]
    run(cmd)

    rendered = probe(out) if out.exists() else None
    return TimelineView(
        path=out, start=start, end=end, frames=frames,
        has_waveform=has_audio,
        width=rendered.width if rendered else strip_w,
        height=rendered.height if rendered else tile_h,
    )


def cut_boundaries(edl, work_dir: Path, *, window: float = 1.5) -> list[tuple[float, float]]:
    """Windows around each seam on the output timeline, for self-review.

    The frames either side of a cut are where flashes, jump cuts and
    subject-lost-frame errors show up, so these are the ranges worth actually
    looking at on a rendered file.
    """
    out: list[tuple[float, float]] = []
    total = edl.total_duration
    for offset in edl.offsets()[1:]:
        out.append((max(0.0, offset - window), min(total, offset + window)))
    return out
