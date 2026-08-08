"""Cut candidates from the waveform alone — no transcript, no model, no network.

Transcript-driven editing is better when a transcript is available: it lets the
agent choose between takes on what was actually said. But it is not always
available, and the most common editing request does not need it. "Tighten this
up" is a waveform problem — find the dead air, remove it, keep everything else.

`silencedetect` gives the spans; the work is in what you do with them:

- **Pad outward.** Cutting exactly at the detected boundary clips the attack of
  the first consonant and the decay of the last, which is the characteristic
  sound of a badly auto-edited video. A little silence on each side is what
  makes a jump cut sound deliberate.
- **Keep a floor of silence.** Removing every pause entirely produces a
  breathless, unnatural read. The gap is shortened, not eliminated.
- **Discard fragments.** A 120ms island between two long silences is a breath
  or a click, not speech, and cutting to it reads as a glitch.

This yields a first-pass EDL you render immediately or hand-tune afterwards.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .edl import EDL, Range
from .ffmpeg import probe

#: Level below which audio counts as silence. -32dBFS sits under room tone and
#: breath on typical recorded speech without swallowing quiet delivery.
NOISE_FLOOR_DB = -32.0

#: Shortest pause treated as a cut candidate. Below roughly a third of a second
#: you are inside the natural rhythm of a sentence, not between phrases.
MIN_SILENCE = 0.35

#: Silence preserved on each side of a kept span. Protects consonant attack and
#: decay, and keeps the result sounding edited rather than clipped.
EDGE_PAD = 0.08

#: Kept spans shorter than this are fragments — breaths, lip noise, a chair.
MIN_KEEP = 0.30

_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


@dataclass
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_silences(
    source: Path,
    *,
    noise_db: float = NOISE_FLOOR_DB,
    min_silence: float = MIN_SILENCE,
) -> list[Span]:
    """Run `silencedetect` and parse the spans it reports.

    ffmpeg writes the results to stderr as a log, not as structured output, so
    this parses the log. A trailing `silence_start` with no matching end means
    the file ends in silence; it is closed at the duration.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(source),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
         "-vn", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    log = proc.stderr or ""

    spans: list[Span] = []
    pending: float | None = None
    for line in log.splitlines():
        if (m := _SILENCE_START.search(line)) is not None:
            pending = max(0.0, float(m.group(1)))
        if (m := _SILENCE_END.search(line)) is not None and pending is not None:
            end = float(m.group(1))
            if end > pending:
                spans.append(Span(pending, end))
            pending = None
    if pending is not None:
        spans.append(Span(pending, probe(source).duration))
    return spans


def keep_spans(
    duration: float,
    silences: list[Span],
    *,
    pad: float = EDGE_PAD,
    min_keep: float = MIN_KEEP,
) -> list[Span]:
    """Invert silences into padded, merged, fragment-free keep spans."""
    if duration <= 0:
        return []
    if not silences:
        return [Span(0.0, duration)]

    # Complement of the silence spans.
    raw: list[Span] = []
    cursor = 0.0
    for s in sorted(silences, key=lambda s: s.start):
        if s.start > cursor:
            raw.append(Span(cursor, min(s.start, duration)))
        cursor = max(cursor, s.end)
    if cursor < duration:
        raw.append(Span(cursor, duration))

    # Pad outward into the surrounding silence, then merge anything that now
    # overlaps — padding two spans separated by a short gap can close it.
    padded: list[Span] = []
    for s in raw:
        start = max(0.0, s.start - pad)
        end = min(duration, s.end + pad)
        if padded and start <= padded[-1].end:
            padded[-1].end = max(padded[-1].end, end)
        else:
            padded.append(Span(start, end))

    return [s for s in padded if s.duration >= min_keep]


def autocut(
    source: Path,
    *,
    name: str | None = None,
    platform: str = "reels",
    noise_db: float = NOISE_FLOOR_DB,
    min_silence: float = MIN_SILENCE,
    pad: float = EDGE_PAD,
    min_keep: float = MIN_KEEP,
    reframe: str = "track",
    grade: str = "none",
    captions: bool = False,
) -> tuple[EDL, dict[str, float]]:
    """Build a tightened EDL from one source using silence detection alone.

    Returns the EDL and a summary of what it removed, so the caller can report
    the saving rather than leaving the user to diff two durations.
    """
    info = probe(source)
    name = name or source.stem
    if not info.has_audio:
        raise ValueError(
            f"{source.name} has no audio track — autocut works from the waveform. "
            "Write ranges by hand, or use a source with sound."
        )

    silences = detect_silences(source, noise_db=noise_db, min_silence=min_silence)
    spans = keep_spans(info.duration, silences, pad=pad, min_keep=min_keep)
    if not spans:
        raise ValueError(
            f"no audio above {noise_db}dB found in {source.name}. "
            "Try a lower --noise threshold, or check the clip actually has sound."
        )

    edl = EDL(
        sources={name: source.name},
        ranges=[
            Range(
                source=name,
                start=round(s.start, 3),
                end=round(s.end, 3),
                # The first kept span is doing the hook's job whether or not
                # anyone chose it; labelling it lets the linter say so.
                beat="HOOK" if i == 0 else "",
                reason="silence-trimmed span",
            )
            for i, s in enumerate(spans)
        ],
        platform=platform,
        reframe=reframe,
        grade=grade,
        intent="auto-cut from silence detection",
    )
    edl.captions.enabled = captions

    kept = sum(s.duration for s in spans)
    stats = {
        "source_duration": round(info.duration, 2),
        "kept_duration": round(kept, 2),
        "removed_s": round(info.duration - kept, 2),
        "removed_pct": round(100 * (1 - kept / info.duration), 1) if info.duration else 0.0,
        "cuts": len(spans),
        "silences_found": len(silences),
    }
    return edl, stats
