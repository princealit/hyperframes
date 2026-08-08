"""Transcripts to `takes.md` — the agent's reading view.

The central economy of this pipeline. Raw word-level JSON for ten minutes of
footage runs to hundreds of kilobytes of timestamps; sampling the frames instead
is worse by orders of magnitude. Neither is a sensible thing to put in front of a
model that only needs to decide which sentences survive.

`takes.md` groups words into phrases, breaking wherever the speaker actually
paused, and prefixes each with its time range. The result reads like a script
with timecodes, costs a few kilobytes, and preserves the one thing cut selection
needs: word-boundary precision, in text.

The agent reads this. It does not read the JSON, and it does not look at frames
unless a specific decision requires it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .captions import Word, find_words
from .ffmpeg import probe

#: A pause at or above this reads as a phrase boundary rather than as the
#: natural micro-gap between words, and is also the cleanest place to cut.
PHRASE_GAP = 0.5

#: Silences at least this long are worth calling out explicitly as cut targets.
NOTABLE_SILENCE = 0.8


@dataclass
class Phrase:
    words: list[Word]
    speaker: str = ""

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class Take:
    name: str
    path: Path
    duration: float
    phrases: list[Phrase] = field(default_factory=list)
    #: Silences worth cutting on: (start, end).
    silences: list[tuple[float, float]] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(p.words) for p in self.phrases)

    @property
    def speech_ratio(self) -> float:
        if self.duration <= 0:
            return 0.0
        spoken = sum(p.end - p.start for p in self.phrases)
        return min(1.0, spoken / self.duration)


def _speaker_of(raw: dict) -> str:
    return str(raw.get("speaker_id") or raw.get("speaker") or "")


def build_phrases(transcript: dict, gap: float = PHRASE_GAP) -> list[Phrase]:
    """Group words into phrases, breaking on pause or speaker change."""
    words = find_words(transcript)
    if not words:
        return []

    # Speaker labels are read from the raw payload alongside the parsed words,
    # matched by index over the same filtering `find_words` applies.
    raw_words = transcript.get("words")
    if raw_words is None:
        raw_words = [w for s in transcript.get("segments") or [] for w in (s.get("words") or [])]
    speakers: list[str] = []
    for w in raw_words or []:
        if not isinstance(w, dict) or w.get("type") == "spacing":
            continue
        text = (w.get("text") or w.get("word") or "").strip()
        if not text or w.get("start") is None or w.get("end") is None:
            continue
        speakers.append(_speaker_of(w))
    if len(speakers) != len(words):
        speakers = [""] * len(words)

    phrases: list[Phrase] = []
    current: list[Word] = [words[0]]
    current_speaker = speakers[0]
    for i in range(1, len(words)):
        pause = words[i].start - words[i - 1].end
        if pause >= gap or speakers[i] != current_speaker:
            phrases.append(Phrase(current, current_speaker))
            current = [words[i]]
            current_speaker = speakers[i]
        else:
            current.append(words[i])
    phrases.append(Phrase(current, current_speaker))
    return phrases


def find_silences(
    phrases: list[Phrase], duration: float, minimum: float = NOTABLE_SILENCE
) -> list[tuple[float, float]]:
    """Gaps between phrases long enough to be worth cutting on."""
    out: list[tuple[float, float]] = []
    if not phrases:
        return out
    if phrases[0].start >= minimum:
        out.append((0.0, phrases[0].start))
    for a, b in zip(phrases, phrases[1:]):
        if b.start - a.end >= minimum:
            out.append((a.end, b.start))
    if duration and duration - phrases[-1].end >= minimum:
        out.append((phrases[-1].end, duration))
    return out


def load_take(name: str, transcript_path: Path, media_path: Path | None = None) -> Take:
    transcript = json.loads(transcript_path.read_text())
    duration = 0.0
    if media_path and media_path.exists():
        try:
            duration = probe(media_path).duration
        except Exception:  # noqa: BLE001 — duration is informational here
            duration = 0.0
    phrases = build_phrases(transcript)
    if not duration and phrases:
        duration = phrases[-1].end
    return Take(
        name=name,
        path=media_path or transcript_path,
        duration=duration,
        phrases=phrases,
        silences=find_silences(phrases, duration),
    )


def pack(takes: list[Take], intent: str = "") -> str:
    """Render takes as the markdown reading view."""
    total = sum(t.duration for t in takes)
    lines = [
        "# Takes",
        "",
        f"{len(takes)} source(s), {total:.1f}s total.",
        "",
        "Each line is one phrase, prefixed with its time range **in the source**. "
        "Phrase breaks fall on pauses of "
        f"{PHRASE_GAP}s or more, which are also the cleanest places to cut. "
        "Use these timestamps directly as EDL range boundaries — they already sit "
        "on word boundaries.",
        "",
    ]
    if intent:
        lines += [f"**Brief:** {intent}", ""]

    for take in takes:
        lines += [
            f"## {take.name}",
            "",
            f"`{take.path.name}` — {take.duration:.1f}s, {take.word_count} words, "
            f"{len(take.phrases)} phrase{'' if len(take.phrases) == 1 else 's'}, "
            f"{take.speech_ratio:.0%} speech",
            "",
        ]
        if not take.phrases:
            lines += ["_no speech detected_", ""]
            continue
        for p in take.phrases:
            speaker = f"{p.speaker} " if p.speaker else ""
            lines.append(f"  [{p.start:07.2f}-{p.end:07.2f}] {speaker}{p.text}")
        lines.append("")
        if take.silences:
            spans = ", ".join(
                f"{a:.2f}-{b:.2f} ({b - a:.1f}s)" for a, b in take.silences[:12]
            )
            more = f", +{len(take.silences) - 12} more" if len(take.silences) > 12 else ""
            lines += [f"  _cut candidates (silence): {spans}{more}_", ""]

    return "\n".join(lines)


def pack_project(
    transcripts_dir: Path, media_dir: Path, intent: str = ""
) -> tuple[str, list[Take]]:
    """Load every cached transcript and pack them into one markdown view."""
    takes: list[Take] = []
    for tp in sorted(transcripts_dir.glob("*.json")):
        media = next(
            (m for m in media_dir.glob(f"{tp.stem}.*") if m.suffix.lower() != ".json"),
            None,
        )
        takes.append(load_take(tp.stem, tp, media))
    return pack(takes, intent), takes
