"""The edit decision list: the one artifact the agent writes and the renderer reads.

An EDL is the contract between reasoning and execution. The agent decides *what*
the video is — which takes, in what order, framed how, captioned in what style —
and writes it here as data. The renderer decides nothing; it executes.

Keeping that boundary sharp is what makes the pipeline debuggable: a bad video is
either a bad EDL (the agent's judgement) or a bad render of a good EDL (a bug),
and you can tell which by reading one JSON file.

This schema is a superset of video-use's. The additions are the vertical-native
parts: a platform target, per-range reframing, and overlays with real geometry
instead of a bare top-left composite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

from .config import Platform, get_platform

SCHEMA_VERSION = 2

Anchor = Literal[
    "top-left", "top-center", "top-right",
    "center-left", "center", "center-right",
    "bottom-left", "bottom-center", "bottom-right",
]


class EDLError(ValueError):
    """The EDL is structurally invalid. Carries every problem, not just the first."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        joined = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"invalid EDL ({len(problems)} problem(s)):\n{joined}")


@dataclass
class Range:
    """One kept slice of one source, and the reasoning that selected it."""

    source: str
    start: float
    end: float
    #: Structural role — HOOK, PROBLEM, PAYOFF, CTA. Drives the retention report.
    beat: str = ""
    #: The words spoken, for readability when auditing the EDL by eye.
    quote: str = ""
    #: Why this take over the alternatives. Written by the agent, for humans.
    reason: str = ""
    #: Per-range reframe override. Falls back to the EDL-level default.
    reframe: str | None = None
    #: Per-range grade override.
    grade: str | None = None
    #: Playback rate. >1 tightens a slow passage without recutting it.
    speed: float = 1.0

    @property
    def duration(self) -> float:
        return (self.end - self.start) / max(self.speed, 1e-6)

    @property
    def source_duration(self) -> float:
        return self.end - self.start


@dataclass
class Overlay:
    """A composited element: motion graphic, logo, lower third, progress bar.

    Geometry is expressed against the *safe area* by default rather than the raw
    frame. `anchor="bottom-center"` means "bottom-centre of the region the
    platform UI does not cover", so an overlay that validates on Reels does not
    silently slide under the TikTok action rail when the target changes.
    """

    file: str
    start_in_output: float
    duration: float
    anchor: Anchor = "center"
    #: Offsets from the anchor point, in output pixels.
    dx: int = 0
    dy: int = 0
    #: Width as a fraction of the safe-area width. None keeps the native size.
    scale: float | None = None
    opacity: float = 1.0
    #: Treat the source's alpha channel as transparency (WebM/ProRes 4444).
    alpha: bool = True
    #: Anchor within the full frame instead of the safe area. Escape hatch for
    #: deliberate full-bleed elements.
    ignore_safe_area: bool = False
    fade_in: float = 0.0
    fade_out: float = 0.0
    label: str = ""

    @property
    def end_in_output(self) -> float:
        return self.start_in_output + self.duration


@dataclass
class CaptionSpec:
    """Caption configuration. Resolved against a style preset at render time."""

    enabled: bool = True
    style: str = "punch"
    # Every field below is None-by-default and only overrides the named preset
    # when explicitly set. A concrete default here would silently mask the
    # preset it is layered over — a `karaoke` spec would render with `punch`'s
    # word count purely because the dataclass had to pick a number.
    #: Words per displayed cue. 2 reads as punchy, 5+ reads as a subtitle track.
    words_per_cue: int | None = None
    case: Literal["upper", "sentence", "natural"] | None = None
    #: Vertical placement as a fraction of the safe area's height, 0=top 1=bottom.
    position: float | None = None
    #: Accent colour for the highlighted word in karaoke styles.
    highlight: str | None = None
    font: str = ""
    font_size: int | None = None


@dataclass
class EDL:
    """A complete edit, ready to render."""

    sources: dict[str, str]
    ranges: list[Range]
    platform: str = "reels"
    version: int = SCHEMA_VERSION
    #: Default reframe mode; individual ranges may override.
    reframe: str = "track"
    #: Face/saliency/centre/auto. Applies to every tracked range.
    detector: str = "auto"
    grade: str = "none"
    overlays: list[Overlay] = field(default_factory=list)
    captions: CaptionSpec = field(default_factory=CaptionSpec)
    #: Background music, mixed under the dialogue.
    music: str | None = None
    music_gain_db: float = -18.0
    #: Free-text intent, carried for the retention report and session memory.
    intent: str = ""
    title: str = ""

    # -- derived ------------------------------------------------------------

    @property
    def target(self) -> Platform:
        return get_platform(self.platform)

    @property
    def total_duration(self) -> float:
        return sum(r.duration for r in self.ranges)

    def offsets(self) -> list[float]:
        """Output-timeline start time of each range.

        Every timing conversion in the renderer and the caption builder keys off
        this. Getting it wrong is the classic cause of captions that drift
        further out of sync with each successive cut.
        """
        out: list[float] = []
        t = 0.0
        for r in self.ranges:
            out.append(t)
            t += r.duration
        return out

    def reframe_for(self, r: Range) -> str:
        return r.reframe or self.reframe

    def grade_for(self, r: Range) -> str:
        return r.grade or self.grade

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "platform": self.platform,
            "title": self.title,
            "intent": self.intent,
            "sources": dict(self.sources),
            "reframe": self.reframe,
            "detector": self.detector,
            "grade": self.grade,
            "ranges": [asdict(r) for r in self.ranges],
            "overlays": [asdict(o) for o in self.overlays],
            "captions": asdict(self.captions),
            "music": self.music,
            "music_gain_db": self.music_gain_db,
            "total_duration_s": round(self.total_duration, 3),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EDL":
        known_range = {f for f in Range.__dataclass_fields__}
        ranges = [
            Range(**{k: v for k, v in r.items() if k in known_range})
            for r in data.get("ranges", [])
        ]
        known_ov = {f for f in Overlay.__dataclass_fields__}
        overlays = [
            Overlay(**{k: v for k, v in o.items() if k in known_ov})
            for o in data.get("overlays") or []
        ]
        cap_raw = data.get("captions")
        if isinstance(cap_raw, dict):
            known_cap = {f for f in CaptionSpec.__dataclass_fields__}
            captions = CaptionSpec(**{k: v for k, v in cap_raw.items() if k in known_cap})
        elif cap_raw is False:
            captions = CaptionSpec(enabled=False)
        else:
            captions = CaptionSpec()

        return cls(
            sources=dict(data.get("sources") or {}),
            ranges=ranges,
            platform=data.get("platform", "reels"),
            version=int(data.get("version", SCHEMA_VERSION)),
            reframe=data.get("reframe", "track"),
            detector=data.get("detector", "auto"),
            grade=data.get("grade", "none"),
            overlays=overlays,
            captions=captions,
            music=data.get("music"),
            music_gain_db=float(data.get("music_gain_db", -18.0)),
            intent=data.get("intent", ""),
            title=data.get("title", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EDL":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"EDL not found: {p}")
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise EDLError([f"not valid JSON: {e}"]) from None
        edl = cls.from_dict(data)
        edl.validate(base_dir=p.parent)
        return edl

    # -- validation ---------------------------------------------------------

    def validate(self, base_dir: Path | None = None) -> None:
        """Check every structural invariant, reporting all failures at once.

        Deliberately exhaustive rather than fail-fast: an agent fixing an EDL
        should see the whole list in one pass instead of rediscovering it one
        render at a time.
        """
        problems: list[str] = []
        base = base_dir or Path.cwd()

        if self.version > SCHEMA_VERSION:
            problems.append(
                f"schema version {self.version} is newer than supported ({SCHEMA_VERSION})"
            )
        try:
            target = self.target
        except KeyError as e:
            problems.append(str(e))
            target = get_platform("reels")

        if not self.sources:
            problems.append("no sources declared")
        for name, path in self.sources.items():
            p = Path(path)
            resolved = p if p.is_absolute() else (base / p)
            if not resolved.exists():
                problems.append(f"source {name!r} not found at {resolved}")

        if not self.ranges:
            problems.append("no ranges — nothing to render")

        for i, r in enumerate(self.ranges):
            tag = f"range[{i}]"
            if r.source not in self.sources:
                problems.append(f"{tag} references undeclared source {r.source!r}")
            if r.end <= r.start:
                problems.append(f"{tag} end ({r.end}) must be greater than start ({r.start})")
            if r.start < 0:
                problems.append(f"{tag} start is negative ({r.start})")
            if r.speed <= 0:
                problems.append(f"{tag} speed must be positive (got {r.speed})")
            # Sub-200ms cuts are almost always an arithmetic slip rather than
            # an intentional frame-flash, and they render as a glitch.
            if 0 < r.source_duration < 0.2:
                problems.append(
                    f"{tag} is only {r.source_duration * 1000:.0f}ms — likely a mistake"
                )

        total = self.total_duration
        if total > target.max_duration_s:
            problems.append(
                f"total {total:.1f}s exceeds {target.label}'s {target.max_duration_s:.0f}s limit"
            )
        if 0 < total < target.min_duration_s:
            problems.append(
                f"total {total:.1f}s is below {target.label}'s {target.min_duration_s:.0f}s minimum"
            )

        for i, o in enumerate(self.overlays):
            tag = f"overlay[{i}]"
            p = Path(o.file)
            resolved = p if p.is_absolute() else (base / p)
            if not resolved.exists():
                problems.append(f"{tag} file not found at {resolved}")
            if o.duration <= 0:
                problems.append(f"{tag} duration must be positive")
            if o.start_in_output < 0:
                problems.append(f"{tag} starts before zero")
            if total and o.start_in_output >= total:
                problems.append(
                    f"{tag} starts at {o.start_in_output:.2f}s, past the {total:.2f}s end"
                )
            if not 0 <= o.opacity <= 1:
                problems.append(f"{tag} opacity must be within [0,1]")
            if o.scale is not None and not 0 < o.scale <= 4:
                problems.append(f"{tag} scale must be within (0,4]")
            if o.fade_in + o.fade_out > o.duration:
                problems.append(f"{tag} fades ({o.fade_in}+{o.fade_out}) exceed its duration")

        if self.captions.position is not None and not 0 <= self.captions.position <= 1:
            problems.append("captions.position must be within [0,1]")
        if self.captions.words_per_cue is not None and self.captions.words_per_cue < 1:
            problems.append("captions.words_per_cue must be at least 1")

        if self.music:
            p = Path(self.music)
            resolved = p if p.is_absolute() else (base / p)
            if not resolved.exists():
                problems.append(f"music track not found at {resolved}")

        if problems:
            raise EDLError(problems)

    def resolve_source(self, name: str, base_dir: Path) -> Path:
        p = Path(self.sources[name])
        return p if p.is_absolute() else (base_dir / p).resolve()
