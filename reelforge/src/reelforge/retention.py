"""Static analysis for an edit, before you spend minutes rendering it.

A compiler for short-form video. Most of what kills a Reel is visible in the EDL
and the transcript: the hook takes four seconds to arrive, one shot runs eleven
seconds without a cut, captions flash past unreadably, an overlay sits under the
action rail. All of that is checkable, and all of it is far cheaper to fix before
the render than after the upload.

Findings are advisory except where they describe something the platform will
actually reject. The linter reports; it never edits. Every finding carries a
concrete fix, because "hook is weak" is not actionable and "first speech starts
at 2.4s — trim the first 2.1s" is.

Nothing here claims to predict performance. These are craft checks with defensible
reasoning, not a model of the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from . import captions as cap
from .config import Platform
from .edl import EDL
from .ffmpeg import probe

Severity = Literal["error", "warning", "note"]

#: Weight subtracted from a nominal 100 for each finding of a given severity.
_PENALTY = {"error": 22, "warning": 9, "note": 2}


@dataclass
class Finding:
    severity: Severity
    code: str
    message: str
    fix: str = ""
    #: Output-timeline position the finding refers to, when it has one.
    at: float | None = None

    def format(self) -> str:
        mark = {"error": "ERROR", "warning": "WARN ", "note": "note "}[self.severity]
        stamp = f"[{self.at:6.2f}s] " if self.at is not None else "         "
        out = f"{mark} {stamp}{self.message}"
        if self.fix:
            out += f"\n                 -> {self.fix}"
        return out


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, float | str] = field(default_factory=dict)

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        fix: str = "",
        at: float | None = None,
    ) -> None:
        self.findings.append(Finding(severity, code, message, fix, at))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def score(self) -> int:
        """A rough 0-100 summary. The findings matter; this is a headline."""
        return max(0, min(100, 100 - sum(_PENALTY[f.severity] for f in self.findings)))

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = [f"score {self.score}/100  "
                 f"({len(self.errors)} error, {len(self.warnings)} warning, "
                 f"{len(self.findings) - len(self.errors) - len(self.warnings)} note)"]
        if self.stats:
            lines.append("")
            for k, v in self.stats.items():
                shown = f"{v:.2f}" if isinstance(v, float) else str(v)
                lines.append(f"  {k:<22} {shown}")
        if self.findings:
            lines.append("")
            order = {"error": 0, "warning": 1, "note": 2}
            for f in sorted(self.findings, key=lambda f: (order[f.severity], f.at or 0)):
                lines.append(f.format())
        else:
            lines.append("\nno findings")
        return "\n".join(lines)


# --- Thresholds -------------------------------------------------------------
#
# Craft heuristics, not measurements of any platform's ranking behaviour.

#: Speech should be underway almost immediately; a vertical feed is scrolled at
#: roughly a second per post, so silence at the head is spent budget.
HOOK_SPEECH_BY = 0.8
#: The window a viewer decides within.
HOOK_WINDOW = 3.0
#: Words in the hook window. Too few reads as slow, too many as gabbled.
HOOK_WORDS = (5, 14)
#: Comfortable delivery band for short-form, in words per second.
PACE_BAND = (2.0, 3.6)
#: Silence longer than this with nothing on screen is dead air.
DEAD_AIR = 1.2
#: A single shot beyond this without a cut, overlay or reframe move goes flat.
MAX_SHOT = 7.0
#: Below this a caption cannot be read at all.
MIN_CUE = 0.25
#: Openers so common they no longer register as a hook.
WEAK_OPENERS = frozenset({
    "so", "um", "uh", "okay", "ok", "hi", "hey", "hello", "yeah", "well",
    "basically", "anyway", "right", "today", "alright",
})


def _first_words(cues: Sequence[cap.Cue], until: float) -> list[cap.Word]:
    return [w for c in cues for w in c.words if w.start < until]


def lint(
    edl: EDL,
    *,
    base_dir: Path | None = None,
    work_dir: Path | None = None,
    rendered: Path | None = None,
) -> Report:
    """Analyse an EDL, and a rendered file when one is available."""
    base = base_dir or Path.cwd()
    work = work_dir or base / ".reelforge"
    report = Report()
    platform = edl.target
    total = edl.total_duration

    report.stats["platform"] = platform.label
    report.stats["duration_s"] = round(total, 2)
    report.stats["segments"] = len(edl.ranges)

    _check_duration(report, edl, platform, total)
    _check_structure(report, edl, total)
    _check_sources(report, edl, base, platform)
    _check_overlays(report, edl, base, platform)

    style = cap.resolve_style(edl.captions)
    cues = _cues(edl, style, work) if edl.captions.enabled else []
    if cues:
        _check_hook(report, cues, edl)
        _check_pace(report, cues, total)
        _check_dead_air(report, cues, edl, total)
        _check_captions(report, cues, style, platform)
    elif edl.captions.enabled:
        report.add(
            "warning", "captions.missing",
            "captions are enabled but no transcripts were found",
            f"run `reelforge transcribe` — expected {work / 'transcripts'}/<source>.json",
        )

    if rendered is not None and rendered.exists():
        _check_rendered(report, rendered, platform, total)

    return report


# --- Individual checks ------------------------------------------------------


def _check_duration(report: Report, edl: EDL, platform: Platform, total: float) -> None:
    if total > platform.max_duration_s:
        report.add(
            "error", "duration.over",
            f"{total:.1f}s exceeds {platform.label}'s {platform.max_duration_s:.0f}s ceiling",
            f"cut {total - platform.max_duration_s:.1f}s",
        )
    elif total < platform.min_duration_s:
        report.add(
            "error", "duration.under",
            f"{total:.1f}s is below {platform.label}'s {platform.min_duration_s:.0f}s minimum",
            "add material",
        )
    lo, hi = platform.sweet_spot_s
    if total > hi:
        report.add(
            "note", "duration.long",
            f"{total:.1f}s is past the {lo:.0f}-{hi:.0f}s band where completion rate holds up",
            "consider tightening, or splitting into parts",
        )


def _check_structure(report: Report, edl: EDL, total: float) -> None:
    beats = [r.beat.strip().upper() for r in edl.ranges if r.beat.strip()]
    if not beats:
        report.add(
            "note", "structure.unlabelled",
            "no ranges carry a beat label",
            "label ranges (HOOK / PROBLEM / PAYOFF / CTA) so structure is auditable",
        )
    elif "HOOK" not in beats:
        report.add(
            "warning", "structure.nohook",
            "no range is labelled HOOK",
            "identify which range is doing the opening work",
        )

    durations = [r.duration for r in edl.ranges]
    for i, (rng, offset) in enumerate(zip(edl.ranges, edl.offsets())):
        # An overlay or a reframe move keeps a long take alive; a static one dies.
        covered = any(
            o.start_in_output < offset + rng.duration and o.end_in_output > offset
            for o in edl.overlays
        )
        if rng.duration > MAX_SHOT and not covered:
            report.add(
                "warning", "pace.longshot",
                f"range[{i}] runs {rng.duration:.1f}s with no cut or overlay",
                "cut it, or place an overlay or b-roll inside it",
                at=offset,
            )

    if len(durations) >= 4:
        mean = sum(durations) / len(durations)
        spread = (sum((d - mean) ** 2 for d in durations) / len(durations)) ** 0.5
        if mean > 0:
            report.stats["shot_length_cv"] = round(spread / mean, 2)
            # Near-identical shot lengths read as metronomic and the eye stops
            # being surprised by the cuts.
            if spread / mean < 0.18:
                report.add(
                    "note", "pace.monotone",
                    f"every shot is about {mean:.1f}s — the rhythm is metronomic",
                    "vary shot lengths; let one breathe and cut another short",
                )


def _check_sources(report: Report, edl: EDL, base: Path, platform: Platform) -> None:
    seen: set[str] = set()
    for rng in edl.ranges:
        if rng.source in seen:
            continue
        seen.add(rng.source)
        try:
            info = probe(edl.resolve_source(rng.source, base))
        except Exception as e:  # noqa: BLE001 — surfaced as a finding, not raised
            report.add("error", "source.unreadable", f"{rng.source}: {e}")
            continue

        sw, sh = info.display_size
        # The crop keeps full height and a 9/16 slice of width, so this is the
        # true horizontal resolution feeding the output.
        if sw / sh > platform.aspect:
            effective = sh * platform.aspect
            if effective < platform.width * 0.9:
                report.add(
                    "warning", "source.upscale",
                    f"{rng.source} yields {effective:.0f}px of width after the 9:16 crop, "
                    f"upscaled to {platform.width}px",
                    "use a higher-resolution source, or `reframe: blur_pad` to avoid cropping",
                )
        if not info.has_audio:
            report.add("note", "source.silent", f"{rng.source} has no audio track")


def _check_overlays(report: Report, edl: EDL, base: Path, platform: Platform) -> None:
    w, h = platform.width, platform.height
    safe = platform.safe_at(w, h)
    for i, ov in enumerate(edl.overlays):
        if ov.ignore_safe_area:
            report.add(
                "note", "overlay.unsafe.optout",
                f"overlay[{i}] opts out of the safe area",
                "confirm it is meant to sit under the platform UI",
                at=ov.start_in_output,
            )
            continue
        path = Path(ov.file)
        path = path if path.is_absolute() else base / path
        if not path.exists():
            continue
        try:
            info = probe(path)
        except Exception:  # noqa: BLE001
            continue
        ow = int(safe.content_box(w, h)[2] * ov.scale) if ov.scale else info.width
        oh = int(info.height * ow / info.width) if info.width else info.height
        _, _, safe_w, safe_h = safe.content_box(w, h)
        if ow > safe_w or oh > safe_h:
            report.add(
                "warning", "overlay.oversize",
                f"overlay[{i}] is {ow}x{oh}, larger than the {safe_w}x{safe_h} safe area",
                f"set scale to about {safe_w / max(ow, 1):.2f}, or set ignore_safe_area",
                at=ov.start_in_output,
            )
        if ov.duration < 1.0:
            report.add(
                "note", "overlay.brief",
                f"overlay[{i}] is on screen for {ov.duration:.2f}s",
                "under a second is rarely long enough to read",
                at=ov.start_in_output,
            )


def _check_hook(report: Report, cues: list[cap.Cue], edl: EDL) -> None:
    first = cues[0].words[0]
    report.stats["first_word_s"] = round(first.start, 2)
    if first.start > HOOK_SPEECH_BY:
        report.add(
            "warning", "hook.late",
            f"first word lands at {first.start:.2f}s",
            f"trim {first.start - 0.15:.2f}s off the head so speech starts immediately",
            at=0.0,
        )

    window = _first_words(cues, HOOK_WINDOW)
    count = len(window)
    report.stats["hook_words"] = count
    lo, hi = HOOK_WORDS
    if count < lo:
        report.add(
            "warning", "hook.thin",
            f"only {count} words in the first {HOOK_WINDOW:.0f}s",
            "open on a complete, concrete claim rather than a run-up",
            at=0.0,
        )
    elif count > hi:
        report.add(
            "note", "hook.dense",
            f"{count} words crammed into the first {HOOK_WINDOW:.0f}s",
            "let the opening line land before the next one starts",
            at=0.0,
        )

    # Checked against the actual first word, not the hook window: a video whose
    # speech starts late still opens on whatever that first word is, and that is
    # precisely the case where a filler opener costs most.
    opener = first.text.strip().lower().strip(".,!?")
    if opener in WEAK_OPENERS:
        report.add(
            "warning", "hook.filler",
            f"the video opens on {first.text.strip()!r}",
            "cut to the first substantive word — the opener is spent attention",
            at=0.0,
        )


def _check_pace(report: Report, cues: list[cap.Cue], total: float) -> None:
    words = [w for c in cues for w in c.words]
    if not words or total <= 0:
        return
    wps = len(words) / total
    report.stats["words_per_second"] = round(wps, 2)
    lo, hi = PACE_BAND
    if wps < lo:
        report.add(
            "warning", "pace.slow",
            f"{wps:.2f} words/sec across the cut (short-form sits at {lo:.1f}-{hi:.1f})",
            "tighten pauses, or speed ranges slightly with `speed: 1.05`",
        )
    elif wps > hi:
        report.add(
            "note", "pace.fast",
            f"{wps:.2f} words/sec is quick even for short-form",
            "make sure the captions can be read at this rate",
        )


def _check_dead_air(report: Report, cues: list[cap.Cue], edl: EDL, total: float) -> None:
    words = sorted((w for c in cues for w in c.words), key=lambda w: w.start)
    if not words:
        return

    def covered(a: float, b: float) -> bool:
        return any(o.start_in_output < b and o.end_in_output > a for o in edl.overlays)

    gaps: list[tuple[float, float]] = []
    if words[0].start > DEAD_AIR:
        gaps.append((0.0, words[0].start))
    for a, b in zip(words, words[1:]):
        if b.start - a.end > DEAD_AIR:
            gaps.append((a.end, b.start))
    if total - words[-1].end > DEAD_AIR:
        gaps.append((words[-1].end, total))

    longest = 0.0
    for start, end in gaps:
        longest = max(longest, end - start)
        if covered(start, end):
            continue
        report.add(
            "warning", "pace.deadair",
            f"{end - start:.1f}s of silence with nothing on screen",
            "cut the gap, or fill it with b-roll or an overlay",
            at=start,
        )
    if gaps:
        report.stats["longest_silence_s"] = round(longest, 2)


def _check_captions(
    report: Report, cues: list[cap.Cue], style: cap.CaptionStyle, platform: Platform
) -> None:
    report.stats["caption_cues"] = len(cues)
    short = [c for c in cues if c.end - c.start < MIN_CUE]
    if short:
        report.add(
            "warning", "captions.flash",
            f"{len(short)} cue(s) are on screen under {MIN_CUE * 1000:.0f}ms",
            "lower words_per_cue, or let the cue extend into the following gap",
            at=short[0].start,
        )

    max_w = cap.caption_max_width(platform, platform.width, platform.height)
    font_px = platform.width * style.size_ratio
    fits = max(6, int(max_w / (0.55 * font_px)))
    longest = max((len(c.text) for c in cues), default=0)
    if longest > fits * 2:
        report.add(
            "note", "captions.long",
            f"the longest cue is {longest} characters and will wrap past two lines",
            f"about {fits} characters fit per line at this size — lower words_per_cue",
        )

    for a, b in zip(cues, cues[1:]):
        if b.start < a.end - 0.02:
            report.add(
                "warning", "captions.overlap",
                f"cues overlap at {b.start:.2f}s",
                "two cards on screen at once; check the transcript timings",
                at=b.start,
            )
            break


def _check_rendered(
    report: Report, rendered: Path, platform: Platform, expected: float
) -> None:
    try:
        info = probe(rendered)
    except Exception as e:  # noqa: BLE001
        report.add("error", "render.unreadable", f"cannot probe {rendered}: {e}")
        return

    if (info.width, info.height) != (platform.width, platform.height):
        report.add(
            "note", "render.size",
            f"rendered at {info.width}x{info.height}, "
            f"{platform.label} expects {platform.width}x{platform.height}",
            "re-render at final quality (draft renders at reduced scale)",
        )
    drift = abs(info.duration - expected)
    # A frame or two of drift is normal container rounding; a second is a bug.
    if drift > 0.5:
        report.add(
            "warning", "render.drift",
            f"rendered {info.duration:.2f}s against an EDL total of {expected:.2f}s",
            "check for a range extending past the end of its source",
        )
    if not info.has_audio:
        report.add("note", "render.silent", "the rendered file has no audio track")


def _cues(edl: EDL, style: cap.CaptionStyle, work: Path) -> list[cap.Cue]:
    from .render import _collect_cues

    try:
        return _collect_cues(edl, style, work, work.parent)
    except Exception:  # noqa: BLE001 — a missing transcript is reported upstream
        return []
