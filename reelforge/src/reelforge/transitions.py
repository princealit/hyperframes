"""Transitions at clip seams.

A cut is the right choice most of the time and stays the default. But when you
assemble several sources, some seams want softening — a dissolve between two
angles of the same moment, a dip to black between sections, a whip between
locations. Without them, multi-clip assembly reads as a slideshow.

Everything here maps onto ffmpeg's `xfade` (video) and `acrossfade` (audio).
The house names exist because `smoothleft` and `hlslice` describe an
implementation, not an intent, and the agent picking a transition should be
choosing an intent.

**A transition consumes timeline time.** Two 3s clips joined by a 0.5s
crossfade run 5.5s, not 6s — the segments overlap. This is the part that breaks
quietly: every downstream offset (captions, overlays, the linter's timings) has
to be computed against the overlapped timeline, which is why `EDL.offsets()`
subtracts transition durations rather than simply accumulating.

Transitions also force a re-encode at the join. With none set, the renderer
keeps its fast path: extract per segment, then stream-copy concat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

#: A cut. The default, and usually correct — most seams should not be softened.
CUT = "cut"


@dataclass(frozen=True)
class Transition:
    """One named seam treatment."""

    name: str
    #: The `xfade` transition this compiles to.
    xfade: str
    #: What it is for, in intent terms. Surfaced to the agent choosing one.
    use: str
    #: Sensible duration in seconds. Short-form wants these brief.
    default_duration: float = 0.35


TRANSITIONS: dict[str, Transition] = {
    CUT: Transition(CUT, "", "Hard cut. The default; use it unless a seam needs softening.", 0.0),

    "crossfade": Transition(
        "crossfade", "fade",
        "Dissolve between shots. Two angles of the same moment, or a gentle "
        "passage of time. Overused it reads as a holiday slideshow.",
        0.35,
    ),
    "dip_black": Transition(
        "dip_black", "fadeblack",
        "Through black. A hard break between sections or chapters — the "
        "strongest 'that ended, this begins' punctuation available.",
        0.45,
    ),
    "dip_white": Transition(
        "dip_white", "fadewhite",
        "Through white. Brighter and more energetic than black; suits product "
        "and beauty content, and reveals.",
        0.35,
    ),
    "whip_left": Transition(
        "whip_left", "hlslice",
        "Whip pan left. Fast, energetic location or subject change. Keep it "
        "short — a slow whip reads as a mistake.",
        0.22,
    ),
    "whip_right": Transition(
        "whip_right", "hrslice",
        "Whip pan right. Mirror of whip_left; alternate direction so repeated "
        "whips do not feel mechanical.",
        0.22,
    ),
    "blur": Transition(
        "blur", "hblur",
        "Blur through. Softer than a whip, punchier than a dissolve. Good "
        "between shots that share no visual anchor.",
        0.3,
    ),
    "slide_left": Transition(
        "slide_left", "slideleft",
        "Push the outgoing shot off left. Reads as forward motion through a "
        "sequence — steps, list items, before/after.",
        0.35,
    ),
    "slide_up": Transition(
        "slide_up", "slideup",
        "Push upward. Native to vertical feeds, where the scroll gesture is "
        "already vertical.",
        0.35,
    ),
    "zoom": Transition(
        "zoom", "zoomin",
        "Zoom through. Punchy and attention-grabbing; suits reveals and a "
        "sharp escalation in energy.",
        0.3,
    ),
    "pixelize": Transition(
        "pixelize", "pixelize",
        "Pixel dissolve. A deliberately digital, glitchy texture.",
        0.35,
    ),
    "circle": Transition(
        "circle", "circleopen",
        "Iris open. Retro and theatrical; suits a punchline or a sign-off.",
        0.4,
    ),
    "dissolve": Transition(
        "dissolve", "dissolve",
        "Grainy dissolve. Rougher than a crossfade — more filmic, less clean.",
        0.4,
    ),
}


def get_transition(name: str) -> Transition:
    try:
        return TRANSITIONS[name]
    except KeyError:
        valid = ", ".join(sorted(TRANSITIONS))
        raise KeyError(
            f"unknown transition {name!r}; expected one of: {valid}"
        ) from None


def catalog() -> list[dict[str, object]]:
    """The transition set as data, for the MCP `list_capabilities` tool."""
    return [
        {"name": t.name, "use": t.use, "default_duration_s": t.default_duration}
        for t in TRANSITIONS.values()
    ]


def resolve_duration(name: str, requested: float | None) -> float:
    """A transition's duration, defaulting to the one that suits it."""
    if name == CUT:
        return 0.0
    t = get_transition(name)
    return t.default_duration if requested is None else max(0.0, requested)


def clamp_durations(
    segment_durations: Sequence[float], names: Sequence[str], durations: Sequence[float]
) -> list[float]:
    """Shrink any transition that would consume more than its neighbours have.

    `xfade` reads `duration` seconds from the tail of one segment and the head
    of the next. Asking for more than either holds truncates the output and
    desynchronises everything after it, so the overlap is capped at just under
    half the shorter neighbour — leaving each segment some frames that are
    solely its own.
    """
    out: list[float] = []
    for i, (name, requested) in enumerate(zip(names, durations)):
        if name == CUT or i == 0:
            out.append(0.0)
            continue
        shorter = min(segment_durations[i - 1], segment_durations[i])
        out.append(max(0.0, min(requested, shorter * 0.45)))
    return out


def timeline_offsets(
    segment_durations: Sequence[float], transition_durations: Sequence[float]
) -> list[float]:
    """Output-timeline start of each segment, with overlaps accounted for.

    Segment i begins where everything before it ended, minus the transition
    overlaps consumed so far. Used for caption and overlay timing, so an error
    here shows up as captions drifting further out of sync at every seam.
    """
    offsets: list[float] = []
    cursor = 0.0
    for i, d in enumerate(segment_durations):
        if i > 0:
            cursor -= transition_durations[i]
        offsets.append(cursor)
        cursor += d
    return offsets


def total_duration(
    segment_durations: Sequence[float], transition_durations: Sequence[float]
) -> float:
    return max(0.0, sum(segment_durations) - sum(transition_durations))


def build_graph(
    segment_durations: Sequence[float],
    names: Sequence[str],
    durations: Sequence[float],
    *,
    has_audio: bool = True,
) -> tuple[str, str, str]:
    """Compile the seam plan into a filtergraph.

    Returns `(filter_complex, video_label, audio_label)`.

    Seams are chained pairwise. A cut joins with the `concat` filter and a
    transition with `xfade`, so a timeline mixing both is handled in one pass
    rather than needing the two to be separated first.

    `xfade`'s `offset` is absolute on the *output* timeline — where the overlap
    begins — which is the running cursor minus this transition's duration.
    """
    n = len(segment_durations)
    if n == 0:
        raise ValueError("no segments to join")
    if n == 1:
        return "", "0:v", "0:a"

    parts: list[str] = []
    v_label = "0:v"
    a_label = "0:a"
    cursor = segment_durations[0]

    for i in range(1, n):
        name = names[i]
        d = durations[i]
        v_out = f"v{i}"
        a_out = f"a{i}"

        if name == CUT or d <= 0:
            parts.append(f"[{v_label}][{i}:v]concat=n=2:v=1:a=0[{v_out}]")
            if has_audio:
                parts.append(f"[{a_label}][{i}:a]concat=n=2:v=0:a=1[{a_out}]")
            cursor += segment_durations[i]
        else:
            xf = get_transition(name).xfade
            offset = max(0.0, cursor - d)
            parts.append(
                f"[{v_label}][{i}:v]xfade=transition={xf}:duration={d:.3f}"
                f":offset={offset:.3f}[{v_out}]"
            )
            if has_audio:
                # `tri` is a linear crossfade: constant-amplitude rather than
                # constant-power, which keeps dialogue from swelling mid-seam.
                parts.append(
                    f"[{a_label}][{i}:a]acrossfade=d={d:.3f}:c1=tri:c2=tri[{a_out}]"
                )
            cursor += segment_durations[i] - d

        v_label = v_out
        if has_audio:
            a_label = a_out

    return ";".join(parts), v_label, a_label
