"""Caption synthesis: word timings in, positioned ASS out.

Two things here that a plain SRT pipeline cannot do.

**Placement is computed, not configured.** video-use pins captions with a fixed
`MarginV=90` and a comment explaining that it clears the Reels UI. That constant
is correct for Reels at 1080x1920 and wrong everywhere else — TikTok's caption
block and action rail are taller, and the whole thing rescales at 720p. Here the
baseline is derived from the target's safe area, so switching platform moves the
captions to where that platform actually leaves room.

**Word-level highlighting.** SRT has no styling, so the best it can do is swap
whole cues. ASS can colour individual words, which is the short-form convention
that makes captions feel synced to delivery rather than merely present.

Timing follows the output timeline throughout: a word's caption time is
`word.start - range.start + range_offset`. Every cue is built through
`build_cues`, so that conversion exists in exactly one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from .config import Platform

CaseMode = Literal["upper", "sentence", "natural"]
Highlight = Literal["none", "word", "pop"]

#: Trailing punctuation stripped for the all-caps styles, where a comma hanging
#: off a two-word card reads as debris rather than grammar.
_TRAILING = ",;:"
_SENTENCE_END = frozenset(".!?")


# --- Colour -----------------------------------------------------------------


def to_ass_colour(value: str, alpha: int = 0) -> str:
    """Convert `#RRGGBB` to ASS `&HAABBGGRR`.

    ASS stores colour byte-reversed against the web convention and puts alpha
    first, where 0 is opaque. Both inversions are easy to get backwards, so the
    conversion lives here and nowhere else.
    """
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        raise ValueError(f"expected a #RRGGBB colour, got {value!r}")
    r, g, b = v[0:2], v[2:4], v[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


# --- Style ------------------------------------------------------------------


@dataclass(frozen=True)
class CaptionStyle:
    """A caption look, in units relative to the output canvas where possible."""

    name: str
    font: str = "DejaVu Sans"
    #: Cap height as a fraction of output width, so a style holds its
    #: proportions across 1080p and 720p renders.
    size_ratio: float = 0.078
    bold: bool = True
    primary: str = "#FFFFFF"
    outline: str = "#000000"
    highlight_colour: str = "#FFD400"
    outline_ratio: float = 0.0055
    shadow_ratio: float = 0.0
    words_per_cue: int = 2
    case: CaseMode = "upper"
    highlight: Highlight = "none"
    #: Vertical position within the safe area; 0 is its top, 1 its bottom.
    position: float = 0.72
    #: Opaque box behind the text instead of an outline.
    boxed: bool = False
    #: Per-cue fade in/out in milliseconds.
    fade_ms: tuple[int, int] = (0, 0)
    #: Maximum characters on a line before wrapping.
    max_chars: int = 18
    letter_spacing: float = 0.0


PRESETS: dict[str, CaptionStyle] = {
    # The short-form default: two words, enormous, unmissable at thumb distance.
    "punch": CaptionStyle(
        name="punch", size_ratio=0.082, words_per_cue=2, case="upper",
        highlight="none", outline_ratio=0.006, fade_ms=(40, 40), max_chars=16,
    ),
    # Word-by-word colour sweep. Reads as tightly synced to the speaker.
    "karaoke": CaptionStyle(
        name="karaoke", size_ratio=0.070, words_per_cue=4, case="upper",
        highlight="word", outline_ratio=0.0055, max_chars=22,
    ),
    # A scale bump on the active word instead of a colour change — louder, and
    # it survives being screenshotted in greyscale.
    "pop": CaptionStyle(
        name="pop", size_ratio=0.072, words_per_cue=3, case="upper",
        highlight="pop", outline_ratio=0.0058, max_chars=20,
    ),
    # Longer cues, sentence case, calmer. For explainers and interviews where
    # the captions support the audio rather than perform alongside it.
    "clean": CaptionStyle(
        name="clean", size_ratio=0.050, words_per_cue=6, case="sentence",
        highlight="none", outline_ratio=0.004, position=0.80, max_chars=34,
        bold=False,
    ),
    # Opaque block. Maximum legibility over busy or bright footage.
    "boxed": CaptionStyle(
        name="boxed", size_ratio=0.052, words_per_cue=5, case="sentence",
        highlight="none", boxed=True, position=0.78, max_chars=30,
    ),
}


def get_style(name: str) -> CaptionStyle:
    try:
        return PRESETS[name]
    except KeyError:
        valid = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown caption style {name!r}; expected one of: {valid}") from None


# --- Cues -------------------------------------------------------------------


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Cue:
    """One displayed card, on the output timeline."""

    start: float
    end: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def apply_case(text: str, mode: CaseMode) -> str:
    if mode == "upper":
        return text.upper()
    if mode == "sentence":
        return text[:1].upper() + text[1:] if text else text
    return text


def chunk_words(words: Sequence[Word], per_cue: int) -> list[list[Word]]:
    """Group words into cues, breaking early at sentence boundaries.

    Running a card across a full stop makes two sentences look like one clause,
    so punctuation wins over the word count.
    """
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if not w.text.strip():
            continue
        current.append(w)
        ends_sentence = bool(w.text) and w.text.rstrip()[-1:] in _SENTENCE_END
        if len(current) >= per_cue or ends_sentence:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def build_cues(
    ranges: Iterable[tuple[Sequence[Word], float, float, float]],
    style: CaptionStyle,
    *,
    min_duration: float = 0.28,
) -> list[Cue]:
    """Map per-source words onto the output timeline and group them into cues.

    Each entry is `(words, range_start, range_end, output_offset)`. Words are
    clipped to the range, shifted by `output_offset - range_start`, then chunked.
    Cues below `min_duration` are extended so a fast-spoken word is still
    readable, without ever overlapping the cue that follows.
    """
    cues: list[Cue] = []
    for words, r_start, r_end, offset in ranges:
        shift = offset - r_start
        kept: list[Word] = []
        for w in words:
            if w.end <= r_start or w.start >= r_end:
                continue
            text = w.text.strip()
            if not text:
                continue
            kept.append(
                Word(
                    text=text,
                    start=max(w.start, r_start) + shift,
                    end=min(w.end, r_end) + shift,
                )
            )
        for group in chunk_words(kept, style.words_per_cue):
            cues.append(Cue(start=group[0].start, end=group[-1].end, words=group))

    cues.sort(key=lambda c: c.start)
    for i, cue in enumerate(cues):
        if cue.end - cue.start >= min_duration:
            continue
        ceiling = cues[i + 1].start if i + 1 < len(cues) else cue.start + min_duration
        cue.end = min(cue.start + min_duration, max(ceiling, cue.start + 0.05))
    return cues


# --- ASS emission -----------------------------------------------------------


def _ass_time(seconds: float) -> str:
    """ASS timestamps are centisecond resolution: `H:MM:SS.cc`."""
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{c:02d}"


def _escape(text: str) -> str:
    """Neutralise ASS markup characters inside caption text."""
    return text.replace("\\", "∖").replace("{", "(").replace("}", ")")


def wrap_tokens(tokens: Sequence[str], max_chars: int) -> list[list[int]]:
    """Greedy wrap, returning token *indices* per line.

    Indices rather than joined strings because the karaoke styles need to wrap
    on visible text and only afterwards wrap individual tokens in styling tags.
    Measuring a line after the markup is applied counts `{\\c&H0000D4FF}` as
    fourteen visible characters and wraps a two-word card onto two lines.
    """
    if not tokens:
        return []
    lines: list[list[int]] = []
    current: list[int] = [0]
    width = len(tokens[0])
    for i in range(1, len(tokens)):
        extra = 1 + len(tokens[i])
        if width + extra <= max_chars:
            current.append(i)
            width += extra
        else:
            lines.append(current)
            current = [i]
            width = len(tokens[i])
    lines.append(current)
    return lines


def _wrap(text: str, max_chars: int) -> str:
    """Greedy wrap of plain text using the ASS line break `\\N`."""
    tokens = text.split()
    lines = wrap_tokens(tokens, max_chars)
    return "\\N".join(" ".join(tokens[i] for i in line) for line in lines)


def caption_position(
    platform: Platform, width: int, height: int, position: float
) -> tuple[int, int]:
    """Resolve the caption anchor point for this target.

    Horizontally the anchor is the *frame* centre, not the safe area's centre.
    On vertical platforms the right-hand action rail makes the safe area
    asymmetric, so centring inside it pushes captions visibly left of centre —
    which reads as a layout mistake rather than as UI avoidance. Captions stay
    optically centred; `caption_max_width` is what keeps them clear of the rail.

    Vertically the anchor *is* derived from the safe area, because that is where
    the platform genuinely covers the frame and no amount of centring helps.
    """
    safe = platform.safe_at(width, height)
    _, y0, _, sh = safe.content_box(width, height)
    cx = width // 2
    cy = y0 + int(sh * min(max(position, 0.0), 1.0))
    return cx, cy


def caption_max_width(platform: Platform, width: int, height: int) -> int:
    """Widest a frame-centred caption can be while clearing the UI both sides.

    Symmetric about the frame centre, so the limit is twice the smaller of the
    two side margins. On Reels the action rail dominates and this comes out
    appreciably narrower than the safe area — which is correct, and is why long
    caption lines have to wrap rather than extend under the icons.
    """
    safe = platform.safe_at(width, height)
    centre = width / 2
    return int(2 * min(centre - safe.left, (width - safe.right) - centre))


def build_ass(
    cues: Sequence[Cue],
    style: CaptionStyle,
    platform: Platform,
    width: int,
    height: int,
) -> str:
    """Render cues to an ASS subtitle document sized for `width`x`height`.

    `PlayResX/Y` are set to the real output size so every dimension below is
    literal output pixels and libass performs no implicit rescaling.
    """
    font_size = max(12, int(width * style.size_ratio))
    outline = max(1, round(width * style.outline_ratio))
    shadow = round(width * style.shadow_ratio)
    cx, cy = caption_position(platform, width, height, style.position)

    primary = to_ass_colour(style.primary)
    outline_col = to_ass_colour(style.outline)
    back_col = to_ass_colour(style.outline, alpha=0 if style.boxed else 0)
    border_style = 3 if style.boxed else 1

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font},{font_size},{primary},{primary},{outline_col},{back_col},{1 if style.bold else 0},0,0,0,100,100,{style.letter_spacing},0,{border_style},{outline},{shadow},5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    fade = ""
    if any(style.fade_ms):
        fade = f"\\fad({style.fade_ms[0]},{style.fade_ms[1]})"
    # \an5 anchors the text box on its own centre, so \pos places the middle of
    # the card at the computed safe-area point regardless of line count.
    base_tags = f"\\an5\\pos({cx},{cy}){fade}"

    lines: list[str] = []

    def emit(start: float, end: float, body: str) -> None:
        if end <= start:
            return
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{body}"
        )

    highlight_col = to_ass_colour(style.highlight_colour)

    # Cap the wrap width at what actually fits between the platform's side UI.
    # 0.55em is a reasonable mean advance for a bold sans; it only has to be
    # close enough to stop a line running under the action rail.
    fitting_chars = max(6, int(caption_max_width(platform, width, height) / (0.55 * font_size)))
    max_chars = min(style.max_chars, fitting_chars)

    for cue in cues:
        tokens = [
            _escape(apply_case(w.text, style.case).rstrip(_TRAILING)) for w in cue.words
        ]
        if style.highlight == "none" or len(cue.words) == 1:
            emit(
                cue.start, cue.end,
                f"{{{base_tags}}}" + "\\N".join(
                    " ".join(tokens[i] for i in line)
                    for line in wrap_tokens(tokens, max_chars)
                ),
            )
            continue

        # One event per word, each rendering the whole card with that word
        # emphasised. Line breaks are computed once from the bare tokens so the
        # card's layout stays fixed while the emphasis moves through it.
        layout = wrap_tokens(tokens, max_chars)
        for i, word in enumerate(cue.words):
            seg_start = word.start if i else cue.start
            seg_end = cue.words[i + 1].start if i + 1 < len(cue.words) else cue.end
            if style.highlight == "word":
                styled = [
                    f"{{\\c{highlight_col}}}{t}{{\\c{primary}}}" if j == i else t
                    for j, t in enumerate(tokens)
                ]
            else:  # "pop" — scale the active word rather than recolouring it
                styled = [
                    f"{{\\fscx112\\fscy112}}{t}{{\\fscx100\\fscy100}}" if j == i else t
                    for j, t in enumerate(tokens)
                ]
            body = "\\N".join(
                " ".join(styled[i] for i in line) for line in layout
            )
            emit(seg_start, seg_end, f"{{{base_tags}}}{body}")

    return header + "\n".join(lines) + "\n"


def build_srt(cues: Sequence[Cue], style: CaptionStyle) -> str:
    """Plain SRT, for uploading as a sidecar caption file.

    Instagram and YouTube both accept an SRT upload, which yields selectable,
    translatable captions alongside the burned-in ones. Styling is dropped;
    only the text and timings survive.
    """
    out: list[str] = []
    for i, cue in enumerate(cues, start=1):
        text = apply_case(cue.text, style.case).rstrip(_TRAILING)
        out += [
            str(i),
            f"{_srt_time(cue.start)} --> {_srt_time(cue.end)}",
            text,
            "",
        ]
    return "\n".join(out)


def _srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def resolve_style(spec: Any) -> CaptionStyle:
    """Merge a `CaptionSpec` from an EDL over its named preset."""
    style = get_style(getattr(spec, "style", "punch") or "punch")
    overrides: dict[str, Any] = {}
    # Only explicitly-set fields override the preset; see `CaptionSpec`.
    if getattr(spec, "words_per_cue", None) is not None:
        overrides["words_per_cue"] = spec.words_per_cue
    if getattr(spec, "case", None) is not None:
        overrides["case"] = spec.case
    if getattr(spec, "position", None) is not None:
        overrides["position"] = spec.position
    if getattr(spec, "highlight", None) is not None:
        overrides["highlight_colour"] = spec.highlight
    if getattr(spec, "font", ""):
        overrides["font"] = spec.font
    size = getattr(spec, "font_size", None)
    if size:
        # An explicit pixel size is honoured by back-converting to the ratio,
        # keeping one representation internally.
        overrides["size_ratio"] = size / 1080
    return replace(style, **overrides) if overrides else style


def find_words(transcript: dict) -> list[Word]:
    """Extract word-level timings from a transcript payload.

    Accepts the ElevenLabs Scribe shape and the common Whisper variants, since
    the transcript may arrive from either. Entries without both timestamps are
    dropped rather than guessed at.
    """
    raw = transcript.get("words")
    if raw is None:
        segments = transcript.get("segments") or []
        raw = [w for seg in segments for w in (seg.get("words") or [])]

    words: list[Word] = []
    for w in raw or []:
        if isinstance(w, dict):
            kind = w.get("type", "word")
            if kind not in ("word", "spacing", None):
                continue
            if kind == "spacing":
                continue
            text = (w.get("text") or w.get("word") or "").strip()
            start, end = w.get("start"), w.get("end")
        else:
            continue
        if not text or start is None or end is None:
            continue
        try:
            words.append(Word(text=text, start=float(start), end=float(end)))
        except (TypeError, ValueError):
            continue
    return words


def load_words(path: str | Path) -> list[Word]:
    import json

    return find_words(json.loads(Path(path).read_text()))


_WORD_RE = re.compile(r"\S+")
