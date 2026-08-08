"""Delivery targets and the platform UI keep-out zones that constrain them.

Everything downstream — reframing, caption placement, overlay compositing,
the retention linter — resolves its numbers from this module. There is one
table, and it is the only place a platform constant is allowed to live.

## On the safe-zone numbers

Instagram, TikTok and YouTube do not publish pixel-exact keep-out rectangles,
and the real ones shift with app version, device notch and caption length.
The values below are *conservative* measurements at the reference resolution
(1080x1920): they reserve slightly more than the observed chrome so that a
composition that validates here is safe across app versions rather than
pixel-fitted to one.

They are expressed in reference-resolution pixels and scaled proportionally
for any other output size, so a 720x1280 render gets the same *proportional*
protection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SafeZone:
    """Keep-out insets in pixels at the platform's reference resolution.

    An element is "safe" when it sits inside the rectangle that remains after
    these insets are removed. `right` is usually the dominant constraint on
    vertical platforms because the like/comment/share/audio rail lives there.
    """

    top: int
    bottom: int
    left: int
    right: int

    def scaled_to(self, ref_w: int, ref_h: int, out_w: int, out_h: int) -> "SafeZone":
        """Rescale insets from the reference resolution to an actual output size."""
        sx = out_w / ref_w
        sy = out_h / ref_h
        return SafeZone(
            top=round(self.top * sy),
            bottom=round(self.bottom * sy),
            left=round(self.left * sx),
            right=round(self.right * sx),
        )

    def content_box(self, w: int, h: int) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) of the usable area inside `w`x`h`."""
        x = self.left
        y = self.top
        cw = max(0, w - self.left - self.right)
        ch = max(0, h - self.top - self.bottom)
        return x, y, cw, ch


@dataclass(frozen=True)
class Platform:
    """A delivery target: canvas, timing envelope, and its UI keep-out zone."""

    key: str
    label: str
    width: int
    height: int
    fps: int
    safe: SafeZone
    #: Hard ceiling the platform enforces on upload.
    max_duration_s: float
    #: Below this the platform (or the algorithm) treats the post as broken.
    min_duration_s: float
    #: The band where completion rate and replays are historically strongest.
    sweet_spot_s: tuple[float, float]
    #: Integrated loudness target. Every major platform normalises to -14 LUFS.
    lufs: float = -14.0
    #: True peak ceiling in dBTP, leaving headroom for lossy transcode.
    true_peak: float = -1.0

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def aspect_label(self) -> str:
        from math import gcd

        g = gcd(self.width, self.height)
        return f"{self.width // g}:{self.height // g}"

    def safe_at(self, w: int, h: int) -> SafeZone:
        """This platform's keep-out zone scaled to an arbitrary output size."""
        return self.safe.scaled_to(self.width, self.height, w, h)

    def resized(self, width: int, height: int) -> "Platform":
        return replace(self, width=width, height=height)


# --- Reference safe zones ---------------------------------------------------
#
# Measured at 1080x1920 against the app chrome, then rounded outward.
#
#   top     status bar + (on Reels) the "Reels" header and back affordance
#   bottom  username, caption, audio ticker, and the CTA/"See more" row
#   right   the vertical action rail: like / comment / share / more / audio disc
#   left    small gutter only — nothing structural lives here
#
# The bottom inset is the one people get wrong most often. A caption baseline
# placed 40px from the bottom edge looks fine in a preview player and is fully
# buried by the real UI.

_IG_REELS_SAFE = SafeZone(top=180, bottom=420, left=48, right=228)
_IG_FEED_SAFE = SafeZone(top=120, bottom=200, left=48, right=48)
_TIKTOK_SAFE = SafeZone(top=200, bottom=520, left=48, right=260)
_SHORTS_SAFE = SafeZone(top=160, bottom=380, left=48, right=200)
_SQUARE_SAFE = SafeZone(top=90, bottom=140, left=48, right=48)


PLATFORMS: dict[str, Platform] = {
    "reels": Platform(
        key="reels",
        label="Instagram Reels",
        width=1080,
        height=1920,
        fps=30,
        safe=_IG_REELS_SAFE,
        max_duration_s=180.0,
        min_duration_s=3.0,
        sweet_spot_s=(7.0, 45.0),
    ),
    "tiktok": Platform(
        key="tiktok",
        label="TikTok",
        width=1080,
        height=1920,
        fps=30,
        safe=_TIKTOK_SAFE,
        max_duration_s=600.0,
        min_duration_s=3.0,
        sweet_spot_s=(9.0, 60.0),
    ),
    "shorts": Platform(
        key="shorts",
        label="YouTube Shorts",
        width=1080,
        height=1920,
        fps=30,
        safe=_SHORTS_SAFE,
        max_duration_s=180.0,
        min_duration_s=3.0,
        sweet_spot_s=(15.0, 60.0),
    ),
    "feed": Platform(
        key="feed",
        label="Instagram Feed (4:5)",
        width=1080,
        height=1350,
        fps=30,
        safe=_IG_FEED_SAFE,
        max_duration_s=60.0,
        min_duration_s=3.0,
        sweet_spot_s=(7.0, 30.0),
    ),
    "square": Platform(
        key="square",
        label="Square 1:1",
        width=1080,
        height=1080,
        fps=30,
        safe=_SQUARE_SAFE,
        max_duration_s=60.0,
        min_duration_s=3.0,
        sweet_spot_s=(7.0, 30.0),
    ),
    "landscape": Platform(
        key="landscape",
        label="Landscape 16:9",
        width=1920,
        height=1080,
        fps=30,
        safe=SafeZone(top=54, bottom=108, left=96, right=96),
        max_duration_s=43200.0,
        min_duration_s=1.0,
        sweet_spot_s=(30.0, 600.0),
    ),
}

DEFAULT_PLATFORM = "reels"


def get_platform(key: str) -> Platform:
    """Look up a delivery target, with a listing of valid keys on failure."""
    try:
        return PLATFORMS[key]
    except KeyError:
        valid = ", ".join(sorted(PLATFORMS))
        raise KeyError(f"unknown platform {key!r}; expected one of: {valid}") from None


# --- Encoder ladder ---------------------------------------------------------
#
# Three tiers, because the three questions you ask of a render are different.
#
#   draft    "are my cut points right?"      — 720p, ultrafast, throwaway
#   preview  "would I post this?"            — full res, honest quality
#   final    "post it"                       — full res, archival-ish
#
# Preview deliberately renders at full resolution: a 720p preview hides caption
# legibility and safe-zone problems, which are exactly what you preview for.


@dataclass(frozen=True)
class Quality:
    key: str
    preset: str
    crf: int
    #: Scale factor applied to the platform's canvas. Draft only.
    scale: float = 1.0


QUALITY: dict[str, Quality] = {
    "draft": Quality("draft", preset="ultrafast", crf=28, scale=2 / 3),
    "preview": Quality("preview", preset="veryfast", crf=23),
    "final": Quality("final", preset="slow", crf=19),
}


def get_quality(key: str) -> Quality:
    try:
        return QUALITY[key]
    except KeyError:
        valid = ", ".join(QUALITY)
        raise KeyError(f"unknown quality {key!r}; expected one of: {valid}") from None
