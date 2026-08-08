"""Colour treatment as ffmpeg filter chains.

Presets are starting points, not a menu to pick from blindly. The mental model
throughout is ASC CDL — per channel, `out = (in * slope + offset) ** power`,
then a global saturation move:

    slope   scales highlights      (gain)
    offset  lifts or crushes black (lift)
    power   bends midtones         (gamma)

`curves` supplies per-channel control, `eq` the global contrast/saturation move.
Any preset name may be replaced by a raw filter string, which is passed through
untouched — the grade field is an escape hatch by design.
"""

from __future__ import annotations

import re

#: Named chains. Each is deliberately gentle: a grade that survives the
#: platform's re-encode is worth more than one that looks strong locally and
#: bands after upload.
PRESETS: dict[str, str] = {
    "none": "",

    # Minimal corrective. Slight contrast and a touch of saturation — the
    # safe default when footage is already decent.
    "punch": "eq=contrast=1.08:saturation=1.10:brightness=0.01",

    # Warm highlights, cool shadows: the teal-and-orange split, dialled back
    # far enough to stay believable on skin.
    "cinematic": (
        "curves=r='0/0.02 0.5/0.52 1/0.98':b='0/0.03 0.5/0.48 1/0.96',"
        "eq=contrast=1.06:saturation=0.94"
    ),

    # Lifted blacks and reduced saturation — a matte, editorial look that
    # reads as intentional on talking heads.
    "matte": (
        "curves=all='0/0.06 0.25/0.28 0.75/0.78 1/0.96',"
        "eq=contrast=0.98:saturation=0.88"
    ),

    # Cool and clean, for screencasts and product shots where warmth reads
    # as a white-balance error rather than a choice.
    "crisp": (
        "curves=b='0/0.01 0.5/0.53 1/1',"
        "eq=contrast=1.10:saturation=1.02,unsharp=5:5:0.4:5:5:0.0"
    ),

    # Strong, saturated, high-contrast. Built for feeds where the first frame
    # competes against everything else on screen.
    "vivid": "eq=contrast=1.16:saturation=1.28:brightness=0.015,unsharp=5:5:0.3:5:5:0.0",

    # Neutral log-ish normalisation for flat-profile footage that would
    # otherwise look washed out straight off the camera.
    "restore_log": (
        "curves=all='0/0 0.1/0.05 0.5/0.52 0.9/0.95 1/1',"
        "eq=contrast=1.20:saturation=1.18"
    ),
}

#: Preset names are bare identifiers; anything containing filter syntax is
#: treated as a raw chain and passed through.
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_\-]+$")


def resolve_grade(value: str | None) -> str:
    """Turn a grade field into a filter chain.

    Accepts a preset name, a raw ffmpeg filter string, or nothing. An unknown
    bare identifier raises rather than silently rendering ungraded — a typo in a
    preset name should not quietly produce a different-looking video.
    """
    if not value:
        return ""
    value = value.strip()
    if not value or value == "none":
        return ""
    if _IDENTIFIER.match(value):
        try:
            return PRESETS[value]
        except KeyError:
            valid = ", ".join(sorted(PRESETS))
            raise KeyError(
                f"unknown grade preset {value!r}; expected one of: {valid} "
                "(or pass a raw ffmpeg filter string)"
            ) from None
    return value


def list_presets() -> list[str]:
    return sorted(PRESETS)
