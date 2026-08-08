"""reelforge — agentic editor for vertical video.

The pipeline, in the order it runs:

    transcribe  word-level ASR, cached against source content
    pack        transcripts -> takes.md, the agent's reading view
    (agent)     reads takes.md, writes edl.json
    lint        retention and correctness report, before rendering
    render      reframe -> concat -> overlays -> captions -> mix -> normalise

The EDL is the boundary. Above it, judgement about what the video should be;
below it, deterministic execution. Nothing in the render path makes an editorial
decision, and nothing in the agent path touches a frame.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "EDL",
    "Range",
    "Overlay",
    "CaptionSpec",
    "render",
    "lint",
    "get_platform",
    "plan_reframe",
]


def __getattr__(name: str):
    """Expose the main API lazily.

    Importing `reelforge` should stay cheap — the CLI's `platforms` and `doctor`
    commands have no reason to pull in numpy or spawn a probe, and an agent
    checking the environment should not pay for the render path.
    """
    if name in ("EDL", "Range", "Overlay", "CaptionSpec"):
        from . import edl as _edl

        return getattr(_edl, name)
    if name == "render":
        from .render import render

        return render
    if name == "lint":
        from .retention import lint

        return lint
    if name == "get_platform":
        from .config import get_platform

        return get_platform
    if name == "plan_reframe":
        from .reframe import plan_reframe

        return plan_reframe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
