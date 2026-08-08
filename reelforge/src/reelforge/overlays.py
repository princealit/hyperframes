"""HyperFrames bridge: motion graphics as first-class overlay slots.

video-use can drive HyperFrames, but only by handing a sub-agent a prose brief
and hoping the result comes back at usable dimensions. That fails in a specific,
repeated way: the composition gets authored at 1920x1080 because that is the
habit, then composited onto a 1080x1920 canvas where it either overflows or is
pillarboxed into a strip.

Here a slot is scaffolded *from* the delivery target. The composition is created
at the exact output size, with a content box already inset to the platform's safe
area, and rendered to WebM with alpha so it composites without a matte. The agent
writes the animation inside a container that is already the right shape.

Everything shells out to the `hyperframes` CLI, so there is no Python-side
dependency on the framework — a project without Node simply cannot make slots,
and everything else still works.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Platform
from .edl import Anchor, Overlay

#: Alpha-capable render formats. WebM (VP9 with yuva420p) is the default because
#: it is far smaller than ProRes 4444 and ffmpeg composites it directly.
ALPHA_FORMATS = {"webm": ".webm", "mov": ".mov"}


class HyperFramesError(RuntimeError):
    pass


def available() -> bool:
    """Whether the HyperFrames CLI can be invoked at all."""
    return shutil.which("npx") is not None


def _run(cmd: list[str], cwd: Path, timeout: int = 900) -> str:
    proc = subprocess.run(
        [str(c) for c in cmd], cwd=str(cwd),
        capture_output=True, text=True, timeout=timeout,
        # The scaffolder otherwise prompts to install agent skills, which hangs
        # a non-interactive run.
        env={**os.environ, "HYPERFRAMES_SKIP_SKILLS": "1", "CI": "1"},
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
        raise HyperFramesError(f"`{' '.join(cmd[:4])}` failed in {cwd}:\n{tail}")
    return proc.stdout


@dataclass
class Slot:
    """One overlay animation, living in its own directory."""

    slot_id: str
    directory: Path
    composition: Path
    duration: float
    width: int
    height: int
    render_path: Path | None = None

    def to_overlay(
        self,
        start_in_output: float,
        *,
        anchor: Anchor = "center",
        scale: float | None = None,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
    ) -> Overlay:
        """Build the EDL entry for this slot's rendered output."""
        if self.render_path is None:
            raise HyperFramesError(f"slot {self.slot_id} has not been rendered yet")
        return Overlay(
            file=str(self.render_path),
            start_in_output=start_in_output,
            duration=self.duration,
            anchor=anchor,
            scale=scale,
            alpha=True,
            fade_in=fade_in,
            fade_out=fade_out,
            label=self.slot_id,
            # A full-canvas slot is authored with the safe area already built in,
            # so anchoring it inside the safe area a second time would inset it
            # twice and shrink it against the frame it was designed for.
            ignore_safe_area=(scale is None),
        )


STARTER_COMPOSITION = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{slot_id}</title>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
<style>
  /* Transparent ground: this renders to WebM with alpha and composites over
     the footage, so anything opaque here becomes a box on the video. */
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  #root {{
    position: relative;
    width: {width}px; height: {height}px;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  /* Pre-inset to {platform} safe area. Keep content inside this box and it
     cannot land under the caption block or the action rail. */
  .safe {{
    position: absolute;
    left: {safe_left}px; top: {safe_top}px;
    width: {safe_w}px; height: {safe_h}px;
  }}
  .card {{
    position: absolute; left: 0; right: 0; top: 50%;
    transform: translateY(-50%);
    text-align: center;
    color: #fff;
    font-size: {font_size}px; font-weight: 800; letter-spacing: -0.02em;
    text-shadow: 0 4px 24px rgba(0,0,0,0.45);
    opacity: 0;
  }}
</style>
</head>
<body>
  <div id="root"
       data-composition-id="{slot_id}"
       data-width="{width}"
       data-height="{height}"
       data-duration="{duration}"
       data-fps="{fps}"
       data-start="0">
    <!-- Every timeline-visible element carries a stable id: the HyperFrames
         linter requires it so Studio and agents have a fixed edit target. -->
    <div class="clip" id="{slot_id}-clip" data-start="0" data-duration="{duration}">
      <div class="safe" id="{slot_id}-safe">
        <div class="card" id="{slot_id}-card">{placeholder}</div>
      </div>
    </div>
  </div>

<script>
  // One paused timeline per composition, built synchronously at load and
  // registered under the root's data-composition-id. The renderer seeks it
  // frame by frame, so nothing may depend on wall-clock time or unseeded
  // randomness — identical seeks must produce identical frames.
  const tl = gsap.timeline({{ paused: true }});
  tl.to("#{slot_id}-card", {{ opacity: 1, y: -12, duration: 0.45, ease: "power3.out" }}, 0.1)
    .to("#{slot_id}-card", {{ opacity: 0, duration: 0.35, ease: "power2.in" }}, {fade_at});
  window.__timelines = window.__timelines || {{}};
  window.__timelines["{slot_id}"] = tl;
</script>
</body>
</html>
"""


def create_slot(
    slot_id: str,
    slots_dir: Path,
    platform: Platform,
    duration: float,
    *,
    width: int | None = None,
    height: int | None = None,
    placeholder: str = "REPLACE ME",
    overwrite: bool = False,
) -> Slot:
    """Scaffold an overlay slot sized to the delivery target.

    The generated composition is a working starting point, not a finished
    graphic: it renders, passes lint, and shows one animated card. The agent is
    expected to replace its contents while keeping the root attributes and the
    timeline registration intact.
    """
    width = width or platform.width
    height = height or platform.height
    directory = slots_dir / f"slot_{slot_id}"
    if directory.exists() and not overwrite:
        composition = directory / "index.html"
        if composition.exists():
            return Slot(slot_id, directory, composition, duration, width, height)
    directory.mkdir(parents=True, exist_ok=True)

    safe = platform.safe_at(width, height)
    sx, sy, sw, sh = safe.content_box(width, height)
    composition = directory / "index.html"
    composition.write_text(
        STARTER_COMPOSITION.format(
            slot_id=slot_id, width=width, height=height,
            duration=f"{duration:.2f}", fps=platform.fps,
            safe_left=sx, safe_top=sy, safe_w=sw, safe_h=sh,
            font_size=max(28, int(width * 0.072)),
            platform=platform.label,
            placeholder=placeholder,
            fade_at=f"{max(0.5, duration - 0.4):.2f}",
        )
    )
    return Slot(slot_id, directory, composition, duration, width, height)


def lint_slot(slot: Slot) -> str:
    """Run the HyperFrames structural check on a slot."""
    if not available():
        raise HyperFramesError("npx not found — Node.js is required for HyperFrames slots")
    # The CLI takes the project directory, not the composition file.
    return _run(["npx", "--yes", "hyperframes", "lint", "."], slot.directory)


def render_slot(
    slot: Slot, *, fmt: str = "webm", output: str | None = None, timeout: int = 900
) -> Path:
    """Render a slot to an alpha-carrying overlay file."""
    if not available():
        raise HyperFramesError("npx not found — Node.js is required for HyperFrames slots")
    if fmt not in ALPHA_FORMATS:
        raise HyperFramesError(
            f"format {fmt!r} does not carry alpha; expected one of: {', '.join(ALPHA_FORMATS)}"
        )
    out_name = output or f"render{ALPHA_FORMATS[fmt]}"
    _run(
        ["npx", "--yes", "hyperframes", "render", ".",
         "--format", fmt, "--output", out_name],
        slot.directory, timeout=timeout,
    )
    path = slot.directory / out_name
    if not path.exists():
        raise HyperFramesError(f"render reported success but {path} is missing")
    slot.render_path = path
    return path


def slot_brief(
    slot: Slot,
    platform: Platform,
    goal: str,
    palette: dict[str, str] | None = None,
) -> str:
    """A self-contained brief for a sub-agent building this slot.

    Sub-agents inherit no context, so every constraint has to be restated. The
    ordering below is deliberate: goal, then the non-negotiables, then the
    aesthetic, then the deliverable. Ambiguity is resolved by instruction rather
    than by a question, because a sub-agent that stops to ask has stalled a
    parallel fan-out.
    """
    palette = palette or {}
    swatches = "\n".join(f"  {k}: {v}" for k, v in palette.items()) or "  (choose and state one)"
    safe = platform.safe_at(slot.width, slot.height)
    sx, sy, sw, sh = safe.content_box(slot.width, slot.height)
    return f"""Build ONE overlay animation. Nothing else.

GOAL
  {goal}

FILE
  Edit {slot.composition} in place. Do not create sibling compositions.

HARD CONSTRAINTS
  - Canvas is exactly {slot.width}x{slot.height} at {platform.fps}fps.
  - Duration is exactly {slot.duration:.2f}s. Root data-duration must match.
  - Background stays transparent. This composites over live footage; any
    opaque fill becomes a visible box.
  - Keep all content inside .safe ({sw}x{sh} at {sx},{sy}). Outside it the
    {platform.label} UI covers the frame.
  - Exactly one paused GSAP timeline, built synchronously, registered at
    window.__timelines["{slot.slot_id}"].
  - Deterministic only: no Date.now(), no unseeded Math.random(), no network
    fetches at render time. The renderer seeks frame by frame and identical
    seeks must produce identical frames.
  - Never ease linearly. Use power2/power3 or back for entrances.
  - Hold the final state at least 0.4s before the end.

PALETTE
{swatches}

DELIVERABLE
  1. `npx --yes hyperframes lint .` passes (run from the slot directory).
  2. `npx --yes hyperframes render . --format webm --output render.webm`.
  3. Confirm duration and dimensions with ffprobe.
  4. Report the rendered path and what you built in two sentences.

If anything is ambiguous, choose the most obvious reading and proceed. Do not
ask questions."""


def write_manifest(slots: list[Slot], path: Path) -> Path:
    """Record slot state so a later session can pick the project back up."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [
            {
                "slot_id": s.slot_id,
                "directory": str(s.directory),
                "composition": str(s.composition),
                "duration": s.duration,
                "width": s.width,
                "height": s.height,
                "render": str(s.render_path) if s.render_path else None,
            }
            for s in slots
        ],
        indent=2,
    ) + "\n")
    return path
