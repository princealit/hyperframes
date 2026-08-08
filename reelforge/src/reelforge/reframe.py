"""Turn any-aspect footage into a vertical frame that keeps the subject in shot.

This is the capability the upstream tools don't have. video-use scales a source
to 1080p and preserves its orientation; landscape footage stays landscape. For
Instagram that is the whole problem — you either crop to 9:16 or you post a
letterboxed clip that reads as reposted desktop content.

Cropping is easy. Cropping *without looking like a security camera* is the work,
and it splits into two halves:

1. **Where is the subject?** Face detection when OpenCV is installed, a
   gradient/motion saliency estimate when it isn't, centre as the floor. All
   three return the same shape, so the rest of the pipeline is detector-agnostic.

2. **How should the camera respond?** Not by following the subject. A crop
   window locked to a detection centroid jitters on every frame and induces
   motion sickness. Real operators hold a frame until the subject drifts far
   enough to matter, then glide — so that is what `VirtualOperator` does:
   a deadband, an eased approach, and a hard velocity ceiling.

The output is a `ReframePlan` carrying an ffmpeg `crop` expression, which the
renderer splices into the per-segment extract filter chain.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .ffmpeg import even, run_binary

ReframeMode = Literal["track", "static", "center", "blur_pad", "fit"]

#: Resolution the tracker analyses at. Detection quality is flat above roughly
#: 480px for faces that occupy a meaningful share of frame, and decode cost is
#: quadratic, so sampling wider only burns time.
ANALYSIS_WIDTH = 480

#: Frames per second sampled for tracking. Subject position is a low-frequency
#: signal — a speaker does not translate meaningfully within 250ms — so 4fps
#: captures the movement that matters at 1/8th the decode of full rate.
ANALYSIS_FPS = 4.0


# --- Geometry ---------------------------------------------------------------


@dataclass(frozen=True)
class CropGeometry:
    """The crop rectangle's fixed dimensions and which axis is free to move."""

    crop_w: int
    crop_h: int
    src_w: int
    src_h: int
    axis: Literal["x", "y", "none"]

    @property
    def travel(self) -> int:
        """Total distance the crop window can move along its free axis."""
        if self.axis == "x":
            return max(0, self.src_w - self.crop_w)
        if self.axis == "y":
            return max(0, self.src_h - self.crop_h)
        return 0


def plan_geometry(src_w: int, src_h: int, target_aspect: float) -> CropGeometry:
    """Largest centred rectangle of `target_aspect` that fits inside the source.

    Whichever source dimension is in surplus becomes the free axis; the other is
    consumed entirely. Dimensions are forced even for yuv420p.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"invalid source dimensions: {src_w}x{src_h}")
    if target_aspect <= 0:
        raise ValueError(f"invalid target aspect: {target_aspect}")

    src_aspect = src_w / src_h
    # A 1px tolerance keeps float noise from inventing a 2px pan on footage
    # that already matches the target.
    if abs(src_aspect - target_aspect) < 1e-3:
        return CropGeometry(even(src_w), even(src_h), src_w, src_h, "none")

    if src_aspect > target_aspect:
        # Source is wider than the target: full height, crop the width.
        crop_h = even(src_h)
        crop_w = even(min(src_w, crop_h * target_aspect))
        axis: Literal["x", "y", "none"] = "x"
    else:
        # Source is taller: full width, crop the height.
        crop_w = even(src_w)
        crop_h = even(min(src_h, crop_w / target_aspect))
        axis = "y"

    crop_w = max(2, min(crop_w, even(src_w)))
    crop_h = max(2, min(crop_h, even(src_h)))
    if crop_w >= even(src_w) and crop_h >= even(src_h):
        axis = "none"
    return CropGeometry(crop_w, crop_h, src_w, src_h, axis)


# --- Frame sampling ---------------------------------------------------------


def sample_frames(
    source: Path,
    start: float,
    duration: float,
    *,
    fps: float = ANALYSIS_FPS,
    width: int = ANALYSIS_WIDTH,
) -> tuple[np.ndarray, float]:
    """Decode a segment to a greyscale array of shape (n, h, w).

    Returns the stack and the interval between samples. Greyscale because every
    detector here is luma-only, and it cuts the pipe volume by two thirds.
    """
    if duration <= 0:
        return np.zeros((0, 1, 1), dtype=np.uint8), 1.0 / fps

    # Height is derived from the scale filter rather than assumed, so the
    # reshape below cannot silently mis-frame the buffer.
    from .ffmpeg import probe

    info = probe(source)
    disp_w, disp_h = info.display_size
    height = even(width * disp_h / disp_w) if disp_w else width

    raw = run_binary([
        "ffmpeg", "-v", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source),
        "-vf", f"fps={fps},scale={width}:{height}",
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "-",
    ])
    frame_bytes = width * height
    n = len(raw) // frame_bytes if frame_bytes else 0
    if n == 0:
        return np.zeros((0, height, width), dtype=np.uint8), 1.0 / fps
    stack = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape(n, height, width)
    return stack, 1.0 / fps


# --- Subject detection ------------------------------------------------------

#: Per-frame normalised subject position in [0,1]^2, or None when unknown.
Track = list[tuple[float, float] | None]


def _opencv():
    """Import OpenCV lazily; it is an optional extra, not a hard dependency."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    return cv2


def detect_faces(frames: np.ndarray) -> Track:
    """Per-frame face centroid via Haar cascades, or an all-None track.

    Frontal and profile cascades both run, because a speaker who turns their
    head vanishes from the frontal detector alone and the resulting gap reads
    as a subject that teleports. Where several faces are found the largest
    wins — in practice the person closest to camera, who is the subject.
    """
    cv2 = _opencv()
    if cv2 is None or len(frames) == 0:
        return [None] * len(frames)

    cascade_dir = Path(getattr(cv2.data, "haarcascades", ""))
    cascades = []
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        path = cascade_dir / name
        if path.exists():
            clf = cv2.CascadeClassifier(str(path))
            if not clf.empty():
                cascades.append(clf)
    if not cascades:
        return [None] * len(frames)

    h, w = frames.shape[1:3]
    # Faces smaller than ~8% of frame width are usually background bystanders
    # or false positives on texture, and following them is worse than not.
    min_size = max(16, int(w * 0.08))
    track: Track = []
    for frame in frames:
        img = np.ascontiguousarray(frame)
        best: tuple[int, int, int, int] | None = None
        best_area = 0
        for clf in cascades:
            found = clf.detectMultiScale(
                img, scaleFactor=1.15, minNeighbors=5, minSize=(min_size, min_size)
            )
            for (fx, fy, fw, fh) in found:
                area = int(fw) * int(fh)
                if area > best_area:
                    best_area = area
                    best = (int(fx), int(fy), int(fw), int(fh))
        if best is None:
            track.append(None)
        else:
            fx, fy, fw, fh = best
            # Bias the vertical anchor above the box centre. Framing on the
            # eyeline rather than the nose is the difference between a portrait
            # and a mugshot.
            cx = (fx + fw / 2) / w
            cy = (fy + fh * 0.42) / h
            track.append((cx, cy))
    return track


def detect_saliency(frames: np.ndarray) -> Track:
    """Detector of last resort: where is the detail and the movement?

    Combines spatial gradient energy (detail attracts the eye and backgrounds
    are typically flatter than subjects) with inter-frame absolute difference
    (things that move are things that matter). Both are reduced to marginal
    distributions over each axis and combined into a weighted centroid.

    Crude next to a face detector, but it needs nothing beyond numpy and
    degrades sensibly on b-roll, screencasts and hands-on product shots where
    there is no face to find.
    """
    n = len(frames)
    if n == 0:
        return []

    f = frames.astype(np.float32)
    # Spatial detail: gradient magnitude, approximated with forward differences.
    gy = np.abs(np.diff(f, axis=1, prepend=f[:, :1, :]))
    gx = np.abs(np.diff(f, axis=2, prepend=f[:, :, :1]))
    detail = gx + gy

    # Temporal motion, with the first frame reusing the second's delta so the
    # array stays aligned with the sample times.
    if n > 1:
        motion = np.abs(np.diff(f, axis=0, prepend=f[:1]))
        motion[0] = motion[1] if n > 1 else 0
    else:
        motion = np.zeros_like(f)

    # Motion is the stronger signal for "the subject" but is noisy frame to
    # frame; detail is stable but happily locks onto a busy background.
    energy = detail + 2.0 * motion

    h, w = frames.shape[1:3]
    xs = (np.arange(w) + 0.5) / w
    ys = (np.arange(h) + 0.5) / h

    track: Track = []
    for e in energy:
        col = e.sum(axis=0)
        row = e.sum(axis=1)
        col_total = col.sum()
        row_total = row.sum()
        if col_total <= 0 or row_total <= 0:
            track.append(None)
            continue
        # Subtract the median so a uniformly-textured frame reads as centred
        # rather than as whichever side carries marginally more noise.
        col = np.clip(col - np.median(col), 0, None)
        row = np.clip(row - np.median(row), 0, None)
        if col.sum() <= 0 or row.sum() <= 0:
            track.append(None)
            continue
        cx = float((col * xs).sum() / col.sum())
        cy = float((row * ys).sum() / row.sum())
        track.append((cx, cy))
    return track


def _coverage(track: Track) -> float:
    return sum(1 for t in track if t is not None) / len(track) if track else 0.0


def build_track(frames: np.ndarray, detector: str = "auto") -> tuple[Track, str]:
    """Resolve a subject track and report which detector actually produced it.

    ``auto`` prefers faces but only trusts them when they appear in at least a
    third of sampled frames. Below that the footage is not a talking head, and
    following the occasional false positive is worse than tracking saliency.
    """
    if len(frames) == 0:
        return [], "none"

    if detector == "center":
        return [(0.5, 0.5)] * len(frames), "center"
    if detector == "saliency":
        return detect_saliency(frames), "saliency"
    if detector == "face":
        return detect_faces(frames), "face"

    faces = detect_faces(frames)
    if _coverage(faces) >= 0.33:
        return faces, "face"
    sal = detect_saliency(frames)
    if _coverage(sal) > 0:
        return sal, "saliency"
    return [(0.5, 0.5)] * len(frames), "center"


def fill_gaps(track: Track) -> list[tuple[float, float]]:
    """Interpolate across frames where detection failed.

    Gaps are usually a blink, a turn or a motion-blurred frame — the subject did
    not actually leave. Interior gaps interpolate linearly between known
    neighbours; leading and trailing gaps hold the nearest known value; a track
    with nothing in it at all falls back to centre.
    """
    n = len(track)
    if n == 0:
        return []
    known = [i for i, t in enumerate(track) if t is not None]
    if not known:
        return [(0.5, 0.5)] * n

    out: list[tuple[float, float]] = [(0.5, 0.5)] * n
    for i in range(n):
        if track[i] is not None:
            out[i] = track[i]  # type: ignore[assignment]
            continue
        prev = next((k for k in reversed(known) if k < i), None)
        nxt = next((k for k in known if k > i), None)
        if prev is None and nxt is None:
            out[i] = (0.5, 0.5)
        elif prev is None:
            out[i] = track[nxt]  # type: ignore[index,assignment]
        elif nxt is None:
            out[i] = track[prev]  # type: ignore[index,assignment]
        else:
            span = nxt - prev
            a = (i - prev) / span
            px, py = track[prev]  # type: ignore[misc]
            nx, ny = track[nxt]  # type: ignore[misc]
            out[i] = (px + (nx - px) * a, py + (ny - py) * a)
    return out


def median_filter(values: Sequence[float], k: int = 5) -> list[float]:
    """Odd-window median, to drop single-sample detector outliers.

    A median is used rather than a mean because one badly wrong detection —
    a face found in a poster on the wall — would drag an average with it.
    """
    n = len(values)
    if n == 0 or k <= 1:
        return list(values)
    k = k if k % 2 else k + 1
    half = k // 2
    arr = np.asarray(values, dtype=np.float64)
    padded = np.pad(arr, (half, half), mode="edge")
    return [float(np.median(padded[i : i + k])) for i in range(n)]


# --- The virtual camera operator --------------------------------------------


@dataclass
class OperatorSettings:
    """How the crop window is allowed to behave.

    Defaults describe a careful operator on a tripod: hold the frame, ignore
    small drift, and when a move is genuinely needed make it slow and eased.
    """

    #: Subject drift, as a fraction of crop width, tolerated before moving.
    #: Roughly the "rule of thirds slack" a human operator allows.
    deadband: float = 0.10
    #: Fraction of remaining error closed per second once moving. Exponential
    #: approach, so the move decelerates into its target.
    approach: float = 2.5
    #: Ceiling on pan speed as a fraction of crop width per second. Above this
    #: a pan reads as a whip and calls attention to the reframe.
    max_speed: float = 0.35
    #: Error, as a fraction of the deadband, at which the move is declared done.
    settle: float = 0.25

    def __post_init__(self) -> None:
        if not 0 <= self.deadband < 1:
            raise ValueError("deadband must be in [0, 1)")
        if self.approach <= 0:
            raise ValueError("approach must be positive")
        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive")


def run_operator(
    targets: Sequence[float],
    dt: float,
    travel: int,
    crop_extent: int,
    settings: OperatorSettings | None = None,
) -> list[float]:
    """Convert per-sample subject positions into crop-window positions.

    `targets` are normalised subject centres along the free axis; the return is
    absolute crop-window offsets in source pixels, clamped to `[0, travel]`.

    The state machine is deliberately small: hold until the error exceeds the
    deadband, then ease toward the target under a speed cap until the error
    settles. That hysteresis — a wide gate to start moving, a narrow one to
    stop — is what stops the camera oscillating around the threshold.
    """
    s = settings or OperatorSettings()
    n = len(targets)
    if n == 0:
        return []
    if travel <= 0:
        return [0.0] * n

    deadband_px = s.deadband * crop_extent
    max_step = s.max_speed * crop_extent * dt

    def desired(norm_centre: float) -> float:
        # Place the crop window so the subject sits at its centre, then clamp
        # to the source. Clamping is what makes a subject near the frame edge
        # drift off-centre rather than the crop running past the footage.
        return min(max(norm_centre * (travel + crop_extent) - crop_extent / 2, 0.0), travel)

    pos = desired(targets[0])
    moving = False
    out: list[float] = []
    for target_norm in targets:
        goal = desired(target_norm)
        err = goal - pos
        if abs(err) > deadband_px:
            moving = True
        elif abs(err) < deadband_px * s.settle:
            moving = False
        if moving:
            step = err * min(1.0, s.approach * dt)
            step = max(-max_step, min(max_step, step))
            pos = min(max(pos + step, 0.0), float(travel))
        out.append(pos)
    return out


# --- Keyframe reduction and expression synthesis ----------------------------


def reduce_keyframes(
    times: Sequence[float], values: Sequence[float], tolerance: float = 1.5
) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification of the position curve.

    The operator emits a sample every 250ms; most are redundant because the
    curve is flat or straight. Dropping points that a straight line already
    predicts to within `tolerance` pixels keeps the generated ffmpeg expression
    to a length that stays readable and cheap to evaluate per frame.
    """
    n = len(times)
    if n != len(values):
        raise ValueError("times and values must be the same length")
    if n <= 2:
        return list(zip(times, values))

    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        t0, v0 = times[lo], values[lo]
        t1, v1 = times[hi], values[hi]
        span = t1 - t0
        worst, worst_i = 0.0, -1
        for i in range(lo + 1, hi):
            predicted = v0 if span <= 0 else v0 + (v1 - v0) * (times[i] - t0) / span
            dev = abs(values[i] - predicted)
            if dev > worst:
                worst, worst_i = dev, i
        if worst > tolerance and worst_i > 0:
            keep[worst_i] = True
            stack.append((lo, worst_i))
            stack.append((worst_i, hi))
    return [(times[i], values[i]) for i in range(n) if keep[i]]


def build_position_expr(keys: Sequence[tuple[float, float]]) -> str:
    """Emit an ffmpeg expression interpolating `keys` piecewise-linearly in `t`.

    crop re-evaluates its x/y expressions per frame with `t` bound to
    presentation time, so a nested `if` chain is enough to drive a pan without
    sendcmd files or per-frame filter rebuilds. Values are held flat before the
    first key and after the last.
    """
    if not keys:
        return "0"
    if len(keys) == 1:
        return f"{keys[0][1]:.2f}"

    # Built inside-out so the innermost else-branch is the final held value.
    expr = f"{keys[-1][1]:.2f}"
    for i in range(len(keys) - 2, -1, -1):
        t0, v0 = keys[i]
        t1, v1 = keys[i + 1]
        span = t1 - t0
        if span <= 1e-6:
            segment = f"{v1:.2f}"
        elif abs(v1 - v0) < 1e-3:
            segment = f"{v0:.2f}"
        else:
            slope = (v1 - v0) / span
            segment = f"({v0:.2f}+{slope:.4f}*(t-{t0:.3f}))"
        expr = f"if(lt(t,{t1:.3f}),{segment},{expr})"
    # Hold the first value for anything before the first key.
    t_first, v_first = keys[0]
    if t_first > 1e-6:
        expr = f"if(lt(t,{t_first:.3f}),{v_first:.2f},{expr})"
    return expr


# --- Plan -------------------------------------------------------------------


@dataclass
class ReframePlan:
    """A resolved reframe: the filter chain plus why it looks the way it does."""

    mode: ReframeMode
    geometry: CropGeometry
    detector: str
    #: (time, position) along the free axis, in source pixels.
    keys: list[tuple[float, float]] = field(default_factory=list)
    #: Filter fragment, crop and scale included, ready for the extract chain.
    filter_chain: str = ""
    #: Total pan distance in pixels — a proxy for how busy the reframe is.
    motion_px: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def is_static(self) -> bool:
        return len(self.keys) <= 1 or self.motion_px < 1.0


def _scale_and_pad(out_w: int, out_h: int) -> str:
    """Fit-to-canvas with black bars — the honest fallback when not cropping."""
    return (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black"
    )


def _blur_pad_chain(out_w: int, out_h: int, blur: int = 40) -> str:
    """Fit the whole frame over a blurred, zoomed copy of itself.

    Preserves the original composition — the right call for a screencast or
    anything with text near the edges, where a crop would sever content — while
    still filling the canvas so the post doesn't read as letterboxed.
    """
    return (
        f"split=2[bg][fg];"
        f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},gblur=sigma={blur},eq=brightness=-0.06[bgb];"
        f"[fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
    )


def plan_reframe(
    source: Path,
    start: float,
    duration: float,
    out_w: int,
    out_h: int,
    *,
    mode: ReframeMode = "track",
    detector: str = "auto",
    settings: OperatorSettings | None = None,
    src_size: tuple[int, int] | None = None,
) -> ReframePlan:
    """Analyse a segment and produce the filter chain that verticalises it."""
    if src_size is None:
        from .ffmpeg import probe

        src_size = probe(source).display_size
    src_w, src_h = src_size
    target_aspect = out_w / out_h
    geom = plan_geometry(src_w, src_h, target_aspect)

    if mode == "fit":
        return ReframePlan(
            mode=mode, geometry=geom, detector="none",
            filter_chain=_scale_and_pad(out_w, out_h),
            notes=["letterboxed to fit; no crop applied"],
        )

    if mode == "blur_pad":
        return ReframePlan(
            mode=mode, geometry=geom, detector="none",
            filter_chain=_blur_pad_chain(out_w, out_h),
            notes=["full frame preserved over a blurred backdrop"],
        )

    scale_tail = f"scale={out_w}:{out_h}"

    if geom.axis == "none" or geom.travel == 0:
        return ReframePlan(
            mode=mode, geometry=geom, detector="none",
            filter_chain=f"crop={geom.crop_w}:{geom.crop_h}:0:0,{scale_tail}"
            if (geom.crop_w, geom.crop_h) != (src_w, src_h) else scale_tail,
            notes=["source already matches the target aspect"],
        )

    if mode == "center":
        pos = geom.travel / 2
        x = f"{pos:.0f}" if geom.axis == "x" else "0"
        y = f"{pos:.0f}" if geom.axis == "y" else "0"
        return ReframePlan(
            mode=mode, geometry=geom, detector="center",
            keys=[(0.0, pos)],
            filter_chain=f"crop={geom.crop_w}:{geom.crop_h}:{x}:{y},{scale_tail}",
            notes=["fixed centre crop"],
        )

    frames, dt = sample_frames(source, start, duration)
    notes: list[str] = []
    if len(frames) == 0:
        pos = geom.travel / 2
        x = f"{pos:.0f}" if geom.axis == "x" else "0"
        y = f"{pos:.0f}" if geom.axis == "y" else "0"
        return ReframePlan(
            mode=mode, geometry=geom, detector="none", keys=[(0.0, pos)],
            filter_chain=f"crop={geom.crop_w}:{geom.crop_h}:{x}:{y},{scale_tail}",
            notes=["could not sample frames; fell back to centre crop"],
        )

    track, used_detector = build_track(frames, detector)
    if used_detector == "face" and _opencv() is None:
        notes.append("OpenCV not installed — install reelforge[vision] for face tracking")
    filled = fill_gaps(track)

    axis_index = 0 if geom.axis == "x" else 1
    raw = [p[axis_index] for p in filled]
    smoothed = median_filter(raw, k=5)

    crop_extent = geom.crop_w if geom.axis == "x" else geom.crop_h
    times = [i * dt for i in range(len(smoothed))]

    if mode == "static":
        # One position for the whole segment: the median of where the subject
        # actually was, which beats the geometric centre whenever they stood
        # off to one side for most of the take.
        target = float(np.median(smoothed))
        positions = run_operator([target] * len(smoothed), dt, geom.travel, crop_extent, settings)
        keys = [(0.0, positions[0])]
        motion = 0.0
    else:
        positions = run_operator(smoothed, dt, geom.travel, crop_extent, settings)
        keys = reduce_keyframes(times, positions)
        motion = sum(abs(b[1] - a[1]) for a, b in zip(keys, keys[1:]))

    expr = build_position_expr(keys)
    x_expr = expr if geom.axis == "x" else "0"
    y_expr = expr if geom.axis == "y" else "0"
    chain = f"crop={geom.crop_w}:{geom.crop_h}:x='{x_expr}':y='{y_expr}',{scale_tail}"

    coverage = _coverage(track)
    notes.append(f"{used_detector} detector, {coverage:.0%} frame coverage, {len(keys)} keyframes")
    if used_detector == "saliency":
        notes.append("no reliable face found; tracked visual saliency instead")

    return ReframePlan(
        mode=mode, geometry=geom, detector=used_detector, keys=keys,
        filter_chain=chain, motion_px=motion, notes=notes,
    )


def vision_available() -> bool:
    """Whether face tracking can run in this environment."""
    cv2 = _opencv()
    if cv2 is None:
        return False
    return bool(shutil.which("ffmpeg")) and Path(
        getattr(cv2.data, "haarcascades", "")
    ).joinpath("haarcascade_frontalface_default.xml").exists()
