"""Thin, typed wrappers over ffmpeg/ffprobe.

Two jobs: keep subprocess plumbing out of the pipeline modules, and make
failures legible. An ffmpeg command that dies inside a 40-line filtergraph
produces a wall of stderr whose one useful line is usually near the end —
`FFmpegError` keeps the tail and the command so the caller can print both.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegError(RuntimeError):
    """An ffmpeg/ffprobe invocation exited non-zero."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        super().__init__(
            f"ffmpeg exited {returncode}\n"
            f"  command: {' '.join(cmd[:8])}{' ...' if len(cmd) > 8 else ''}\n"
            f"  stderr tail:\n{tail}"
        )


def require_ffmpeg() -> None:
    """Fail early and actionably when the toolchain is missing."""
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not found on PATH. "
            "Install with: brew install ffmpeg (macOS) / apt-get install ffmpeg (Debian)."
        )


def run(cmd: list[str], *, capture: bool = True) -> str:
    """Run a command, raising `FFmpegError` with a readable tail on failure."""
    cmd = [str(c) for c in cmd]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr or "")
    return proc.stdout or ""


def run_binary(cmd: list[str]) -> bytes:
    """Run a command and return raw stdout bytes (for piped rawvideo)."""
    cmd = [str(c) for c in cmd]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr.decode("utf-8", "replace"))
    return proc.stdout


# --- Probing ----------------------------------------------------------------

#: Transfer characteristics that mean the source carries HDR and must be
#: tone-mapped rather than merely bit-depth-reduced. PQ (HDR10) and HLG —
#: HLG being the iPhone default, which is why this matters so often.
HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    color_transfer: str
    #: Display rotation from container metadata, in degrees.
    rotation: int
    audio_sample_rate: int | None
    video_codec: str

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    @property
    def display_size(self) -> tuple[int, int]:
        """Size after container rotation is applied.

        A phone clip is frequently stored 1920x1080 with a 90-degree rotation
        flag; every geometry decision must be made against the *displayed*
        1080x1920, not the stored dimensions.
        """
        if self.rotation in (90, 270):
            return self.height, self.width
        return self.width, self.height


def _parse_fps(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            d = float(den)
            return float(num) / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def _parse_rotation(stream: dict) -> int:
    """Rotation lives in two different places depending on the muxer."""
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return int(float(tags["rotate"])) % 360
        except (TypeError, ValueError):
            pass
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                # Display-matrix rotation is reported as the negative of the
                # clockwise display rotation.
                return int(-float(sd["rotation"])) % 360
            except (TypeError, ValueError):
                pass
    return 0


def probe(path: str | Path) -> MediaInfo:
    """Read stream geometry, timing and colour metadata from a media file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"media not found: {p}")

    raw = run([
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json", str(p),
    ])
    data = json.loads(raw)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"no video stream in {p}")

    duration = 0.0
    for candidate in (video.get("duration"), (data.get("format") or {}).get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    fps = _parse_fps(video.get("avg_frame_rate") or "") or _parse_fps(
        video.get("r_frame_rate") or ""
    )

    sample_rate = None
    if audio is not None:
        try:
            sample_rate = int(audio["sample_rate"])
        except (KeyError, TypeError, ValueError):
            sample_rate = None

    return MediaInfo(
        path=p,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        duration=duration,
        has_audio=audio is not None,
        color_transfer=(video.get("color_transfer") or "").strip(),
        rotation=_parse_rotation(video),
        audio_sample_rate=sample_rate,
        video_codec=video.get("codec_name", ""),
    )


# --- Shared filter fragments ------------------------------------------------

#: Tone-map HDR (PQ/HLG) into clean Rec.709 SDR.
#:
#: Reducing bit depth alone leaves HLG/PQ transfer metadata on 8-bit values.
#: Players that honour the metadata — notably every social upload transcode —
#: then read SDR values as HDR and the result blows out. macOS QuickTime hides
#: this locally, so it typically ships undetected.
TONEMAP_CHAIN = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,"
    "format=yuv420p"
)


def audio_edge_fades(duration: float, fade: float = 0.03) -> str:
    """Short fades at both edges of a segment.

    A hard splice between two segments lands a step discontinuity in the
    waveform, which is audible as a click at every cut. 30ms is under the
    perceptual threshold for a fade but comfortably above the click.
    """
    fade = min(fade, max(duration / 4, 0.001))
    out_start = max(0.0, duration - fade)
    return f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={out_start:.3f}:d={fade:.3f}"


def even(n: float) -> int:
    """Round to the nearest even integer.

    yuv420p subsamples chroma 2x2, so odd width or height is not encodable.
    """
    return int(round(n / 2)) * 2
