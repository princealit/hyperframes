"""Talking heads: a reference likeness plus a script becomes a new video.

Give it a photo or a clip of someone and words they never said, and it returns
them saying those words — mouth matched, and for a video reference the original
body movement preserved.

Three stages, each a swappable provider:

    reference ─┬─ still ──→ [animate] ──┐
               └─ video ────────────────┼─→ [lipsync] ──→ talking video
    script ────→ [voice] ───────────────┘

**voice** clones a target voice from a short reference recording and speaks new
text in it. **animate** only runs for stills, turning one frame into footage
with head and body motion — lipsyncing a motionless photo produces a mouth
moving on a mannequin. **lipsync** drives the mouth to match the audio.

The output is an ordinary MP4, so it flows straight into the rest of reelforge:
reframe to 9:16, caption, cut against other footage, render.

## What is verified and what is not

Fish Speech's request schema is taken from its own source (`ServeTTSRequest`),
and is identical for the hosted API and a self-hosted `tools/api_server.py`, so
one provider serves both via `base_url`.

The sync.so request shape could **not** be verified — its documentation is
unreachable from the environment this was written in. It is therefore declared
as data in `SYNC_SHAPE` rather than hard-coded through the call path, so a
mismatch is a one-line correction instead of a rewrite, and provider errors
surface the API's own response body rather than a generic failure.

Nothing here has been executed against a live endpoint. `preflight()` exists to
tell you which credentials actually work before you build on them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .ffmpeg import media_duration, probe, run
from .generate import GenerationError

Reference = Literal["image", "video"]

VOICES_DIRNAME = "voices"
AVATAR_DIRNAME = "avatars"

#: Poll interval and ceiling for asynchronous provider jobs. Lipsync on a long
#: clip genuinely takes minutes, so the ceiling is generous; the interval backs
#: off so a slow job does not generate hundreds of requests.
POLL_INITIAL = 3.0
POLL_MAX = 15.0
POLL_TIMEOUT = 1800.0

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"})


class AvatarError(RuntimeError):
    pass


def _requests():
    try:
        import requests
    except ImportError:
        raise AvatarError(
            "network providers need `requests` — pip install 'reelforge[transcribe]'"
        ) from None
    return requests


def classify_reference(path: Path) -> Reference:
    """Is this reference a still or moving footage?

    Decided by probing rather than by extension: a single-frame MP4 is a still
    in everything but its container, and treating it as video skips the
    animation stage that makes it watchable.
    """
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix not in VIDEO_SUFFIXES:
        raise AvatarError(f"unsupported reference type: {path.name}")
    try:
        info = probe(path)
    except Exception as e:  # noqa: BLE001
        raise AvatarError(f"could not read reference {path.name}: {e}") from None
    # A container holding under a quarter-second is a still frame in practice.
    return "image" if info.duration < 0.25 else "video"


# --- Voice ------------------------------------------------------------------


@dataclass
class VoiceProfile:
    """A cloned voice: reference audio, and what is said in it.

    Fish Speech clones in context — it is handed the reference audio and its
    transcript at generation time rather than being fine-tuned. The transcript
    matters: it tells the model which sounds in the reference map to which
    graphemes, and cloning quality drops noticeably without it.
    """

    name: str
    #: 15-30s of clean speech. More is not better; noise and music are worse.
    reference_audio: Path
    #: What the reference recording actually says, verbatim.
    reference_text: str = ""
    provider: str = "fish"
    #: A voice already registered with the provider, used instead of uploading.
    voice_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reference_audio"] = str(self.reference_audio)
        return d

    @classmethod
    def load(cls, path: Path) -> "VoiceProfile":
        data = json.loads(Path(path).read_text())
        data["reference_audio"] = Path(data["reference_audio"])
        return cls(**data)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def prepare_reference_audio(source: Path, out_path: Path, *, max_seconds: float = 30.0) -> Path:
    """Normalise a reference recording for cloning.

    Mono 44.1kHz, loudness-levelled, and trimmed. The trim is not arbitrary:
    cloning quality plateaus around half a minute and long references cost
    tokens and latency on every single generation thereafter, since the
    reference is resent with each request.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(source),
        "-t", f"{max_seconds:.2f}",
        "-vn", "-ac", "1", "-ar", "44100",
        # Levelling only — no denoise or EQ. Whatever colour the reference has
        # is the voice being cloned, and processing it away clones the
        # processing instead.
        "-af", "loudnorm=I=-18:TP=-2:LRA=11",
        "-c:a", "pcm_s16le", str(out_path),
    ])
    return out_path


def _fish_endpoint() -> tuple[str, dict[str, str]]:
    """Resolve the Fish endpoint and auth headers.

    `FISH_BASE_URL` points at a self-hosted `tools/api_server.py`; unset, it
    targets the hosted API. The request schema is the same either way — that is
    why one provider covers both — but only the hosted one needs a key.
    """
    base = os.environ.get("FISH_BASE_URL", "https://api.fish.audio").rstrip("/")
    headers = {"content-type": "application/json"}
    key = os.environ.get("FISH_API_KEY", "")
    if "fish.audio" in base:
        if not key:
            raise AvatarError(
                "FISH_API_KEY is not set. Get one at fish.audio, or point "
                "FISH_BASE_URL at a self-hosted server (which needs no key)."
            )
        headers["authorization"] = f"Bearer {key}"
    elif key:
        headers["authorization"] = f"Bearer {key}"
    return base, headers


def speak(
    text: str,
    voice: VoiceProfile,
    out_path: Path,
    *,
    model: str = "s1",
    fmt: str = "wav",
    seed: int | None = None,
    timeout: int = 600,
) -> Path:
    """Speak `text` in the cloned voice.

    Schema is `ServeTTSRequest` from the Fish Speech source: either a
    `reference_id` for a voice already held by the provider, or a `references`
    array carrying the raw reference audio inline as base64 with its transcript.

    Language is not a parameter — the model infers it from the script, so
    Persian text in Persian script generates Persian. That is the reason to
    prefer Fish here over a model with a fixed language list: nothing has to
    claim Farsi support for Farsi text to work, and no phonetic transliteration
    into a neighbouring language is required.
    """
    # Endpoint and credentials first: a missing key should report the missing
    # key, not a missing HTTP library.
    base, headers = _fish_endpoint()
    requests = _requests()
    # The model is selected by header on the hosted API and ignored by a
    # self-hosted server, which serves whatever weights it was started with.
    headers["model"] = model

    payload: dict[str, Any] = {
        "text": text,
        "format": fmt,
        "normalize": True,
        "chunk_length": 200,
    }
    if seed is not None:
        payload["seed"] = seed

    if voice.voice_id:
        payload["reference_id"] = voice.voice_id
    else:
        if not voice.reference_audio.exists():
            raise AvatarError(f"reference audio not found: {voice.reference_audio}")
        payload["references"] = [{
            "audio": base64.b64encode(voice.reference_audio.read_bytes()).decode(),
            "text": voice.reference_text,
        }]
        if not voice.reference_text:
            # Not fatal, but the clone will be measurably worse.
            payload["references"][0]["text"] = ""

    try:
        resp = requests.post(
            f"{base}/v1/tts", headers=headers, json=payload, timeout=timeout
        )
    except Exception as e:  # noqa: BLE001
        raise AvatarError(f"speech request failed: {e}") from None
    if resp.status_code >= 300:
        raise AvatarError(f"speech failed (HTTP {resp.status_code}): {resp.text[:400]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    if out_path.stat().st_size < 1024:
        raise AvatarError(
            f"speech returned {out_path.stat().st_size} bytes — likely an error "
            "payload rather than audio"
        )
    return out_path


# --- Driver preparation -----------------------------------------------------


def extend_driver(source: Path, out_path: Path, target_duration: float) -> Path:
    """Make a driver clip at least `target_duration` long, by ping-pong looping.

    The stage everyone skips, and the one that produces the most obviously
    broken output when skipped. A lipsync provider given a 6-second reference
    and 25 seconds of audio truncates to the shorter of the two, so most of the
    script silently vanishes.

    The loop is a ping-pong — forward, then reversed, repeating — rather than a
    plain repeat, because a plain loop cuts from the last frame back to the
    first and lands a visible jump on every cycle. Playing it back and forth
    makes the seam continuous: the motion reverses, which reads as the subject
    shifting rather than as an edit.
    """
    info = probe(source)
    if info.duration >= target_duration:
        return source

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Each ping-pong cycle is two passes of the source.
    cycles = int(target_duration / (info.duration * 2)) + 1
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(source),
        "-filter_complex",
        f"[0:v]split[fwd][r];[r]reverse[rev];[fwd][rev]concat=n=2:v=1:a=0,"
        f"loop=loop={cycles}:size=32767:start=0,trim=duration={target_duration:.3f},"
        f"setpts=N/FRAME_RATE/TB[v]",
        "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(out_path),
    ])
    return out_path


def animate_still(
    image: Path,
    out_path: Path,
    duration: float,
    *,
    prompt: str = "subtle natural head movement, breathing, slight body sway, "
                  "speaking to camera, static background",
    provider: str = "auto",
    work_dir: Path | None = None,
) -> Path:
    """Turn a still into footage with motion, ready to be lipsynced.

    Lipsyncing a static photo animates a mouth on an otherwise frozen face, and
    the result reads as a puppet rather than a person. A few seconds of head and
    shoulder movement is what sells it, and `extend_driver` loops that into
    however long the script runs.

    The prompt deliberately asks for *subtle* motion and a static background:
    large gestures fight the lipsync, and a moving background makes the loop
    seam visible.
    """
    from .generate import GenRequest, generate, resolve_provider

    work = work_dir or out_path.parent
    p = resolve_provider("video", provider)
    if p.offline:
        raise AvatarError(
            "animating a still needs a real image-to-video provider "
            "(REPLICATE_API_TOKEN), or supply a video reference instead of a photo"
        )

    req = GenRequest(
        kind="video", prompt=prompt, duration=duration,
        options={"model": os.environ.get(
            "REELFORGE_ANIMATE_MODEL", "kwaivgi/kling-v1.6-standard"
        ), "start_image": str(image), "image": str(image)},
    )
    asset = generate(req, work, provider=p.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if asset.path != out_path:
        out_path.write_bytes(asset.path.read_bytes())
    return out_path


# --- Lipsync ----------------------------------------------------------------

#: sync.so request shape, declared as data because it could not be verified
#: against live documentation. If the API disagrees, correct it here — nothing
#: else in the call path encodes the shape.
SYNC_SHAPE: dict[str, Any] = {
    "base_url": "https://api.sync.so",
    "create_path": "/v2/generate",
    "status_path": "/v2/generate/{id}",
    "auth_header": "x-api-key",
    "default_model": "lipsync-2",
    # Field names read from the create response.
    "id_field": "id",
    "status_field": "status",
    "output_field": "outputUrl",
    "terminal_ok": ("COMPLETED",),
    "terminal_fail": ("FAILED", "REJECTED", "CANCELED", "TIMED_OUT"),
}


def _upload_target(path: Path | str) -> str:
    """Resolve a local file to something a remote provider can fetch.

    Hosted lipsync APIs take URLs, not uploads, so local files need to be
    somewhere reachable first. `REELFORGE_PUBLIC_BASE` maps a local directory
    onto a public base URL for exactly this; without it, this is the stage that
    fails, and it fails clearly rather than sending a `file://` path nobody can
    read.

    The URL check happens on the raw input, before any `Path` conversion:
    `Path("https://host/x")` collapses the double slash to `https:/host/x`, so
    round-tripping a URL through `Path` silently corrupts it.
    """
    text = str(path)
    if text.startswith(("http://", "https://")):
        return text
    base = os.environ.get("REELFORGE_PUBLIC_BASE", "").rstrip("/")
    root = os.environ.get("REELFORGE_PUBLIC_DIR", "")
    if base and root:
        try:
            rel = Path(path).resolve().relative_to(Path(root).resolve())
            return f"{base}/{rel.as_posix()}"
        except ValueError:
            pass
    raise AvatarError(
        f"{Path(text).name} is a local file and the lipsync API fetches by URL. "
        "Either pass an https URL, or set REELFORGE_PUBLIC_DIR and "
        "REELFORGE_PUBLIC_BASE so local files resolve to public URLs."
    )


def lipsync(
    driver: Path | str,
    audio: Path | str,
    out_path: Path,
    *,
    model: str | None = None,
    sync_mode: str = "loop",
    timeout: float = POLL_TIMEOUT,
    shape: dict[str, Any] | None = None,
) -> Path:
    """Drive the mouth in `driver` to match `audio`.

    Submits the job, polls with backoff, downloads the result. Both inputs must
    be URLs the provider can fetch; see `_upload_target`.

    `sync_mode` decides what happens when the two durations disagree — `loop`
    repeats the driver, `bounce` ping-pongs it, `cut_off` truncates. Preparing
    the driver locally with `extend_driver` first is better than relying on it,
    because the local version controls how the loop looks.
    """
    # Credentials and inputs are validated before the transport is imported, so
    # a missing key reports the missing key rather than a missing library.
    cfg = {**SYNC_SHAPE, **(shape or {})}
    key = os.environ.get("SYNC_API_KEY", "")
    if not key:
        raise AvatarError(
            "SYNC_API_KEY is not set. Get one at sync.so and put it in the "
            "environment or a .env file at the project root."
        )
    # Resolved before Path conversion — see `_upload_target`.
    driver_url = _upload_target(driver)
    audio_url = _upload_target(audio)
    requests = _requests()

    payload = {
        "model": model or cfg["default_model"],
        "input": [
            {"type": "video", "url": driver_url},
            {"type": "audio", "url": audio_url},
        ],
        "options": {"sync_mode": sync_mode},
    }
    headers = {cfg["auth_header"]: key, "Content-Type": "application/json"}

    try:
        resp = requests.post(
            f"{cfg['base_url']}{cfg['create_path']}",
            headers=headers, json=payload, timeout=120,
        )
    except Exception as e:  # noqa: BLE001
        raise AvatarError(f"lipsync request failed: {e}") from None
    if resp.status_code >= 300:
        raise AvatarError(
            f"lipsync submission failed (HTTP {resp.status_code}): {resp.text[:400]}"
        )

    data = resp.json()
    job_id = data.get(cfg["id_field"])
    if not job_id:
        raise AvatarError(f"no job id in response: {json.dumps(data)[:300]}")

    status_url = f"{cfg['base_url']}{cfg['status_path'].format(id=job_id)}"
    deadline = time.monotonic() + timeout
    interval = POLL_INITIAL
    last: dict[str, Any] = {}

    while time.monotonic() < deadline:
        time.sleep(interval)
        interval = min(interval * 1.5, POLL_MAX)
        try:
            poll = requests.get(status_url, headers={cfg["auth_header"]: key}, timeout=60)
        except Exception as e:  # noqa: BLE001
            raise AvatarError(f"lipsync polling failed: {e}") from None
        if poll.status_code >= 300:
            raise AvatarError(
                f"lipsync status failed (HTTP {poll.status_code}): {poll.text[:300]}"
            )
        last = poll.json()
        status = str(last.get(cfg["status_field"], "")).upper()
        if status in cfg["terminal_ok"]:
            break
        if status in cfg["terminal_fail"]:
            raise AvatarError(
                f"lipsync job {job_id} ended {status}: "
                f"{last.get('error') or json.dumps(last)[:300]}"
            )
    else:
        raise AvatarError(f"lipsync job {job_id} did not finish within {timeout:.0f}s")

    url = last.get(cfg["output_field"])
    if not url:
        raise AvatarError(f"job completed without an output URL: {json.dumps(last)[:300]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    got = requests.get(url, timeout=600)
    if got.status_code >= 300:
        raise AvatarError(f"could not download result: HTTP {got.status_code}")
    out_path.write_bytes(got.content)
    return out_path


# --- Orchestration ----------------------------------------------------------


@dataclass
class TalkingHeadResult:
    output: Path
    reference_kind: Reference
    duration: float
    speech: Path
    driver: Path
    voice: str
    steps: list[str] = field(default_factory=list)


def talking_head(
    reference: Path,
    script: str,
    voice: VoiceProfile,
    out_path: Path,
    *,
    work_dir: Path | None = None,
    animate_seconds: float = 5.0,
    model: str | None = None,
    animate_provider: str = "auto",
    verbose: bool = True,
) -> TalkingHeadResult:
    """Reference plus script becomes a video of that likeness speaking it.

    Order matters. Speech is generated *first* because its duration determines
    everything downstream: how long the driver has to be, and therefore how much
    motion has to be generated or looped. Generating video to a guessed length
    and discovering the narration overruns it means paying for the video twice.
    """
    work = work_dir or out_path.parent / ".reelforge" / AVATAR_DIRNAME
    work.mkdir(parents=True, exist_ok=True)
    steps: list[str] = []

    def log(msg: str) -> None:
        steps.append(msg)
        if verbose:
            print(msg)

    reference = Path(reference)
    if not reference.exists():
        raise AvatarError(f"reference not found: {reference}")
    kind = classify_reference(reference)
    log(f"reference  {reference.name} ({kind})")

    # 1. Speech first — its length drives every later decision.
    digest = hashlib.sha256(f"{script}{voice.name}".encode()).hexdigest()[:12]
    speech = work / f"speech_{digest}.wav"
    if speech.exists():
        log(f"speech     cached ({media_duration(speech):.2f}s)")
    else:
        speak(script, voice, speech)
        log(f"speech     {media_duration(speech):.2f}s in '{voice.name}'")
    speech_len = media_duration(speech)
    if speech_len <= 0:
        raise AvatarError("generated speech has zero duration")

    # 2. A still needs motion before it can be lipsynced.
    if kind == "image":
        animated = work / f"animated_{reference.stem}.mp4"
        if not animated.exists():
            animate_still(
                reference, animated, min(animate_seconds, 10.0),
                provider=animate_provider, work_dir=work,
            )
        log(f"animate    still -> {media_duration(animated):.2f}s of motion")
        base_driver = animated
    else:
        base_driver = reference

    # 3. The driver must cover the audio or the tail is silently discarded.
    driver = base_driver
    if probe(base_driver).duration < speech_len:
        driver = extend_driver(
            base_driver, work / f"driver_{digest}.mp4", speech_len + 0.5
        )
        log(
            f"extend     driver {probe(base_driver).duration:.2f}s -> "
            f"{probe(driver).duration:.2f}s (ping-pong loop)"
        )

    # 4. Lipsync.
    lipsync(driver, speech, out_path, model=model)
    duration = media_duration(out_path)
    log(f"lipsync    {out_path.name} ({duration:.2f}s)")

    return TalkingHeadResult(
        output=out_path, reference_kind=kind, duration=duration,
        speech=speech, driver=driver, voice=voice.name, steps=steps,
    )


# --- Preflight --------------------------------------------------------------


def preflight(timeout: int = 20) -> dict[str, Any]:
    """Test every configured credential and report which actually work.

    Worth its own function because this stack has five independent providers,
    each failing differently — a bad key, an unreachable host and an
    out-of-credit account all present as "it didn't work" at the point of use,
    hours into a pipeline. Better to find out in one call, before building.

    Never raises: a dead provider is a result, not an error.
    """
    out: dict[str, Any] = {}

    def probe_endpoint(name: str, url: str, headers: dict[str, str], env: str) -> None:
        if env and not os.environ.get(env):
            out[name] = {"configured": False, "detail": f"{env} not set"}
            return
        try:
            requests = _requests()
        except AvatarError as e:
            out[name] = {"configured": True, "ok": False, "detail": str(e)}
            return
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            # 401/403 means reachable but the credential was rejected — a very
            # different problem from an unreachable host, so they are separated.
            out[name] = {
                "configured": True,
                "ok": r.status_code < 300,
                "status": r.status_code,
                "detail": (
                    "ok" if r.status_code < 300
                    else "credential rejected" if r.status_code in (401, 403)
                    else r.text[:160]
                ),
            }
        except Exception as e:  # noqa: BLE001
            out[name] = {"configured": True, "ok": False,
                         "detail": f"unreachable: {type(e).__name__}: {e}"}

    fish_base = os.environ.get("FISH_BASE_URL", "https://api.fish.audio").rstrip("/")
    fish_key = os.environ.get("FISH_API_KEY", "")
    probe_endpoint(
        "fish", f"{fish_base}/v1/tts",
        {"authorization": f"Bearer {fish_key}"} if fish_key else {},
        "" if "fish.audio" not in fish_base else "FISH_API_KEY",
    )
    probe_endpoint(
        "sync", f"{SYNC_SHAPE['base_url']}{SYNC_SHAPE['create_path']}",
        {SYNC_SHAPE["auth_header"]: os.environ.get("SYNC_API_KEY", "")},
        "SYNC_API_KEY",
    )
    probe_endpoint(
        "replicate", "https://api.replicate.com/v1/account",
        {"Authorization": f"Bearer {os.environ.get('REPLICATE_API_TOKEN', '')}"},
        "REPLICATE_API_TOKEN",
    )
    probe_endpoint(
        "openrouter", "https://openrouter.ai/api/v1/key",
        {"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"},
        "OPENROUTER_API_KEY",
    )
    probe_endpoint(
        "together", "https://api.together.xyz/v1/models",
        {"Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY', '')}"},
        "TOGETHER_API_KEY",
    )

    out["public_url_mapping"] = {
        "configured": bool(
            os.environ.get("REELFORGE_PUBLIC_BASE")
            and os.environ.get("REELFORGE_PUBLIC_DIR")
        ),
        "detail": (
            "local files can be exposed to URL-fetching providers"
            if os.environ.get("REELFORGE_PUBLIC_BASE")
            else "REELFORGE_PUBLIC_DIR/REELFORGE_PUBLIC_BASE unset — lipsync "
                 "cannot read local files and will need https URLs"
        ),
    }
    return out
