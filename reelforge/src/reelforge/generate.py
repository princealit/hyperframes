"""Asset generation behind a provider abstraction.

OpenMontage's answer to generation is breadth: a hundred-odd tools and a scored
registry across a dozen vendors. That is the right shape for a system whose job
is choosing a provider, and the wrong shape here — reelforge needs generated
assets to drop into an EDL, not a procurement layer.

So this is deliberately thin. Four asset kinds — image, video, speech, music —
one interface, and providers registered against it. What the pipeline consumes
is a file path plus a ledger entry; where the bytes came from is the provider's
problem.

Two things earn their place beyond the API calls:

**Caching by request hash.** Generation is slow and metered. The same prompt,
provider and parameters resolve to the same asset directory, so re-running an
edit does not re-bill or re-wait for assets that already exist.

**Offline providers.** `mock` synthesises deterministic placeholder media with
ffmpeg and `espeak` does real local speech. Neither needs a key or a network,
which means the whole generate-to-render path is developable and testable
without spending anything — and stays usable when the network is not.

Cloud providers are declared with their real request shapes but are exercised
only when a key is present.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Literal

from .ffmpeg import run

Kind = Literal["image", "video", "speech", "music"]

#: Where generated assets and the ledger live, under the project work dir.
ASSETS_DIRNAME = "assets"
LEDGER_NAME = "ledger.json"


class GenerationError(RuntimeError):
    pass


@dataclass
class GenRequest:
    kind: Kind
    prompt: str
    #: Seconds. Meaningful for video, music and (as a target) speech.
    duration: float = 4.0
    width: int = 1080
    height: int = 1920
    #: Provider-specific extras: voice id, model name, style, seed.
    options: dict[str, Any] = field(default_factory=dict)

    def key(self, provider: str) -> str:
        """Stable hash of everything that affects the output."""
        payload = json.dumps(
            {
                "kind": self.kind,
                "prompt": self.prompt,
                "duration": round(self.duration, 3),
                "size": [self.width, self.height],
                "options": self.options,
                "provider": provider,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class Asset:
    path: Path
    kind: Kind
    prompt: str
    provider: str
    key: str
    duration: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


# --- Provider registry ------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    name: str
    kinds: tuple[Kind, ...]
    #: True when it runs locally with no key and no network.
    offline: bool
    #: Environment variable holding the credential, for cloud providers.
    env_key: str | None
    note: str
    #: (request, output_dir) -> produced file.
    run: Callable[[GenRequest, Path], Path]


_PROVIDERS: dict[str, Provider] = {}


def register(provider: Provider) -> Provider:
    _PROVIDERS[provider.name] = provider
    return provider


def available_providers(kind: Kind | None = None) -> list[dict[str, Any]]:
    """Providers, with whether each is actually usable right now."""
    out = []
    for p in _PROVIDERS.values():
        if kind and kind not in p.kinds:
            continue
        usable = p.offline or bool(p.env_key and os.environ.get(p.env_key))
        out.append({
            "name": p.name,
            "kinds": list(p.kinds),
            "offline": p.offline,
            "requires": p.env_key,
            "usable": usable,
            "note": p.note,
        })
    return out


def resolve_provider(kind: Kind, name: str = "auto") -> Provider:
    """Pick a provider, preferring a usable cloud one, then offline.

    `auto` never silently fails: if no cloud credential is present it falls back
    to the offline provider so the pipeline still produces a file, and the caller
    can see from the asset which provider actually ran.
    """
    if name != "auto":
        p = _PROVIDERS.get(name)
        if p is None:
            valid = ", ".join(sorted(_PROVIDERS))
            raise GenerationError(f"unknown provider {name!r}; expected one of: {valid}")
        if kind not in p.kinds:
            raise GenerationError(
                f"provider {name!r} does not produce {kind}; it handles: "
                f"{', '.join(p.kinds)}"
            )
        if not p.offline and p.env_key and not os.environ.get(p.env_key):
            raise GenerationError(
                f"provider {name!r} needs {p.env_key} in the environment or a .env file"
            )
        return p

    candidates = [p for p in _PROVIDERS.values() if kind in p.kinds]
    for p in candidates:
        if not p.offline and p.env_key and os.environ.get(p.env_key):
            return p
    for p in candidates:
        if p.offline:
            return p
    raise GenerationError(f"no provider available for {kind}")


# --- Offline providers ------------------------------------------------------


def _wrap(text: str, width: int = 22) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:6]


def _placeholder_colour(key: str) -> tuple[str, str]:
    """Derive a stable colour pair from the request hash.

    Deterministic so a given prompt always yields the same placeholder — an
    edit reviewed with placeholders stays visually recognisable on re-run.
    """
    h = int(key[:6], 16)
    hues = [
        ("#1B2A4A", "#7AA7FF"), ("#3A1B2E", "#FF7AA7"), ("#1B3A2E", "#7AFFC2"),
        ("#3A2E1B", "#FFC27A"), ("#2E1B3A", "#C27AFF"), ("#3A1B1B", "#FF9A7A"),
    ]
    return hues[h % len(hues)]


def _font() -> str | None:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        if Path(p).exists():
            return p
    return None


def _mock_run(req: GenRequest, out_dir: Path) -> Path:
    """Synthesise a labelled placeholder with ffmpeg.

    Not a stand-in for a real generation — a legible marker that carries its own
    prompt, so a rough cut assembled from placeholders is still reviewable for
    timing and structure before anything is paid for.
    """
    bg, fg = _placeholder_colour(req.key("mock"))
    font = _font()

    if req.kind == "image":
        out = out_dir / "asset.png"
        chain = f"color=c={bg}:s={req.width}x{req.height}"
        if font:
            lines = _wrap(req.prompt)
            size = max(28, req.width // 18)
            draws = [
                f"drawtext=fontfile={font}:text='{line.replace(chr(39), '')}'"
                f":fontcolor={fg}:fontsize={size}"
                f":x=(w-text_w)/2:y=(h-text_h)/2+{(i - len(lines) / 2) * size * 1.3:.0f}"
                for i, line in enumerate(lines)
            ]
            chain += "," + ",".join(draws)
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", chain,
             "-frames:v", "1", str(out)])
        return out

    if req.kind == "video":
        out = out_dir / "asset.mp4"
        # A slow drift so the placeholder is visibly moving footage rather than
        # a still, which matters when judging pacing in a rough cut.
        chain = f"color=c={bg}:s={req.width}x{req.height}:d={req.duration}:r=30"
        if font:
            lines = _wrap(req.prompt)
            size = max(28, req.width // 18)
            draws = [
                f"drawtext=fontfile={font}:text='{line.replace(chr(39), '')}'"
                f":fontcolor={fg}:fontsize={size}"
                f":x=(w-text_w)/2+20*sin(t):y=(h-text_h)/2+"
                f"{(i - len(lines) / 2) * size * 1.3:.0f}"
                for i, line in enumerate(lines)
            ]
            chain += "," + ",".join(draws)
        run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", chain,
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t", f"{req.duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out),
        ])
        return out

    if req.kind == "music":
        out = out_dir / "asset.m4a"
        # Two detuned sines and a slow tremolo: obviously placeholder, but with
        # enough movement to check a music bed's level and ducking behaviour.
        h = int(req.key("mock")[:4], 16)
        root = 110 + (h % 6) * 20
        run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i",
            f"sine=frequency={root}:duration={req.duration}",
            "-f", "lavfi", "-i",
            f"sine=frequency={root * 1.5:.1f}:duration={req.duration}",
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2,tremolo=f=2:d=0.4,volume=0.5[a]",
            "-map", "[a]", "-c:a", "aac", "-b:a", "160k", str(out),
        ])
        return out

    # speech
    out = out_dir / "asset.m4a"
    words = max(1, len(req.prompt.split()))
    est = max(1.0, words / 2.6)
    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"sine=frequency=180:duration={est:.2f}",
        "-af", "tremolo=f=6:d=0.7,volume=0.4",
        "-c:a", "aac", "-b:a", "128k", str(out),
    ])
    return out


def _espeak_run(req: GenRequest, out_dir: Path) -> Path:
    """Real speech, locally, with no key.

    Robotic, and the wrong choice for a voiceover anyone will hear. Right for
    building and timing a cut, because the durations are real — a narration
    track laid against picture needs the actual length, not an estimate.
    """
    if req.kind != "speech":
        raise GenerationError("espeak only produces speech")
    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        raise GenerationError(
            "espeak-ng not installed — apt-get install espeak-ng, or "
            "brew install espeak-ng"
        )
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    wav = out_dir / "_raw.wav"
    out = out_dir / "asset.m4a"
    voice = req.options.get("voice", "en-us")
    speed = str(req.options.get("speed", 150))
    proc = subprocess.run(
        [binary, "-v", str(voice), "-s", speed, "-w", str(wav), req.prompt],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not wav.exists():
        raise GenerationError(f"espeak failed: {proc.stderr[:300]}")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(wav),
         "-ar", "48000", "-ac", "1", "-c:a", "aac", "-b:a", "128k", str(out)])
    wav.unlink(missing_ok=True)
    return out


# --- Cloud providers --------------------------------------------------------


def _http_json(url: str, headers: dict[str, str], payload: dict, timeout: int = 300) -> dict:
    try:
        import requests
    except ImportError:
        raise GenerationError(
            "cloud providers need `requests` — pip install 'reelforge[transcribe]'"
        ) from None
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise GenerationError(f"request failed: {e}") from None
    if resp.status_code >= 300:
        raise GenerationError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _elevenlabs_run(req: GenRequest, out_dir: Path) -> Path:
    """ElevenLabs text-to-speech and music."""
    try:
        import requests
    except ImportError:
        raise GenerationError("needs `requests` — pip install 'reelforge[transcribe]'") from None

    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        raise GenerationError("ELEVENLABS_API_KEY is not set")
    out = out_dir / "asset.mp3"

    if req.kind == "speech":
        voice = req.options.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        payload: dict[str, Any] = {
            "text": req.prompt,
            "model_id": req.options.get("model_id", "eleven_multilingual_v2"),
        }
    elif req.kind == "music":
        url = "https://api.elevenlabs.io/v1/music"
        payload = {
            "prompt": req.prompt,
            "music_length_ms": int(req.duration * 1000),
        }
    else:
        raise GenerationError("elevenlabs provides speech and music only")

    try:
        resp = requests.post(
            url, headers={"xi-api-key": key, "accept": "audio/mpeg"},
            json=payload, timeout=600,
        )
    except requests.RequestException as e:
        raise GenerationError(f"request failed: {e}") from None
    if resp.status_code >= 300:
        raise GenerationError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    out.write_bytes(resp.content)
    return out


def _replicate_run(req: GenRequest, out_dir: Path) -> Path:
    """Replicate-hosted image and video models.

    Replicate is used rather than a dozen direct vendor integrations because it
    normalises auth and polling across models — the breadth OpenMontage gets
    from its registry, without the registry.
    """
    try:
        import requests
    except ImportError:
        raise GenerationError("needs `requests` — pip install 'reelforge[transcribe]'") from None

    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        raise GenerationError("REPLICATE_API_TOKEN is not set")

    defaults = {
        "image": "black-forest-labs/flux-1.1-pro",
        "video": "kwaivgi/kling-v1.6-standard",
    }
    model = req.options.get("model") or defaults.get(req.kind)
    if not model:
        raise GenerationError("replicate provides image and video only")

    payload = {
        "input": {
            "prompt": req.prompt,
            "aspect_ratio": req.options.get("aspect_ratio", "9:16"),
            **{k: v for k, v in req.options.items() if k not in ("model", "aspect_ratio")},
        }
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        # Blocks until the prediction resolves instead of requiring a poll loop.
        "Prefer": "wait",
    }
    data = _http_json(
        f"https://api.replicate.com/v1/models/{model}/predictions",
        headers, payload, timeout=900,
    )
    output = data.get("output")
    if isinstance(output, list):
        output = output[0] if output else None
    if not output:
        raise GenerationError(f"no output returned: {json.dumps(data)[:300]}")

    suffix = ".mp4" if req.kind == "video" else ".png"
    out = out_dir / f"asset{suffix}"
    resp = requests.get(str(output), timeout=600)
    if resp.status_code >= 300:
        raise GenerationError(f"could not download output: HTTP {resp.status_code}")
    out.write_bytes(resp.content)
    return out


def _openai_run(req: GenRequest, out_dir: Path) -> Path:
    """OpenAI image generation and text-to-speech."""
    try:
        import requests
    except ImportError:
        raise GenerationError("needs `requests` — pip install 'reelforge[transcribe]'") from None

    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise GenerationError("OPENAI_API_KEY is not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    if req.kind == "speech":
        out = out_dir / "asset.mp3"
        resp = requests.post(
            "https://api.openai.com/v1/audio/speech", headers=headers,
            json={
                "model": req.options.get("model", "gpt-4o-mini-tts"),
                "voice": req.options.get("voice", "alloy"),
                "input": req.prompt,
            }, timeout=300,
        )
        if resp.status_code >= 300:
            raise GenerationError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        out.write_bytes(resp.content)
        return out

    if req.kind == "image":
        import base64

        size = "1024x1536" if req.height > req.width else (
            "1536x1024" if req.width > req.height else "1024x1024"
        )
        data = _http_json(
            "https://api.openai.com/v1/images/generations", headers,
            {"model": req.options.get("model", "gpt-image-1"),
             "prompt": req.prompt, "size": size, "n": 1},
            timeout=600,
        )
        entry = (data.get("data") or [{}])[0]
        out = out_dir / "asset.png"
        if entry.get("b64_json"):
            out.write_bytes(base64.b64decode(entry["b64_json"]))
        elif entry.get("url"):
            out.write_bytes(requests.get(entry["url"], timeout=300).content)
        else:
            raise GenerationError("no image returned")
        return out

    raise GenerationError("openai provides image and speech only")


# Registration order matters for `auto`: cloud providers are considered first,
# and the offline ones act as the floor.
register(Provider("replicate", ("image", "video"), False, "REPLICATE_API_TOKEN",
                  "FLUX, Kling, Veo and others via Replicate.", _replicate_run))
register(Provider("openai", ("image", "speech"), False, "OPENAI_API_KEY",
                  "gpt-image-1 and gpt-4o-mini-tts.", _openai_run))
register(Provider("elevenlabs", ("speech", "music"), False, "ELEVENLABS_API_KEY",
                  "High-quality voices and generated music beds.", _elevenlabs_run))
register(Provider("espeak", ("speech",), True, None,
                  "Local synthetic speech. Robotic, but real durations for "
                  "timing a cut with no key and no network.", _espeak_run))
register(Provider("mock", ("image", "video", "speech", "music"), True, None,
                  "Deterministic labelled placeholders. Reviewable rough cuts "
                  "with nothing spent.", _mock_run))


# --- Ledger and orchestration -----------------------------------------------


def _ledger_path(work_dir: Path) -> Path:
    return work_dir / ASSETS_DIRNAME / LEDGER_NAME


def read_ledger(work_dir: Path) -> list[dict[str, Any]]:
    p = _ledger_path(work_dir)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return []


def _record(work_dir: Path, asset: Asset) -> None:
    entries = [e for e in read_ledger(work_dir) if e.get("key") != asset.key]
    entries.append(asset.to_dict())
    p = _ledger_path(work_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2) + "\n")


def generate(
    req: GenRequest,
    work_dir: Path,
    *,
    provider: str = "auto",
    force: bool = False,
) -> Asset:
    """Produce one asset, reusing a cached result when the request is identical."""
    p = resolve_provider(req.kind, provider)
    key = req.key(p.name)
    out_dir = work_dir / ASSETS_DIRNAME / f"{req.kind}_{key}"

    if out_dir.exists() and not force:
        existing = sorted(f for f in out_dir.glob("asset.*") if f.is_file())
        if existing:
            asset = Asset(existing[0], req.kind, req.prompt, p.name, key, cached=True)
            asset.duration = _duration_of(existing[0])
            return asset

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = p.run(req, out_dir)
    except GenerationError:
        raise
    except Exception as e:  # noqa: BLE001 — normalised for the caller
        raise GenerationError(f"{p.name} failed to generate: {e}") from None

    asset = Asset(path, req.kind, req.prompt, p.name, key, duration=_duration_of(path))
    (out_dir / "request.json").write_text(
        json.dumps({"prompt": req.prompt, "kind": req.kind,
                    "provider": p.name, "options": req.options}, indent=2)
    )
    _record(work_dir, asset)
    return asset


def _duration_of(path: Path) -> float:
    from .ffmpeg import media_duration

    try:
        return media_duration(path)
    except Exception:  # noqa: BLE001 — a still genuinely has no duration
        return 0.0


def still_to_clip(image: Path, out_path: Path, duration: float, *, fps: int = 30,
                  zoom: bool = True) -> Path:
    """Turn a generated still into a clip the timeline can actually use.

    A slow push is applied by default. A still held motionless in a feed reads
    as a loading error; the drift is what makes it read as a shot. The zoom is
    done at 2x scale and then downsampled, because `zoompan` on a
    native-resolution source visibly stair-steps.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(duration * fps))
    if zoom:
        vf = (
            f"scale=-2:2160,"
            f"zoompan=z='min(zoom+0.0008,1.12)':d={frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':fps={fps},"
            f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
        )
    else:
        vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"

    run([
        "ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(image),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t", f"{duration:.3f}", "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac", "-shortest", str(out_path),
    ])
    return out_path
