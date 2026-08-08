"""Word-level transcription, cached against source content.

Word-level timings are the whole substrate. Phrase-level output (an SRT, or
Whisper's default segments) throws away the sub-second gap data that cut
selection depends on — you cannot snap a cut to a word boundary you were never
given, and you cannot find the 400ms silence that makes the cleanest edit point.

Two backends, same output shape:

**scribe** — hosted ElevenLabs Scribe. Verbatim, so stumbles and fillers survive
as the editorial signal they are, with speaker diarization. Needs a key, and the
audio leaves the machine.

**whisper** — local faster-whisper. No key, nothing uploaded, slower on CPU. It
also normalises: "um", "uh" and false starts are largely scrubbed. That is fine
for a single clean take and a real loss for choosing between takes, because the
thing you were selecting *on* is the thing it removed.

`auto` prefers Scribe when a key is present and falls back to local.

Caching is keyed on a hash of the audio content *and* the backend that produced
it, so re-transcription happens when the footage genuinely changes or the
backend does — never because a file was copied or touched.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .ffmpeg import run

Backend = Literal["auto", "scribe", "whisper"]

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
SCRIBE_MODEL = "scribe_v1"

#: Default local model. `base` transcribes roughly 6-10x faster than real time
#: on a modern CPU and its word timings are accurate enough to cut on. Step up
#: to `small` when the audio is noisy or accented; `tiny` is rarely worth it.
WHISPER_MODEL = "base"

#: Extensions treated as transcribable media when scanning a directory.
MEDIA_SUFFIXES = frozenset({
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
})


class TranscriptionError(RuntimeError):
    pass


@dataclass
class TranscribeResult:
    source: Path
    output: Path
    words: int
    cached: bool
    backend: str = ""


def content_key(path: Path) -> str:
    """Hash enough of the file to identify its content cheaply.

    Head and tail plus size, rather than the whole file: a multi-gigabyte take
    should not be read end to end just to decide whether it changed, and a
    re-encode that preserves both ends and the exact byte count is not a case
    worth defending against here.
    """
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(1 << 20))
        if size > (1 << 21):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


def extract_audio(source: Path, out_path: Path, *, wav: bool = False) -> Path:
    """Downmix to 16kHz mono before transcription.

    Every ASR model resamples to mono 16kHz internally, so uploading a 4K take's
    original audio costs upload time proportional to file size for no accuracy
    gain. Opus for the hosted path (a 2GB source becomes a few megabytes); WAV
    for the local path, where there is no upload and decoding is the only cost.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codec = ["-c:a", "pcm_s16le"] if wav else ["-c:a", "libopus", "-b:a", "32k"]
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", *codec, str(out_path),
    ])
    return out_path


# --- Backends ---------------------------------------------------------------


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: Backend, api_key: str | None = None) -> str:
    """Decide which backend will actually run, and say why it cannot."""
    key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if backend == "scribe":
        if not key:
            raise TranscriptionError(
                "backend 'scribe' needs ELEVENLABS_API_KEY. Set it in the environment "
                "or a .env file at the project root, or use --backend whisper to run "
                "locally with no key."
            )
        return "scribe"
    if backend == "whisper":
        if not whisper_available():
            raise TranscriptionError(
                "backend 'whisper' needs faster-whisper — install with: "
                "pip install 'reelforge[local]'"
            )
        return "whisper"
    if key:
        return "scribe"
    if whisper_available():
        return "whisper"
    raise TranscriptionError(
        "no transcription backend available. Either set ELEVENLABS_API_KEY, or "
        "install the local backend with: pip install 'reelforge[local]'"
    )


def _scribe(
    audio: Path,
    *,
    api_key: str,
    num_speakers: int | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    try:
        import requests
    except ImportError:
        raise TranscriptionError(
            "the `requests` package is required for the hosted backend — "
            "install with: pip install 'reelforge[transcribe]'"
        ) from None

    data = {
        "model_id": SCRIBE_MODEL,
        # Word granularity is the point of this module; anything coarser
        # discards the boundaries cut selection relies on.
        "timestamps_granularity": "word",
        "diarize": "true",
        "tag_audio_events": "true",
    }
    if num_speakers:
        data["num_speakers"] = str(num_speakers)
    if language:
        data["language_code"] = language

    try:
        with audio.open("rb") as fh:
            resp = requests.post(
                SCRIBE_URL,
                headers={"xi-api-key": api_key},
                data=data,
                files={"file": (audio.name, fh, "audio/ogg")},
                timeout=900,
            )
    except requests.RequestException as e:
        raise TranscriptionError(f"transcription request failed: {e}") from None

    if resp.status_code != 200:
        raise TranscriptionError(
            f"transcription failed (HTTP {resp.status_code}): {resp.text[:400]}"
        )
    return resp.json()


def _whisper(
    audio: Path,
    *,
    model_size: str = WHISPER_MODEL,
    language: str | None = None,
) -> dict[str, Any]:
    """Local transcription, emitted in the Scribe payload shape.

    Deliberately runs without VAD filtering. VAD would drop silent spans, and
    silence is not noise here — the gaps between phrases are precisely the cut
    candidates the rest of the pipeline reads. Filtering them out would make the
    transcript smaller and the edit worse.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TranscriptionError(
            "faster-whisper is not installed — pip install 'reelforge[local]'"
        ) from None

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(audio),
        word_timestamps=True,
        vad_filter=False,
        language=language,
        # Whisper can loop on repeated phrasing when it conditions on its own
        # prior output; disabling it trades a little fluency for reliability.
        condition_on_previous_text=False,
    )

    words: list[dict[str, Any]] = []
    for segment in segments:
        for w in segment.words or []:
            text = (w.word or "").strip()
            if not text or w.start is None or w.end is None:
                continue
            words.append({
                "text": text,
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "type": "word",
                # Whisper does not diarize; a single label keeps the shape
                # consistent for `pack`, which groups on speaker change.
                "speaker_id": "S0",
            })

    return {
        "language_code": getattr(info, "language", language or "") or "",
        "language_probability": getattr(info, "language_probability", None),
        "text": " ".join(w["text"] for w in words),
        "words": words,
    }


# --- Orchestration ----------------------------------------------------------


def transcribe_file(
    source: Path,
    out_dir: Path,
    *,
    backend: Backend = "auto",
    api_key: str | None = None,
    name: str | None = None,
    num_speakers: int | None = None,
    language: str | None = None,
    model_size: str = WHISPER_MODEL,
    force: bool = False,
) -> TranscribeResult:
    """Transcribe one file, reusing a cached transcript when nothing changed."""
    name = name or source.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"

    chosen = resolve_backend(backend, api_key)
    stamp = f"{chosen}:{model_size}" if chosen == "whisper" else chosen
    key = content_key(source)

    if out_path.exists() and not force:
        try:
            existing = json.loads(out_path.read_text())
            meta = existing.get("_reelforge", {})
            if meta.get("content_key") == key and meta.get("backend") == stamp:
                return TranscribeResult(
                    source, out_path, len(existing.get("words", [])), True, chosen
                )
        except (json.JSONDecodeError, OSError):
            pass

    audio_dir = out_dir / "_audio"
    if chosen == "scribe":
        audio = extract_audio(source, audio_dir / f"{name}.opus")
        payload = _scribe(
            audio,
            api_key=api_key or os.environ["ELEVENLABS_API_KEY"],
            num_speakers=num_speakers,
            language=language,
        )
    else:
        audio = extract_audio(source, audio_dir / f"{name}.wav", wav=True)
        payload = _whisper(audio, model_size=model_size, language=language)

    payload["_reelforge"] = {
        "content_key": key,
        "source": str(source),
        "backend": stamp,
    }
    out_path.write_text(json.dumps(payload, indent=1))
    audio.unlink(missing_ok=True)
    return TranscribeResult(source, out_path, len(payload.get("words", [])), False, chosen)


def transcribe_dir(
    sources: list[Path],
    out_dir: Path,
    *,
    backend: Backend = "auto",
    api_key: str | None = None,
    workers: int = 4,
    force: bool = False,
    num_speakers: int | None = None,
    model_size: str = WHISPER_MODEL,
) -> list[TranscribeResult]:
    """Transcribe several sources, in parallel where that helps.

    The hosted backend is network-bound, so threads overlap usefully. The local
    backend is CPU-bound and already threads internally, so running several
    models at once mostly contends for the same cores and inflates peak memory —
    it is deliberately serialised.
    """
    chosen = resolve_backend(backend, api_key)
    if chosen == "whisper":
        workers = 1

    results: list[TranscribeResult] = []
    errors: list[str] = []

    def one(path: Path) -> None:
        try:
            results.append(
                transcribe_file(
                    path, out_dir, backend=backend, api_key=api_key, force=force,
                    num_speakers=num_speakers, model_size=model_size,
                )
            )
        except Exception as e:  # noqa: BLE001 — collected and reported together
            errors.append(f"{path.name}: {e}")

    if workers <= 1:
        for path in sources:
            one(path)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, sources))

    if errors and not results:
        raise TranscriptionError("all transcriptions failed:\n  " + "\n  ".join(errors))
    for e in errors:
        print(f"  warning: {e}")
    return sorted(results, key=lambda r: r.source.name)


def find_media(directory: Path) -> list[Path]:
    """List transcribable media in a directory, skipping our own artifacts."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in MEDIA_SUFFIXES
        and not p.name.startswith(".")
        and p.parent.name != ".reelforge"
    )


def load_dotenv(path: Path) -> None:
    """Minimal .env loader, so a key in the project root is picked up.

    Existing environment variables win — an explicitly exported key should not
    be silently replaced by a stale file.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
