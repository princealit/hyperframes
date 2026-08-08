"""Word-level transcription, cached against source content.

Word-level timings are the whole substrate. Phrase-level output (an SRT, or
Whisper's default segments) throws away the sub-second gap data that cut
selection depends on — you cannot snap a cut to a word boundary you were never
given, and you cannot find the 400ms silence that makes the cleanest edit point.

Verbatim matters too. A transcriber that helpfully removes "um" also removes the
signal that a take had a stumble, which is exactly what take selection needs.

Caching is keyed on a hash of the audio content, not the filename or mtime, so
re-transcription happens when the footage genuinely changes and never because a
file was copied or touched.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg import run

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_MODEL = "scribe_v1"

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


def extract_audio(source: Path, out_path: Path) -> Path:
    """Downmix to 16kHz mono Opus before upload.

    ASR models work from a mono 16kHz signal regardless of what is uploaded, so
    sending a 4K video's original audio wastes upload time proportional to the
    file size for no accuracy gain. This routinely turns a 2GB upload into
    a few megabytes.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", "32k",
        str(out_path),
    ])
    return out_path


def transcribe_file(
    source: Path,
    out_dir: Path,
    *,
    api_key: str | None = None,
    name: str | None = None,
    num_speakers: int | None = None,
    language: str | None = None,
    force: bool = False,
) -> TranscribeResult:
    """Transcribe one file, reusing a cached transcript when the content matches."""
    try:
        import requests
    except ImportError:
        raise TranscriptionError(
            "the `requests` package is required — install with: pip install 'reelforge[transcribe]'"
        ) from None

    name = name or source.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    key = content_key(source)

    if out_path.exists() and not force:
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("_reelforge", {}).get("content_key") == key:
                return TranscribeResult(source, out_path, len(existing.get("words", [])), True)
        except (json.JSONDecodeError, OSError):
            pass

    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise TranscriptionError(
            "ELEVENLABS_API_KEY is not set. Put it in the environment or in a .env "
            "file at the project root. Get one at elevenlabs.io/app/settings/api-keys"
        )

    audio = extract_audio(source, out_dir / "_audio" / f"{name}.opus")
    data = {
        "model_id": DEFAULT_MODEL,
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
        raise TranscriptionError(f"transcription request failed for {source.name}: {e}") from None

    if resp.status_code != 200:
        detail = resp.text[:400]
        raise TranscriptionError(
            f"transcription failed for {source.name} (HTTP {resp.status_code}): {detail}"
        )

    payload = resp.json()
    payload["_reelforge"] = {
        "content_key": key,
        "source": str(source),
        "model": DEFAULT_MODEL,
    }
    out_path.write_text(json.dumps(payload, indent=1))
    audio.unlink(missing_ok=True)
    return TranscribeResult(source, out_path, len(payload.get("words", [])), False)


def transcribe_dir(
    sources: list[Path],
    out_dir: Path,
    *,
    api_key: str | None = None,
    workers: int = 4,
    force: bool = False,
    num_speakers: int | None = None,
) -> list[TranscribeResult]:
    """Transcribe several sources concurrently.

    The work is network-bound, so threads are the right tool and four is enough
    to saturate a typical uplink without tripping rate limits.
    """
    results: list[TranscribeResult] = []
    errors: list[str] = []

    def one(path: Path) -> None:
        try:
            results.append(
                transcribe_file(
                    path, out_dir, api_key=api_key, force=force, num_speakers=num_speakers
                )
            )
        except Exception as e:  # noqa: BLE001 — collected and reported together
            errors.append(f"{path.name}: {e}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
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
