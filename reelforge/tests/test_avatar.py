"""Tests for the talking-head layer.

No provider here has been reachable from the environment this was written in,
so the network calls are exercised through monkeypatched transports. What that
still covers is everything that actually goes wrong in practice: the *order* of
the stages, the driver-length arithmetic, reference classification, and the
error messages a user hits when a credential or a URL mapping is missing.

`extend_driver` and `prepare_reference_audio` are real ffmpeg and are tested for
real.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from reelforge.avatar import (
    SYNC_SHAPE,
    AvatarError,
    VoiceProfile,
    classify_reference,
    extend_driver,
    preflight,
    prepare_reference_audio,
    talking_head,
)
from reelforge.ffmpeg import media_duration, probe

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def _clip(path: Path, duration: float = 3.0, size: str = "320x320") -> Path:
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=#102030:s={size}:d={duration}:r=25",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return path


def _still(path: Path) -> Path:
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=#405060:s=320x320", "-frames:v", "1", str(path),
    ], check=True, capture_output=True)
    return path


# --- reference classification -----------------------------------------------


@needs_ffmpeg
def test_photo_is_classified_as_an_image(tmp_path):
    assert classify_reference(_still(tmp_path / "face.png")) == "image"


@needs_ffmpeg
def test_clip_is_classified_as_video(tmp_path):
    assert classify_reference(_clip(tmp_path / "take.mp4")) == "video"


@needs_ffmpeg
def test_single_frame_container_is_treated_as_a_still(tmp_path):
    """A one-frame MP4 is a photo in everything but its extension.

    Treating it as video skips the animation stage, and lipsyncing a frozen
    frame produces a mouth moving on a mannequin.
    """
    path = tmp_path / "oneframe.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x320:d=0.04:r=25",
        "-frames:v", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    assert classify_reference(path) == "image"


def test_unsupported_reference_is_rejected(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("hello")
    with pytest.raises(AvatarError, match="unsupported"):
        classify_reference(doc)


# --- driver preparation -----------------------------------------------------


@needs_ffmpeg
def test_driver_is_extended_to_cover_the_audio(tmp_path):
    src = _clip(tmp_path / "short.mp4", duration=3.0)
    out = extend_driver(src, tmp_path / "long.mp4", 14.0)
    assert probe(out).duration >= 13.5


@needs_ffmpeg
def test_a_long_enough_driver_is_left_alone(tmp_path):
    src = _clip(tmp_path / "long_enough.mp4", duration=8.0)
    # No re-encode, no new file — the source is returned untouched.
    assert extend_driver(src, tmp_path / "unused.mp4", 5.0) == src
    assert not (tmp_path / "unused.mp4").exists()


@needs_ffmpeg
def test_the_loop_seam_does_not_jump(tmp_path):
    """Ping-pong, not repeat.

    A plain loop cuts from the last frame back to the first, landing a visible
    jump every cycle. Reversing instead keeps the motion continuous. Measured on
    a moving marker: a hard cut would show a per-frame jump near the full width
    of its travel.
    """
    import numpy as np

    src = tmp_path / "marker.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=#102030:s=320x320:d=3:r=25",
        "-f", "lavfi", "-i", "color=c=#FFCC55:s=40x40:d=3:r=25",
        "-filter_complex", "[0:v][1:v]overlay=x='20+250*(t/3)':y=140[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", str(src),
    ], check=True, capture_output=True)

    out = extend_driver(src, tmp_path / "looped.mp4", 12.0)
    raw = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(out),
        "-vf", "fps=10,scale=160:160", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 160, 160).astype(int)

    xs = []
    for f in frames:
        mask = f > 180
        if mask.any():
            xs.append(float(np.argwhere(mask)[:, 1].mean() / 160))
    assert len(xs) > 50
    # Smooth throughout; a loop cut would spike toward the marker's full travel.
    assert max(abs(b - a) for a, b in zip(xs, xs[1:])) < 0.15


@needs_ffmpeg
def test_reference_audio_is_normalised_and_trimmed(tmp_path):
    long_audio = tmp_path / "raw.wav"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=200:duration=60",
        "-ac", "2", "-ar", "48000", str(long_audio),
    ], check=True, capture_output=True)

    out = prepare_reference_audio(long_audio, tmp_path / "ref.wav", max_seconds=30.0)
    # Cloning quality plateaus around 30s, and the reference is resent on every
    # generation, so the trim is a cost control as much as a quality one.
    assert media_duration(out) == pytest.approx(30.0, abs=0.5)


# --- voice profiles ---------------------------------------------------------


def test_voice_profile_round_trips(tmp_path):
    voice = VoiceProfile(
        name="ali", reference_audio=tmp_path / "ref.wav",
        reference_text="salaam, this is a test recording",
    )
    path = voice.save(tmp_path / "voice.json")
    back = VoiceProfile.load(path)
    assert back.name == "ali"
    assert back.reference_text == voice.reference_text
    assert isinstance(back.reference_audio, Path)


# --- orchestration ----------------------------------------------------------


@pytest.fixture
def stub_stack(monkeypatch, tmp_path):
    """Replace the two network stages, keeping the real sequencing."""
    calls: dict[str, object] = {}

    def fake_speak(text, voice, out_path, **kw):
        calls["speak"] = {"text": text, "voice": voice.name}
        subprocess.run([
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=180:duration=11",
            str(out_path),
        ], check=True, capture_output=True)
        return out_path

    def fake_lipsync(driver, audio, out_path, **kw):
        calls["lipsync"] = {
            "driver": str(driver),
            "driver_duration": probe(Path(driver)).duration,
            "audio_duration": media_duration(Path(audio)),
        }
        shutil.copy2(driver, out_path)
        return out_path

    monkeypatch.setattr("reelforge.avatar.speak", fake_speak)
    monkeypatch.setattr("reelforge.avatar.lipsync", fake_lipsync)
    return calls


@needs_ffmpeg
def test_video_reference_runs_speech_then_extend_then_lipsync(stub_stack, tmp_path):
    reference = _clip(tmp_path / "me.mp4", duration=4.0)
    voice = VoiceProfile(name="ali", reference_audio=tmp_path / "ref.wav")

    result = talking_head(
        reference, "some new words entirely", voice,
        tmp_path / "out.mp4", work_dir=tmp_path / "work", verbose=False,
    )
    assert result.reference_kind == "video"
    assert stub_stack["speak"]["text"] == "some new words entirely"

    # The whole point of the ordering: speech length is known before the driver
    # is prepared, so the driver is grown to cover it rather than truncating.
    seen = stub_stack["lipsync"]
    assert seen["driver_duration"] >= seen["audio_duration"]
    assert seen["driver_duration"] > 4.0


@needs_ffmpeg
def test_a_still_reference_is_animated_before_lipsync(stub_stack, monkeypatch, tmp_path):
    animated: dict[str, bool] = {}

    def fake_animate(image, out_path, duration, **kw):
        animated["ran"] = True
        _clip(out_path, duration=duration)
        return out_path

    monkeypatch.setattr("reelforge.avatar.animate_still", fake_animate)
    reference = _still(tmp_path / "face.png")
    voice = VoiceProfile(name="ali", reference_audio=tmp_path / "ref.wav")

    result = talking_head(
        reference, "hello there", voice, tmp_path / "out.mp4",
        work_dir=tmp_path / "work", verbose=False,
    )
    assert result.reference_kind == "image"
    assert animated.get("ran"), "a still must be animated before lipsync"
    assert stub_stack["lipsync"]["driver_duration"] >= stub_stack["lipsync"]["audio_duration"]


@needs_ffmpeg
def test_speech_is_cached_between_runs(stub_stack, tmp_path):
    reference = _clip(tmp_path / "me.mp4", duration=12.0)
    voice = VoiceProfile(name="ali", reference_audio=tmp_path / "ref.wav")
    work = tmp_path / "work"

    talking_head(reference, "identical script", voice, tmp_path / "a.mp4",
                 work_dir=work, verbose=False)
    stub_stack.pop("speak", None)
    talking_head(reference, "identical script", voice, tmp_path / "b.mp4",
                 work_dir=work, verbose=False)
    # Second run reuses the audio rather than paying for it again.
    assert "speak" not in stub_stack


@needs_ffmpeg
def test_a_missing_reference_fails_before_any_provider_is_called(tmp_path):
    voice = VoiceProfile(name="ali", reference_audio=tmp_path / "ref.wav")
    with pytest.raises(AvatarError, match="reference not found"):
        talking_head(tmp_path / "nope.mp4", "words", voice,
                     tmp_path / "out.mp4", verbose=False)


# --- credentials and configuration ------------------------------------------


def test_lipsync_without_a_key_says_which_one(monkeypatch, tmp_path):
    from reelforge.avatar import lipsync

    monkeypatch.delenv("SYNC_API_KEY", raising=False)
    with pytest.raises(AvatarError, match="SYNC_API_KEY"):
        lipsync(tmp_path / "d.mp4", tmp_path / "a.wav", tmp_path / "o.mp4")


def test_local_files_without_a_public_mapping_fail_clearly(monkeypatch, tmp_path):
    from reelforge.avatar import _upload_target

    monkeypatch.delenv("REELFORGE_PUBLIC_BASE", raising=False)
    monkeypatch.delenv("REELFORGE_PUBLIC_DIR", raising=False)
    with pytest.raises(AvatarError, match="REELFORGE_PUBLIC_DIR"):
        _upload_target(tmp_path / "clip.mp4")


def test_https_inputs_pass_straight_through():
    from reelforge.avatar import _upload_target

    url = "https://example.com/clip.mp4"
    assert _upload_target(url) == url


def test_a_url_is_not_corrupted_by_path_normalisation():
    """Regression: `Path("https://h/x")` collapses `//` to `https:/h/x`.

    The URL check therefore has to happen on the raw input, before any Path
    conversion, or a perfectly valid URL is silently mangled into a
    404 the provider reports as a missing file.
    """
    from reelforge.avatar import _upload_target

    assert str(Path("https://example.com/clip.mp4")) == "https:/example.com/clip.mp4"
    assert _upload_target("https://example.com/clip.mp4") == "https://example.com/clip.mp4"


def test_public_mapping_rewrites_a_local_path(monkeypatch, tmp_path):
    from reelforge.avatar import _upload_target

    monkeypatch.setenv("REELFORGE_PUBLIC_DIR", str(tmp_path))
    monkeypatch.setenv("REELFORGE_PUBLIC_BASE", "https://cdn.example.com/media")
    target = tmp_path / "nested" / "clip.mp4"
    target.parent.mkdir()
    target.write_bytes(b"x")
    assert _upload_target(target) == "https://cdn.example.com/media/nested/clip.mp4"


def test_speech_without_a_fish_key_says_so(monkeypatch, tmp_path):
    from reelforge.avatar import speak

    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.delenv("FISH_BASE_URL", raising=False)
    voice = VoiceProfile(name="x", reference_audio=tmp_path / "r.wav")
    with pytest.raises(AvatarError, match="FISH_API_KEY"):
        speak("hello", voice, tmp_path / "out.wav")


def test_a_self_hosted_endpoint_needs_no_key(monkeypatch, tmp_path):
    """Same schema either way — that is why one provider covers both."""
    from reelforge.avatar import _fish_endpoint

    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.setenv("FISH_BASE_URL", "http://localhost:8080")
    base, headers = _fish_endpoint()
    assert base == "http://localhost:8080"
    assert "authorization" not in headers


def test_preflight_reports_unconfigured_providers_without_raising(monkeypatch):
    for var in ("SYNC_API_KEY", "FISH_API_KEY", "REPLICATE_API_TOKEN",
                "OPENROUTER_API_KEY", "TOGETHER_API_KEY",
                "REELFORGE_PUBLIC_BASE", "REELFORGE_PUBLIC_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("FISH_BASE_URL", raising=False)

    report = preflight(timeout=1)
    assert report["sync"]["configured"] is False
    assert "SYNC_API_KEY" in report["sync"]["detail"]
    assert report["public_url_mapping"]["configured"] is False


def test_sync_shape_declares_everything_the_caller_reads():
    """The shape is data precisely so a mismatch is a one-line fix."""
    required = {
        "base_url", "create_path", "status_path", "auth_header",
        "default_model", "id_field", "status_field", "output_field",
        "terminal_ok", "terminal_fail",
    }
    assert required <= set(SYNC_SHAPE)
    assert SYNC_SHAPE["create_path"].startswith("/")
    assert "{id}" in SYNC_SHAPE["status_path"]
    assert not set(SYNC_SHAPE["terminal_ok"]) & set(SYNC_SHAPE["terminal_fail"])
