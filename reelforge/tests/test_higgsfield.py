"""Tests for the Higgsfield talking-head planner.

Unlike the `avatar` stack, this path has been run against the live API — voice,
video and the two-call chain between them all executed successfully, and the
costs asserted below are the API's own `get_cost` figures rather than a
published price list.

What is tested here is the deterministic half: cost arithmetic, segment
splitting, duration reconciliation, and the EDL assembly that joins a finished
talking head back into the normal pipeline. The network calls themselves belong
to the agent, so there is nothing to mock.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from reelforge.higgsfield import (
    COST_TALKING_5S_720P,
    COST_TTS,
    MAX_SEGMENT_S,
    TALKING_MODEL,
    HiggsfieldError,
    plan,
    plan_speech,
    plan_talking_head,
    reconcile,
    split_script,
    to_edl,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)

FARSI = "سلام، این یک آزمایش است. اگر این صدا درست به نظر می‌رسد، یعنی مسیر کار می‌کند."


def _clip(path: Path, duration: float = 5.0) -> Path:
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=#203040:s=270x480:d={duration}:r=25",
        "-f", "lavfi", "-i", f"sine=frequency=200:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return path


# --- individual calls -------------------------------------------------------


def test_speech_call_selects_the_cloned_voice_not_a_preset():
    """A preset is a stranger's voice; `element` is the one cloned in-workspace."""
    call = plan_speech("hello there", "ff76a844-1506-4c07-8f65-8279b34dc81b")
    assert call.tool == "generate_audio"
    assert call.params["voice_type"] == "element"
    assert call.params["voice_id"] == "ff76a844-1506-4c07-8f65-8279b34dc81b"
    assert call.est_credits == COST_TTS


def test_speech_carries_persian_script_through_untouched():
    """No language flag, no transliteration — the model reads the script.

    This is the whole reason for the model choice, so it is worth asserting that
    nothing in the plumbing mangles or romanises the text.
    """
    call = plan_speech(FARSI, "voice-1")
    assert call.params["prompt"] == FARSI
    assert "language" not in call.params


def test_empty_script_is_rejected():
    with pytest.raises(HiggsfieldError, match="empty script"):
        plan_speech("   ", "voice-1")


def test_video_call_wires_photo_and_audio_as_the_two_references():
    """The whole collapse: one call takes both, so there is no lipsync stage."""
    call = plan_talking_head("img-uuid", "audio-job-id", duration=5)
    roles = {m["role"]: m["value"] for m in call.params["medias"]}
    assert roles == {"start_image": "img-uuid", "audio_references": "audio-job-id"}
    assert call.params["model"] == TALKING_MODEL
    assert call.params["aspect_ratio"] == "9:16"


def test_video_defaults_to_vertical():
    assert plan_talking_head("i", "a").params["aspect_ratio"] == "9:16"


def test_duration_beyond_the_model_ceiling_is_refused():
    """Silently clamping would truncate the script with no warning."""
    with pytest.raises(HiggsfieldError, match="outside"):
        plan_talking_head("i", "a", duration=40)


def test_the_prompt_asks_for_restrained_motion():
    """Large gestures pull the face off-axis and the mouth stops tracking."""
    prompt = plan_talking_head("i", "a").params["prompt"]
    assert "static background" in prompt
    assert "subtle" in prompt or "natural" in prompt


# --- whole-job planning -----------------------------------------------------


def test_a_short_script_is_two_calls_and_costs_under_ten_credits():
    """The headline claim: photo + script -> talking video in two calls."""
    p = plan("This is a short line to camera.", "voice-1", "img-1")
    assert [c.tool for c in p.calls] == ["generate_audio", "generate_video"]
    assert p.segments == 1
    assert p.est_credits == pytest.approx(COST_TTS + COST_TALKING_5S_720P, abs=0.01)


def test_a_long_script_is_split_into_segments_and_says_so():
    script = " ".join(["word"] * 200)  # ~87s at 2.3 w/s
    p = plan(script, "voice-1", "img-1")
    assert p.segments > 1
    assert sum(1 for c in p.calls if c.tool == "generate_video") == p.segments
    # Silent truncation is the failure mode being guarded against.
    assert any("exceeds" in n for n in p.notes)


def test_the_cost_estimate_is_the_sum_of_its_calls():
    p = plan("a short script here", "voice-1", "img-1")
    assert p.est_credits == pytest.approx(sum(c.est_credits for c in p.calls))


def test_the_summary_shows_the_price_before_anything_is_spent():
    text = plan("hello", "voice-1", "img-1").summary()
    assert "credits" in text
    assert "generate_audio" in text and "generate_video" in text


# --- splitting --------------------------------------------------------------


def test_splitting_falls_on_sentence_boundaries():
    """A seam inside a clause reads as a glitch; a sentence already has a pause."""
    script = "First sentence here. Second sentence here. Third sentence here."
    chunks = split_script(script, max_seconds=2.0, words_per_second=2.0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip().endswith((".", "!", "?", "؟", "۔"))


def test_splitting_handles_persian_punctuation():
    """Non-Latin script support is the reason for this stack; splitting must match."""
    script = "سلام چطوری؟ من خوبم. تو چطور؟"
    chunks = split_script(script, max_seconds=1.5, words_per_second=2.0)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_a_short_script_is_not_split():
    assert split_script("Just one line.") == ["Just one line."]


def test_splitting_never_loses_words():
    script = "Alpha one two. Beta three four. Gamma five six. Delta seven eight."
    chunks = split_script(script, max_seconds=1.5, words_per_second=2.0)
    assert " ".join(chunks).split() == script.split()


# --- reconciliation ---------------------------------------------------------


def test_measured_audio_overrides_the_word_count_guess():
    """The estimate is guesswork; once the audio exists its length is a fact.

    Generating video to the guess either truncates the speech or pads it with
    silence, so the plan is rewritten before any video is bought.
    """
    p = plan("short", "voice-1", "img-1")
    fixed = reconcile(p, actual_audio_seconds=12.0)
    videos = [c for c in fixed.calls if c.tool == "generate_video"]
    assert sum(c.params["duration"] for c in videos) >= 12
    assert any("reconciled" in n for n in fixed.notes)


def test_reconciling_a_long_take_produces_enough_segments():
    p = plan("short", "voice-1", "img-1")
    fixed = reconcile(p, actual_audio_seconds=38.0)
    videos = [c for c in fixed.calls if c.tool == "generate_video"]
    assert len(videos) >= 3
    assert sum(c.params["duration"] for c in videos) >= 38
    assert all(c.params["duration"] <= MAX_SEGMENT_S for c in videos)


def test_reconciling_keeps_the_speech_call():
    p = plan("short", "voice-1", "img-1")
    fixed = reconcile(p, actual_audio_seconds=6.0)
    assert sum(1 for c in fixed.calls if c.tool == "generate_audio") == 1


# --- assembly back into the pipeline ----------------------------------------


@needs_ffmpeg
def test_finished_segments_become_a_normal_edl(tmp_path):
    """Once it is an EDL it is just footage — caption, grade, cut, render."""
    segs = [_clip(tmp_path / f"seg_{i}.mp4", 4.0) for i in range(3)]
    edl = to_edl(segs, tmp_path)

    assert edl["version"] == 2
    assert edl["platform"] == "reels"
    assert len(edl["ranges"]) == 3
    assert edl["ranges"][0]["beat"] == "HOOK"
    assert all(r["beat"] == "BODY" for r in edl["ranges"][1:])
    assert all(r["end"] > 0 for r in edl["ranges"])


@needs_ffmpeg
def test_the_generated_edl_actually_loads_and_lints(tmp_path):
    """End of the join: the assembled EDL is valid input to the real pipeline."""
    import json

    from reelforge.edl import EDL

    segs = [_clip(tmp_path / f"seg_{i}.mp4", 5.0) for i in range(2)]
    path = tmp_path / "edl.json"
    path.write_text(json.dumps(to_edl(segs, tmp_path), indent=2))

    loaded = EDL.load(path)
    assert len(loaded.ranges) == 2
    assert loaded.platform == "reels"


def test_assembling_nothing_is_an_error(tmp_path):
    with pytest.raises(HiggsfieldError, match="no segments"):
        to_edl([], tmp_path)


# --- the live-verified numbers ----------------------------------------------


def test_costs_match_what_the_live_api_quoted():
    """Asserted from the API's own get_cost preflight, run 2026-08-09.

    Pinned so a silent upstream price change shows up as a failing test rather
    than as a surprise on the bill.
    """
    assert COST_TTS == 0.2
    assert COST_TALKING_5S_720P == 7.5
    # The headline: a five-second talking head costs under eight credits.
    assert COST_TTS + COST_TALKING_5S_720P < 8.0
