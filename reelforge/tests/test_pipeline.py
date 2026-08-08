"""Integration tests against real ffmpeg.

These generate synthetic footage rather than shipping fixtures, so the suite
stays small and the inputs are exactly controlled — when a test says the subject
is at 14% of frame width at t=0, that is true by construction rather than by
someone's reading of a sample clip.

Skipped wholesale when ffmpeg is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from reelforge.edl import EDL, CaptionSpec, Range
from reelforge.ffmpeg import probe
from reelforge.reframe import plan_reframe
from reelforge.render import render

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

DURATION = 6.0


@pytest.fixture(scope="module")
def moving_subject(tmp_path_factory):
    """A 1920x1080 clip with a bright block sliding left to right.

    The block spans x=120..1500, so its centre travels from roughly 14% to 80%
    of frame width — far enough that a fixed centre crop must lose it.
    """
    path = tmp_path_factory.mktemp("media") / "subject.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=#101018:s=1920x1080:d={DURATION}:r=30",
        "-f", "lavfi", "-i", f"color=c=#FFD37A:s=300x420:d={DURATION}:r=30",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={DURATION}",
        "-filter_complex",
        "[0:v][1:v]overlay=x='120+(W-w-240)*(t/6)':y=330[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return path


def _subject_track(path, samples=12):
    """Where is the bright block within each sampled frame, and how much of it?"""
    raw = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"fps={samples / DURATION},scale=216:384",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 384, 216)
    out = []
    for f in frames:
        mask = f > 150
        share = mask.sum() / mask.size
        cx = float(np.argwhere(mask)[:, 1].mean() / 216) if mask.any() else None
        out.append((share, cx))
    return out


# --- reframing --------------------------------------------------------------


def test_probe_reads_geometry(moving_subject):
    info = probe(moving_subject)
    assert (info.width, info.height) == (1920, 1080)
    assert info.has_audio
    assert info.duration == pytest.approx(DURATION, abs=0.2)


def test_reframe_plan_tracks_the_subject(moving_subject):
    plan = plan_reframe(moving_subject, 0.0, DURATION, 1080, 1920,
                        mode="track", detector="saliency")
    assert plan.geometry.axis == "x"
    assert plan.geometry.crop_w == 608
    # The subject crosses most of the frame, so the crop must genuinely travel.
    assert plan.motion_px > 400
    positions = [v for _, v in plan.keys]
    assert positions[-1] > positions[0]


def test_static_mode_emits_a_single_position(moving_subject):
    plan = plan_reframe(moving_subject, 0.0, DURATION, 1080, 1920,
                        mode="static", detector="saliency")
    assert len(plan.keys) == 1
    assert plan.is_static


def test_center_mode_needs_no_analysis(moving_subject):
    plan = plan_reframe(moving_subject, 0.0, DURATION, 1080, 1920, mode="center")
    assert plan.detector == "center"
    assert plan.keys[0][1] == pytest.approx(plan.geometry.travel / 2)


def test_blur_pad_preserves_the_whole_frame(moving_subject):
    plan = plan_reframe(moving_subject, 0.0, DURATION, 1080, 1920, mode="blur_pad")
    assert "gblur" in plan.filter_chain
    assert "crop=608" not in plan.filter_chain


def test_tracking_beats_a_fixed_crop_at_keeping_the_subject(moving_subject, tmp_path):
    """The core claim of the project, measured rather than asserted."""
    plan = plan_reframe(moving_subject, 0.0, DURATION, 1080, 1920,
                        mode="track", detector="saliency")
    tracked = tmp_path / "tracked.mp4"
    centred = tmp_path / "centred.mp4"
    for out, chain in (
        (tracked, plan.filter_chain),
        (centred, "crop=608:1080:656:0,scale=1080:1920"),
    ):
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(moving_subject),
            "-vf", chain, "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-an", str(out),
        ], check=True, capture_output=True)

    tracked_track = _subject_track(tracked)
    centred_track = _subject_track(centred)

    # Tracked: the subject is present in every frame.
    assert all(share > 0.02 for share, _ in tracked_track)
    # Centred: it leaves frame entirely at least once.
    assert any(share < 0.001 for share, _ in centred_track)
    # And tracked keeps it far closer to centre throughout.
    tracked_dev = max(abs(cx - 0.5) for _, cx in tracked_track if cx is not None)
    centred_dev = max(abs(cx - 0.5) for _, cx in centred_track if cx is not None)
    assert tracked_dev < centred_dev


# --- render -----------------------------------------------------------------


@pytest.fixture
def project(tmp_path, moving_subject):
    """A minimal project directory with footage and a cached transcript."""
    shutil.copy2(moving_subject, tmp_path / "A.mp4")
    transcripts = tmp_path / ".reelforge" / "transcripts"
    transcripts.mkdir(parents=True)
    words, t = [], 0.35
    for w in "we tried this for ninety days and the results surprised us".split():
        d = 0.16 + len(w) * 0.035
        words.append({"text": w, "start": round(t, 3), "end": round(t + d, 3),
                      "type": "word", "speaker_id": "S0"})
        t += d + 0.055
    (transcripts / "A.json").write_text(json.dumps({"words": words}))
    return tmp_path


def _edl(**kw):
    base = dict(
        sources={"A": "A.mp4"},
        ranges=[
            Range(source="A", start=0.3, end=2.6, beat="HOOK"),
            Range(source="A", start=3.0, end=5.9, beat="PAYOFF"),
        ],
        platform="reels",
        detector="saliency",
    )
    base.update(kw)
    return EDL(**base)


def test_render_produces_a_vertical_file_of_the_right_length(project):
    result = render(_edl(), project / "out.mp4", quality="draft",
                    base_dir=project, verbose=False)
    info = probe(result.output)
    assert info.height > info.width                      # vertical
    assert info.width / info.height == pytest.approx(9 / 16, abs=0.01)
    assert info.duration == pytest.approx(5.2, abs=0.3)  # 2.3 + 2.9
    assert info.has_audio


def test_render_at_final_quality_uses_the_full_canvas(project):
    result = render(_edl(), project / "final.mp4", quality="final",
                    base_dir=project, verbose=False)
    info = probe(result.output)
    assert (info.width, info.height) == (1080, 1920)


def test_captions_are_generated_and_burned(project):
    result = render(_edl(captions=CaptionSpec(style="karaoke")),
                    project / "capped.mp4", quality="draft",
                    base_dir=project, verbose=False)
    assert result.caption_cues > 0
    assert result.captions_path is not None and result.captions_path.exists()
    assert result.srt_path is not None and result.srt_path.exists()

    # Sample the caption band and confirm bright glyph pixels landed there.
    info = probe(result.output)
    raw = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", "1.0", "-i", str(result.output),
        "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ], capture_output=True, check=True).stdout
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width)
    lo = int(info.height * 0.55)
    hi = int(info.height * 0.68)
    assert (frame[lo:hi, :] > 230).sum() > 200


def test_disabling_captions_skips_them(project):
    result = render(_edl(captions=CaptionSpec(enabled=False)),
                    project / "bare.mp4", quality="draft",
                    base_dir=project, verbose=False)
    assert result.caption_cues == 0
    assert result.captions_path is None


def test_speed_shortens_the_output(project):
    fast = _edl(ranges=[Range(source="A", start=0.3, end=5.3, speed=2.0)])
    result = render(fast, project / "fast.mp4", quality="draft",
                    base_dir=project, verbose=False)
    assert probe(result.output).duration == pytest.approx(2.5, abs=0.4)


def test_switching_platform_changes_the_canvas(project):
    result = render(_edl(platform="square"), project / "sq.mp4", quality="final",
                    base_dir=project, verbose=False)
    info = probe(result.output)
    assert info.width == info.height == 1080


def test_per_range_reframe_override_is_honoured(project):
    edl = _edl(reframe="track", ranges=[
        Range(source="A", start=0.3, end=2.6, beat="HOOK"),
        Range(source="A", start=3.0, end=5.9, beat="PAYOFF", reframe="blur_pad"),
    ])
    result = render(edl, project / "mixed.mp4", quality="draft",
                    base_dir=project, verbose=False)
    assert result.reframe_plans[0].mode == "track"
    assert result.reframe_plans[1].mode == "blur_pad"


def test_loudness_normalisation_hits_the_platform_target(project):
    result = render(_edl(), project / "loud.mp4", quality="final",
                    base_dir=project, loudnorm=True, verbose=False)
    proc = subprocess.run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(result.output),
        "-af", "loudnorm=I=-14:TP=-1:LRA=11:print_format=json",
        "-vn", "-f", "null", "-",
    ], capture_output=True, text=True)
    payload = proc.stderr[proc.stderr.rfind("{") : proc.stderr.rfind("}") + 1]
    measured = float(json.loads(payload)["input_i"])
    # Two-pass linear normalisation should land within a dB of target.
    assert measured == pytest.approx(-14.0, abs=1.5)


def test_the_linter_accepts_a_real_render(project):
    from reelforge.retention import lint

    edl = _edl()
    result = render(edl, project / "linted.mp4", quality="final",
                    base_dir=project, verbose=False)
    report = lint(edl, base_dir=project, work_dir=project / ".reelforge",
                  rendered=result.output)
    assert report.ok, report.format()
    assert "render.drift" not in {f.code for f in report.findings}


# --- autocut (real silencedetect) -------------------------------------------


@pytest.fixture(scope="module")
def gapped_audio(tmp_path_factory):
    """Tone, silence, tone, silence, tone — known boundaries by construction."""
    path = tmp_path_factory.mktemp("media") / "gapped.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=#202030:s=1280x720:d=9:r=30",
        "-f", "lavfi", "-i",
        "sine=frequency=300:duration=9:sample_rate=48000",
        "-filter_complex",
        # Audible on 0-2, 4-6 and 7.5-9; silent elsewhere.
        "[1:a]volume='if(between(t,0,2)+between(t,4,6)+between(t,7.5,9),1,0)':eval=frame[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return path


def test_silences_are_detected_where_they_were_placed(gapped_audio):
    from reelforge.autocut import detect_silences

    silences = detect_silences(gapped_audio, min_silence=0.3)
    assert len(silences) >= 2
    # A silence should cover the 2-4s window that was muted.
    assert any(s.start < 2.4 and s.end > 3.6 for s in silences)


def test_autocut_removes_dead_air_and_keeps_the_audible_spans(gapped_audio):
    from reelforge.autocut import autocut

    edl, stats = autocut(gapped_audio, platform="reels")
    assert stats["removed_s"] > 2.0
    assert stats["cuts"] >= 2
    assert edl.total_duration < stats["source_duration"]
    assert edl.ranges[0].beat == "HOOK"


def test_autocut_output_renders(gapped_audio, tmp_path):
    from reelforge.autocut import autocut

    shutil.copy2(gapped_audio, tmp_path / gapped_audio.name)
    edl, _ = autocut(tmp_path / gapped_audio.name, platform="reels")
    edl.validate(base_dir=tmp_path)
    result = render(edl, tmp_path / "cut.mp4", quality="draft",
                    base_dir=tmp_path, verbose=False)
    info = probe(result.output)
    assert info.height > info.width
    assert info.duration == pytest.approx(edl.total_duration, abs=0.4)


def test_autocut_refuses_a_silent_source(tmp_path):
    from reelforge.autocut import autocut

    path = tmp_path / "silent.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:d=3:r=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-an", str(path),
    ], check=True, capture_output=True)
    with pytest.raises(ValueError, match="no audio track"):
        autocut(path)
