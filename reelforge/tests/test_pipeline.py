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

from pathlib import Path

from reelforge.edl import EDL, CaptionSpec, Overlay, Range
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


# --- multi-source assembly and overlay compositing --------------------------


@pytest.fixture(scope="module")
def two_sources(tmp_path_factory):
    """Two clips that differ in resolution, frame rate and colour.

    Different geometry per source is the point: each must be reframed on its own
    terms and still concatenate cleanly. The distinct base colours make it
    possible to prove from the pixels which source a given output frame came
    from, rather than trusting the segment list.
    """
    d = tmp_path_factory.mktemp("multi")
    specs = [
        ("A.mp4", "#1B3A6B", "1920x1080", 30, 200),
        ("B.mp4", "#6B1B2E", "1280x720", 25, 330),
    ]
    for name, colour, size, fps, freq in specs:
        subprocess.run([
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c={colour}:s={size}:d=5:r={fps}",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration=5",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(d / name),
        ], check=True, capture_output=True)
    return d


@pytest.fixture(scope="module")
def alpha_overlay(tmp_path_factory):
    """A ProRes 4444 badge with genuine transparency.

    Built with alphamerge rather than a transparent `color` source: lavfi's
    `color` negotiates to a format without an alpha plane, so `black@0.0`
    silently produces an opaque frame and the overlay would composite as a box.
    """
    path = tmp_path_factory.mktemp("overlay") / "badge.mov"
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not Path(font).exists():
        pytest.skip("DejaVu font not available")
    draw = "text='BADGE':fontsize=110:x=(w-text_w)/2:y=(h-text_h)/2"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=700x220:d=3:r=30",
        "-filter_complex",
        f"[0:v]drawtext=fontfile={font}:{draw}:fontcolor=#FFD400[fg];"
        f"[0:v]drawtext=fontfile={font}:{draw}:fontcolor=white,format=gray[mk];"
        f"[fg][mk]alphamerge,format=yuva444p10le[o]",
        "-map", "[o]", "-c:v", "prores_ks", "-profile:v", "4444",
        "-pix_fmt", "yuva444p10le", str(path),
    ], check=True, capture_output=True)
    return path


def _mean_rgb(path, t):
    raw = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path),
        "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(int).mean(0)


def _badge_span(path, t, height=1920, width=1080):
    """Vertical extent and pixel count of the #FFD400 badge in one frame."""
    raw = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path),
        "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ], capture_output=True, check=True).stdout
    f = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).astype(int)
    m = (f[:, :, 0] > 200) & (f[:, :, 1] > 150) & (f[:, :, 2] < 90)
    if m.sum() < 50:
        return None, 0
    ys = np.argwhere(m)[:, 0]
    return (int(ys.min()), int(ys.max())), int(m.sum())


def test_sources_of_different_size_and_fps_assemble(two_sources, tmp_path):
    edl = EDL(
        sources={"A": "A.mp4", "B": "B.mp4"},
        ranges=[
            Range(source="A", start=0.2, end=2.2, beat="HOOK"),
            Range(source="B", start=1.0, end=3.0, beat="TURN"),
            Range(source="A", start=3.0, end=4.8, beat="PAYOFF"),
        ],
        platform="reels", detector="saliency",
        captions=CaptionSpec(enabled=False),
    )
    out = tmp_path / "multi.mp4"
    result = render(edl, out, quality="draft", base_dir=two_sources, verbose=False)

    # Each source is cropped on its own geometry.
    assert result.reframe_plans[0].geometry.crop_h == 1080
    assert result.reframe_plans[1].geometry.crop_h == 720
    info = probe(out)
    assert info.duration == pytest.approx(5.8, abs=0.4)

    # Prove the cut from the pixels: the middle segment is the other source.
    assert _mean_rgb(out, 1.0)[2] > _mean_rgb(out, 1.0)[0]   # A reads blue
    assert _mean_rgb(out, 2.6)[0] > _mean_rgb(out, 2.6)[2]   # B reads red
    assert _mean_rgb(out, 4.6)[2] > _mean_rgb(out, 4.6)[0]   # back to A


def test_overlays_composite_with_alpha_at_the_anchors_given(
    two_sources, alpha_overlay, tmp_path
):
    shutil.copy2(alpha_overlay, two_sources / "badge.mov")
    edl = EDL(
        sources={"A": "A.mp4"},
        ranges=[Range(source="A", start=0.0, end=5.0, beat="HOOK")],
        platform="reels", detector="saliency",
        captions=CaptionSpec(enabled=False),
        overlays=[
            Overlay(file="badge.mov", start_in_output=0.4, duration=2.0,
                    anchor="top-center", scale=0.9, fade_in=0.2),
            Overlay(file="badge.mov", start_in_output=3.4, duration=1.4,
                    anchor="bottom-center", scale=0.6, dy=-120, opacity=0.85),
        ],
    )
    out = tmp_path / "over.mp4"
    render(edl, out, quality="final", base_dir=two_sources, verbose=False)

    top_span, top_px = _badge_span(out, 1.4)
    gap_span, gap_px = _badge_span(out, 3.0)
    low_span, low_px = _badge_span(out, 4.0)

    assert top_px > 1000, "top overlay did not composite"
    assert gap_px == 0, "overlay visible outside its window"
    assert low_px > 500, "bottom overlay did not composite"

    # Anchoring: the first sits high, the second low, both inside the safe area.
    platform = edl.target
    safe = platform.safe_at(1080, 1920)
    assert top_span[0] >= safe.top - 2
    assert low_span[1] <= 1920 - safe.bottom
    assert top_span[1] < low_span[0], "anchors did not separate the overlays"

    # Alpha held: a smaller, more transparent overlay covers fewer pixels.
    assert low_px < top_px


def test_overlay_outside_the_timeline_is_rejected(two_sources):
    edl = EDL(
        sources={"A": "A.mp4"},
        ranges=[Range(source="A", start=0.0, end=4.0)],
        overlays=[Overlay(file="A.mp4", start_in_output=99.0, duration=1.0)],
    )
    with pytest.raises(Exception):
        edl.validate(base_dir=two_sources)


# --- transitions (real render) ----------------------------------------------


def test_transitions_shorten_the_render_by_the_overlap(two_sources, tmp_path):
    edl = EDL(
        sources={"A": "A.mp4", "B": "B.mp4"},
        ranges=[
            Range(source="A", start=0.5, end=2.5, beat="HOOK"),
            Range(source="B", start=0.5, end=2.5, transition="crossfade",
                  transition_duration=0.6),
            Range(source="A", start=2.5, end=4.5, transition="dip_black",
                  transition_duration=0.5),
        ],
        platform="reels", reframe="center", captions=CaptionSpec(enabled=False),
    )
    # 6s of segments minus 1.1s of overlap.
    assert edl.total_duration == pytest.approx(4.9)
    out = tmp_path / "trans.mp4"
    render(edl, out, quality="draft", base_dir=two_sources, verbose=False)
    assert probe(out).duration == pytest.approx(4.9, abs=0.35)


def test_crossfade_actually_blends_the_two_sources(two_sources, tmp_path):
    edl = EDL(
        sources={"A": "A.mp4", "B": "B.mp4"},
        ranges=[
            Range(source="A", start=0.5, end=2.5),
            Range(source="B", start=0.5, end=2.5, transition="crossfade",
                  transition_duration=0.6),
        ],
        platform="reels", reframe="center", captions=CaptionSpec(enabled=False),
    )
    out = tmp_path / "x.mp4"
    render(edl, out, quality="final", base_dir=two_sources, verbose=False)

    before = _mean_rgb(out, 0.8)     # A alone: blue
    middle = _mean_rgb(out, 1.65)    # mid-seam
    after = _mean_rgb(out, 2.3)      # B alone: red
    # Mid-seam sits between the two, rather than being either one.
    assert before[2] > before[0]
    assert after[0] > after[2]
    assert before[0] < middle[0] < after[0]


def test_dip_black_actually_reaches_black(two_sources, tmp_path):
    edl = EDL(
        sources={"A": "A.mp4", "B": "B.mp4"},
        ranges=[
            Range(source="A", start=0.5, end=2.5),
            Range(source="B", start=0.5, end=2.5, transition="dip_black",
                  transition_duration=0.6),
        ],
        platform="reels", reframe="center", captions=CaptionSpec(enabled=False),
    )
    out = tmp_path / "dip.mp4"
    render(edl, out, quality="final", base_dir=two_sources, verbose=False)

    def luma(t):
        raw = subprocess.run([
            "ffmpeg", "-v", "error", "-ss", str(t), "-i", str(out),
            "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-",
        ], capture_output=True, check=True).stdout
        return float(np.frombuffer(raw, dtype=np.uint8).mean())

    # Seam spans 1.4 -> 2.0; the midpoint must be near black.
    assert luma(1.0) > 40
    assert luma(1.7) < 15
    assert luma(2.3) > 40


def test_captions_stay_aligned_across_a_transition(project, tmp_path):
    """Transitions eat timeline time; caption offsets must account for it."""
    plain = EDL(
        sources={"A": "A.mp4"},
        ranges=[Range(source="A", start=0.3, end=2.6), Range(source="A", start=3.0, end=5.9)],
        platform="reels", detector="saliency",
    )
    faded = EDL(
        sources={"A": "A.mp4"},
        ranges=[
            Range(source="A", start=0.3, end=2.6),
            Range(source="A", start=3.0, end=5.9, transition="crossfade",
                  transition_duration=0.5),
        ],
        platform="reels", detector="saliency",
    )
    # The second range starts half a second earlier once the seam overlaps.
    assert faded.offsets()[1] == pytest.approx(plain.offsets()[1] - 0.5)

    r = render(faded, tmp_path / "cap.mp4", quality="draft",
               base_dir=project, verbose=False)
    assert r.caption_cues > 0
    assert probe(r.output).duration == pytest.approx(faded.total_duration, abs=0.35)


# --- timeline_view ----------------------------------------------------------


def test_timeline_view_renders_a_strip_over_a_waveform(moving_subject, tmp_path):
    from reelforge.timeline_view import timeline_view

    out = tmp_path / "view.png"
    view = timeline_view(moving_subject, 0.0, 5.0, out, frames=8)
    assert out.exists()
    assert view.has_waveform
    # Filmstrip plus waveform stacked, so it is wider than tall.
    assert view.width > view.height
    assert view.frames == 8


def test_timeline_view_works_without_audio(tmp_path):
    from reelforge.timeline_view import timeline_view

    silent = tmp_path / "silent.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=#334455:s=640x360:d=4:r=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-an", str(silent),
    ], check=True, capture_output=True)
    view = timeline_view(silent, 0.0, 4.0, tmp_path / "v.png", frames=4)
    assert view.path.exists()
    assert view.has_waveform is False


def test_timeline_view_rejects_an_empty_range(moving_subject, tmp_path):
    from reelforge.timeline_view import timeline_view

    with pytest.raises(ValueError, match="empty range"):
        timeline_view(moving_subject, 3.0, 3.0, tmp_path / "x.png")


def test_cut_boundaries_land_on_the_seams(two_sources):
    from reelforge.timeline_view import cut_boundaries

    edl = EDL(
        sources={"A": "A.mp4"},
        ranges=[Range(source="A", start=0, end=2),
                Range(source="A", start=2, end=4),
                Range(source="A", start=0, end=2)],
    )
    windows = cut_boundaries(edl, Path("."), window=1.0)
    assert len(windows) == 2                       # one per seam, not per range
    assert windows[0][0] < 2.0 < windows[0][1]


# --- generation -------------------------------------------------------------


def test_mock_provider_produces_usable_media(tmp_path):
    from reelforge.generate import GenRequest, generate

    work = tmp_path / ".reelforge"
    img = generate(GenRequest("image", "a lone lighthouse"), work, provider="mock")
    assert img.path.exists() and img.path.suffix == ".png"

    vid = generate(GenRequest("video", "waves", duration=2.0), work, provider="mock")
    assert probe(vid.path).duration == pytest.approx(2.0, abs=0.3)

    mus = generate(GenRequest("music", "ambient", duration=3.0), work, provider="mock")
    assert mus.duration == pytest.approx(3.0, abs=0.3)


def test_generation_is_cached_by_request(tmp_path):
    from reelforge.generate import GenRequest, generate

    work = tmp_path / ".reelforge"
    first = generate(GenRequest("image", "same prompt"), work, provider="mock")
    second = generate(GenRequest("image", "same prompt"), work, provider="mock")
    assert first.cached is False
    assert second.cached is True
    assert first.path == second.path


def test_audio_only_assets_report_a_real_duration(tmp_path):
    """Regression: probe() requires video, so audio durations read as 0.0."""
    from reelforge.generate import GenRequest, generate

    work = tmp_path / ".reelforge"
    mus = generate(GenRequest("music", "bed", duration=4.0), work, provider="mock")
    assert mus.duration > 1.0


@pytest.mark.skipif(
    shutil.which("espeak-ng") is None and shutil.which("espeak") is None,
    reason="espeak not installed",
)
def test_espeak_produces_real_speech_with_a_real_duration(tmp_path):
    from reelforge.generate import GenRequest, generate

    work = tmp_path / ".reelforge"
    short = generate(GenRequest("speech", "Hello."), work, provider="espeak")
    long = generate(
        GenRequest("speech", "This sentence is considerably longer than the other "
                             "one and should therefore take more time to say."),
        work, provider="espeak",
    )
    assert short.duration > 0.2
    # Real synthesis, not an estimate: more words genuinely takes longer.
    assert long.duration > short.duration * 2


def test_still_becomes_a_moving_clip(tmp_path):
    from reelforge.generate import GenRequest, generate, still_to_clip

    work = tmp_path / ".reelforge"
    img = generate(GenRequest("image", "a lighthouse"), work, provider="mock")
    clip = still_to_clip(img.path, tmp_path / "clip.mp4", 3.0)
    info = probe(clip)
    assert info.duration == pytest.approx(3.0, abs=0.3)
    assert (info.width, info.height) == (1080, 1920)

    # The push means consecutive frames differ; a static still would not.
    def frame(t):
        raw = subprocess.run([
            "ffmpeg", "-v", "error", "-ss", str(t), "-i", str(clip),
            "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-",
        ], capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.uint8).astype(int)

    assert np.abs(frame(0.2) - frame(2.6)).mean() > 0.5


def test_a_whole_video_can_be_built_from_generated_assets(tmp_path):
    """No source footage at all: stills, music, transitions, render."""
    from reelforge.generate import GenRequest, generate, still_to_clip

    work = tmp_path / ".reelforge"
    sources = {}
    for i, prompt in enumerate(["a lighthouse", "dark rocks", "a calm sunrise"]):
        img = generate(GenRequest("image", prompt), work, provider="mock")
        still_to_clip(img.path, tmp_path / f"shot{i}.mp4", 2.0)
        sources[f"S{i}"] = f"shot{i}.mp4"
    music = generate(GenRequest("music", "ambient", duration=8.0), work, provider="mock")

    edl = EDL(
        sources=sources,
        ranges=[
            Range(source="S0", start=0, end=2.0, beat="HOOK"),
            Range(source="S1", start=0, end=2.0, transition="crossfade"),
            Range(source="S2", start=0, end=2.0, transition="dip_white"),
        ],
        platform="reels", reframe="center", music=str(music.path),
        captions=CaptionSpec(enabled=False),
    )
    edl.validate(base_dir=tmp_path)
    result = render(edl, tmp_path / "generated.mp4", quality="draft",
                    base_dir=tmp_path, verbose=False)
    info = probe(result.output)
    assert info.height > info.width
    assert info.has_audio
    assert info.duration == pytest.approx(edl.total_duration, abs=0.4)
