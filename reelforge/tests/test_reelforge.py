"""Tests for the deterministic core.

Everything here runs without ffmpeg, a network, or footage. The parts that do
need media — frame sampling, the render pipeline — are exercised by
`tests/test_pipeline.py`, which skips itself when ffmpeg is absent.
"""

from __future__ import annotations

import json
import math

import pytest

from reelforge import captions as cap
from reelforge.config import SafeZone, get_platform, get_quality
from reelforge.edl import EDL, CaptionSpec, EDLError, Overlay, Range
from reelforge.grade import resolve_grade
from reelforge.reframe import (
    OperatorSettings,
    build_position_expr,
    fill_gaps,
    median_filter,
    plan_geometry,
    reduce_keyframes,
    run_operator,
)
from reelforge.render import overlay_position


# --- geometry ---------------------------------------------------------------


def test_landscape_to_vertical_crops_width_keeps_height():
    g = plan_geometry(1920, 1080, 9 / 16)
    assert g.axis == "x"
    assert g.crop_h == 1080
    assert g.crop_w == 608  # 1080 * 9/16, rounded even
    assert g.travel == 1920 - 608


def test_vertical_to_square_crops_height():
    g = plan_geometry(1080, 1920, 1.0)
    assert g.axis == "y"
    assert (g.crop_w, g.crop_h) == (1080, 1080)
    assert g.travel == 840


def test_matching_aspect_leaves_nothing_to_pan():
    g = plan_geometry(1080, 1920, 9 / 16)
    assert g.axis == "none"
    assert g.travel == 0


def test_crop_dimensions_are_always_even():
    # yuv420p subsamples chroma 2x2; an odd dimension is not encodable.
    for w, h in [(1919, 1081), (1280, 719), (999, 555), (4096, 2160)]:
        g = plan_geometry(w, h, 9 / 16)
        assert g.crop_w % 2 == 0 and g.crop_h % 2 == 0


def test_crop_never_exceeds_the_source():
    for w, h in [(640, 480), (1080, 1080), (3840, 2160), (720, 1280)]:
        for ar in (9 / 16, 1.0, 4 / 5, 16 / 9):
            g = plan_geometry(w, h, ar)
            assert g.crop_w <= w + 1 and g.crop_h <= h + 1


@pytest.mark.parametrize("w,h", [(0, 100), (100, 0), (-5, 10)])
def test_invalid_dimensions_rejected(w, h):
    with pytest.raises(ValueError):
        plan_geometry(w, h, 9 / 16)


# --- position expression ----------------------------------------------------


def _eval_expr(expr: str, t: float) -> float:
    """Evaluate a generated ffmpeg expression in Python.

    The subset emitted here — `if`, `lt`, arithmetic — maps cleanly onto Python,
    so the interpolation can be checked without invoking ffmpeg.
    """
    import re

    py = expr
    while "if(" in py:
        start = py.index("if(")
        depth, i = 0, start + 2
        while i < len(py):
            if py[i] == "(":
                depth += 1
            elif py[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        inner = py[start + 3 : i]
        parts, depth, last = [], 0, 0
        for j, ch in enumerate(inner):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(inner[last:j])
                last = j + 1
        parts.append(inner[last:])
        cond, a, b = parts
        py = py[:start] + f"(({a}) if ({cond}) else ({b}))" + py[i + 1 :]
    py = re.sub(r"lt\(([^,]+),([^)]+)\)", r"(\1)<(\2)", py)
    return float(eval(py, {"t": t, "__builtins__": {}}))  # noqa: S307 — generated input


def test_single_keyframe_is_constant():
    assert build_position_expr([(0.0, 656.0)]) == "656.00"


def test_expression_interpolates_linearly_between_keys():
    expr = build_position_expr([(0.0, 100.0), (2.0, 300.0)])
    assert _eval_expr(expr, 0.0) == pytest.approx(100.0, abs=0.5)
    assert _eval_expr(expr, 1.0) == pytest.approx(200.0, abs=0.5)
    assert _eval_expr(expr, 2.0) == pytest.approx(300.0, abs=0.5)


def test_expression_holds_flat_outside_the_keyframe_range():
    expr = build_position_expr([(1.0, 100.0), (2.0, 300.0)])
    assert _eval_expr(expr, 0.0) == pytest.approx(100.0, abs=0.5)   # before first
    assert _eval_expr(expr, 9.0) == pytest.approx(300.0, abs=0.5)   # after last


def test_expression_survives_duplicate_timestamps():
    # Degenerate spans would divide by zero if not guarded.
    expr = build_position_expr([(0.0, 10.0), (0.0, 20.0), (1.0, 30.0)])
    assert math.isfinite(_eval_expr(expr, 0.5))


def test_empty_keyframes_yield_zero():
    assert build_position_expr([]) == "0"


# --- keyframe reduction -----------------------------------------------------


def test_reduction_keeps_endpoints_and_the_corner():
    times = [i * 0.25 for i in range(20)]
    values = [100.0] * 10 + [100.0 + 40 * i for i in range(10)]
    keys = reduce_keyframes(times, values, tolerance=1.5)
    assert len(keys) < len(times)
    assert keys[0] == (times[0], values[0])
    assert keys[-1] == (times[-1], values[-1])


def test_a_straight_line_reduces_to_two_points():
    times = [i * 0.25 for i in range(12)]
    values = [10.0 * i for i in range(12)]
    assert len(reduce_keyframes(times, values, tolerance=1.0)) == 2


def test_reduction_preserves_the_curve_within_tolerance():
    times = [i * 0.2 for i in range(30)]
    values = [200 + 150 * math.sin(t) for t in times]
    keys = reduce_keyframes(times, values, tolerance=2.0)
    expr = build_position_expr(keys)
    for t, v in zip(times, values):
        assert abs(_eval_expr(expr, t) - v) <= 3.0


# --- the virtual operator ---------------------------------------------------


def test_operator_ignores_jitter_inside_the_deadband():
    # ±1% wobble on a 608px crop is ~19px peak-to-peak, well inside the
    # 10% deadband, so the camera should not move at all.
    targets = [0.5 + (0.01 if i % 2 else -0.01) for i in range(40)]
    pos = run_operator(targets, 0.25, travel=1312, crop_extent=608)
    assert max(pos) - min(pos) < 1.0


def test_operator_follows_a_genuine_move():
    targets = [0.3] * 8 + [0.7] * 24
    pos = run_operator(targets, 0.25, travel=1312, crop_extent=608)
    assert pos[-1] > pos[0] + 300


def test_operator_respects_the_speed_ceiling():
    settings = OperatorSettings(max_speed=0.2)
    targets = [0.0] + [1.0] * 30
    pos = run_operator(targets, 0.25, travel=1312, crop_extent=608, settings=settings)
    cap_px = 0.2 * 608 * 0.25
    assert all(abs(b - a) <= cap_px + 1e-6 for a, b in zip(pos, pos[1:]))


def test_operator_stays_within_the_source():
    for target in (0.0, 1.0, -0.5, 1.5):
        pos = run_operator([target] * 20, 0.25, travel=1312, crop_extent=608)
        assert all(0.0 <= p <= 1312 for p in pos)


def test_operator_handles_no_travel():
    assert run_operator([0.5] * 5, 0.25, travel=0, crop_extent=1080) == [0.0] * 5


def test_operator_settings_validate():
    with pytest.raises(ValueError):
        OperatorSettings(deadband=1.5)
    with pytest.raises(ValueError):
        OperatorSettings(max_speed=0)


# --- track post-processing --------------------------------------------------


def test_gaps_interpolate_between_known_neighbours():
    track = [(0.0, 0.5), None, None, (0.3, 0.5)]
    filled = fill_gaps(track)
    assert filled[1][0] == pytest.approx(0.1)
    assert filled[2][0] == pytest.approx(0.2)


def test_leading_and_trailing_gaps_hold_the_nearest_value():
    filled = fill_gaps([None, (0.4, 0.5), None])
    assert filled[0] == (0.4, 0.5)
    assert filled[2] == (0.4, 0.5)


def test_an_empty_track_falls_back_to_centre():
    assert fill_gaps([None, None]) == [(0.5, 0.5), (0.5, 0.5)]


def test_median_filter_removes_a_single_outlier():
    values = [10.0, 10.0, 900.0, 10.0, 10.0]
    assert max(median_filter(values, k=5)) < 50


# --- safe zones -------------------------------------------------------------


def test_safe_zone_scales_proportionally():
    zone = SafeZone(top=180, bottom=420, left=48, right=228)
    half = zone.scaled_to(1080, 1920, 540, 960)
    assert (half.top, half.bottom, half.left, half.right) == (90, 210, 24, 114)


def test_content_box_excludes_the_insets():
    p = get_platform("reels")
    x, y, w, h = p.safe_at(1080, 1920).content_box(1080, 1920)
    assert (x, y) == (48, 180)
    assert w == 1080 - 48 - 228
    assert h == 1920 - 180 - 420


def test_reels_reserves_more_on_the_right_for_the_action_rail():
    p = get_platform("reels")
    assert p.safe.right > p.safe.left


def test_every_platform_has_a_coherent_safe_area():
    for key in ("reels", "tiktok", "shorts", "feed", "square", "landscape"):
        p = get_platform(key)
        _, _, w, h = p.safe_at(p.width, p.height).content_box(p.width, p.height)
        assert w > 0 and h > 0, key
        assert p.sweet_spot_s[0] < p.sweet_spot_s[1], key


def test_unknown_platform_lists_the_valid_keys():
    with pytest.raises(KeyError, match="reels"):
        get_platform("nope")


def test_unknown_quality_rejected():
    with pytest.raises(KeyError):
        get_quality("ultra")


# --- captions ---------------------------------------------------------------


def test_ass_colour_is_byte_reversed_with_alpha_first():
    assert cap.to_ass_colour("#FFD400") == "&H0000D4FF"
    assert cap.to_ass_colour("#000000") == "&H00000000"
    assert cap.to_ass_colour("#FFF") == "&H00FFFFFF"


def test_bad_colour_rejected():
    with pytest.raises(ValueError):
        cap.to_ass_colour("chartreuse")


def _words(spec):
    return [cap.Word(t, s, e) for t, s, e in spec]


def test_chunking_respects_the_word_count():
    words = _words([(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(6)])
    assert [len(c) for c in cap.chunk_words(words, 2)] == [2, 2, 2]


def test_chunking_breaks_early_at_a_sentence_end():
    words = _words([("hi.", 0, 0.3), ("there", 0.4, 0.7), ("now", 0.8, 1.0)])
    chunks = cap.chunk_words(words, 3)
    assert [w.text for w in chunks[0]] == ["hi."]


def test_cues_map_onto_the_output_timeline():
    # A range starting at 10s placed at output offset 0 must yield cues at ~0s.
    words = _words([("a", 10.0, 10.4), ("b", 10.5, 10.9)])
    cues = cap.build_cues([(words, 10.0, 11.0, 0.0)], cap.get_style("punch"))
    assert cues[0].start == pytest.approx(0.0, abs=0.01)
    assert cues[0].end == pytest.approx(0.9, abs=0.01)


def test_cue_offsets_accumulate_across_ranges():
    # The classic drift bug: the second range's cues must be shifted by the
    # first range's duration, not by its source position.
    a = _words([("one", 5.0, 5.4)])
    b = _words([("two", 30.0, 30.4)])
    style = cap.get_style("punch")
    cues = cap.build_cues([(a, 5.0, 6.0, 0.0), (b, 30.0, 31.0, 1.0)], style)
    assert cues[0].start == pytest.approx(0.0, abs=0.01)
    assert cues[1].start == pytest.approx(1.0, abs=0.01)


def test_words_outside_the_range_are_dropped():
    words = _words([("before", 0.0, 0.5), ("inside", 5.2, 5.6), ("after", 9.0, 9.5)])
    cues = cap.build_cues([(words, 5.0, 6.0, 0.0)], cap.get_style("punch"))
    assert [w.text for c in cues for w in c.words] == ["inside"]


def test_very_short_cues_are_extended_to_stay_readable():
    words = _words([("x", 0.0, 0.05)])
    cues = cap.build_cues([(words, 0.0, 1.0, 0.0)], cap.get_style("punch"))
    assert cues[0].end - cues[0].start >= 0.2


def test_extension_never_overlaps_the_following_cue():
    words = _words([("a", 0.0, 0.04), ("b", 0.10, 0.9)])
    cues = cap.build_cues([(words, 0.0, 1.0, 0.0)], cap.get_style("punch"))
    for x, y in zip(cues, cues[1:]):
        assert x.end <= y.start + 1e-6


def test_wrapping_measures_visible_text_not_markup():
    # Regression: the karaoke path once wrapped on the styled string, so
    # `{\c&H0000D4FF}WE{\c&H00FFFFFF}` counted as ~30 characters and a
    # two-word card broke onto two lines.
    lines = cap.wrap_tokens(["WE", "TRIED"], 18)
    assert lines == [[0, 1]]


def test_wrapping_breaks_when_genuinely_too_long():
    lines = cap.wrap_tokens(["ABCDEFGH", "IJKLMNOP", "QRSTUVWX"], 18)
    assert len(lines) == 2


def test_captions_centre_on_the_frame_not_the_safe_box():
    # Centring inside the asymmetric safe area pushes captions visibly left.
    p = get_platform("reels")
    cx, _ = cap.caption_position(p, 1080, 1920, 0.72)
    assert cx == 540


def test_caption_vertical_position_derives_from_the_safe_area():
    reels = cap.caption_position(get_platform("reels"), 1080, 1920, 0.72)[1]
    tiktok = cap.caption_position(get_platform("tiktok"), 1080, 1920, 0.72)[1]
    # TikTok reserves more at the bottom, so its captions must sit higher.
    assert tiktok < reels


def test_caption_max_width_is_symmetric_about_centre():
    p = get_platform("reels")
    width = cap.caption_max_width(p, 1080, 1920)
    assert width == 2 * (1080 - p.safe.right - 540)
    assert width < 1080 - p.safe.left - p.safe.right


def test_spec_fields_do_not_override_the_preset_unless_set():
    # Regression: CaptionSpec defaults once silently replaced preset values.
    style = cap.resolve_style(CaptionSpec(style="karaoke"))
    assert style.words_per_cue == cap.get_style("karaoke").words_per_cue == 4


def test_explicit_spec_fields_do_override():
    style = cap.resolve_style(CaptionSpec(style="karaoke", words_per_cue=2))
    assert style.words_per_cue == 2


def test_ass_document_is_well_formed():
    words = _words([("hello", 0.0, 0.4), ("world", 0.5, 0.9)])
    cues = cap.build_cues([(words, 0.0, 1.0, 0.0)], cap.get_style("punch"))
    doc = cap.build_ass(cues, cap.get_style("punch"), get_platform("reels"), 1080, 1920)
    assert "PlayResX: 1080" in doc and "PlayResY: 1920" in doc
    assert "[V4+ Styles]" in doc and "[Events]" in doc
    assert doc.count("Dialogue:") == len(cues)


def test_karaoke_emits_one_event_per_word():
    words = _words([("a", 0.0, 0.3), ("b", 0.35, 0.6), ("c", 0.65, 0.9)])
    style = cap.get_style("karaoke")
    cues = cap.build_cues([(words, 0.0, 1.0, 0.0)], style)
    doc = cap.build_ass(cues, style, get_platform("reels"), 1080, 1920)
    assert doc.count("Dialogue:") == 3


def test_caption_text_cannot_inject_ass_markup():
    words = _words([("{\\c&HFF0000}", 0.0, 0.4)])
    cues = cap.build_cues([(words, 0.0, 1.0, 0.0)], cap.get_style("punch"))
    doc = cap.build_ass(cues, cap.get_style("punch"), get_platform("reels"), 1080, 1920)
    body = doc.split("Dialogue:")[1]
    # The override block belongs to us; the payload must be neutralised.
    assert "\\c&HFF0000" not in body


def test_srt_round_trips_timing():
    words = _words([("hello", 1.5, 2.0)])
    cues = cap.build_cues([(words, 0.0, 3.0, 0.0)], cap.get_style("punch"))
    srt = cap.build_srt(cues, cap.get_style("punch"))
    assert "00:00:01,500 --> 00:00:02,000" in srt


def test_find_words_accepts_scribe_and_whisper_shapes():
    scribe = {"words": [{"text": "hi", "start": 0, "end": 0.3, "type": "word"},
                        {"text": " ", "start": 0.3, "end": 0.31, "type": "spacing"}]}
    whisper = {"segments": [{"words": [{"word": "hi", "start": 0, "end": 0.3}]}]}
    assert len(cap.find_words(scribe)) == 1
    assert len(cap.find_words(whisper)) == 1


def test_find_words_drops_entries_without_timings():
    assert cap.find_words({"words": [{"text": "hi"}]}) == []


# --- grades -----------------------------------------------------------------


def test_named_preset_resolves_to_a_chain():
    assert "eq=" in resolve_grade("punch")


def test_none_and_empty_resolve_to_no_filter():
    assert resolve_grade("none") == ""
    assert resolve_grade(None) == ""


def test_raw_filter_strings_pass_through():
    raw = "eq=contrast=1.5:saturation=0.2"
    assert resolve_grade(raw) == raw


def test_a_typo_in_a_preset_name_is_loud():
    # Silently rendering ungraded would be worse than failing.
    with pytest.raises(KeyError):
        resolve_grade("cinematicc")


# --- overlay anchoring ------------------------------------------------------


def test_anchor_expressions_reference_overlay_dimensions():
    p = get_platform("reels")
    x, y = overlay_position(Overlay(file="x", start_in_output=0, duration=1,
                                    anchor="bottom-right"), p, 1080, 1920)
    assert "-w" in x and "-h" in y


def test_top_left_anchor_lands_on_the_safe_origin():
    p = get_platform("reels")
    x, y = overlay_position(Overlay(file="x", start_in_output=0, duration=1,
                                    anchor="top-left"), p, 1080, 1920)
    assert x == "48" and y == "180"


def test_ignoring_the_safe_area_anchors_to_the_frame():
    p = get_platform("reels")
    x, y = overlay_position(Overlay(file="x", start_in_output=0, duration=1,
                                    anchor="top-left", ignore_safe_area=True), p, 1080, 1920)
    assert x == "0" and y == "0"


def test_offsets_are_applied_to_the_anchor():
    p = get_platform("reels")
    x, y = overlay_position(Overlay(file="x", start_in_output=0, duration=1,
                                    anchor="top-left", dx=10, dy=-20), p, 1080, 1920)
    assert "10" in x and "-20" in y


# --- EDL --------------------------------------------------------------------


def _edl(tmp_path, **kw):
    (tmp_path / "a.mp4").write_bytes(b"stub")
    base = dict(
        sources={"A": "a.mp4"},
        ranges=[Range(source="A", start=0.0, end=10.0, beat="HOOK")],
    )
    base.update(kw)
    return EDL(**base)


def test_duration_accounts_for_speed(tmp_path):
    edl = _edl(tmp_path, ranges=[Range(source="A", start=0.0, end=10.0, speed=2.0)])
    assert edl.total_duration == pytest.approx(5.0)


def test_offsets_are_cumulative(tmp_path):
    edl = _edl(tmp_path, ranges=[
        Range(source="A", start=0.0, end=3.0),
        Range(source="A", start=20.0, end=22.0),
        Range(source="A", start=40.0, end=41.0),
    ])
    assert edl.offsets() == pytest.approx([0.0, 3.0, 5.0])


def test_validation_reports_every_problem_at_once(tmp_path):
    edl = _edl(tmp_path, ranges=[
        Range(source="MISSING", start=0.0, end=1.0),
        Range(source="A", start=5.0, end=2.0),
    ])
    with pytest.raises(EDLError) as e:
        edl.validate(base_dir=tmp_path)
    assert len(e.value.problems) >= 2


def test_missing_source_file_is_caught(tmp_path):
    edl = EDL(sources={"A": "nope.mp4"},
              ranges=[Range(source="A", start=0.0, end=1.0)])
    with pytest.raises(EDLError, match="not found"):
        edl.validate(base_dir=tmp_path)


def test_over_length_edit_is_rejected(tmp_path):
    edl = _edl(tmp_path, ranges=[Range(source="A", start=0.0, end=200.0)],
               platform="reels")
    with pytest.raises(EDLError, match="exceeds"):
        edl.validate(base_dir=tmp_path)


def test_sub_frame_range_is_flagged(tmp_path):
    edl = _edl(tmp_path, ranges=[Range(source="A", start=0.0, end=0.05)])
    with pytest.raises(EDLError, match="likely a mistake"):
        edl.validate(base_dir=tmp_path)


def test_overlay_past_the_end_is_rejected(tmp_path):
    (tmp_path / "ov.webm").write_bytes(b"stub")
    edl = _edl(tmp_path, overlays=[
        Overlay(file="ov.webm", start_in_output=99.0, duration=1.0)
    ])
    with pytest.raises(EDLError, match="past the"):
        edl.validate(base_dir=tmp_path)


def test_overlay_fades_cannot_exceed_its_duration(tmp_path):
    (tmp_path / "ov.webm").write_bytes(b"stub")
    edl = _edl(tmp_path, overlays=[
        Overlay(file="ov.webm", start_in_output=0.0, duration=1.0,
                fade_in=0.8, fade_out=0.8)
    ])
    with pytest.raises(EDLError, match="fades"):
        edl.validate(base_dir=tmp_path)


def test_round_trip_preserves_the_edit(tmp_path):
    edl = _edl(tmp_path, platform="tiktok", grade="cinematic",
               captions=CaptionSpec(style="karaoke", words_per_cue=3),
               # 5s of source at 1.25x is 4s of output, clear of the 3s floor.
               ranges=[Range(source="A", start=1.0, end=6.0, beat="HOOK",
                             reason="cleanest take", speed=1.25)])
    path = edl.save(tmp_path / "edl.json")
    back = EDL.load(path)
    assert back.platform == "tiktok"
    assert back.grade == "cinematic"
    assert back.captions.words_per_cue == 3
    assert back.ranges[0].speed == 1.25
    assert back.ranges[0].reason == "cleanest take"
    assert back.total_duration == pytest.approx(edl.total_duration)


def test_unknown_fields_are_ignored_on_load(tmp_path):
    # Forward compatibility: a newer writer's extra keys must not break loading.
    data = _edl(tmp_path).to_dict()
    data["ranges"][0]["future_field"] = "x"
    data["some_new_top_level"] = 1
    assert EDL.from_dict(data).ranges[0].source == "A"


def test_malformed_json_is_reported_clearly(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    with pytest.raises(EDLError, match="not valid JSON"):
        EDL.load(p)


def test_per_range_overrides_fall_back_to_the_edl_default(tmp_path):
    edl = _edl(tmp_path, reframe="track", grade="punch", ranges=[
        Range(source="A", start=0.0, end=2.0),
        Range(source="A", start=2.0, end=4.0, reframe="blur_pad", grade="matte"),
    ])
    assert edl.reframe_for(edl.ranges[0]) == "track"
    assert edl.reframe_for(edl.ranges[1]) == "blur_pad"
    assert edl.grade_for(edl.ranges[1]) == "matte"


# --- retention --------------------------------------------------------------


def test_linter_flags_a_late_hook(tmp_path):
    from reelforge.retention import lint

    (tmp_path / "a.mp4").write_bytes(b"stub")
    work = tmp_path / ".reelforge" / "transcripts"
    work.mkdir(parents=True)
    (work / "A.json").write_text(json.dumps({"words": [
        {"text": "so", "start": 3.0, "end": 3.4, "type": "word"},
        {"text": "anyway", "start": 3.5, "end": 4.0, "type": "word"},
    ]}))
    edl = EDL(sources={"A": "a.mp4"},
              ranges=[Range(source="A", start=0.0, end=10.0, beat="HOOK")])
    report = lint(edl, base_dir=tmp_path, work_dir=tmp_path / ".reelforge")
    codes = {f.code for f in report.findings}
    assert "hook.late" in codes
    assert "hook.filler" in codes


def test_linter_scores_a_clean_edit_highly(tmp_path):
    from reelforge.retention import lint

    (tmp_path / "a.mp4").write_bytes(b"stub")
    work = tmp_path / ".reelforge" / "transcripts"
    work.mkdir(parents=True)
    words = [{"text": f"w{i}", "start": 0.1 + i * 0.33, "end": 0.35 + i * 0.33,
              "type": "word"} for i in range(60)]
    (work / "A.json").write_text(json.dumps({"words": words}))
    edl = EDL(sources={"A": "a.mp4"}, ranges=[
        Range(source="A", start=0.0, end=6.0, beat="HOOK"),
        Range(source="A", start=6.0, end=11.0, beat="PAYOFF"),
        Range(source="A", start=11.0, end=15.0, beat="CTA"),
    ])
    report = lint(edl, base_dir=tmp_path, work_dir=tmp_path / ".reelforge")
    codes = {f.code for f in report.findings}
    # The stub source cannot be probed, and the linter is right to say so; every
    # transcript-driven check should nonetheless come back clean.
    assert {f.code for f in report.errors} == {"source.unreadable"}
    assert not codes & {
        "hook.late", "hook.filler", "hook.thin",
        "pace.slow", "pace.deadair", "pace.longshot", "captions.flash",
    }


def test_linter_flags_a_long_uncovered_shot(tmp_path):
    from reelforge.retention import lint

    (tmp_path / "a.mp4").write_bytes(b"stub")
    edl = EDL(sources={"A": "a.mp4"},
              ranges=[Range(source="A", start=0.0, end=20.0, beat="HOOK")],
              captions=CaptionSpec(enabled=False))
    report = lint(edl, base_dir=tmp_path, work_dir=tmp_path / ".reelforge")
    assert "pace.longshot" in {f.code for f in report.findings}


def test_report_score_decreases_with_severity():
    from reelforge.retention import Report

    clean, noisy = Report(), Report()
    noisy.add("error", "x", "bad")
    assert clean.score == 100
    assert noisy.score < clean.score
    assert not noisy.ok
