import pytest

from pipeline.detection_scale import DETECTION_FLOOR_INPUT_PX, detection_scale


class TestDetectionScale:
    """Issue #113 — a survivorship-bias-free scene flag for detection recall.

    ``detection_scale`` answers "how small is the smallest person this scene's
    geometry lets the detector resolve", from config + frame size + the ~60 px
    model-input floor #101 measured on ``magazyn-hall-v1``. It is a pure function
    so the math is covered here, without the full analyze pipeline.
    """

    def test_magazyn_640_anchor_flags_high_risk(self):
        # #113's headline anchor: magazyn (3840x2160) at the shipped 640 input.
        # width binds (640/3840 = 0.1667), so a person must be >= 360 native px
        # (~1/6 of a 2160 frame) to reach the floor. That undercuts the mid field.
        scale = detection_scale([3840, 2160], (640, 640), [])

        assert scale["input_scale"] == pytest.approx(0.16667, abs=1e-4)
        assert scale["floor_input_px"] == 60
        assert scale["resolvable_height_native_px"] == pytest.approx(360.0)
        assert scale["resolvable_height_frac"] == pytest.approx(0.16667, abs=1e-4)
        assert scale["recall_risk"] == "high"

    def test_reports_median_native_height_and_count_of_detections(self):
        # The detected-height median is carried as supporting context. It is the
        # median of native bbox heights, and detections_measured is the n behind it.
        scale = detection_scale([3840, 2160], (640, 640), [200.0, 400.0, 300.0])

        assert scale["median_detected_height_native_px"] == pytest.approx(300.0)
        assert scale["detections_measured"] == 3

    def test_magazyn_1280x736_anchor_clears_to_normal(self):
        # Same hall at 1280x736 (#100/#101): scale doubles to 0.333, the resolvable
        # floor halves to ~180 native px (~1/12 of frame), and the flag clears.
        scale = detection_scale([3840, 2160], (1280, 736), [])

        assert scale["input_scale"] == pytest.approx(0.33333, abs=1e-4)
        assert scale["resolvable_height_native_px"] == pytest.approx(180.0)
        assert scale["resolvable_height_frac"] == pytest.approx(0.08333, abs=1e-4)
        assert scale["recall_risk"] == "normal"

    def test_no_detections_yields_null_median_but_still_scores_scale(self):
        # A run that found nobody still has a scene geometry to report; only the
        # detected-height median is unknown.
        scale = detection_scale([3840, 2160], (640, 640), [])

        assert scale["detections_measured"] == 0
        assert scale["median_detected_height_native_px"] is None
        assert scale["recall_risk"] == "high"

    def test_returns_none_when_no_frame_was_analysed(self):
        # source_frame is None when the run analysed zero frames — there is no
        # scene to describe, so the whole block is null rather than fabricated.
        assert detection_scale(None, (640, 640), []) is None

    def test_floor_is_the_documented_101_constant(self):
        # The reported floor is the module constant #101 measured, not a literal
        # the caller could drift from.
        scale = detection_scale([3840, 2160], (640, 640), [])

        assert scale["floor_input_px"] == DETECTION_FLOOR_INPUT_PX == 60
