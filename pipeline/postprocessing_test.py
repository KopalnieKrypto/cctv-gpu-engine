"""Tests for YOLO-pose ONNX output parsing, NMS, and coordinate scaling."""

import numpy as np
import pytest

from pipeline.postprocessing import (
    CONFIDENCE_THRESHOLD,
    Detection,
    Keypoint,
    nms,
    postprocess,
)


def _det(x1: float, y1: float, x2: float, y2: float, conf: float) -> Detection:
    return Detection(
        bbox=[x1, y1, x2, y2],
        confidence=conf,
        keypoints=[Keypoint(0.0, 0.0, 0.0)] * 17,
    )


def _make_yolo_output(
    boxes: list[tuple[float, float, float, float, float]],
    keypoints_per_box: list[list[tuple[float, float, float]]] | None = None,
) -> np.ndarray:
    """Build a synthetic YOLO-pose output tensor [1, 56, N].

    Each box is (cx, cy, w, h, conf) in 640-space; keypoints are 17 (x, y, vis)
    triples in 640-space. If keypoints_per_box is None, all keypoints are zero.
    """
    n = len(boxes)
    data = np.zeros((1, 56, n), dtype=np.float32)
    for i, (cx, cy, w, h, conf) in enumerate(boxes):
        data[0, 0, i] = cx
        data[0, 1, i] = cy
        data[0, 2, i] = w
        data[0, 3, i] = h
        data[0, 4, i] = conf
        if keypoints_per_box is not None:
            for k, (kx, ky, kv) in enumerate(keypoints_per_box[i]):
                base = 5 + k * 3
                data[0, base, i] = kx
                data[0, base + 1, i] = ky
                data[0, base + 2, i] = kv
    return data


class TestPostprocessParsing:
    def test_parses_single_high_confidence_detection_with_17_keypoints(self):
        # synthetic output: 1 person centered in 640x640 with high confidence
        kps = [(320.0 + i * 5, 320.0 + i * 5, 0.9) for i in range(17)]
        output = _make_yolo_output(
            boxes=[(320.0, 320.0, 100.0, 200.0, 0.85)],
            keypoints_per_box=[kps],
        )

        detections = postprocess(output, orig_w=640, orig_h=640)

        assert len(detections) == 1
        det = detections[0]
        assert isinstance(det, Detection)
        assert det.confidence == pytest.approx(0.85, abs=1e-5)
        # bbox in xyxy format (orig=640, no scaling here)
        assert det.bbox == pytest.approx([270.0, 220.0, 370.0, 420.0], abs=1e-4)
        # 17 COCO keypoints
        assert len(det.keypoints) == 17
        for k, kp in enumerate(det.keypoints):
            assert isinstance(kp, Keypoint)
            assert kp.x == pytest.approx(320.0 + k * 5, abs=1e-4)
            assert kp.y == pytest.approx(320.0 + k * 5, abs=1e-4)
            assert kp.vis == pytest.approx(0.9, abs=1e-5)


class TestPostprocessConfidenceFilter:
    def test_filters_out_detections_below_threshold(self):
        # 3 candidate boxes: above, below, and exactly on the threshold
        output = _make_yolo_output(
            boxes=[
                (100.0, 100.0, 50.0, 100.0, 0.9),     # keep (well above)
                (200.0, 200.0, 50.0, 100.0, 0.10),    # drop (well below)
                (300.0, 300.0, 50.0, 100.0, CONFIDENCE_THRESHOLD - 0.001),  # drop (just below)
            ]
        )

        detections = postprocess(output, orig_w=640, orig_h=640)

        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.9, abs=1e-5)

    def test_returns_empty_list_when_no_detections_pass_threshold(self):
        output = _make_yolo_output(
            boxes=[(100.0, 100.0, 50.0, 100.0, 0.05)]
        )

        detections = postprocess(output, orig_w=640, orig_h=640)

        assert detections == []

    def test_handles_empty_output_tensor(self):
        # YOLO can return zero candidate boxes — must not crash
        output = np.zeros((1, 56, 0), dtype=np.float32)

        detections = postprocess(output, orig_w=1920, orig_h=1080)

        assert detections == []


class TestPostprocessCoordinateScaling:
    def test_scales_bbox_and_keypoints_to_non_square_original_image(self):
        # Person centered in 640-space, model thinks they are at (320, 320)
        # with bbox 100×200. Original image is 1920×1080 (sx=3.0, sy=1.6875).
        kps = [(320.0, 320.0, 0.9)] + [(0.0, 0.0, 0.0)] * 16
        output = _make_yolo_output(
            boxes=[(320.0, 320.0, 100.0, 200.0, 0.9)],
            keypoints_per_box=[kps],
        )

        detections = postprocess(output, orig_w=1920, orig_h=1080)

        assert len(detections) == 1
        det = detections[0]
        # bbox: x scaled by 3.0, y scaled by 1.6875
        # x1=(320-50)*3=810, x2=(320+50)*3=1110
        # y1=(320-100)*1.6875=371.25, y2=(320+100)*1.6875=708.75
        assert det.bbox == pytest.approx([810.0, 371.25, 1110.0, 708.75], abs=1e-3)
        # keypoint 0 at center scales: 320*3=960, 320*1.6875=540
        kp = det.keypoints[0]
        assert kp.x == pytest.approx(960.0, abs=1e-3)
        assert kp.y == pytest.approx(540.0, abs=1e-3)
        # visibility is NOT scaled — it's a confidence, not a coord
        assert kp.vis == pytest.approx(0.9, abs=1e-5)


class TestNMS:
    def test_suppresses_overlapping_box_with_lower_confidence(self):
        # Two boxes with significant overlap (IoU > 0.45) — keep higher conf
        high = _det(100, 100, 200, 200, conf=0.9)
        low = _det(110, 110, 210, 210, conf=0.7)  # IoU ≈ 0.68

        kept = nms([low, high])  # input order should not matter

        assert len(kept) == 1
        assert kept[0].confidence == pytest.approx(0.9)

    def test_keeps_all_when_no_overlap(self):
        a = _det(0, 0, 100, 100, conf=0.9)
        b = _det(200, 200, 300, 300, conf=0.85)
        c = _det(400, 400, 500, 500, conf=0.8)

        kept = nms([a, b, c])

        assert len(kept) == 3
        confs = sorted(d.confidence for d in kept)
        assert confs == pytest.approx([0.8, 0.85, 0.9])

    def test_keeps_partially_overlapping_boxes_below_iou_threshold(self):
        # Two boxes 100×100 with 30px x-overlap → intersection 30×100=3000,
        # union = 100*100*2 - 3000 = 17000 → IoU ≈ 0.176 < 0.45
        a = _det(0, 0, 100, 100, conf=0.9)
        b = _det(70, 0, 170, 100, conf=0.85)

        kept = nms([a, b])

        assert len(kept) == 2

    def test_returns_empty_list_for_empty_input(self):
        assert nms([]) == []


class TestPostprocessAppliesNMS:
    def test_postprocess_suppresses_duplicate_boxes_via_nms(self):
        # Two near-identical detections — postprocess should return one
        kps = [(0.0, 0.0, 0.0)] * 17
        output = _make_yolo_output(
            boxes=[
                (320.0, 320.0, 100.0, 200.0, 0.95),
                (322.0, 318.0, 100.0, 200.0, 0.80),
            ],
            keypoints_per_box=[kps, kps],
        )

        detections = postprocess(output, orig_w=640, orig_h=640)

        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.95, abs=1e-5)
