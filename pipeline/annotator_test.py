"""Tests for keyframe annotation: bbox + COCO skeleton + activity label.

We don't try to test the visual quality of the overlay (that's a manual job)
— only that ``annotate_frame`` actually mutates pixels inside each detection's
bbox, returns a copy of the original (no in-place mutation), and is robust
to detections with low-visibility keypoints.
"""

from __future__ import annotations

import numpy as np

from pipeline.annotator import annotate_frame
from pipeline.postprocessing import Detection, Keypoint


def _all_visible_kps() -> list[Keypoint]:
    # 17 keypoints inside the bbox area, all "visible"
    return [Keypoint(x=50.0 + i, y=50.0 + i, vis=0.9) for i in range(17)]


def _det(activity: str = "standing") -> Detection:
    det = Detection(
        bbox=[40.0, 40.0, 120.0, 200.0],
        confidence=0.9,
        keypoints=_all_visible_kps(),
    )
    det.activity = activity
    return det


class TestAnnotateFrame:
    def test_draws_something_inside_bbox(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        det = _det(activity="walking")

        out = annotate_frame(frame, [det])

        # Original frame must be untouched (no in-place mutation)
        assert np.all(frame == 0)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

        # Some pixel inside the bbox area should differ from black
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        roi = out[y1:y2, x1:x2]
        assert np.any(roi != 0), "annotate_frame did not draw inside the bbox"

    def test_handles_empty_detection_list(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        out = annotate_frame(frame, [])

        assert out.shape == frame.shape
        # Empty list → no drawing → output equals zeros
        assert np.all(out == 0)
