"""Tests for the per-frame detections JSONL dump (issue #35)."""

from __future__ import annotations

import json

from pipeline.detections_dump import frame_to_jsonl_line
from pipeline.postprocessing import Detection, Keypoint


def _detection(activity: str = "standing", confidence: float = 0.9) -> Detection:
    kps = [Keypoint(x=float(i), y=float(i * 2), vis=0.95) for i in range(17)]
    det = Detection(bbox=[100.0, 200.0, 300.0, 600.0], confidence=confidence, keypoints=kps)
    det.activity = activity
    return det


class TestFrameToJsonlLine:
    def test_serializes_one_frame_to_a_single_line_json_object(self) -> None:
        line = frame_to_jsonl_line(
            timestamp_s=12.5,
            frame_idx=7,
            detections=[_detection(activity="walking", confidence=0.87)],
        )

        # One line: no embedded newline (JSONL invariant).
        assert "\n" not in line

        obj = json.loads(line)
        assert obj["timestamp_s"] == 12.5
        assert obj["frame_idx"] == 7
        assert obj["person_count"] == 1

        person = obj["persons"][0]
        assert person["activity"] == "walking"
        assert person["confidence"] == 0.87
        assert person["bbox"] == [100.0, 200.0, 300.0, 600.0]
        assert len(person["keypoints"]) == 17
        assert set(person["keypoints"][0]) == {"x", "y", "vis"}
