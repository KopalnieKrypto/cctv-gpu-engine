"""Tests for the canonical structured JSON report artifact (issue #72).

The engine emits ``result.json`` (schema_version 1) and the platform renders
it natively in React. These tests pin the contract: exact top-level keys and
types, the four activity buckets, per-minute timeline shape, base64-JPEG
keyframes, and the absence of any brand / English presentation strings.
"""

from __future__ import annotations

import base64
import json

import numpy as np

from pipeline.aggregator import Keyframe, ReportData, TimelineBin, ZoneReport
from pipeline.postprocessing import Detection, Keypoint
from pipeline.report_json import render_report_json


def _solid_frame(color: tuple[int, int, int] = (10, 20, 30)) -> np.ndarray:
    return np.full((40, 60, 3), color, dtype=np.uint8)


def _det(activity: str = "standing") -> Detection:
    det = Detection(
        bbox=[5.0, 5.0, 25.0, 35.0],
        confidence=0.9,
        keypoints=[Keypoint(10.0 + i, 10.0 + i, 0.9) for i in range(17)],
    )
    det.activity = activity
    return det


def _make_report_data() -> ReportData:
    return ReportData(
        video_duration_s=125.0,
        total_frames=125,
        peak_persons=4,
        avg_persons=2.5,
        dominant_activity="walking",
        person_minutes={"sitting": 0.5, "standing": 1.0, "walking": 2.5, "running": 0.0},
        timeline=[
            TimelineBin(minute=0, walking=60, standing=10),
            TimelineBin(minute=1, walking=30, sitting=20),
        ],
        keyframes=[
            Keyframe(
                timestamp_s=10.0,
                person_count=4,
                frame=_solid_frame((10, 20, 30)),
                detections=[_det("walking")],
            ),
        ],
        zones=[
            ZoneReport(
                zone_id="bending-1",
                name="Giętarka 1",
                person_minutes={"sitting": 3.0, "standing": 0.5, "walking": 0.0, "running": 0.0},
            ),
        ],
    )


class TestRenderReportJson:
    def test_emits_json_bytes_with_schema_version_and_summary_fields(self):
        payload = json.loads(render_report_json(_make_report_data()))

        assert payload["schema_version"] == 2
        assert payload["video_duration_s"] == 125.0
        assert payload["total_frames"] == 125
        assert payload["peak_persons"] == 4
        assert payload["avg_persons"] == 2.5
        assert payload["dominant_activity"] == "walking"

    def test_person_minutes_has_all_four_activity_buckets_as_floats(self):
        payload = json.loads(render_report_json(_make_report_data()))

        pm = payload["person_minutes"]
        assert set(pm.keys()) == {"sitting", "standing", "walking", "running"}
        assert all(isinstance(v, float) for v in pm.values())
        assert pm["walking"] == 2.5
        assert pm["running"] == 0.0

    def test_timeline_is_per_minute_rows_with_minute_and_four_int_counts(self):
        payload = json.loads(render_report_json(_make_report_data()))

        timeline = payload["timeline"]
        assert len(timeline) == 2
        first = timeline[0]
        assert set(first.keys()) == {"minute", "sitting", "standing", "walking", "running"}
        assert first["minute"] == 0
        assert first["walking"] == 60
        assert first["standing"] == 10
        assert first["sitting"] == 0
        assert all(isinstance(first[a], int) for a in ("sitting", "standing", "walking", "running"))
        assert timeline[1]["minute"] == 1
        assert timeline[1]["sitting"] == 20

    def test_keyframe_carries_timestamp_count_activities_and_jpeg_field(self):
        payload = json.loads(render_report_json(_make_report_data()))

        assert len(payload["keyframes"]) == 1
        kf = payload["keyframes"][0]
        assert set(kf.keys()) == {"timestamp_s", "person_count", "activities", "image_b64_jpeg"}
        assert kf["timestamp_s"] == 10.0
        assert kf["person_count"] == 4
        assert kf["activities"] == ["walking"]
        assert isinstance(kf["image_b64_jpeg"], str) and kf["image_b64_jpeg"]

    def test_keyframe_image_b64_decodes_to_a_jpeg(self):
        payload = json.loads(render_report_json(_make_report_data()))

        raw = base64.b64decode(payload["keyframes"][0]["image_b64_jpeg"])
        # JPEG SOI marker (and *not* the PNG magic header we used to emit).
        assert raw[:3] == b"\xff\xd8\xff"
        assert raw[:8] != b"\x89PNG\r\n\x1a\n"

    def test_multi_person_keyframe_dedups_activities_in_first_seen_order(self):
        data = _make_report_data()
        data.keyframes = [
            Keyframe(
                timestamp_s=5.0,
                person_count=3,
                frame=_solid_frame(),
                detections=[_det("walking"), _det("sitting"), _det("walking")],
            ),
        ]

        payload = json.loads(render_report_json(data))

        assert payload["keyframes"][0]["activities"] == ["walking", "sitting"]

    def test_zones_section_carries_per_zone_person_minutes(self):
        payload = json.loads(render_report_json(_make_report_data()))

        assert len(payload["zones"]) == 1
        zone = payload["zones"][0]
        assert set(zone.keys()) == {"zone_id", "name", "person_minutes"}
        assert zone["zone_id"] == "bending-1"
        assert zone["name"] == "Giętarka 1"
        assert set(zone["person_minutes"]) == {"sitting", "standing", "walking", "running"}
        assert zone["person_minutes"]["sitting"] == 3.0
        assert all(isinstance(v, float) for v in zone["person_minutes"].values())

    def test_zones_section_is_empty_list_when_no_zones_configured(self):
        data = _make_report_data()
        data.zones = []

        payload = json.loads(render_report_json(data))

        assert payload["zones"] == []

    def test_canonical_output_has_no_brand_or_presentation_strings(self):
        """Acceptance criterion #2 — presentation is the platform's job.

        The engine artifact must be pure data: no brand/footer/lang strings
        and no HTML. Guards against a regression that re-embeds the old
        Jinja report's ``Kopalnie Krypto`` / ``ML Compute Exchange`` chrome.
        """
        raw = render_report_json(_make_report_data())

        # It parses as JSON and nothing else (no HTML document leaked in).
        json.loads(raw)
        haystack = raw.decode("utf-8")
        for banned in ("Kopalnie", "ML Compute Exchange", "Generated by", "<html", "<!DOCTYPE"):
            assert banned.lower() not in haystack.lower(), f"leaked presentation string: {banned}"
