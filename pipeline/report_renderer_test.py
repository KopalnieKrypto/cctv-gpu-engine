"""Tests for the standalone HTML report renderer.

Behavioural tests only — they verify the rendered string contains the right
sections, embeds Chart.js inline (no external script src), and base64-encodes
keyframe images so the report can be opened with zero network access.
"""

from __future__ import annotations

import base64
import re

import numpy as np

from pipeline.aggregator import Keyframe, ReportData, TimelineBin
from pipeline.postprocessing import Detection, Keypoint
from pipeline.report_renderer import render_report


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
    )


class TestRenderReport:
    def test_returns_html_with_doctype_and_summary_numbers(self):
        html = render_report(_make_report_data())

        assert html.startswith("<!DOCTYPE html>") or "<!doctype html>" in html.lower()
        # Summary fields surfaced somewhere in the body
        assert "125" in html  # video_duration_s / total_frames
        assert "walking" in html.lower()  # dominant activity label
        assert "4" in html  # peak persons

    def test_embeds_chartjs_inline_with_no_external_script_src(self):
        html = render_report(_make_report_data())

        # No <script src=...> at all (no network requests)
        assert not re.search(r"<script[^>]*\bsrc\s*=", html, re.IGNORECASE)
        # Chart.js content is present (use a recognizable token from the file)
        assert "Chart.js" in html or "chart.js" in html

    def test_keyframes_are_inlined_as_base64_png(self):
        data = _make_report_data()

        html = render_report(data)

        # No external <img src="http..."> for keyframes
        assert not re.search(r'<img[^>]*src\s*=\s*["\']https?://', html, re.IGNORECASE)
        # Each keyframe encoded as a data: URL
        assert html.count("data:image/png;base64,") >= len(data.keyframes)

    def test_keyframe_base64_decodes_to_a_png(self):
        html = render_report(_make_report_data())

        match = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
        assert match is not None
        png_bytes = base64.b64decode(match.group(1))
        # PNG magic header
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
