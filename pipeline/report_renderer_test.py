"""Tests for the standalone HTML report renderer.

Behavioural tests only — they verify the rendered string contains the right
sections, embeds Chart.js inline (no external script src), and base64-encodes
keyframe images so the report can be opened with zero network access.
"""

from __future__ import annotations

import base64
import re

import cv2
import numpy as np

from pipeline.aggregator import Keyframe, ReportData, TimelineBin
from pipeline.annotator import annotate_frame
from pipeline.postprocessing import Detection, Keypoint
from pipeline.report_renderer import JPEG_QUALITY, render_report


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

    def test_keyframes_are_inlined_as_base64_jpeg(self):
        data = _make_report_data()

        html = render_report(data)

        # No external <img src="http..."> for keyframes
        assert not re.search(r'<img[^>]*src\s*=\s*["\']https?://', html, re.IGNORECASE)
        # Each keyframe encoded as a JPEG data: URL (issue #65 — PNG on
        # photographic surveillance frames is ~an order of magnitude bigger).
        assert html.count("data:image/jpeg;base64,") >= len(data.keyframes)
        # And no PNG data URLs linger (template + encoder must agree).
        assert "data:image/png" not in html

    def test_keyframe_base64_decodes_to_a_jpeg(self):
        html = render_report(_make_report_data())

        match = re.search(r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)", html)
        assert match is not None
        jpeg_bytes = base64.b64decode(match.group(1))
        # JPEG SOI magic header (FF D8 FF).
        assert jpeg_bytes[:3] == b"\xff\xd8\xff"

    def test_jpeg_quality_constant_is_module_level(self):
        """The quality lives in one reviewable module-level constant (issue
        #65), not buried as a magic number in the encoder call."""
        assert JPEG_QUALITY == 85


def _noisy_1080p_frame(rng) -> np.ndarray:
    """A deterministic, photographic-style noisy 1080p frame.

    A low-resolution random field upscaled to 1080p gives smooth gradients
    like a real scene; additive noise keeps it non-trivial (a solid colour
    would let PNG cheat and defeat the point). This compresses like real
    footage — calibration measured ~7.6x PNG/JPEG on frames of this kind,
    matching the ~6.6x seen on the real ground-truth clip.
    """
    low = rng.integers(0, 256, size=(36, 64, 3), dtype=np.uint8)
    base = cv2.resize(low, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    noise = rng.normal(0, 12, size=(1080, 1920, 3))
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class TestReportSizeAndLegibility:
    """Numeric size AC (issue #65) + a q=85 overlay-legibility spot-check."""

    def test_report_size_bounded(self):
        """A report with 8 photographic 1080p keyframes must stay under an
        explicit ceiling. Bound derived from the calibration: JPEG q=85 on
        this synthetic footage is ~6.4 MiB for 8 frames (PNG would be ~49
        MiB), so 10 MiB clears JPEG with margin while a PNG regression — the
        exact bug this issue fixes — blows straight past it.
        """
        rng = np.random.default_rng(1234)
        keyframes = []
        for i in range(8):
            frame = _noisy_1080p_frame(rng)
            det = _det("walking")
            keyframes.append(
                Keyframe(
                    timestamp_s=float(i * 10),
                    person_count=1,
                    frame=frame,
                    detections=[det],
                )
            )
        data = ReportData(
            video_duration_s=80.0,
            total_frames=80,
            peak_persons=1,
            avg_persons=1.0,
            dominant_activity="walking",
            person_minutes={"sitting": 0.0, "standing": 0.0, "walking": 1.3, "running": 0.0},
            timeline=[TimelineBin(minute=0, walking=60)],
            keyframes=keyframes,
        )

        html = render_report(data)

        max_bytes = 10 * 1024 * 1024  # 10 MiB ceiling (see docstring)
        size = len(html.encode("utf-8"))
        assert size < max_bytes, (
            f"report is {size / 1024 / 1024:.1f} MiB for 8x 1080p keyframes — "
            f"over the {max_bytes / 1024 / 1024:.0f} MiB ceiling. Are keyframes "
            f"PNG again? (calibration: JPEG ~6.4 MiB vs PNG ~49 MiB.)"
        )

    def test_skeleton_overlays_remain_legible_at_q85(self):
        """Spot-check the visual AC automatically: the coloured bbox/skeleton
        overlay must survive q=85 encoding. Over the drawn overlay pixels the
        'walking' orange signature (high R, low B) must persist strongly —
        a washed-out overlay would collapse this gap toward the background's
        ~0. Measured ~192 at q=85; assert a conservative floor.
        """
        frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)  # smooth gray
        kps = [Keypoint(x=900 + (i % 5) * 30, y=300 + (i // 5) * 90, vis=0.95) for i in range(17)]
        det = Detection(bbox=[850.0, 250.0, 1100.0, 800.0], confidence=0.9, keypoints=kps)
        det.activity = "walking"  # BGR (0, 165, 255) → R high, B low

        annotated = annotate_frame(frame, [det])
        overlay_mask = np.any(annotated != frame, axis=2)
        assert overlay_mask.sum() > 0

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        assert ok
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        b = dec[..., 0][overlay_mask].astype(float).mean()
        r = dec[..., 2][overlay_mask].astype(float).mean()
        assert r - b > 100, (
            f"orange overlay washed out at q={JPEG_QUALITY}: R-B gap {r - b:.0f} "
            f"(background ~0) — skeleton/bbox no longer legible."
        )
