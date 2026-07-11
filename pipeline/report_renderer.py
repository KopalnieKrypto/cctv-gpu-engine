"""Standalone HTML report renderer.

Renders a :class:`pipeline.aggregator.ReportData` into a self-contained HTML
string using a Jinja2 template. The output:

* embeds the vendored Chart.js bundle inline (no ``<script src=>``),
* annotates each keyframe with bbox + skeleton + activity label,
* encodes annotated keyframes as base64 JPEG inside ``<img>`` data URLs,

so the resulting file works offline in any modern browser.

Keyframes are JPEG (issue #65), not PNG: PNG is lossless and on photographic
1080p surveillance frames it runs ~an order of magnitude larger than JPEG
q=85 (calibrated on real footage: ~6.6x, and ~7.6x on synthetic 1080p),
before base64's +33%. q=85 is the standard choice for camera frames and keeps
skeleton overlays legible; the canonical ``result.json`` artifact
(``report_json.py``) already encodes its keyframes the same way.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
from jinja2 import Environment, FileSystemLoader, select_autoescape

from pipeline.aggregator import ACTIVITIES, ReportData
from pipeline.annotator import annotate_frame

_TEMPLATE_DIR = Path(__file__).parent
_TEMPLATE_NAME = "report_template.html"
_VENDOR_CHARTJS = _TEMPLATE_DIR / "vendor" / "chart.umd.min.js"

# JPEG quality for base64-embedded keyframes (issue #65). 85 is the standard
# photographic sweet spot — near-lossless to the eye, keeps skeleton overlays
# legible, ~an order of magnitude smaller than lossless PNG. Bump with care:
# lower loses overlay legibility, higher balloons the report.
JPEG_QUALITY = 85


def _load_chartjs_source() -> str:
    if not _VENDOR_CHARTJS.exists():
        raise RuntimeError(
            f"Vendored Chart.js missing at {_VENDOR_CHARTJS}. "
            "Run setup-models.sh or restore the file from git."
        )
    return _VENDOR_CHARTJS.read_text(encoding="utf-8")


def _encode_keyframe_to_base64_jpeg(frame_bgr) -> str:
    annotated = frame_bgr  # caller pre-annotates
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed for keyframe JPEG encoding")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_report(data: ReportData) -> str:
    """Render ``data`` into a complete standalone HTML document string."""
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template(_TEMPLATE_NAME)

    # Annotate keyframes once at render time so the aggregator can stay
    # GPU-independent (annotation pulls in OpenCV draw calls).
    keyframes_b64 = []
    for kf in data.keyframes:
        annotated = annotate_frame(kf.frame, kf.detections)
        # De-duplicate while preserving order — single-person frames have
        # one activity, multi-person frames may show multiple ("sitting +
        # walking") which the meta line surfaces verbatim.
        seen: set[str] = set()
        activities = []
        for d in kf.detections:
            if d.activity and d.activity not in seen:
                seen.add(d.activity)
                activities.append(d.activity)
        keyframes_b64.append(
            {
                "timestamp_s": kf.timestamp_s,
                "person_count": kf.person_count,
                "activities": activities,
                "b64": _encode_keyframe_to_base64_jpeg(annotated),
            }
        )

    timeline_labels = [f"{b.minute:d}m" for b in data.timeline]
    timeline_data = {
        activity: [getattr(b, activity) for b in data.timeline] for activity in ACTIVITIES
    }
    activity_labels = list(ACTIVITIES)
    activity_minutes = [data.person_minutes.get(a, 0.0) for a in ACTIVITIES]

    return template.render(
        data=data,
        chartjs_source=_load_chartjs_source(),
        keyframes_b64=keyframes_b64,
        activity_labels=activity_labels,
        activity_minutes=activity_minutes,
        timeline_labels=timeline_labels,
        timeline_data=timeline_data,
    )
