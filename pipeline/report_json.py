"""Canonical structured JSON report artifact (issue #72).

Serializes a :class:`pipeline.aggregator.ReportData` into the platform's
``result.json`` contract (``schema_version`` 1). Presentation — branding,
i18n, layout — is the platform's job (native React report), so this artifact
carries *only* data plus base64-JPEG keyframes. No brand/footer/lang strings.
"""

from __future__ import annotations

import base64
import json

import cv2

from pipeline.aggregator import ACTIVITIES, Keyframe, ReportData
from pipeline.annotator import annotate_frame

SCHEMA_VERSION = 1


def _encode_keyframe_to_base64_jpeg(frame_bgr) -> str:
    """Encode an (annotated) BGR frame as base64 JPEG (issue #65 — not PNG)."""
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed for keyframe JPEG encoding")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _keyframe_to_dict(kf: Keyframe) -> dict:
    # Bake the detection overlay (bbox + skeleton + activity label) into the
    # JPEG: the contract carries no bbox/keypoints, so this is the only way
    # the platform can show overlays.
    annotated = annotate_frame(kf.frame, kf.detections)
    # De-duplicate activities while preserving first-seen order — a
    # multi-person frame may show several ("sitting" + "walking").
    seen: set[str] = set()
    activities: list[str] = []
    for d in kf.detections:
        if d.activity and d.activity not in seen:
            seen.add(d.activity)
            activities.append(d.activity)
    return {
        "timestamp_s": kf.timestamp_s,
        "person_count": kf.person_count,
        "activities": activities,
        "image_b64_jpeg": _encode_keyframe_to_base64_jpeg(annotated),
    }


def report_data_to_dict(data: ReportData) -> dict:
    """Serialize ``data`` into the canonical ``result.json`` dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "video_duration_s": data.video_duration_s,
        "total_frames": data.total_frames,
        "peak_persons": data.peak_persons,
        "avg_persons": data.avg_persons,
        "dominant_activity": data.dominant_activity,
        # Always emit all four buckets so the platform never branches on
        # missing keys — an activity that never occurred is 0.0, not absent.
        "person_minutes": {a: float(data.person_minutes.get(a, 0.0)) for a in ACTIVITIES},
        "timeline": [
            {
                "minute": b.minute,
                "sitting": b.sitting,
                "standing": b.standing,
                "walking": b.walking,
                "running": b.running,
            }
            for b in data.timeline
        ],
        "keyframes": [_keyframe_to_dict(kf) for kf in data.keyframes],
    }


def render_report_json(data: ReportData) -> bytes:
    """Render ``data`` into canonical ``result.json`` bytes (UTF-8)."""
    return json.dumps(report_data_to_dict(data)).encode("utf-8")
