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

from pipeline.aggregator import ACTIVITIES, Keyframe, ReportData, ShiftSummary, ZoneReport
from pipeline.annotator import annotate_frame
from pipeline.conversation import ZoneConversation
from pipeline.presence import Absence, Interval, ZonePresence

# 6 (issue #34): top-level classifier/model diagnostics. 5 (issue #81): each
# zone carries a ``conversation`` block alongside
# ``presence``, completing the work / conversation / absent mode set. 4 (issue
# #80) added the anchored-worker ``presence`` block; 3 (issue #79) added the
# ``shift`` gating summary; 2 (issue #78) added the per-zone ``zones[]`` section;
# 1 was the original posture-only contract (issue #72). Bump whenever the
# top-level shape changes.
SCHEMA_VERSION = 6


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


def _interval_to_dict(iv: Interval) -> dict:
    return {"start_s": iv.start_s, "end_s": iv.end_s, "duration_s": iv.duration_s}


def _absence_to_dict(a: Absence) -> dict:
    return {
        "start_s": a.start_s,
        "end_s": a.end_s,
        "duration_s": a.duration_s,
        "flagged": a.flagged,
    }


def _presence_to_dict(presence: ZonePresence | None) -> dict | None:
    # ``null`` when no anchored-worker analysis ran for this zone (no zone
    # config, or tracking disabled). Otherwise the anchor id, total present /
    # absent / work seconds, and the interval lists — absences carry a
    # ``flagged`` bool for those past the zone's ``flag_after_s`` (issue #80).
    if presence is None:
        return None
    return {
        "anchored_track_id": presence.anchored_track_id,
        "present_s": float(presence.present_s),
        "absent_s": float(presence.absent_s),
        "work_s": float(presence.work_s),
        "presence_intervals": [_interval_to_dict(iv) for iv in presence.presence_intervals],
        "absence_intervals": [_absence_to_dict(a) for a in presence.absence_intervals],
        "work_intervals": [_interval_to_dict(iv) for iv in presence.work_intervals],
    }


def _conversation_to_dict(conversation: ZoneConversation | None) -> dict | None:
    # ``null`` when no conversation analysis ran for this zone (no zone config,
    # or tracking disabled). Otherwise the total conversing seconds plus the
    # interval list — two idle, proximate tracks standing together (issue #81).
    if conversation is None:
        return None
    return {
        "conversation_s": float(conversation.conversation_s),
        "intervals": [_interval_to_dict(iv) for iv in conversation.intervals],
    }


def _zone_to_dict(zone: ZoneReport) -> dict:
    # Emit all four buckets so the platform never branches on a missing key —
    # an activity that never occurred in this zone is 0.0, not absent.
    return {
        "zone_id": zone.zone_id,
        "name": zone.name,
        "person_minutes": {a: float(zone.person_minutes.get(a, 0.0)) for a in ACTIVITIES},
        "presence": _presence_to_dict(zone.presence),
        "conversation": _conversation_to_dict(zone.conversation),
    }


def _shift_to_dict(shift: ShiftSummary | None) -> dict | None:
    # ``null`` when no shift schedule gated the run; otherwise the analysed
    # windows/breaks as [start, end] pairs plus the total excluded footage.
    if shift is None:
        return None
    return {
        "windows": [[start, end] for start, end in shift.windows],
        "breaks": [[start, end] for start, end in shift.breaks],
        "excluded_duration_s": float(shift.excluded_duration_s),
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
        # Per-zone posture breakdown (issue #78); [] when no zones config ran.
        "zones": [_zone_to_dict(z) for z in data.zones],
        # Shift-window gating summary (issue #79); null when no shift gated it.
        "shift": _shift_to_dict(data.shift),
        "diagnostics": data.diagnostics,
    }


def render_report_json(data: ReportData) -> bytes:
    """Render ``data`` into canonical ``result.json`` bytes (UTF-8)."""
    return json.dumps(report_data_to_dict(data)).encode("utf-8")
