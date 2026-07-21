"""Recording health for the heartbeat ``status`` dict (issue #93).

The box knows whether it is actually recording — ``Recorder._run`` stashes the
ffmpeg/RTSP failure in ``status().message`` — but every heartbeat sent a
hardcoded ``status={}``, so ``appliances.last_error`` was permanently NULL and
the fault banner the platform already ships had never once rendered.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

# The wire vocabulary. Each entry is (slug, stderr fragments that imply it),
# matched case-insensitively in order — so a message carrying several
# fragments resolves to the earliest, most specific cause.
#
#   disk_full         the buffer volume is full. Named explicitly by DD-09
#                     §7.6:686 and box-wide rather than per-camera.
#   rtsp_auth         camera rejected our credentials.
#   rtsp_timeout      camera addressable but not answering in time.
#   rtsp_unreachable  nothing listening / no path to the host — the everyday
#                     unplugged-camera and changed-IP case on a camera LAN.
#   ffmpeg_no_output  ffmpeg ran and exited without producing a usable chunk
#                     (e.g. a codec the MP4 muxer rejects). Synthesised by
#                     ``Recorder._run``, not emitted by ffmpeg itself.
#
# Anything unmatched yields ``None``: the platform then stores its own
# documented ``"recording_failed"`` fallback rather than us inventing a
# parallel catch-all slug. Adding a slug here is safe without a platform
# change — ``deriveApplianceLastError`` passes ``reason`` through verbatim.
_FAILURE_SLUGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("disk_full", ("no space left on device",)),
    ("rtsp_auth", ("401 unauthorized", "403 forbidden")),
    ("rtsp_timeout", ("connection timed out", "timed out")),
    ("rtsp_unreachable", ("connection refused", "no route to host", "network is unreachable")),
    ("ffmpeg_no_output", ("with no output",)),
)


def classify_recorder_failure(message: str) -> str | None:
    """Map a recorder's raw ffmpeg/RTSP stderr onto a stable slug.

    Returns ``None`` when nothing matches. The raw message is only ever
    *read* here — it never reaches the return value, so RTSP credentials
    embedded in a URL cannot ride the wire to the platform."""
    lowered = message.lower()
    for slug, fragments in _FAILURE_SLUGS:
        if any(fragment in lowered for fragment in fragments):
            return slug
    return None


def _read_status(cam_id: str, handle: Any) -> Any | None:
    """Read a handle's status snapshot, or ``None`` if it cannot be read.

    Mirrors the guards ``_describe_recorder_death`` already applies. An
    unreadable handle counts as *no evidence of failure*, not as a failure:
    inventing one would light the platform's fault banner on a working box.
    Either way it must never raise — telemetry is an addition to the beat, and
    a throwing handle must not cost the appliance its whole platform session."""
    status = getattr(handle, "status", None)
    if not callable(status):
        return None
    try:
        return status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("recording_status: camera_id=%s status() raised: %s", cam_id, exc)
        return None


def _has_failed(handle: Any, snapshot: Any) -> bool:
    """Whether this recorder is genuinely broken, as opposed to between runs.

    Two distinct shapes, and each is invisible to the other's check:

    * ``state="failed"`` — ``Recorder._run`` walked its output dir, found no
      non-empty chunk, and parked the stderr in ``message``.
    * ``state="recording"`` on a thread that is gone —
      ``BackgroundRecorder._target`` swallowed an exception out of ``_run``,
      so the state machine is frozen mid-recording (the #66 shape).

    Deliberately *not* a failure: ``state="idle"`` with a dead thread. That is
    a recorder that reached its ``-t duration_s``, wrote its chunks and exited
    for ``reconcile_recorders`` to respawn — the steady-state churn of every
    healthy appliance.

    Handles with no ``is_running`` are treated as alive, matching
    ``reconcile_recorders``, so opaque test fakes behave consistently in both."""
    state = getattr(snapshot, "state", "")
    if state == "failed":
        return True
    is_running = getattr(handle, "is_running", None)
    return state == "recording" and callable(is_running) and not is_running()


def _rollup(reasons: list[str | None]) -> str | None:
    """Collapse per-camera reasons into the single slug ``last_error`` holds.

    Policy: **highest severity wins**, ranked by ``_FAILURE_SLUGS`` order.
    ``disk_full`` leads because it is a property of the box rather than of one
    camera — it explains every other camera's failure too — and an
    unclassifiable reason always loses to one we can name, so a single novel
    stderr cannot mask a legible cause on another camera.

    Severity beats recency (the alternative the issue weighed) on two counts:
    ``RecorderStatus`` carries no failure timestamp, so recency would need new
    state; and severity is *stable* — the same failing set yields the same slug
    every beat instead of flapping between cameras on the admin page."""
    ranked = [slug for slug, _ in _FAILURE_SLUGS if slug in reasons]
    return ranked[0] if ranked else None


def recording_status(active_recorders: Mapping[str, Any]) -> dict[str, Any]:
    """Roll the live recorder set up into the heartbeat's ``status`` dict.

    Emits one of three shapes::

        {"recordingStatus": "idle",             "reason": None}
        {"recordingStatus": "recording",        "reason": None}
        {"recordingStatus": "recording_failed", "reason": <slug or None>}

    **Send this on every beat, healthy ones included.** The platform's
    ``deriveApplianceLastError`` is tri-state: a missing ``recordingStatus``
    means "no evidence" and leaves ``appliances.last_error`` untouched, so a
    beat can never erase a real fault. The flip side is that a healthy beat is
    the *only* thing that clears one — emit ``status`` only when broken and an
    appliance's first-ever failure stays pinned to its admin page forever.

    ``idle`` (no approved cameras) is deliberately distinct from ``recording``
    but lands on the same platform branch: anything other than
    ``recording_failed`` clears the error. A freshly-installed box with
    nothing enabled is not broken.

    Called with the handles as they stand *before* the beat's
    ``reconcile_recorders``, which is the right vantage point — a recorder
    that crashed since the last beat is still in ``active_recorders`` with its
    failure intact, and reports healthy again only once a respawn has taken.
    """
    if not active_recorders:
        return {"recordingStatus": "idle", "reason": None}

    reasons: list[str | None] = []
    for cam_id, handle in active_recorders.items():
        snapshot = _read_status(cam_id, handle)
        if snapshot is not None and _has_failed(handle, snapshot):
            reasons.append(classify_recorder_failure(getattr(snapshot, "message", "") or ""))

    if not reasons:
        return {"recordingStatus": "recording", "reason": None}
    return {"recordingStatus": "recording_failed", "reason": _rollup(reasons)}
