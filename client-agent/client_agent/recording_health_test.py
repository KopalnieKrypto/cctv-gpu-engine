"""Recording-health rollup for the heartbeat ``status`` dict (issue #93)."""

from dataclasses import dataclass

import pytest

from client_agent.recording_health import classify_recorder_failure, recording_status


@dataclass
class FakeHandle:
    """Stand-in for a recorder handle as ``active_recorders`` holds it.

    Mirrors the parts of ``BackgroundRecorder`` the rollup actually reads:
    ``status()`` (state + ffmpeg message) and ``is_running()``."""

    state: str = "recording"
    message: str = ""
    running: bool = True

    def status(self):
        @dataclass
        class _Snapshot:
            state: str
            message: str

        return _Snapshot(state=self.state, message=self.message)

    def is_running(self) -> bool:
        return self.running


def test_healthy_box_reports_recording() -> None:
    """A box whose recorders are all producing footage says so on every beat.

    Healthy beats are not optional: ``deriveApplianceLastError`` treats a
    missing ``recordingStatus`` as "no evidence" and leaves the stored value
    alone, so a healthy beat is the *only* thing that ever clears a resolved
    fault. Emitting ``status`` solely when broken would pin the first-ever
    failure to the appliance page forever."""
    assert recording_status({"cam-1": FakeHandle()}) == {
        "recordingStatus": "recording",
        "reason": None,
    }


def test_appliance_with_no_approved_cameras_is_idle_not_failed() -> None:
    """A box nobody has approved a camera on is idle, not broken.

    Reporting ``recording_failed`` here would light the fault banner on every
    freshly-installed appliance before the operator has enabled anything. The
    slug still has to be *something* rather than an omitted key: the platform
    reads any value other than ``recording_failed`` as "healthy, clear the
    error", so ``idle`` truthfully clears a fault left over from before the
    last camera was disabled."""
    assert recording_status({}) == {"recordingStatus": "idle", "reason": None}


def test_failed_recorder_reports_the_slug_for_its_ffmpeg_message() -> None:
    """A dead recorder surfaces *why* it died, as a slug.

    ``Recorder._run`` already parked the RTSP error in ``message``; until now
    it went to a ``logger.warning`` and died in journald on a box nobody
    watches."""
    handle = FakeHandle(
        state="failed",
        message="method DESCRIBE failed: 401 Unauthorized",
        running=False,
    )
    assert recording_status({"cam-1": handle}) == {
        "recordingStatus": "recording_failed",
        "reason": "rtsp_auth",
    }


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        # Real ffmpeg/RTSP stderr shapes, one per documented slug.
        ("av_interleaved_write_frame(): No space left on device", "disk_full"),
        ("method DESCRIBE failed: 401 Unauthorized", "rtsp_auth"),
        (
            "Connection to tcp://192.168.88.31:554?timeout=0 failed: Connection timed out",
            "rtsp_timeout",
        ),
        (
            "Connection to tcp://192.168.88.31:554 failed: Connection refused",
            "rtsp_unreachable",
        ),
        ("Connection to tcp://192.168.88.31:554 failed: No route to host", "rtsp_unreachable"),
        # Not ffmpeg's — ours, synthesised by ``Recorder._run`` when the run
        # produced no non-empty chunk (e.g. a codec the MP4 muxer rejects).
        ("ffmpeg exited 1 with no output", "ffmpeg_no_output"),
    ],
)
def test_classifier_maps_real_stderr_onto_the_documented_vocabulary(stderr, expected) -> None:
    """The wire vocabulary is a closed set of slugs, not prose.

    Raw stderr is unbounded, uncomparable across boxes, and PII-adjacent —
    RTSP URLs carry credentials. The box reports the observation, the platform
    decides what it means, so the vocabulary can be retuned backend-side
    without a fleet redeploy."""
    assert classify_recorder_failure(stderr) == expected


def test_unclassified_failure_never_leaks_raw_stderr() -> None:
    """An unrecognised failure reports ``reason: None``, never the message.

    ffmpeg stderr routinely echoes the input URL, and an appliance RTSP URL
    carries ``user:password@``. Passing the message through would ship camera
    credentials to the platform and store them in ``appliances.last_error``,
    where they would render on the admin page. ``None`` is not a loss: the
    platform substitutes its own documented ``"recording_failed"``."""
    secret = "rtsp://admin:hunter2@192.168.88.31:554/unicast/c1/s0/live"
    message = f"Could not read from {secret}: Invalid data found when processing input"

    assert classify_recorder_failure(message) is None

    status = recording_status({"cam-1": FakeHandle(state="failed", message=message)})
    assert status == {"recordingStatus": "recording_failed", "reason": None}
    # Belt and braces: no fragment of the message survives anywhere in the
    # emitted dict, however the shape changes later.
    for token in ("admin", "hunter2", "192.168.88.31", "rtsp://"):
        assert token not in repr(status)


def test_multi_camera_failure_rolls_up_to_the_highest_severity_reason() -> None:
    """``last_error`` is one string on the appliance row; failures are
    per-camera. Ties break by severity, not by dict order.

    Severity wins over recency because ``disk_full`` is a property of the box
    — it explains every other camera's failure too — whereas one camera's
    ``rtsp_auth`` explains only itself. It is also stable: the same failing
    set yields the same slug on every beat, so the admin page does not flap
    between two cameras' causes every 30 s."""
    status = recording_status(
        {
            # Insertion order deliberately puts the *least* severe first, so
            # a first-match-wins implementation fails this test.
            "cam-a": FakeHandle(state="failed", message="method DESCRIBE failed: 401 Unauthorized"),
            "cam-b": FakeHandle(state="failed", message="Connection refused"),
            "cam-c": FakeHandle(state="failed", message="No space left on device"),
        }
    )
    assert status == {"recordingStatus": "recording_failed", "reason": "disk_full"}


def test_classified_reason_outranks_an_unclassifiable_one() -> None:
    """A cause we can name beats one we cannot.

    Otherwise a single unrecognised stderr on one camera would mask a
    perfectly legible ``rtsp_auth`` on another and the operator would be sent
    back to SSHing the box for journald."""
    status = recording_status(
        {
            "cam-a": FakeHandle(state="failed", message="something we have never seen"),
            "cam-b": FakeHandle(state="failed", message="method DESCRIBE failed: 401 Unauthorized"),
        }
    )
    assert status == {"recordingStatus": "recording_failed", "reason": "rtsp_auth"}


def test_finished_recording_awaiting_respawn_is_not_a_failure() -> None:
    """A recorder that reached its ``-t duration_s`` and exited cleanly is
    normal churn, not a fault.

    ``Recorder._run`` flips back to ``idle`` after writing its chunks and the
    thread ends; ``reconcile_recorders`` respawns it on the next beat. Keying
    the fault off ``is_running()`` alone would therefore paint every appliance
    ``recording_failed`` once per recording window."""
    finished = FakeHandle(state="idle", running=False)
    assert recording_status({"cam-1": finished}) == {
        "recordingStatus": "recording",
        "reason": None,
    }


def test_thread_that_died_while_claiming_to_record_is_a_failure() -> None:
    """A dead thread whose state machine still says ``recording`` is broken.

    ``BackgroundRecorder._target`` catches an exception escaping ``_run``,
    logs it and returns — leaving ``state="recording"`` frozen on a thread
    that is gone (the Wi-Fi-blip incident behind #66). ``state`` alone cannot
    see this; liveness alone cannot see the clean-exit case above. Both
    checks together are what make the signal honest."""
    zombie = FakeHandle(state="recording", running=False)
    assert recording_status({"cam-1": zombie}) == {
        "recordingStatus": "recording_failed",
        "reason": None,
    }


def test_unreadable_handle_does_not_break_the_heartbeat() -> None:
    """An opaque or throwing handle degrades; it never raises into the beat.

    Same guard ``_describe_recorder_death`` already applies. Telemetry is an
    addition to the heartbeat, never a precondition for it — a handle we
    cannot interrogate must not cost the box its config, its runtime settings
    and its whole platform session. Unreadable is treated as healthy: we have
    no evidence of failure, and inventing one would light the fault banner on
    a working box."""

    class NoStatus:
        pass

    class Throws:
        def status(self):
            raise RuntimeError("status() exploded")

    healthy = recording_status({"cam-1": NoStatus(), "cam-2": Throws()})
    assert healthy == {"recordingStatus": "recording", "reason": None}

    # A genuinely failed camera alongside them is still reported.
    mixed = recording_status(
        {
            "cam-1": Throws(),
            "cam-2": FakeHandle(state="failed", message="No space left on device"),
        }
    )
    assert mixed == {"recordingStatus": "recording_failed", "reason": "disk_full"}


def test_status_snapshot_missing_its_fields_is_tolerated() -> None:
    """A snapshot lacking ``state``/``message`` degrades rather than raising.

    ``_describe_recorder_death`` already reads both through ``getattr``
    defaults; this path has to agree, or the same handle that merely produces
    a vague log line over there takes the whole heartbeat down over here."""

    class Bare:
        def status(self):
            return object()

    assert recording_status({"cam-1": Bare()}) == {"recordingStatus": "recording", "reason": None}

    class FailedNoMessage:
        def status(self):
            from types import SimpleNamespace

            return SimpleNamespace(state="failed")

    assert recording_status({"cam-1": FailedNoMessage()}) == {
        "recordingStatus": "recording_failed",
        "reason": None,
    }
