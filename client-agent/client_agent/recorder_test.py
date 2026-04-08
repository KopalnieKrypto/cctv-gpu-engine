"""Tests for the client-agent RTSP recorder (issue #8).

The recorder owns three concerns: (a) probing an RTSP URL to give the
operator fast yes/no feedback, (b) building the right ffmpeg command for
short vs segmented (long) recordings, and (c) running one recording at a
time and uploading the resulting chunks to R2 in the same handshake the
MP4-upload path uses.

Mocks live only at system boundaries: ffmpeg (subprocess) and R2. The
``runner`` callable mirrors ``subprocess.run``'s contract just enough that
production code can pass ``subprocess.run`` directly with no shim.
"""

from __future__ import annotations

import subprocess
from typing import Any

from client_agent.recorder import (
    Recorder,
    RecorderBusy,
    build_ffmpeg_cmd,
    probe_rtsp,
)


def _ok_runner(returncode: int = 0, stderr: str = "") -> Any:
    """Build a fake ``subprocess.run``-shaped callable.

    The recorder only inspects ``returncode`` and ``stderr`` on the result —
    keeping the fake this small means the test contract reads as "what
    matters about a subprocess result" rather than "what subprocess.run
    happens to expose today".
    """

    def _run(cmd, **kwargs):  # noqa: ANN001 — mirrors subprocess.run
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stderr=stderr)

    return _run


# ----- 1. probe_rtsp success -----


def test_probe_rtsp_returns_ok_when_runner_exits_zero() -> None:
    """A successful ffmpeg probe (returncode 0) is the operator's
    "connection works" signal — the rest of the start flow depends on it."""
    result = probe_rtsp("rtsp://camera.local/stream", timeout=5, runner=_ok_runner())

    assert result.ok is True


# ----- 2. probe_rtsp failure carries stderr through to the operator -----


def test_probe_rtsp_returns_failure_with_stderr_message() -> None:
    """A non-zero exit means ffmpeg refused the URL. The operator needs
    the stderr line in the response so they can tell "wrong creds" from
    "wrong port" from "DNS doesn't resolve" — the alternative is a
    binary go/no-go that hides the actual error."""
    runner = _ok_runner(returncode=1, stderr="Connection refused")

    result = probe_rtsp("rtsp://nope.local/stream", timeout=5, runner=runner)

    assert result.ok is False
    assert "Connection refused" in result.message


# ----- 3. probe_rtsp timeout — no hung ffmpeg -----


def test_probe_rtsp_returns_timeout_when_runner_raises_timeout_expired() -> None:
    """Acceptance criterion (#8 testing focus): "Invalid RTSP URL → clear
    'connection failed', no hung ffmpeg". When the underlying ffmpeg blocks
    on a dead host, ``subprocess.run`` raises ``TimeoutExpired``. The
    probe must convert that into a ProbeResult instead of letting the
    exception escape into the Flask handler — otherwise the operator
    sees a 500 traceback instead of "connection failed"."""

    def _hanging_runner(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    result = probe_rtsp("rtsp://blackhole.local/stream", timeout=1, runner=_hanging_runner)

    assert result.ok is False
    assert "timeout" in result.message.lower()


# ----- 4. build_ffmpeg_cmd: 1h recording uses -t, no segmenting -----


def test_build_ffmpeg_cmd_short_recording_uses_t_flag() -> None:
    """A 1-hour recording fits in a single MP4. We use ``-t 3600`` and
    skip the segment muxer because it would needlessly fragment the
    upload into a one-element list."""
    cmd = build_ffmpeg_cmd(
        url="rtsp://camera.local/stream",
        duration_s=3600,
        output_dir="/tmp/rec",
    )

    assert "-t" in cmd
    assert "3600" in cmd
    assert "-f" not in cmd or "segment" not in cmd
    assert "-segment_time" not in cmd


# ----- 5. build_ffmpeg_cmd: 4h recording is segmented into 1h chunks -----


def test_build_ffmpeg_cmd_long_recording_uses_segment_muxer() -> None:
    """A 4-hour recording must be segmented into 1h chunks (SPEC §7.3) so
    each chunk is uploadable on a flaky LAN. ``-reset_timestamps 1`` keeps
    each chunk independently playable instead of starting at offset 3600s.
    The output template uses ``%03d`` so chunks sort lexically the same as
    chronologically — important for the upload step's ordering."""
    cmd = build_ffmpeg_cmd(
        url="rtsp://camera.local/stream",
        duration_s=4 * 3600,
        output_dir="/tmp/rec",
    )

    assert "-f" in cmd
    f_index = cmd.index("-f")
    assert cmd[f_index + 1] == "segment"
    assert "-segment_time" in cmd
    st_index = cmd.index("-segment_time")
    assert cmd[st_index + 1] == "3600"
    assert "-reset_timestamps" in cmd
    # Total duration still bounded so we don't accidentally record forever.
    assert "-t" in cmd
    assert str(4 * 3600) in cmd
    # Output template must contain a numeric placeholder so segments don't
    # all clobber the same filename.
    assert any("%" in arg and "d" in arg for arg in cmd)


# ----- 6. build_ffmpeg_cmd: common flags shared by both branches -----


def test_build_ffmpeg_cmd_always_uses_tcp_transport_and_stream_copy() -> None:
    """Stream copy is non-negotiable: re-encoding would push CPU usage on
    a customer LAN box (no GPU). TCP transport is the SPEC default
    because UDP packet loss on a busy LAN produces unplayable MP4s.
    Both flags must appear in *every* recording, short or long, and the
    output path must live under the requested ``output_dir`` so callers
    can sandbox each recording in its own temp dir."""
    for duration in (1800, 3600, 4 * 3600, 8 * 3600):
        cmd = build_ffmpeg_cmd(
            url="rtsp://camera.local/stream",
            duration_s=duration,
            output_dir="/tmp/rec-xyz",
        )
        assert "-rtsp_transport" in cmd
        t_index = cmd.index("-rtsp_transport")
        assert cmd[t_index + 1] == "tcp"
        assert "-c" in cmd
        c_index = cmd.index("-c")
        assert cmd[c_index + 1] == "copy"
        # Output (last positional) sits under the requested directory.
        assert cmd[-1].startswith("/tmp/rec-xyz/")


# ===== Recorder class — single-recording state machine =====


class FakeR2ForRecorder:
    """Just enough of the client-agent R2 surface for the recorder.

    The recorder only needs ``upload_input_chunk`` (per-chunk upload) and
    ``put_status`` (the SPEC §6.2 handshake). Everything else stays out so
    the contract reads as "what does a recording need from R2".
    """

    def __init__(self) -> None:
        self.uploaded: list[tuple[str, bytes]] = []
        self.statuses: dict[str, dict[str, Any]] = {}

    def upload_input_chunk(self, job_id: str, fileobj: Any) -> str:
        # Drain the file-like the same way boto3's upload_fileobj would,
        # so the test can assert on the exact bytes that left the host.
        data = fileobj.read()
        self.uploaded.append((job_id, data))
        return f"surveillance-jobs/{job_id}/input/chunk_{len(self.uploaded):03d}.mp4"

    def put_status(self, job_id: str, status: dict[str, Any]) -> None:
        self.statuses[job_id] = status


def _make_runner_that_writes_chunks(
    chunks: dict[str, bytes],
    returncode: int = 0,
    stderr: str = "",
):
    """Build a runner that simulates ffmpeg by dropping files into the
    recording dir before "exiting". The runner inspects the cmd to find
    the output template (last positional) and resolves the dir from it,
    then materialises the requested chunk files there. This is the
    cleanest way to fake ffmpeg without conditionally branching on cmd
    shape — both the short and long branches end up with files on disk."""

    def _run(cmd, **kwargs):  # noqa: ANN001
        from pathlib import Path

        output_template = cmd[-1]
        out_dir = Path(output_template).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, data in chunks.items():
            (out_dir / name).write_bytes(data)
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stderr=stderr)

    return _run


# ----- 7. Recorder.start invokes runner with the right cmd and transitions state -----


def test_recorder_start_invokes_ffmpeg_and_uploads_chunks(tmp_path) -> None:  # noqa: ANN001
    """The happy path: ``start`` runs ffmpeg, then uploads every produced
    chunk via ``upload_input_chunk`` and writes a single ``status.json``
    with ``status: pending``. After the call returns, the recorder is
    back in ``idle`` so the operator can launch another recording.

    Uses an injected ``output_dir_factory`` so the test pins the temp dir
    under pytest's ``tmp_path`` instead of the real ``/tmp``."""
    fake_r2 = FakeR2ForRecorder()
    runner_calls: list[list[str]] = []

    def _capturing_runner(cmd, **kwargs):  # noqa: ANN001
        runner_calls.append(list(cmd))
        return _make_runner_that_writes_chunks({"recording.mp4": b"fake-mp4-bytes"})(cmd, **kwargs)

    rec = Recorder(
        uploader=fake_r2,
        runner=_capturing_runner,
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-rec-001",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=3600)

    # ffmpeg was invoked once with the cmd build_ffmpeg_cmd would produce.
    assert len(runner_calls) == 1
    assert runner_calls[0][0] == "ffmpeg"
    assert "rtsp://camera.local/stream" in runner_calls[0]
    # Chunk reached R2 with the right job_id and the bytes ffmpeg "wrote".
    assert fake_r2.uploaded == [("job-rec-001", b"fake-mp4-bytes")]
    # status.json handshake — pending so the worker picks it up.
    assert "job-rec-001" in fake_r2.statuses
    status = fake_r2.statuses["job-rec-001"]
    assert status["status"] == "pending"
    assert status["job_id"] == "job-rec-001"
    # Back to idle so the next /start works.
    assert rec.status().state == "idle"


# ----- 8. Recorder.start raises RecorderBusy when already running -----


def test_recorder_rejects_concurrent_start(tmp_path) -> None:  # noqa: ANN001
    """Acceptance criterion (#8): "second recording while one active →
    rejected". The recorder owns a single ffmpeg slot — letting two run
    in parallel would double-spend the LAN's RTSP allowance and corrupt
    both recordings."""
    fake_r2 = FakeR2ForRecorder()

    # A runner that pauses the "ffmpeg" call long enough for the second
    # start to race in. We do this by having the runner itself call back
    # into the recorder before "exiting" — easier than wiring threads.
    busy_observed: list[bool] = []

    def _runner(cmd, **kwargs):  # noqa: ANN001
        # While we're "inside ffmpeg", state should be 'recording' and
        # a second start must raise.
        try:
            rec.start(url="rtsp://other/stream", duration_s=3600)
            busy_observed.append(False)
        except RecorderBusy:
            busy_observed.append(True)
        # Then act like a normal successful ffmpeg run.
        return _make_runner_that_writes_chunks({"recording.mp4": b"x"})(cmd, **kwargs)

    rec = Recorder(
        uploader=fake_r2,
        runner=_runner,
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-busy-1",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=3600)

    assert busy_observed == [True], "second start should have raised RecorderBusy"
    # Only the first job's chunk reached R2.
    assert len(fake_r2.uploaded) == 1
    assert fake_r2.uploaded[0][0] == "job-busy-1"


# ----- 9. Recorder.start: invalid URL → no chunks → failed, no R2 calls -----


def test_recorder_marks_failed_when_ffmpeg_produces_no_chunks(tmp_path) -> None:  # noqa: ANN001
    """Acceptance criterion (#8): "Invalid RTSP URL → clear 'connection
    failed', no hung ffmpeg". When ffmpeg exits non-zero AND leaves zero
    files on disk, there's nothing to upload — and crucially we must
    *not* write a status.json, because that would create a phantom job
    the worker would later fail loudly. Better to fail at the recorder
    and surface the error in the UI."""
    fake_r2 = FakeR2ForRecorder()

    def _runner(cmd, **kwargs):  # noqa: ANN001
        # Don't write any chunk files — simulate "ffmpeg refused the URL".
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stderr="rtsp://nope: Connection refused"
        )

    rec = Recorder(
        uploader=fake_r2,
        runner=_runner,
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-doomed",
    )

    rec.start(url="rtsp://nope.local/stream", duration_s=3600)

    assert fake_r2.uploaded == []
    assert fake_r2.statuses == {}, "no status.json should leak for a failed recording"
    snapshot = rec.status()
    assert snapshot.state == "failed"
    assert "Connection refused" in snapshot.message


# ----- 10. Partial recording (camera offline mid-recording) is still uploaded -----


def test_recorder_uploads_partial_chunks_when_ffmpeg_fails_late(tmp_path) -> None:  # noqa: ANN001
    """Acceptance criterion (#8): "camera goes offline mid-recording →
    partial MP4 still valid, uploaded". ffmpeg's segment muxer flushes
    each finished chunk before EOF, so an interrupted 4h recording still
    leaves the first N chunks playable. We must upload those instead of
    discarding them — losing 2 hours of footage because hour 3 dropped
    is the worst possible failure mode for a surveillance system.

    Multiple chunks land in the recorder dir; we assert all were
    uploaded *in lexical order* (chunk_000 → chunk_001 → ...) and a
    single status.json was written."""
    fake_r2 = FakeR2ForRecorder()

    def _runner(cmd, **kwargs):  # noqa: ANN001
        # Simulate ffmpeg flushing two segments before crashing.
        return _make_runner_that_writes_chunks(
            {"chunk_000.mp4": b"hour-1", "chunk_001.mp4": b"hour-2"},
            returncode=1,
            stderr="RTSP/1.0 200 OK ... Input/output error",
        )(cmd, **kwargs)

    rec = Recorder(
        uploader=fake_r2,
        runner=_runner,
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-partial",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=4 * 3600)

    assert [job_id for job_id, _ in fake_r2.uploaded] == ["job-partial", "job-partial"]
    assert [data for _, data in fake_r2.uploaded] == [b"hour-1", b"hour-2"]
    assert fake_r2.statuses["job-partial"]["status"] == "pending"
    assert rec.status().state == "idle"


# ----- 11. Temp dir is cleaned up after a successful upload -----


def test_recorder_cleans_up_temp_dir_after_upload(tmp_path) -> None:  # noqa: ANN001
    """Acceptance criterion (#8): "8h recording ~32GB → verify temp
    cleanup after upload". The customer LAN box can't afford to leak
    32GB per recording — after the chunks are safely in R2, the local
    files must go."""
    from pathlib import Path

    fake_r2 = FakeR2ForRecorder()
    out_dir = tmp_path / "job-cleanup"

    def _runner(cmd, **kwargs):  # noqa: ANN001
        return _make_runner_that_writes_chunks({"recording.mp4": b"some bytes"})(cmd, **kwargs)

    rec = Recorder(
        uploader=fake_r2,
        runner=_runner,
        output_dir_factory=lambda job_id: str(out_dir),
        job_id_factory=lambda: "job-cleanup",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=3600)

    assert not Path(out_dir).exists(), "recorder must remove the temp dir after upload"
