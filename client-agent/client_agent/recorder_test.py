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
import threading
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
        self.keys: list[str] = []
        self.statuses: dict[str, dict[str, Any]] = {}

    def upload_input_chunk(
        self, job_id: str, fileobj: Any, chunk_name: str = "chunk_001.mp4"
    ) -> str:
        # Drain the file-like the same way boto3's upload_fileobj would,
        # so the test can assert on the exact bytes that left the host.
        data = fileobj.read()
        self.uploaded.append((job_id, data))
        # Mirror the production key convention exactly — the destination is
        # derived from ``chunk_name``, NOT fabricated from call order. The
        # old fake invented ``chunk_{len:03d}.mp4`` which masked issue #50
        # (every real chunk overwriting ``chunk_001.mp4`` in R2).
        key = f"surveillance-jobs/{job_id}/input/{chunk_name}"
        self.keys.append(key)
        return key

    def put_status(self, job_id: str, status: dict[str, Any]) -> None:
        self.statuses[job_id] = status


class FakePopen:
    """``subprocess.Popen``-shaped fake for the recorder's recording boundary.

    Issue #52 moved recording from a blocking ``subprocess.run`` to a
    ``subprocess.Popen`` handle the recorder keeps so ``stop`` can SIGTERM
    the ffmpeg child. This fake mirrors just the slice the recorder uses:
    ``communicate`` / ``wait`` / ``poll`` / ``terminate`` / ``kill`` /
    ``returncode``.

    Two lifecycles:

    * ``auto_exit=True`` (default) — ffmpeg "ran its course" (``-t`` elapsed)
      and exited on its own: ``communicate``/``wait`` return immediately.
      Used by the single-threaded happy-path tests.
    * ``auto_exit=False`` — a long capture that only ends when the recorder
      signals it: ``communicate``/``wait`` block on an Event that
      ``terminate``/``kill`` set. Lets a ``stop`` test drive the in-flight
      case from another thread.

    Chunk files are materialised at construction (ffmpeg opens its output
    while capturing), so both the upload path and the "partial chunks left
    on disk" path see files regardless of how the process ends.

    ``ignore_terminate=True`` models a wedged ffmpeg that shrugs off SIGTERM,
    forcing ``stop`` to escalate to ``kill()`` after the grace window.
    """

    def __init__(
        self,
        cmd,  # noqa: ANN001
        *,
        chunks: dict[str, bytes] | None = None,
        returncode: int = 0,
        stderr: str = "",
        auto_exit: bool = True,
        ignore_terminate: bool = False,
    ) -> None:
        from pathlib import Path

        self.args = cmd
        self.cmd = cmd
        self.terminated = False
        self.killed = False
        self.returncode: int | None = None
        self._pending_returncode = returncode
        self._stderr = stderr
        self._ignore_terminate = ignore_terminate
        self._exited = threading.Event()
        self._reap_lock = threading.Lock()
        # Set the instant the recorder starts blocking in ``wait``/``communicate``.
        # Lets a stop-test synchronise on "ffmpeg is genuinely in flight" instead
        # of racing the spawn window.
        self.entered_wait = threading.Event()
        if chunks:
            out_dir = Path(cmd[-1]).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, data in chunks.items():
                (out_dir / name).write_bytes(data)
        if auto_exit:
            self._exited.set()

    def _reap(self) -> None:
        with self._reap_lock:
            if self.returncode is None:
                self.returncode = self._pending_returncode

    def poll(self):  # noqa: ANN201
        return self.returncode

    def wait(self, timeout=None):  # noqa: ANN001, ANN201
        self.entered_wait.set()
        if not self._exited.wait(timeout):
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        self._reap()
        return self.returncode

    def communicate(self, timeout=None):  # noqa: ANN001, ANN201
        self.wait(timeout)
        return ("", self._stderr)

    def terminate(self) -> None:
        self.terminated = True
        if not self._ignore_terminate:
            self._exited.set()

    def kill(self) -> None:
        self.killed = True
        self._exited.set()


def _popen_factory(
    chunks: dict[str, bytes] | None = None,
    *,
    returncode: int = 0,
    stderr: str = "",
    auto_exit: bool = True,
    ignore_terminate: bool = False,
):
    """Build a ``subprocess.Popen``-shaped factory that yields :class:`FakePopen`.

    Every spawned process is recorded on ``.created`` so a test can assert
    on ``terminate``/``kill`` after the recording ends."""
    created: list[FakePopen] = []

    def _factory(cmd, **kwargs):  # noqa: ANN001, ANN202
        proc = FakePopen(
            cmd,
            chunks=chunks,
            returncode=returncode,
            stderr=stderr,
            auto_exit=auto_exit,
            ignore_terminate=ignore_terminate,
        )
        created.append(proc)
        return proc

    _factory.created = created  # type: ignore[attr-defined]
    return _factory


# ----- 7. Recorder.start invokes runner with the right cmd and transitions state -----


def test_recorder_start_invokes_ffmpeg_and_uploads_chunks(tmp_path) -> None:  # noqa: ANN001
    """The happy path: ``start`` runs ffmpeg, then uploads every produced
    chunk via ``upload_input_chunk`` and writes a single ``status.json``
    with ``status: pending``. After the call returns, the recorder is
    back in ``idle`` so the operator can launch another recording.

    Uses an injected ``output_dir_factory`` so the test pins the temp dir
    under pytest's ``tmp_path`` instead of the real ``/tmp``."""
    fake_r2 = FakeR2ForRecorder()
    popen_calls: list[list[str]] = []

    def _capturing_factory(cmd, **kwargs):  # noqa: ANN001
        popen_calls.append(list(cmd))
        return FakePopen(cmd, chunks={"recording.mp4": b"fake-mp4-bytes"})

    rec = Recorder(
        uploader=fake_r2,
        popen_factory=_capturing_factory,
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-rec-001",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=3600)

    # ffmpeg was invoked once with the cmd build_ffmpeg_cmd would produce.
    assert len(popen_calls) == 1
    assert popen_calls[0][0] == "ffmpeg"
    assert "rtsp://camera.local/stream" in popen_calls[0]
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

    # A factory that observes state *while ffmpeg is spawning* by re-entering
    # the recorder before returning the handle — easier than wiring threads.
    # The outer start has already reserved state='recording' by the time the
    # factory runs, so the second start must raise.
    busy_observed: list[bool] = []

    def _factory(cmd, **kwargs):  # noqa: ANN001
        try:
            rec.start(url="rtsp://other/stream", duration_s=3600)
            busy_observed.append(False)
        except RecorderBusy:
            busy_observed.append(True)
        # Then act like a normal successful ffmpeg run.
        return FakePopen(cmd, chunks={"recording.mp4": b"x"})

    rec = Recorder(
        uploader=fake_r2,
        popen_factory=_factory,
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

    rec = Recorder(
        uploader=fake_r2,
        # No chunk files — simulate "ffmpeg refused the URL".
        popen_factory=_popen_factory(returncode=1, stderr="rtsp://nope: Connection refused"),
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-doomed",
    )

    rec.start(url="rtsp://nope.local/stream", duration_s=3600)

    assert fake_r2.uploaded == []
    assert fake_r2.statuses == {}, "no status.json should leak for a failed recording"
    snapshot = rec.status()
    assert snapshot.state == "failed"
    assert "Connection refused" in snapshot.message


# ----- 9b. Recorder.start: 0-byte chunk → treated as failure, no R2 upload -----


def test_recorder_rejects_zero_byte_chunks(tmp_path) -> None:  # noqa: ANN001
    """ffmpeg can create the output file before trying to mux — a codec
    mismatch (e.g. pcm_mulaw in MP4) causes it to exit immediately,
    leaving a 0-byte file on disk. The recorder must NOT upload empty
    files: doing so creates a phantom job in R2 that the GPU worker
    downloads, feeds to ffprobe, and fails loudly on "moov atom not found".
    Instead, treat 0-byte chunks the same as "no chunks" → state=failed,
    no R2 writes."""
    fake_r2 = FakeR2ForRecorder()

    rec = Recorder(
        uploader=fake_r2,
        # Simulate ffmpeg creating an empty file and failing.
        popen_factory=_popen_factory(
            {"recording.mp4": b""},
            returncode=1,
            stderr="Could not find tag for codec pcm_mulaw in stream #1",
        ),
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-zerobyte",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=300)

    assert fake_r2.uploaded == [], "0-byte chunks must not be uploaded"
    assert fake_r2.statuses == {}, "no status.json should leak for a 0-byte recording"
    snapshot = rec.status()
    assert snapshot.state == "failed"
    assert "pcm_mulaw" in snapshot.message


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

    rec = Recorder(
        uploader=fake_r2,
        # Simulate ffmpeg flushing two segments before crashing.
        popen_factory=_popen_factory(
            {"chunk_000.mp4": b"hour-1", "chunk_001.mp4": b"hour-2"},
            returncode=1,
            stderr="RTSP/1.0 200 OK ... Input/output error",
        ),
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

    rec = Recorder(
        uploader=fake_r2,
        popen_factory=_popen_factory({"recording.mp4": b"some bytes"}),
        output_dir_factory=lambda job_id: str(out_dir),
        job_id_factory=lambda: "job-cleanup",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=3600)

    assert not Path(out_dir).exists(), "recorder must remove the temp dir after upload"


# ----- 12. chunks_uploaded survives the idle transition -----


def test_recorder_status_preserves_chunks_uploaded_after_idle(tmp_path) -> None:  # noqa: ANN001
    """The final state-flip back to ``idle`` must preserve
    ``chunks_uploaded`` so the UI can render "last recording: N chunks"
    without re-listing R2. Discovered during e2e validation on cctv-vps:
    a successful recording reported ``chunks_uploaded=0`` because the
    idle reset clobbered the counter."""
    fake_r2 = FakeR2ForRecorder()

    rec = Recorder(
        uploader=fake_r2,
        popen_factory=_popen_factory(
            {"chunk_000.mp4": b"a", "chunk_001.mp4": b"b", "chunk_002.mp4": b"c"}
        ),
        output_dir_factory=lambda jid: str(tmp_path / jid),
        job_id_factory=lambda: "job-count",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=4 * 3600)

    snapshot = rec.status()
    assert snapshot.state == "idle"
    assert snapshot.chunks_uploaded == 3


# ----- 13. buffer mode (#27): camera_id routes output, no upload, no cleanup -----


def test_recorder_buffer_mode_routes_to_camera_id_and_leaves_chunks(tmp_path) -> None:  # noqa: ANN001
    """When ``camera_id`` is passed to ``start``, the recorder is in
    continuous-buffer mode (issue #27, Slice 1c.2):

    * ``output_dir_factory`` is invoked with ``camera_id`` (not ``job_id``),
      so the caller can route writes to ``{BUFFER_DIR}/{camera_id}/``.
    * Chunks are **left on disk** — the rolling buffer is the source of
      truth for the task poller. No ``upload_input_chunk``, no
      ``put_status``, no ``rmtree`` of the output dir.

    The legacy job_id-keyed upload flow (no ``camera_id`` passed) is the
    other path and is exercised by every other test in this file."""
    fake_r2 = FakeR2ForRecorder()
    factory_calls: list[str] = []

    def _factory(key: str) -> str:
        factory_calls.append(key)
        return str(tmp_path / "buffer" / key)

    rec = Recorder(
        uploader=fake_r2,
        popen_factory=_popen_factory({"chunk_000.mp4": b"aaa", "chunk_001.mp4": b"bbb"}),
        output_dir_factory=_factory,
        # job_id_factory should NOT be called in buffer mode — wire it to a
        # sentinel that would explode if invoked.
        job_id_factory=lambda: (_ for _ in ()).throw(AssertionError("job_id used in buffer mode")),
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=4 * 3600, camera_id="cam-front-door")

    # Factory routed by camera_id, not a generated job_id.
    assert factory_calls == ["cam-front-door"]
    # No R2 traffic in buffer mode — poller does the upload later.
    assert fake_r2.uploaded == []
    assert fake_r2.statuses == {}
    # Chunks still on disk for the poller to find.
    chunk_dir = tmp_path / "buffer" / "cam-front-door"
    assert sorted(p.name for p in chunk_dir.glob("chunk_*.mp4")) == [
        "chunk_000.mp4",
        "chunk_001.mp4",
    ]
    # State machine back to idle so the next start (or per-camera respawn) works.
    assert rec.status().state == "idle"


# ----- 14. issue #50: multi-chunk recording writes distinct R2 keys -----


def test_multichunk_recording_uploads_distinct_r2_keys(tmp_path) -> None:  # noqa: ANN001
    """Regression for issue #50: a segmented (multi-chunk) recording must
    upload each chunk to its OWN R2 key. The bug was that every chunk was
    written to ``chunk_001.mp4``, so an 8h recording left only its final
    hour in R2 — silent data loss that the ``chunks_uploaded`` counter
    happily reported as a full upload.

    The fake mirrors the production signature and records the *real*
    destination key (``surveillance-jobs/{job}/input/{chunk_name}``) instead
    of fabricating one from call order, so this test actually exercises the
    key the recorder asks for. Order-preserving because the worker downloads
    the ``input/`` prefix lexically sorted."""
    fake_r2 = FakeR2ForRecorder()

    rec = Recorder(
        uploader=fake_r2,
        popen_factory=_popen_factory(
            {"chunk_000.mp4": b"h0", "chunk_001.mp4": b"h1", "chunk_002.mp4": b"h2"}
        ),
        output_dir_factory=lambda jid: str(tmp_path / jid),
        job_id_factory=lambda: "job-multi",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=4 * 3600)

    assert fake_r2.keys == [
        "surveillance-jobs/job-multi/input/chunk_000.mp4",
        "surveillance-jobs/job-multi/input/chunk_001.mp4",
        "surveillance-jobs/job-multi/input/chunk_002.mp4",
    ]
    # Distinctness is the crux — the bug produced three identical keys.
    assert len(set(fake_r2.keys)) == 3


# ===== stop() actually terminates the ffmpeg child (issue #52) =====
#
# Before #52, ``stop`` only flipped in-memory state to idle; the blocking
# ``subprocess.run`` had no handle so ffmpeg kept capturing for up to
# ``duration_s``. With the ``popen_factory`` boundary the recorder holds the
# handle and ``stop`` sends it SIGTERM. These tests drive an *in-flight*
# recording from a real thread (the fake's ``communicate`` blocks until the
# recorder signals it) — hermetic (no ffmpeg, no network) but concurrent.


def _wait_until(pred, *, timeout: float = 2.0, step: float = 0.01) -> bool:
    """Spin until ``pred()`` is truthy or ``timeout`` elapses.

    Used to synchronise the test thread with the recording thread without a
    fixed sleep (which would be either flaky or needlessly slow)."""
    import time

    elapsed = 0.0
    while not pred() and elapsed < timeout:
        time.sleep(step)
        elapsed += step
    return pred()


def test_stop_terminates_inflight_ffmpeg(tmp_path) -> None:  # noqa: ANN001
    """Acceptance criterion (#52): ``stop`` must terminate the ffmpeg child,
    not just flip state. Legacy (job_id) mode: a long recording is in flight
    on a background thread; the fake ffmpeg blocks until signalled. After
    ``stop`` the underlying process got ``terminate()``, the state machine is
    back to ``idle``, and — crucially — the killed partial recording is NOT
    uploaded to R2 (issue point #2: stop must not leave a phantom job)."""
    fake_r2 = FakeR2ForRecorder()
    factory = _popen_factory({"recording.mp4": b"partial"}, auto_exit=False)

    rec = Recorder(
        uploader=fake_r2,
        popen_factory=factory,
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-stop",
    )

    thread = threading.Thread(
        target=lambda: rec.start(url="rtsp://camera.local/stream", duration_s=8 * 3600),
        daemon=True,
    )
    thread.start()

    # Wait until ffmpeg is actually spawned and blocking (state=recording).
    assert _wait_until(
        lambda: bool(factory.created) and factory.created[0].entered_wait.is_set()
    ), "recording never reached the in-flight state"
    assert rec.status().state == "recording"

    rec.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive(), "start thread should return once ffmpeg is terminated"
    proc = factory.created[0]
    assert proc.terminated is True, "stop must SIGTERM the ffmpeg child"
    assert rec.status().state == "idle"
    # The operator killed this recording — its partial output must not be
    # uploaded (would create a phantom job the GPU worker later fails on).
    assert fake_r2.uploaded == []
    assert fake_r2.statuses == {}


def test_stop_kills_after_grace(tmp_path) -> None:  # noqa: ANN001
    """A wedged ffmpeg that ignores SIGTERM must not let ``stop`` hang the
    web thread. After ``stop_grace_s`` the recorder escalates to SIGKILL so
    the operator's "stop" always completes and the camera always frees up.
    The fake ignores ``terminate()`` (models the wedged encoder), so the
    grace ``wait`` times out and ``kill()`` is the only thing that unblocks
    the recording thread."""
    fake_r2 = FakeR2ForRecorder()
    factory = _popen_factory({"recording.mp4": b"x"}, auto_exit=False, ignore_terminate=True)

    rec = Recorder(
        uploader=fake_r2,
        popen_factory=factory,
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-wedged",
        stop_grace_s=0.1,  # keep the test fast; production default is 5s
    )

    thread = threading.Thread(
        target=lambda: rec.start(url="rtsp://camera.local/stream", duration_s=8 * 3600),
        daemon=True,
    )
    thread.start()

    assert _wait_until(
        lambda: bool(factory.created) and factory.created[0].entered_wait.is_set()
    ), "recording never reached the in-flight state"

    rec.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    proc = factory.created[0]
    assert proc.terminated is True, "stop must try SIGTERM first"
    assert proc.killed is True, "stop must escalate to SIGKILL when SIGTERM is ignored"
    assert rec.status().state == "idle"


def test_stop_in_buffer_mode_stops_chunk_production(tmp_path) -> None:  # noqa: ANN001
    """Buffer mode (issue #27) is the appliance's continuous-capture path:
    ffmpeg writes 60s chunks into the rolling buffer indefinitely. When the
    platform disables the camera the recorder must actually stop producing
    chunks — before #52 ffmpeg kept writing for up to ``duration_s`` (default
    1h). Through the production wrapper (:class:`BackgroundRecorder`): after
    ``stop`` the ffmpeg child is terminated and ``is_running()`` — the signal
    ``reconcile_recorders`` polls — reports ``False`` so the slot is free."""
    from client_agent.recorder import BackgroundRecorder

    factory = _popen_factory({"chunk_000.mp4": b"aaa"}, auto_exit=False)
    inner = Recorder(
        uploader=None,  # buffer mode never uploads — the poller does
        popen_factory=factory,
        output_dir_factory=lambda cam_id: str(tmp_path / "buffer" / cam_id),
    )
    bg = BackgroundRecorder(inner)

    bg.start(url="rtsp://camera.local/stream", duration_s=8 * 3600, camera_id="cam-front")

    assert _wait_until(
        lambda: bool(factory.created) and factory.created[0].entered_wait.is_set()
    ), "buffer recording never reached the in-flight state"
    assert bg.is_running() is True

    bg.stop()

    assert _wait_until(lambda: not bg.is_running()), "recorder still running after stop"
    proc = factory.created[0]
    assert proc.terminated is True, "buffer-mode stop must terminate the ffmpeg child"
    assert bg.is_running() is False


# ----- BackgroundRecorder thread survival + is_running contract -----


def test_background_recorder_is_running_false_when_thread_exits_via_exception(
    tmp_path,
) -> None:
    """An exception inside the inner ``Recorder.start`` (ffmpeg crash, RTSP
    drop, codec error) must NOT silently kill the daemon thread with no
    trace. The wrapper logs a warning AND ``is_running()`` reports False so
    the next ``reconcile_recorders`` cycle respawns. Source incident:
    Wi-Fi blip dropped RTSP and the recorder vanished; buffer chunks stopped
    growing; appliance kept heartbeating happily; tasks failed with
    ``time range outside buffer`` until the operator restarted the
    appliance. Bullet-proof postcondition: ``is_running()`` must reflect
    reality so reconcile self-heals."""

    from client_agent.recorder import BackgroundRecorder, Recorder

    class _Boom(Recorder):
        # ``BackgroundRecorder`` reserves the slot synchronously then runs the
        # ffmpeg work via ``_run`` on the daemon thread (issue #52 TOCTOU fix),
        # so the simulated crash must come from ``_run`` to exercise the
        # daemon-thread exception path.
        def _run(  # type: ignore[override]
            self, job_id: str, *, url: str, duration_s: int, camera_id: str | None
        ) -> str:
            raise RuntimeError("simulated ffmpeg crash on RTSP drop")

    inner = _Boom(
        uploader=None,  # type: ignore[arg-type]
        runner=lambda *a, **kw: None,  # type: ignore[arg-type,return-value]
        output_dir_factory=lambda _id: str(tmp_path / "buf" / _id),
    )
    bg = BackgroundRecorder(inner)

    bg.start(url="rtsp://camera.local/1", duration_s=3600, camera_id="cam-1")

    # Give the daemon thread a moment to exit.
    deadline = 2.0
    step = 0.02
    elapsed = 0.0
    while bg.is_running() and elapsed < deadline:
        import time as _t

        _t.sleep(step)
        elapsed += step

    assert bg.is_running() is False, "dead recorder thread must report is_running()=False"


def test_background_recorder_is_running_false_before_start() -> None:
    """A never-started BackgroundRecorder reports ``is_running()=False`` so
    reconcile spawns on first visit instead of treating None as alive."""
    from client_agent.recorder import BackgroundRecorder, Recorder

    inner = Recorder(
        uploader=None,  # type: ignore[arg-type]
        runner=lambda *a, **kw: None,  # type: ignore[arg-type,return-value]
        output_dir_factory=lambda _id: "/tmp/never-used",
    )
    bg = BackgroundRecorder(inner)
    assert bg.is_running() is False


def test_concurrent_start_second_caller_gets_busy(tmp_path) -> None:  # noqa: ANN001
    """Issue #52 (#4): ``BackgroundRecorder.start`` must claim the single
    recording slot SYNCHRONOUSLY on the caller's (HTTP) thread. The old code
    only pre-checked a stale ``status()`` then spawned a daemon thread, so two
    near-simultaneous ``/start`` calls both passed the check and the loser
    raised ``RecorderBusy`` invisibly *inside* its daemon thread — its HTTP
    caller had already been told the recording was scheduled.

    A purely sequential test can't expose that race (the daemon reliably flips
    state before the second call), so we pin the *mechanism*: the slot claim
    (the ``idle → recording`` flip, which computes the ``job_id``) must happen
    on the caller's own thread before ``start`` returns. We observe that via
    the injected ``job_id_factory``:

    * fixed → the factory ran on the test (caller) thread → the reservation is
      synchronous → a second caller is rejected on its own thread.
    * broken → the factory ran on the daemon thread (or hadn't run yet when
      ``start`` returned) → the rejection would surface only in the daemon.
    """
    import pytest

    from client_agent.recorder import BackgroundRecorder

    caller = threading.current_thread()
    reserve_threads: list[threading.Thread] = []

    def _job_id_factory() -> str:
        reserve_threads.append(threading.current_thread())
        return "job-1"

    # First recording stays in flight (blocks) so the slot is genuinely busy
    # while the second caller races in. Legacy mode so the reservation calls
    # ``job_id_factory`` (buffer mode would key on camera_id instead).
    factory = _popen_factory({"recording.mp4": b"x"}, auto_exit=False)
    inner = Recorder(
        uploader=FakeR2ForRecorder(),
        popen_factory=factory,
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=_job_id_factory,
    )
    bg = BackgroundRecorder(inner)

    bg.start(url="rtsp://camera.local/1", duration_s=8 * 3600)

    # The slot was claimed synchronously, on THIS thread, before start returned.
    assert reserve_threads == [caller], "reservation must run on the caller thread"
    assert bg.status().state == "recording"

    # A second caller must be rejected on THIS thread, not inside a daemon —
    # and without invoking job_id_factory again (busy check comes first).
    with pytest.raises(RecorderBusy):
        bg.start(url="rtsp://camera.local/2", duration_s=8 * 3600)
    assert reserve_threads == [caller], "a rejected start must not claim a job_id"

    # Tear down: stop the first recording so its daemon thread exits cleanly.
    bg.stop()
