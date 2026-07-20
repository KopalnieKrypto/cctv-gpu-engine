"""Tests for the client-agent RTSP recorder (issue #8).

The recorder owns three concerns: (a) probing an RTSP URL to give the
operator fast yes/no feedback, (b) building the right ffmpeg command for
short vs segmented (long) recordings, and (c) running one recording at a
time and leaving the resulting chunks on disk for the task poller (buffer
mode) — the recorder no longer uploads to R2 (issue #29).

Mocks live only at the ffmpeg (subprocess) boundary. The ``runner``
callable mirrors ``subprocess.run``'s contract just enough that production
code can pass ``subprocess.run`` directly with no shim.
"""

from __future__ import annotations

import os
import subprocess
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from client_agent.recorder import (
    BUFFER_CHUNK_TEMPLATE,
    BUFFER_SEGMENT_SECONDS,
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
    output into a one-element list."""
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
    each chunk is a finalized, playable file on a flaky LAN. ``-reset_timestamps
    1`` keeps each chunk independently playable instead of starting at offset
    3600s. The output template uses ``%03d`` so chunks sort lexically the same
    as chronologically — important for the poller's ordering."""
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


# ----- 7. buffer mode: segment names survive a recorder respawn (issue #90) -----


def test_build_ffmpeg_cmd_buffer_mode_names_segments_uniquely_across_respawns() -> None:
    """Buffer-mode chunk names must not collide between recorder runs.

    The production recorder respawns roughly hourly (``-t 3600`` is a
    liveness cadence, not a retention knob — see #85). With the old
    ``chunk_%03d.mp4`` template every respawn restarted the counter at 000
    and ffmpeg **overwrote in place**, destroying in-retention history: an
    appliance configured for ``buffer_hours=5`` held ~1 h of footage
    (observed on cctv-vps-camera, #90).

    So the naming contract is *time*-derived, not counter-derived. We assert
    it the way ffmpeg realises it: expand the emitted template with
    ``strftime`` at two wallclock instants one respawn apart and require
    distinct names. A counter-based template collapses both to the same
    filename and fails here."""
    cmd = build_ffmpeg_cmd(
        url="rtsp://camera.local/stream",
        duration_s=3600,
        output_dir="/tmp/buf/cam-1",
        buffer_mode=True,
    )
    template = Path(cmd[-1]).name

    first_run = datetime(2026, 7, 20, 12, 19, 0, tzinfo=UTC)
    respawn = datetime(2026, 7, 20, 15, 0, 41, tzinfo=UTC)

    assert first_run.strftime(template) != respawn.strftime(template)


# ----- 8. buffer mode: names stay discoverable by RollingBuffer (issue #90) -----


def test_buffer_mode_chunk_names_are_found_by_rolling_buffer(tmp_path) -> None:  # noqa: ANN001
    """The recorder's filenames and the buffer's ``chunk_*.mp4`` glob are one
    contract across two modules, and #90 changed one side of it.

    Nothing in :mod:`client_agent.buffer` was touched, so this test is what
    justifies that: expand the real template into real files and drive the
    buffer's two public queries against them. If a future rename drops the
    ``chunk_`` prefix, ``chunks_in_range`` silently returns nothing and every
    task reports "no footage" — a failure mode with no other alarm on it.

    Retention is asserted here too, deliberately. Unique names removed the
    accidental ~1 h cap that overwriting used to provide, so ``trim_old_chunks``
    is now the only thing bounding disk (#90 acceptance, #51)."""
    from client_agent.buffer import RollingBuffer

    cam_dir = tmp_path / "cam-1"
    cam_dir.mkdir(parents=True)

    now = datetime(2026, 7, 20, 15, 0, 0, tzinfo=UTC)
    template = Path(
        build_ffmpeg_cmd(
            url="rtsp://camera.local/stream",
            duration_s=3600,
            output_dir=str(cam_dir),
            buffer_mode=True,
        )[-1]
    ).name

    # Six segments one minute apart, the oldest 5 minutes back — as ffmpeg
    # would name them, with mtime set to the segment's close time.
    written: list[Path] = []
    for age_min in range(5, -1, -1):
        closed_at = now - timedelta(minutes=age_min)
        path = cam_dir / closed_at.strftime(template)
        path.write_bytes(b"segment")
        os.utime(path, (closed_at.timestamp(), closed_at.timestamp()))
        written.append(path)

    assert len({p.name for p in written}) == 6, "template produced colliding names"

    buffer = RollingBuffer(base_dir=tmp_path, buffer_hours=1, segment_seconds=60)

    # Discovery: the glob still matches, and mtime ordering still works.
    # The window sits mid-segment (…:30) on purpose — landing it exactly on a
    # boundary would also pull in the neighbouring chunk, since overlap is
    # inclusive at both ends, and obscure what this test is actually about.
    assert buffer.has_recorded("cam-1") is True
    found = buffer.chunks_in_range(
        "cam-1",
        start=now - timedelta(seconds=150),
        end=now - timedelta(seconds=90),
    )
    assert [c.path.name for c in found] == [
        (now - timedelta(minutes=2)).strftime(template),
        (now - timedelta(minutes=1)).strftime(template),
    ]

    # Retention: with a 2-minute window the three oldest segments go.
    buffer.set_buffer_hours(0)
    deleted = buffer.trim_old_chunks("cam-1", now=now - timedelta(minutes=2))
    assert deleted == 4
    assert sorted(p.name for p in cam_dir.glob("chunk_*.mp4")) == [
        (now - timedelta(minutes=1)).strftime(template),
        now.strftime(template),
    ]


# ----- 9. buffer mode: the flags the rest of the appliance assumes -----


def test_build_ffmpeg_cmd_buffer_mode_emits_strftime_segments_bounded_by_duration() -> None:
    """Pin the whole buffer-mode flag set — the only branch that runs in
    production, and until #90 the only one with no test at all. That gap is
    how a filename template that overwrote live footage reached a customer box.

    Each flag has a distinct failure mode if dropped:

    * ``-strftime 1`` — without it ffmpeg treats ``%Y…`` as literal text and
      every segment fights over one filename. Strictly worse than the #90 bug.
    * ``-segment_time 60`` — must equal ``BUFFER_SEGMENT_SECONDS``, which the
      appliance also feeds to ``RollingBuffer(segment_seconds=)``. Disagreement
      silently skews every chunk's inferred start time.
    * ``-t duration_s`` — the respawn cadence (#85). Dropping it lets one
      ffmpeg run forever, so a wedged RTSP connection is never noticed.
    * ``-reset_timestamps 1`` — each chunk must start at 0 to be independently
      trimmable.
    """
    cmd = build_ffmpeg_cmd(
        url="rtsp://camera.local/stream",
        duration_s=3600,
        output_dir="/tmp/buf/cam-1",
        buffer_mode=True,
    )

    def _value_after(flag: str) -> str:
        assert flag in cmd, f"buffer mode must pass {flag}"
        return cmd[cmd.index(flag) + 1]

    assert _value_after("-f") == "segment"
    assert _value_after("-strftime") == "1"
    assert _value_after("-segment_time") == str(BUFFER_SEGMENT_SECONDS)
    assert _value_after("-reset_timestamps") == "1"
    # Respawn cadence stays bounded and stays decoupled from buffer_hours (#85).
    assert _value_after("-t") == "3600"
    assert cmd[-1] == f"/tmp/buf/cam-1/{BUFFER_CHUNK_TEMPLATE}"


# ----- 10. buffer mode: lexical order == chronological order -----


def test_buffer_mode_chunk_names_sort_lexically_in_time_order() -> None:
    """``%03d`` gave lexical == chronological sort for free; the timestamp
    template must not quietly give that up.

    ``Recorder._run`` and the mediamtx integration test both reach for a plain
    ``sorted(glob(...))``, and a human reading ``ls`` output expects the same.
    Zero-padded big-endian fields preserve it — a format like ``%-d/%-m`` or a
    day-first ordering would not, and the breakage would only show up as
    mis-ordered concat footage."""
    base = datetime(2026, 7, 20, 23, 58, 0, tzinfo=UTC)
    # Spans a minute, an hour, a day, a month and a year rollover — every
    # boundary where a badly-ordered format flips lexical and chronological
    # order apart.
    instants = [
        base,
        base + timedelta(minutes=1),
        base + timedelta(minutes=2),
        base + timedelta(hours=1),
        base + timedelta(days=1),
        datetime(2026, 12, 31, 23, 59, 0, tzinfo=UTC),
        datetime(2027, 1, 1, 0, 0, 0, tzinfo=UTC),
    ]
    names = [t.strftime(BUFFER_CHUNK_TEMPLATE) for t in instants]

    assert names == sorted(names)
    assert len(set(names)) == len(names)


# ===== Recorder class — single-recording state machine =====


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
    while capturing), so both the "chunks left on disk" path and the
    "partial chunks" path see files regardless of how the process ends.

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


# ----- 7. Recorder.start runs ffmpeg and leaves the chunk on disk -----


def test_recorder_start_records_and_leaves_chunk_on_disk(tmp_path) -> None:  # noqa: ANN001
    """The happy path: ``start`` runs ffmpeg, then leaves the produced chunk
    on disk (buffer mode is the only mode since #29 — the poller does the
    upload later via presigned URLs). After the call returns, the recorder is
    back in ``idle`` so the operator can launch another recording.

    Uses an injected ``output_dir_factory`` so the test pins the temp dir
    under pytest's ``tmp_path`` instead of the real ``/tmp``."""

    popen_calls: list[list[str]] = []

    def _capturing_factory(cmd, **kwargs):  # noqa: ANN001
        popen_calls.append(list(cmd))
        return FakePopen(cmd, chunks={"recording.mp4": b"fake-mp4-bytes"})

    out_dir = tmp_path / "job-rec-001"
    rec = Recorder(
        popen_factory=_capturing_factory,
        output_dir_factory=lambda job_id: str(out_dir),
        job_id_factory=lambda: "job-rec-001",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=3600)

    # ffmpeg was invoked once with the cmd build_ffmpeg_cmd would produce.
    assert len(popen_calls) == 1
    assert popen_calls[0][0] == "ffmpeg"
    assert "rtsp://camera.local/stream" in popen_calls[0]
    # Chunk stays on disk for the poller — the recorder never uploads.
    assert (out_dir / "recording.mp4").read_bytes() == b"fake-mp4-bytes"
    # Back to idle so the next start works, and the produced-chunk count is
    # surfaced for the UI.
    snapshot = rec.status()
    assert snapshot.state == "idle"
    assert snapshot.chunks_uploaded == 1


# ----- 8. Recorder.start raises RecorderBusy when already running -----


def test_recorder_rejects_concurrent_start(tmp_path) -> None:  # noqa: ANN001
    """Acceptance criterion (#8): "second recording while one active →
    rejected". The recorder owns a single ffmpeg slot — letting two run
    in parallel would double-spend the LAN's RTSP allowance and corrupt
    both recordings."""

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
        popen_factory=_factory,
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-busy-1",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=3600)

    assert busy_observed == [True], "second start should have raised RecorderBusy"
    # Only the first job produced a chunk on disk.
    assert (tmp_path / "job-busy-1" / "recording.mp4").exists()


# ----- 9. Recorder.start: invalid URL → no chunks → failed -----


def test_recorder_marks_failed_when_ffmpeg_produces_no_chunks(tmp_path) -> None:  # noqa: ANN001
    """Acceptance criterion (#8): "Invalid RTSP URL → clear 'connection
    failed', no hung ffmpeg". When ffmpeg exits non-zero AND leaves zero
    files on disk, there's nothing for the poller to pick up — the recorder
    flips to ``failed`` and surfaces the stderr so the UI can show *why*."""

    rec = Recorder(
        # No chunk files — simulate "ffmpeg refused the URL".
        popen_factory=_popen_factory(returncode=1, stderr="rtsp://nope: Connection refused"),
        output_dir_factory=lambda job_id: str(tmp_path / job_id),
        job_id_factory=lambda: "job-doomed",
    )

    rec.start(url="rtsp://nope.local/stream", duration_s=3600)

    snapshot = rec.status()
    assert snapshot.state == "failed"
    assert "Connection refused" in snapshot.message


# ----- 9b. Recorder.start: 0-byte chunk → treated as failure -----


def test_recorder_rejects_zero_byte_chunks(tmp_path) -> None:  # noqa: ANN001
    """ffmpeg can create the output file before trying to mux — a codec
    mismatch (e.g. pcm_mulaw in MP4) causes it to exit immediately,
    leaving a 0-byte file on disk. The recorder must NOT leave empty
    files for the poller: an empty segment can't be trimmed and would fail
    downstream on "moov atom not found". Instead, treat 0-byte chunks the
    same as "no chunks" → state=failed."""

    rec = Recorder(
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

    snapshot = rec.status()
    assert snapshot.state == "failed"
    assert "pcm_mulaw" in snapshot.message


# ----- 9c. Buffer mode: history from earlier runs is not counted as this run's -----


def test_recorder_ignores_pre_existing_chunks_when_judging_the_run(tmp_path) -> None:  # noqa: ANN001
    """A buffer-mode camera dir accumulates across respawns, so the recorder
    must judge a run by what *this* run wrote — not by whatever is on disk.

    The camera dir is keyed by ``camera_id`` and (since #90) holds uniquely
    named history from every prior run. A run that fails instantly — camera
    unplugged, RTSP credentials rotated — writes nothing, yet the dir is still
    full of yesterday's chunks. Judging on the whole directory reports that
    failure as ``idle`` with a healthy-looking chunk count, and the operator's
    status page shows green while the camera records nothing.

    Pre-#90 the ``%03d`` wrap capped the dir at 60 files so the miscount was
    merely wrong; unique names make it wrong *and* unbounded."""
    cam_dir = tmp_path / "buffer" / "cam-front-door"
    cam_dir.mkdir(parents=True)
    # Yesterday's history — real, non-empty, and none of it this run's work.
    for name in ("chunk_20260719-100000.mp4", "chunk_20260719-100100.mp4"):
        (cam_dir / name).write_bytes(b"yesterday")

    rec = Recorder(
        # ffmpeg exits immediately having written nothing.
        popen_factory=_popen_factory(returncode=1, stderr="rtsp://cam: Connection refused"),
        output_dir_factory=lambda cam_id: str(tmp_path / "buffer" / cam_id),
    )

    rec.start(url="rtsp://cam.local/stream", duration_s=3600, camera_id="cam-front-door")

    snapshot = rec.status()
    assert snapshot.state == "failed"
    assert "Connection refused" in snapshot.message
    # History is untouched — the poller still owns it, and only retention deletes.
    assert len(list(cam_dir.glob("chunk_*.mp4"))) == 2


def test_recorder_counts_only_chunks_written_by_this_run(tmp_path) -> None:  # noqa: ANN001
    """``chunks_uploaded`` reports this run's output, not the buffer's depth.

    The UI renders it as "last recording: N chunks". Counting the whole camera
    dir made it a disk-occupancy number that only ever grew, which reads as a
    wildly productive recording no matter what ffmpeg actually did."""
    cam_dir = tmp_path / "buffer" / "cam-side"
    cam_dir.mkdir(parents=True)
    for name in ("chunk_20260719-100000.mp4", "chunk_20260719-100100.mp4"):
        (cam_dir / name).write_bytes(b"yesterday")

    rec = Recorder(
        popen_factory=_popen_factory(
            {"chunk_20260720-150000.mp4": b"new", "chunk_20260720-150100.mp4": b"new"}
        ),
        output_dir_factory=lambda cam_id: str(tmp_path / "buffer" / cam_id),
    )

    rec.start(url="rtsp://cam.local/stream", duration_s=3600, camera_id="cam-side")

    snapshot = rec.status()
    assert snapshot.state == "idle"
    assert snapshot.chunks_uploaded == 2, "counted pre-existing history as this run's output"
    # All four files coexist — unique names mean the new run appended.
    assert len(list(cam_dir.glob("chunk_*.mp4"))) == 4


# ----- 10. Partial recording (camera offline mid-recording) is kept on disk -----


def test_recorder_keeps_partial_chunks_when_ffmpeg_fails_late(tmp_path) -> None:  # noqa: ANN001
    """Acceptance criterion (#8): "camera goes offline mid-recording →
    partial MP4 still valid, kept". ffmpeg's segment muxer flushes each
    finished chunk before EOF, so an interrupted 4h recording still leaves
    the first N chunks playable. We must keep those on disk instead of
    discarding them — losing 2 hours of footage because hour 3 dropped is
    the worst possible failure mode for a surveillance system.

    Multiple chunks land in the recorder dir; we assert both survive on disk
    and the recorder returns to ``idle`` reporting the produced count."""

    out_dir = tmp_path / "job-partial"
    rec = Recorder(
        # Simulate ffmpeg flushing two segments before crashing.
        popen_factory=_popen_factory(
            {"chunk_000.mp4": b"hour-1", "chunk_001.mp4": b"hour-2"},
            returncode=1,
            stderr="RTSP/1.0 200 OK ... Input/output error",
        ),
        output_dir_factory=lambda job_id: str(out_dir),
        job_id_factory=lambda: "job-partial",
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=4 * 3600)

    assert (out_dir / "chunk_000.mp4").read_bytes() == b"hour-1"
    assert (out_dir / "chunk_001.mp4").read_bytes() == b"hour-2"
    snapshot = rec.status()
    assert snapshot.state == "idle"
    assert snapshot.chunks_uploaded == 2


# ----- 11. chunks_uploaded survives the idle transition -----


def test_recorder_status_preserves_chunks_uploaded_after_idle(tmp_path) -> None:  # noqa: ANN001
    """The final state-flip back to ``idle`` must preserve
    ``chunks_uploaded`` (the produced-chunk count) so the UI can render
    "last recording: N chunks". Discovered during e2e validation on
    cctv-vps: a successful recording reported ``chunks_uploaded=0`` because
    the idle reset clobbered the counter."""

    rec = Recorder(
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


# ----- 12. buffer mode (#27): camera_id routes output, chunks left on disk -----


def test_recorder_buffer_mode_routes_to_camera_id_and_leaves_chunks(tmp_path) -> None:  # noqa: ANN001
    """When ``camera_id`` is passed to ``start``, the recorder keys output on
    the camera (issue #27, Slice 1c.2):

    * ``output_dir_factory`` is invoked with ``camera_id`` (not a generated
      job_id), so the caller can route writes to ``{BUFFER_DIR}/{camera_id}/``.
    * Chunks are **left on disk** — the rolling buffer is the source of
      truth for the task poller. No cleanup of the output dir.

    Since #29 this is the recorder's only behaviour; the ``camera_id=None``
    path keeps chunks the same way, only keyed by a generated job_id."""
    factory_calls: list[str] = []

    def _factory(key: str) -> str:
        factory_calls.append(key)
        return str(tmp_path / "buffer" / key)

    rec = Recorder(
        popen_factory=_popen_factory({"chunk_000.mp4": b"aaa", "chunk_001.mp4": b"bbb"}),
        output_dir_factory=_factory,
        # job_id_factory should NOT be called in buffer mode — wire it to a
        # sentinel that would explode if invoked.
        job_id_factory=lambda: (_ for _ in ()).throw(AssertionError("job_id used in buffer mode")),
    )

    rec.start(url="rtsp://camera.local/stream", duration_s=4 * 3600, camera_id="cam-front-door")

    # Factory routed by camera_id, not a generated job_id.
    assert factory_calls == ["cam-front-door"]
    # Chunks still on disk for the poller to find.
    chunk_dir = tmp_path / "buffer" / "cam-front-door"
    assert sorted(p.name for p in chunk_dir.glob("chunk_*.mp4")) == [
        "chunk_000.mp4",
        "chunk_001.mp4",
    ]
    # State machine back to idle so the next start (or per-camera respawn) works.
    assert rec.status().state == "idle"


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
    not just flip state. A long recording is in flight on a background thread;
    the fake ffmpeg blocks until signalled. After ``stop`` the underlying
    process got ``terminate()`` and the state machine is back to ``idle``."""
    factory = _popen_factory({"recording.mp4": b"partial"}, auto_exit=False)

    rec = Recorder(
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


def test_stop_kills_after_grace(tmp_path) -> None:  # noqa: ANN001
    """A wedged ffmpeg that ignores SIGTERM must not let ``stop`` hang the
    web thread. After ``stop_grace_s`` the recorder escalates to SIGKILL so
    the operator's "stop" always completes and the camera always frees up.
    The fake ignores ``terminate()`` (models the wedged encoder), so the
    grace ``wait`` times out and ``kill()`` is the only thing that unblocks
    the recording thread."""
    factory = _popen_factory({"recording.mp4": b"x"}, auto_exit=False, ignore_terminate=True)

    rec = Recorder(
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
