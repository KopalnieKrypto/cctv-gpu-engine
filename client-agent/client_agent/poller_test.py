"""Tests for the task poller (issue #27, Slice 1c.2).

The poller is the single thread that translates "task in the platform
queue" into "trimmed mp4 ready for upload". It owns the four-state
machine ``claimed → recording → uploading → (cleanup)`` and the
failure terminal state ``failed``.

Tests are hermetic — the platform is a fake with a scripted task queue,
the buffer is a fake that returns canned chunks, and ``trim_and_concat``
is replaced with a function that records its call args. No subprocess,
no network, no filesystem state shared between tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from client_agent.buffer import BufferChunk
from client_agent.platform import Task


@dataclass
class FakePlatform:
    """In-memory platform fake for the poller tests.

    Scripts a queue of tasks (``next_tasks``) the poller will see on
    successive ``fetch_next_task`` calls, and records every status
    update so tests can assert the full transition sequence."""

    next_tasks: list[Task | None] = field(default_factory=list)
    status_calls: list[tuple[str, str, str | None]] = field(default_factory=list)

    def fetch_next_task(self) -> Task | None:
        if not self.next_tasks:
            return None
        return self.next_tasks.pop(0)

    def update_task_status(
        self,
        task_id: str,
        *,
        status: str,
        error: str | None = None,
        chunk_r2_key: str | None = None,
    ) -> None:
        self.status_calls.append((task_id, status, error))


@dataclass
class FakeBuffer:
    """In-memory buffer fake — returns canned chunks per camera."""

    chunks_by_camera: dict[str, list[BufferChunk]] = field(default_factory=dict)

    def chunks_in_range(
        self, camera_id: str, *, start: datetime, end: datetime
    ) -> list[BufferChunk]:
        # Naïve: return whatever was canned, ignoring the actual range —
        # we exercise range filtering in buffer_test.py, here we only
        # care about the poller's edge-case branching.
        return list(self.chunks_by_camera.get(camera_id, []))


@dataclass
class FakeUploader:
    """In-memory uploader fake matching :class:`PresignedUploader`'s
    public surface. Scripts a result list per task_id so tests can
    pin both success (all True) and partial-failure flows."""

    results_by_task: dict[str, list[object]] = field(default_factory=dict)
    upload_calls: list[tuple[str, list[Path]]] = field(default_factory=list)

    def upload_chunks(self, task_id: str, chunks: list[Path]) -> list[object]:
        self.upload_calls.append((task_id, list(chunks)))
        return list(self.results_by_task.get(task_id, []))


# ----- 1. happy path: claim → recording → trim → uploading -----


def test_run_once_happy_path_transitions_through_recording_uploading_uploaded(
    tmp_path: Path,
) -> None:
    """Per #28 the poller now owns the full ``claim → recording →
    uploading → uploaded`` transition. The previous slice (#27) stopped
    at ``uploading`` as a stub; ``uploaded`` is reached only after the
    injected :class:`PresignedUploader` reports every chunk succeeded.

    The trimmed output is handed to the uploader as a single-element
    list (multipart splitting is deferred per DD-09; for now one trimmed
    mp4 = one PUT). Using ``upload_chunks`` (plural) at the boundary
    keeps the contract stable for a future multipart slice."""
    from client_agent.poller import TaskPoller
    from client_agent.uploader import UploadResult

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk = BufferChunk(
        path=tmp_path / "chunk_001.mp4",
        start=t10,
        end=t10 + timedelta(hours=1),
    )
    platform = FakePlatform(
        next_tasks=[
            Task(
                id="task-1",
                camera_id="cam-1",
                start_time=t10 + timedelta(minutes=15),
                end_time=t10 + timedelta(minutes=45),
            )
        ]
    )
    buffer = FakeBuffer(chunks_by_camera={"cam-1": [chunk]})
    trim_calls: list[dict] = []

    def fake_trim(*, chunks, start, end, output, runner):
        trim_calls.append({"chunks": chunks, "start": start, "end": end, "output": output})
        output.write_bytes(b"fake-trimmed")

    uploader = FakeUploader(
        results_by_task={
            "task-1": [UploadResult(chunk_n=0, success=True, key="tenants/t/x/task-1/chunk_0.mp4")]
        }
    )

    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=fake_trim,
        output_dir=tmp_path / "out",
        uploader=uploader,
    )
    handled = poller.run_once()

    assert handled is True
    assert platform.status_calls == [
        ("task-1", "recording", None),
        ("task-1", "uploading", None),
        ("task-1", "uploaded", None),
    ]
    assert len(trim_calls) == 1
    assert trim_calls[0]["chunks"] == [chunk]
    assert trim_calls[0]["start"] == t10 + timedelta(minutes=15)
    assert trim_calls[0]["end"] == t10 + timedelta(minutes=45)
    # Uploader was handed the trimmed mp4 (single-element list — multipart deferred).
    assert len(uploader.upload_calls) == 1
    assert uploader.upload_calls[0][0] == "task-1"
    assert uploader.upload_calls[0][1] == [trim_calls[0]["output"]]


# ----- 2. idle: no task → no status calls, returns False -----


def test_run_once_returns_false_when_queue_idle(tmp_path: Path) -> None:
    """204 from the platform → ``run_once`` returns ``False`` and emits
    no status update. The blocking ``run()`` loop uses the boolean to
    decide whether to sleep before polling again (idle → back off,
    busy → poll again immediately)."""
    from client_agent.poller import TaskPoller

    platform = FakePlatform(next_tasks=[])
    buffer = FakeBuffer()

    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=lambda **kw: None,
        output_dir=tmp_path / "out",
        uploader=FakeUploader(),
    )
    handled = poller.run_once()

    assert handled is False
    assert platform.status_calls == []


# ----- 3. empty buffer: failed status with human-readable error -----


def test_run_once_empty_buffer_marks_task_failed(tmp_path: Path) -> None:
    """Recorder hasn't booted yet for this camera, so the buffer returns
    no chunks. The poller must not invoke ffmpeg (no input would produce
    an opaque error in journald); instead it marks the task ``failed``
    with ``error="buffer empty"``. The error string is the operator-facing
    message the platform UI surfaces, so it has to read in English with
    no Python tracebacks or path leakage."""
    from client_agent.poller import TaskPoller

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    platform = FakePlatform(
        next_tasks=[
            Task(
                id="task-empty",
                camera_id="cam-no-recorder",
                start_time=t10,
                end_time=t10 + timedelta(minutes=30),
            )
        ]
    )
    buffer = FakeBuffer()  # No canned chunks for cam-no-recorder.
    trim_calls: list[dict] = []

    def fake_trim(**kw):
        trim_calls.append(kw)

    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=fake_trim,
        output_dir=tmp_path / "out",
        uploader=FakeUploader(),
    )
    handled = poller.run_once()

    assert handled is True
    # No trim call — failed before invoking ffmpeg.
    assert trim_calls == []
    # Transitions: claim → recording → failed (uploading never reached).
    assert platform.status_calls == [
        ("task-empty", "recording", None),
        ("task-empty", "failed", "buffer empty"),
    ]


# ----- 4. stale buffer: time range outside window → failed -----


def test_run_once_time_range_outside_buffer_marks_task_failed(tmp_path: Path) -> None:
    """The buffer has chunks (recorder ran) but they don't cover the
    requested window — caller asked for footage older than ``BUFFER_HOURS``
    or wholly in the future of what is recorded. The poller's failure
    message has to distinguish this from the empty-buffer case so the
    platform UI can hint the operator at the right fix (extend retention
    vs. wait for the recorder to boot)."""
    from client_agent.poller import TaskPoller

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    # Buffer has a chunk covering 09:00 → 10:00. Task asks for 14:00 →
    # 14:30 — fully outside.
    chunk = BufferChunk(
        path=tmp_path / "chunk_old.mp4",
        start=t10 - timedelta(hours=1),
        end=t10,
    )
    platform = FakePlatform(
        next_tasks=[
            Task(
                id="task-stale",
                camera_id="cam-1",
                start_time=t10 + timedelta(hours=4),
                end_time=t10 + timedelta(hours=4, minutes=30),
            )
        ]
    )

    # FakeBuffer returns the chunk for any range — emulate the real
    # RollingBuffer's behavior of returning [] when no chunk overlaps by
    # giving the poller a buffer that returns nothing for the requested
    # range. We use a custom buffer here because the poller's failure
    # branch is "chunks_in_range returned []", which matches the empty
    # case. Distinguish stale from empty by having the camera dir on
    # disk (i.e. some chunks have been recorded) — the poller checks
    # ``has_camera`` to choose the error message.
    class StaleBuffer(FakeBuffer):
        def has_recorded(self, camera_id: str) -> bool:
            return camera_id == "cam-1"

    buffer = StaleBuffer(chunks_by_camera={"cam-1": []})
    # Existing chunk lives on disk but doesn't overlap the requested range.
    chunk.path.write_bytes(b"")

    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=lambda **kw: None,
        output_dir=tmp_path / "out",
        uploader=FakeUploader(),
    )
    handled = poller.run_once()

    assert handled is True
    assert platform.status_calls == [
        ("task-stale", "recording", None),
        ("task-stale", "failed", "time range outside buffer"),
    ]


# ----- 5. upload failure: any chunk fails → status=failed with aggregated error -----


def test_run_once_upload_failure_marks_task_failed_with_chunk_errors(tmp_path: Path) -> None:
    """The trimmed mp4 makes it through ``trim_fn`` but the uploader's
    :class:`UploadResult` list has at least one ``success=False`` entry.
    Per #28 the poller turns this into ``status=failed`` with an error
    message that names which chunk_n failed and why, so an operator
    looking at the platform UI knows whether the bug is on R2's side
    (always 5xx) or on the platform's side (403 tenant mismatch).

    The ``uploading`` status is still emitted before the failure —
    that's the contract the UI uses to show a "uploading…" spinner,
    and removing it would hide that the trim succeeded."""
    from client_agent.poller import TaskPoller
    from client_agent.uploader import UploadResult

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk = BufferChunk(
        path=tmp_path / "chunk_001.mp4",
        start=t10,
        end=t10 + timedelta(hours=1),
    )
    platform = FakePlatform(
        next_tasks=[
            Task(
                id="task-fail",
                camera_id="cam-1",
                start_time=t10 + timedelta(minutes=15),
                end_time=t10 + timedelta(minutes=45),
            )
        ]
    )
    buffer = FakeBuffer(chunks_by_camera={"cam-1": [chunk]})

    def fake_trim(*, chunks, start, end, output, runner):
        output.write_bytes(b"trim-ok")

    uploader = FakeUploader(
        results_by_task={
            "task-fail": [
                UploadResult(
                    chunk_n=0,
                    success=False,
                    error="R2 PUT returned 500 after 3 attempt(s)",
                )
            ]
        }
    )

    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=fake_trim,
        output_dir=tmp_path / "out",
        uploader=uploader,
    )
    handled = poller.run_once()

    assert handled is True
    assert [call[0] for call in platform.status_calls] == [
        "task-fail",
        "task-fail",
        "task-fail",
    ]
    assert [call[1] for call in platform.status_calls] == ["recording", "uploading", "failed"]
    # Final transition carries an error that names the chunk and the cause.
    failed_call = platform.status_calls[-1]
    assert failed_call[2] is not None
    assert "chunk 0" in failed_call[2]
    assert "500" in failed_call[2]


# ----- 5b. trimmed output is unlinked after a successful upload -----


def test_trim_output_removed_after_successful_upload(tmp_path: Path) -> None:
    """Once the trimmed mp4 is PUT to R2, the local copy has no second
    use — leaving it behind fills the appliance disk one task at a time
    (issue #51). After a successful ``run_once`` the trim output must be
    gone. The uuid-suffixed path is generated inside ``run_once`` so we
    capture it from the trim callback."""
    from client_agent.poller import TaskPoller
    from client_agent.uploader import UploadResult

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk = BufferChunk(path=tmp_path / "chunk_001.mp4", start=t10, end=t10 + timedelta(hours=1))
    platform = FakePlatform(
        next_tasks=[
            Task(
                id="task-1",
                camera_id="cam-1",
                start_time=t10 + timedelta(minutes=15),
                end_time=t10 + timedelta(minutes=45),
            )
        ]
    )
    buffer = FakeBuffer(chunks_by_camera={"cam-1": [chunk]})
    captured: dict[str, Path] = {}

    def fake_trim(*, chunks, start, end, output, runner):
        captured["output"] = output
        output.write_bytes(b"fake-trimmed")

    uploader = FakeUploader(
        results_by_task={"task-1": [UploadResult(chunk_n=0, success=True, key="k")]}
    )
    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=fake_trim,
        output_dir=tmp_path / "out",
        uploader=uploader,
    )

    poller.run_once()

    # The uploader saw the file (it was PUT), but the local copy is gone.
    assert uploader.upload_calls[0][1] == [captured["output"]]
    assert not captured["output"].exists()


# ----- 5c. trimmed output is unlinked even when the upload fails -----


def test_trim_output_removed_after_failed_upload(tmp_path: Path) -> None:
    """A failed upload re-queues on the platform side (the task goes back
    to ``failed`` and the operator/platform retries), so the local trim
    output has no second use here either — it must not linger and leak
    disk while the box keeps taking new tasks (issue #51)."""
    from client_agent.poller import TaskPoller
    from client_agent.uploader import UploadResult

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk = BufferChunk(path=tmp_path / "chunk_001.mp4", start=t10, end=t10 + timedelta(hours=1))
    platform = FakePlatform(
        next_tasks=[
            Task(
                id="task-fail",
                camera_id="cam-1",
                start_time=t10 + timedelta(minutes=15),
                end_time=t10 + timedelta(minutes=45),
            )
        ]
    )
    buffer = FakeBuffer(chunks_by_camera={"cam-1": [chunk]})
    captured: dict[str, Path] = {}

    def fake_trim(*, chunks, start, end, output, runner):
        captured["output"] = output
        output.write_bytes(b"fake-trimmed")

    uploader = FakeUploader(
        results_by_task={
            "task-fail": [UploadResult(chunk_n=0, success=False, error="R2 PUT returned 500")]
        }
    )
    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=fake_trim,
        output_dir=tmp_path / "out",
        uploader=uploader,
    )

    poller.run_once()

    assert not captured["output"].exists()


# ----- 5d. trim_fn raises → task marked failed, never wedged (issue #54) -----


def test_trim_failure_marks_task_failed(tmp_path: Path) -> None:
    """Once a task is flipped to ``recording``, *any* exception before a
    terminal status escapes ``run_once`` — ``run()`` only logs it, so the
    platform-side task stays wedged in ``recording`` forever until a human
    intervenes (issue #54). ffmpeg crashing (``trim_fn`` raising) is the
    most likely such path and has no local try/except today.

    The poller must funnel the exception into ``update_task_status(failed)``
    with a non-empty error and return ``True`` (the task *was* handled — it
    reached a terminal state). ``uploading`` must never be emitted because
    trim never produced an mp4 to upload."""
    from client_agent.poller import TaskPoller

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk = BufferChunk(path=tmp_path / "chunk_001.mp4", start=t10, end=t10 + timedelta(hours=1))
    platform = FakePlatform(
        next_tasks=[
            Task(
                id="task-trimfail",
                camera_id="cam-1",
                start_time=t10 + timedelta(minutes=15),
                end_time=t10 + timedelta(minutes=45),
            )
        ]
    )
    buffer = FakeBuffer(chunks_by_camera={"cam-1": [chunk]})

    def exploding_trim(*, chunks, start, end, output, runner):
        raise RuntimeError("ffmpeg exited 1: Invalid data found when processing input")

    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=exploding_trim,
        output_dir=tmp_path / "out",
        uploader=FakeUploader(),
    )
    handled = poller.run_once()

    assert handled is True
    # recording → failed; uploading never reached (no mp4 was produced).
    assert [call[1] for call in platform.status_calls] == ["recording", "failed"]
    failed_call = platform.status_calls[-1]
    assert failed_call[0] == "task-trimfail"
    assert failed_call[2] is not None  # operator-facing error present


# ----- 5d-bis. real trim raising on ffmpeg exit reaches platform as failed (#57) -----


def test_trim_failure_reaches_platform_as_failed(tmp_path: Path) -> None:
    """Composition test wiring #57 into #54: the *real* ``trim_and_concat``
    (not a hand-rolled exploding fake) is given a runner that returns a
    non-zero ffmpeg exit. trim must raise (the #57 fix), and the poller must
    funnel that into ``status=failed`` — never emitting ``uploading`` for a
    trim that produced no mp4, and never invoking the uploader."""
    from client_agent.ffmpeg_trim import trim_and_concat
    from client_agent.poller import TaskPoller

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk = BufferChunk(path=tmp_path / "chunk_001.mp4", start=t10, end=t10 + timedelta(hours=1))
    chunk.path.write_bytes(b"fake")
    platform = FakePlatform(
        next_tasks=[
            Task(
                id="task-trim-exit1",
                camera_id="cam-1",
                start_time=t10 + timedelta(minutes=15),
                end_time=t10 + timedelta(minutes=45),
            )
        ]
    )
    buffer = FakeBuffer(chunks_by_camera={"cam-1": [chunk]})

    def failing_runner(cmd, **kwargs):
        class _R:
            returncode = 1
            stdout = ""
            stderr = "Invalid data found when processing input"

        return _R()

    uploader = FakeUploader()
    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=trim_and_concat,
        output_dir=tmp_path / "out",
        uploader=uploader,
        runner=failing_runner,
    )
    handled = poller.run_once()

    assert handled is True
    # recording → failed; uploading never reached (trim produced no mp4).
    assert [call[1] for call in platform.status_calls] == ["recording", "failed"]
    assert platform.status_calls[-1][2] is not None  # operator-facing error present
    # The uploader was never asked to upload a missing/partial file.
    assert uploader.upload_calls == []


# ----- 5e. uploader raises → task marked failed (belt-and-suspenders, #54) -----


def test_uploader_exception_marks_task_failed(tmp_path: Path) -> None:
    """The uploader's contract is "no exceptions bubble" (it returns a
    failed :class:`UploadResult`), and #54's uploader fix restores that.
    But the poller must not *depend* on that contract holding — a future
    uploader bug (or any raise past its boundary) after ``status=uploading``
    would otherwise wedge the task at ``uploading`` (issue #54). This is the
    belt-and-suspenders twin of the trim-failure case: a raising uploader
    lands at the same ``failed`` funnel.

    Unlike the trim-failure path, ``uploading`` *is* emitted first (the trim
    succeeded and the poller announced the upload before the uploader blew
    up), so the transition sequence is recording → uploading → failed."""
    from client_agent.poller import TaskPoller

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk = BufferChunk(path=tmp_path / "chunk_001.mp4", start=t10, end=t10 + timedelta(hours=1))
    platform = FakePlatform(
        next_tasks=[
            Task(
                id="task-uploadraise",
                camera_id="cam-1",
                start_time=t10 + timedelta(minutes=15),
                end_time=t10 + timedelta(minutes=45),
            )
        ]
    )
    buffer = FakeBuffer(chunks_by_camera={"cam-1": [chunk]})

    def fake_trim(*, chunks, start, end, output, runner):
        output.write_bytes(b"trim-ok")

    class ExplodingUploader:
        def upload_chunks(self, task_id: str, chunks: list[Path]) -> list[object]:
            raise ConnectionError("[Errno 65] No route to host")

    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=fake_trim,
        output_dir=tmp_path / "out",
        uploader=ExplodingUploader(),
    )
    handled = poller.run_once()

    assert handled is True
    assert [call[1] for call in platform.status_calls] == ["recording", "uploading", "failed"]
    failed_call = platform.status_calls[-1]
    assert failed_call[0] == "task-uploadraise"
    assert failed_call[2] is not None


# ----- 6. run() loop is resilient to transient exceptions (Wi-Fi blip) -----


def test_run_loop_survives_transient_exception_in_run_once(tmp_path: Path) -> None:
    """``run_once`` can raise on a Wi-Fi blip (httpx ConnectError) or a
    platform 5xx hiccup — but the daemon thread MUST NOT die, or the
    appliance silently stops claiming tasks until the operator restarts
    the python process. Mirrors the heartbeat loop's try/except in
    appliance.py. Before the fix, the bare ``while True`` propagated the
    first exception out of the thread and the queue went unserved.

    Hits the loop in-process by replacing the thread's sleep with one that
    stops the loop after enough iterations to prove the loop did NOT die
    after the first throw."""
    from client_agent.poller import TaskPoller

    iterations: list[int] = []

    class Stop(Exception):
        pass

    def fake_sleep(_seconds: float) -> None:
        iterations.append(len(iterations))
        if len(iterations) >= 3:
            raise Stop()

    fetches = [0]

    class FlakyPlatform(FakePlatform):
        def fetch_next_task(self) -> Task | None:
            fetches[0] += 1
            if fetches[0] == 1:
                raise ConnectionError("[Errno 65] No route to host")
            return None

    poller = TaskPoller(
        platform=FlakyPlatform(),
        buffer=FakeBuffer(),
        trim_fn=lambda **kw: None,
        output_dir=tmp_path / "out",
        uploader=FakeUploader(),
        sleep=fake_sleep,
    )
    try:
        poller.run()
    except Stop:
        pass

    # The transient ConnectionError on iteration 1 must NOT have killed the
    # loop — iterations 2 and 3 must have happened (proven by fake_sleep
    # being called ≥ 3 times before we tripped the Stop sentinel).
    assert len(iterations) >= 3
    assert fetches[0] >= 3
