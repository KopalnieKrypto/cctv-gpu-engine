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
