"""Task poller for the client appliance (issue #27, Slice 1c.2).

Sits between the platform's task queue and the local rolling buffer:

1. Pull the next task (``PlatformClient.fetch_next_task``).
2. Mark it ``recording`` (callback).
3. Locate the chunks in the buffer that overlap the task's
   ``[start_time, end_time]`` window.
4. Hand them to :func:`ffmpeg_trim.trim_and_concat` to produce a single
   trimmed MP4.
5. Mark the task ``uploading`` (callback). The actual presigned-URL
   upload lives in the next slice (gpu-exchange #22); this slice stops
   at "trim ready, ready for upload".

Failure modes (empty buffer, time range outside window) terminate at
``failed`` with a human-readable error in the callback body so the
platform UI can show the operator what went wrong.

Single-flight per MVP: the poller is one blocking thread; concurrent
tasks are out of scope (the recorder can only chase one camera at a
time anyway). The ``run()`` entrypoint is intentionally thin so the
appliance harness can wrap it with retry / shutdown semantics later
without dragging logic into this module."""

from __future__ import annotations

import logging
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from client_agent.buffer import BufferChunk
from client_agent.platform import Task

logger = logging.getLogger(__name__)


class _PlatformLike(Protocol):
    """Narrow surface of :class:`client_agent.platform.PlatformClient`
    the poller needs. Declared as a Protocol so tests can pass a fake
    without touching the real HTTP client or its retry semantics."""

    def fetch_next_task(self) -> Task | None: ...
    def update_task_status(
        self, task_id: str, *, status: str, error: str | None = None
    ) -> None: ...


class _BufferLike(Protocol):
    def chunks_in_range(
        self, camera_id: str, *, start: datetime, end: datetime
    ) -> list[BufferChunk]: ...

    # Optional: distinguishes "recorder never ran for this camera"
    # (empty buffer) from "recorder ran but nothing covers the window"
    # (stale / out-of-range). The poller uses the distinction to pick a
    # human-readable error message; implementations without it fall back
    # to the generic "buffer empty".
    def has_recorded(self, camera_id: str) -> bool: ...


class TaskPoller:
    """Single-flight task pump against the platform queue."""

    def __init__(
        self,
        *,
        platform: _PlatformLike,
        buffer: _BufferLike,
        trim_fn: Callable[..., None],
        output_dir: Path,
        poll_interval_s: int = 5,
        runner: Callable[..., Any] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._platform = platform
        self._buffer = buffer
        self._trim_fn = trim_fn
        self._output_dir = output_dir
        self._poll_interval_s = poll_interval_s
        self._runner = runner
        self._sleep = sleep

    def run_once(self) -> bool:
        """Execute one poll-claim-trim cycle.

        Returns ``True`` if a task was handled (success *or* failure),
        ``False`` if the queue was idle. The blocking ``run()`` loop
        uses the boolean to decide its inter-poll sleep."""
        task = self._platform.fetch_next_task()
        if task is None:
            return False

        self._platform.update_task_status(task.id, status="recording")

        chunks = self._buffer.chunks_in_range(
            task.camera_id, start=task.start_time, end=task.end_time
        )
        if not chunks:
            # Distinguish "recorder never ran" (no camera dir, no history)
            # from "recorder ran but nothing covers the window" so the
            # operator gets a fix-hint, not a generic error. ``has_recorded``
            # is best-effort: if the buffer impl doesn't expose it we fall
            # back to the generic message.
            has_history = bool(
                getattr(self._buffer, "has_recorded", lambda _c: False)(task.camera_id)
            )
            error = "time range outside buffer" if has_history else "buffer empty"
            self._platform.update_task_status(task.id, status="failed", error=error)
            return True

        self._output_dir.mkdir(parents=True, exist_ok=True)
        output = self._output_dir / f"{task.id}-{uuid.uuid4().hex[:8]}.mp4"
        self._trim_fn(
            chunks=chunks,
            start=task.start_time,
            end=task.end_time,
            output=output,
            runner=self._runner,
        )

        self._platform.update_task_status(task.id, status="uploading")
        return True

    def run(self) -> None:
        """Blocking poll loop — production entrypoint.

        Iterates ``run_once`` forever, sleeping ``poll_interval_s`` only
        when the queue was idle (so a backlog drains as fast as the
        appliance can trim). Kept out of unit tests because verifying a
        ``while True`` would require a fake clock + interrupt protocol
        that exists nowhere else in the codebase; the smoke test against
        mediamtx exercises this loop end-to-end."""
        while True:
            handled = self.run_once()
            if not handled:
                self._sleep(self._poll_interval_s)
