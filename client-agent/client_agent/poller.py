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
        self,
        task_id: str,
        *,
        status: str,
        error: str | None = None,
        actual_start: datetime | None = None,
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


class _UploaderLike(Protocol):
    """Narrow surface of :class:`client_agent.uploader.PresignedUploader`.

    Only :meth:`upload_chunks` is used at the poller boundary — we hand
    over a list of trimmed mp4 paths and trust the uploader's retry /
    refresh-on-expiry logic. The result objects only need a ``success``
    bool and ``error`` string for our purposes; declared as ``object``
    in the return type so this Protocol does not import from uploader
    (and create an import cycle)."""

    def upload_chunks(self, task_id: str, chunks: list[Path]) -> list[Any]: ...


class TaskPoller:
    """Single-flight task pump against the platform queue."""

    def __init__(
        self,
        *,
        platform: _PlatformLike,
        buffer: _BufferLike,
        trim_fn: Callable[..., datetime | None],
        output_dir: Path,
        uploader: _UploaderLike,
        poll_interval_s: int = 5,
        runner: Callable[..., Any] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._platform = platform
        self._buffer = buffer
        self._trim_fn = trim_fn
        self._output_dir = output_dir
        self._uploader = uploader
        self._poll_interval_s = poll_interval_s
        self._runner = runner
        self._sleep = sleep

    def set_poll_interval_s(self, seconds: int) -> None:
        """Re-time the idle backoff at runtime (issue #85).

        The poller runs one long-lived ``run()`` loop, so an admin editing
        ``polling_interval_seconds`` in the platform UI changes the interval
        in place — the next idle iteration sleeps the new value. A plain int
        assignment is atomic in CPython, so no lock is needed for the
        cross-thread write (the runtime-config applier runs on the heartbeat
        thread); worst case the change lands one iteration late."""
        self._poll_interval_s = seconds

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
        try:
            # The trim returns where the delivered clip *actually* starts.
            # It equals ``task.start_time`` whenever the buffer covered the
            # window, and the oldest chunk's start when the ``-ss`` clamp
            # shortened it (#91).
            actual_start = self._trim_fn(
                chunks=chunks,
                start=task.start_time,
                end=task.end_time,
                output=output,
                runner=self._runner,
            )

            self._platform.update_task_status(task.id, status="uploading")
            results = self._uploader.upload_chunks(task.id, [output])
            failed = [r for r in results if not getattr(r, "success", False)]
            if failed:
                # Aggregate every failed chunk's error into one operator-
                # facing message. Keeps the platform-side error column
                # bounded but loses nothing diagnostic — each chunk_n is
                # tagged so the operator can spot a "chunk 7 always 503"
                # pattern across retries.
                error = "; ".join(
                    f"chunk {getattr(r, 'chunk_n', '?')}: {getattr(r, 'error', 'unknown')}"
                    for r in failed
                )
                self._platform.update_task_status(task.id, status="failed", error=error)
                return True

            # The platform's `uploaded` status variant requires `chunk_r2_key`
            # (the R2 key the appliance just PUT into). For MVP we trim into a
            # single concatenated chunk before upload, so results[0].key is the
            # whole task's payload. When upload_chunks is expanded to multi-
            # chunk batches the platform's schema also gains a multi-key shape;
            # for now `results[0].key` is the contract.
            #
            # ``actual_start`` rides along on the same call: it describes the
            # artifact just uploaded, so the platform can stamp
            # ``recording_start`` from what was delivered instead of what was
            # requested (#91 / gpu-exchange#154). ``None`` (a trim_fn that
            # predates the return value) simply omits the field, and the
            # platform falls back to ``start_time`` as before.
            uploaded_key = getattr(results[0], "key", None) if results else None
            self._platform.update_task_status(
                task.id,
                status="uploaded",
                chunk_r2_key=uploaded_key,
                actual_start=actual_start,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            # Any exception between status=recording and a terminal status
            # (ffmpeg crash in trim_fn, an uploader that raises instead of
            # returning a failed result) must flip the platform-side task to
            # ``failed`` — otherwise it wedges in recording/uploading forever
            # and only a human can unstick it (issue #54). Same wedge family
            # as the #42 timeout fix, one layer up. The task is still
            # "handled" (it reached a terminal state), so return True.
            self._platform.update_task_status(
                task.id, status="failed", error=f"task processing failed: {exc}"
            )
            return True
        finally:
            # Whatever the outcome, the trimmed mp4 has no second local use:
            # on success it is already in R2, on failure the platform re-queues
            # the task. Leaving it behind fills the appliance disk one task at a
            # time (issue #51). ``missing_ok`` covers the trim-never-wrote case.
            output.unlink(missing_ok=True)

    def run(self) -> None:
        """Blocking poll loop — production entrypoint.

        Iterates ``run_once`` forever, sleeping ``poll_interval_s`` only
        when the queue was idle (so a backlog drains as fast as the
        appliance can trim). Kept out of unit tests because verifying a
        ``while True`` would require a fake clock + interrupt protocol
        that exists nowhere else in the codebase; the smoke test against
        mediamtx exercises this loop end-to-end.

        Wraps ``run_once`` in a broad ``except`` so transient network errors
        (httpx ``ConnectError`` / ``ReadError`` from a Wi-Fi blip, DNS hiccups,
        platform 5xx) don't kill the daemon thread. Mirrors the heartbeat
        loop's ``except`` block in ``appliance.py``. Before this guard, a
        single transient failure would leave the appliance heartbeating
        normally but never polling tasks again until the operator
        restarted the python process."""
        while True:
            try:
                handled = self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("task poller iteration failed: %s", exc)
                handled = False
            if not handled:
                self._sleep(self._poll_interval_s)
