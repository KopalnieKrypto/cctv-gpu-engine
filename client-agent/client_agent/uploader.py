"""Presigned-URL upload manager for the client appliance (issues #28, #128).

Sits between :class:`client_agent.poller.TaskPoller` (which drops a
trimmed chunk on local disk) and Cloudflare R2. Holds **no R2
credentials**: every PUT goes through a fresh presigned URL fetched
on demand from the platform. The platform-side issuer binds each URL
to ``tenants/{tid}/appliance-uploads/{task_id}/chunk_N.mp4``, so a
compromised appliance with a valid Bearer token still cannot scribble
outside its task scope (privacy boundary per DD-09).

A chunk goes up as an R2 multipart upload (gpu-exchange#248): R2 takes
at most 5 GiB in one PUT, and 3 h from a 4K camera is 6-7 GB. Parts of
``upload_chunk_bytes`` are read straight from the trimmed file and
retried one by one; on completion R2 joins them into one object under
the key, so nothing downstream sees the split.

Public surface is two methods:

* :meth:`PresignedUploader.upload_chunk` — one chunk, retry-aware per
  part, refresh-on-expiry, returns a :class:`UploadResult` (the poller
  needs to mark the task ``failed`` cleanly).
* :meth:`PresignedUploader.upload_chunks` — many chunks in parallel
  via a :class:`ThreadPoolExecutor`. Results come back in the input
  order regardless of completion order so the poller's error-summary
  is stable.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import httpx

from client_agent.platform import MultipartUpload, PlatformClient, PlatformRequestError

logger = logging.getLogger(__name__)

# Mirror :data:`client_agent.platform._DEFAULT_BACKOFFS`: 3 PUT attempts
# with 1s/2s sleeps between them. Kept as a module constant (not a class
# arg) because no caller has ever needed a different schedule and the
# parallel symmetry with the platform client is itself documentation.
_PUT_BACKOFFS: tuple[int, ...] = (1, 2)
_PUT_ATTEMPTS = 3
# Generous per-phase timeout for R2 PUT — parts are tens of MB and the
# appliance often sits on a residential/asymmetric uplink. httpx default
# is 5s which is essentially "always times out". Write=300s lets a
# ~200MB part land on a 6Mbps uplink (the slowest realistic case).
_PUT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=300.0, pool=10.0)

# Cold-start default for the platform-delivered ``upload_chunk_bytes`` (#85),
# the multipart part size: 50 MiB, matching the platform's default.
_DEFAULT_UPLOAD_CHUNK_BYTES = 52_428_800
# R2's multipart bounds: every part but the last at least 5 MiB, all but the
# last the same size, at most 10 000 parts.
_MIN_PART_BYTES = 5 * 1024 * 1024
_MAX_PARTS = 10_000

# How often the panel's upload bar may move (#127).
_PROGRESS_INTERVAL_S = 10.0

_R2_ERROR_CODE = re.compile(r"<Code>([^<]+)</Code>")


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        # A missing chunk fails in upload_chunk with its usual error; the
        # progress bar just has nothing to measure.
        return 0


def _r2_error_code(response: httpx.Response) -> str:
    """R2's S3 error code (``EntityTooLarge``, ``InvalidPart``) with a
    leading space, or nothing - the status alone does not say why."""
    match = _R2_ERROR_CODE.search(response.text)
    return f" {match.group(1)}" if match else ""


class _PartReader:
    """One part of a chunk file, read from the handle's current position,
    counting the bytes httpx pulls for the PUT.

    No ``fileno`` on purpose: httpx would size the body from the whole file.
    The caller sends the part's own Content-Length instead — R2 refuses a
    chunked presigned PUT. ``__iter__`` is what makes httpx accept the
    object as a stream; it then pulls through ``read``."""

    def __init__(self, fh: BinaryIO, length: int, on_read: Callable[[int], None]) -> None:
        self._fh = fh
        self._left = length
        self._on_read = on_read

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > self._left:
            size = self._left
        data = self._fh.read(size)
        self._left -= len(data)
        if data:
            self._on_read(len(data))
        return data

    def __iter__(self) -> Iterator[bytes]:
        return iter(lambda: self.read(65_536), b"")


class _PartFailed(Exception):
    """A part that cannot go up; the message becomes the chunk's error."""


class _UploadProgress:
    """Bytes sent across one task's chunks, reported to the platform at most
    once per interval (#127).

    Cosmetic by contract: a report that fails (or raises) switches reporting
    off for the rest of the upload and never touches the upload's result."""

    def __init__(
        self,
        *,
        report: Callable[[float], bool],
        total_bytes: int,
        clock: Callable[[], float],
    ) -> None:
        self._report = report
        self._total_bytes = total_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._sent: dict[tuple[int, int], int] = {}
        # Counting from now makes the first report wait a full interval — a 0%
        # the moment the PUT starts reads in the panel as a stalled task.
        self._last_report = clock()
        self._enabled = total_bytes > 0

    def restart(self, part: tuple[int, int]) -> None:
        """A retried or refreshed PUT re-reads the part from its first byte."""
        with self._lock:
            self._sent[part] = 0

    def add(self, part: tuple[int, int], nbytes: int) -> None:
        with self._lock:
            self._sent[part] = self._sent.get(part, 0) + nbytes
            now = self._clock()
            if not self._enabled or now - self._last_report < _PROGRESS_INTERVAL_S:
                return
            self._last_report = now
            pct = min(100.0, round(sum(self._sent.values()) * 100 / self._total_bytes, 1))
        # Reported outside the lock so the other chunk threads keep streaming.
        try:
            accepted = self._report(pct)
        except Exception:  # noqa: BLE001
            logger.warning("upload progress report raised; continuing without it", exc_info=True)
            accepted = False
        if not accepted:
            with self._lock:
                self._enabled = False


@dataclass(frozen=True)
class UploadResult:
    """Outcome of one chunk's upload.

    Held as a value object rather than an exception so partial failures
    in :meth:`PresignedUploader.upload_chunks` are addressable as data
    — the poller composes a single ``status=failed`` error from the
    list of failed chunks rather than catching mid-batch."""

    chunk_n: int
    success: bool
    key: str | None = None
    error: str | None = None


class PresignedUploader:
    """Upload chunks to R2 as multipart uploads through platform-issued URLs."""

    def __init__(
        self,
        *,
        platform: PlatformClient,
        sleep: Callable[[float], None] = time.sleep,
        max_workers: int = 4,
        http_put: Callable[..., httpx.Response] = httpx.put,
        upload_chunk_bytes: int = _DEFAULT_UPLOAD_CHUNK_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._platform = platform
        self._sleep = sleep
        self._max_workers = max_workers
        self._clock = clock
        # The platform-delivered part size (#85). Public because the
        # runtime-config applier and the tests read it directly.
        self.upload_chunk_bytes = upload_chunk_bytes
        # Injectable so tests can simulate transport failures deterministically.
        # Production default is ``httpx.put`` (respx patches the transport for
        # the URL-pattern tests, so the default flows through respx unchanged).
        self._http_put = http_put

    def set_upload_chunk_bytes(self, nbytes: int) -> None:
        """Re-point the part size at runtime (issue #85).

        The platform ships ``upload_chunk_bytes`` on every register/heartbeat;
        the runtime-config applier calls this on-change. An upload in flight
        keeps the size it started with — R2 wants every part but the last the
        same size. Plain assignment is atomic in CPython, so no lock for the
        cross-thread write."""
        self.upload_chunk_bytes = nbytes

    def upload_chunks(self, task_id: str, chunks: list[Path]) -> list[UploadResult]:
        """Upload many chunks in parallel; results return in input order.

        Submits one task per chunk to a :class:`ThreadPoolExecutor` of
        ``max_workers`` size. Result list mirrors the input list's order
        (using ``executor.map`` on an enumerated input — futures complete
        in any order, but ``map`` yields them in submission order). The
        poller relies on the order match to attribute failures to chunks
        by index.

        Each chunk goes through the same :meth:`upload_chunk` path with
        its own retry / refresh budget — a failure on chunk 2 does *not*
        cancel chunks 0/1 or 3+, so the operator gets the fullest
        possible diagnostic in the final ``status=failed`` payload."""

        progress = self._progress_for(task_id, chunks)
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            return list(
                ex.map(
                    lambda pair: self.upload_chunk(task_id, pair[0], pair[1], progress=progress),
                    list(enumerate(chunks)),
                )
            )

    def _progress_for(self, task_id: str, chunks: list[Path]) -> _UploadProgress:
        return _UploadProgress(
            report=lambda pct: self._platform.report_task_progress(task_id, pct),
            total_bytes=sum(_file_size(path) for path in chunks),
            clock=self._clock,
        )

    def upload_chunk(
        self,
        task_id: str,
        chunk_n: int,
        local_path: Path,
        *,
        progress: _UploadProgress | None = None,
    ) -> UploadResult:
        """Upload one chunk as an R2 multipart upload; return the result.

        Parts go up one after another, each through a presigned URL fetched
        right before it. Retry policy per part: ``_PUT_ATTEMPTS`` PUTs with
        ``_PUT_BACKOFFS`` (1s/2s) sleeps on a 5xx or a transport error. A
        ``403 SignatureDoesNotMatch`` refetches the part's URL once (stale
        key or expiry); a second one is a configuration bug. Any other
        status is terminal. A failed upload is aborted, so R2 drops the
        parts it already holds instead of keeping them for 7 days."""
        try:
            upload = self._platform.open_multipart_upload(task_id, chunk_n)
        except PlatformRequestError as exc:
            # Tenant isolation / unknown task — the platform opened nothing,
            # so there is nothing to PUT and nothing to abort.
            return UploadResult(chunk_n=chunk_n, success=False, error=str(exc))
        tracker = progress if progress is not None else self._progress_for(task_id, [local_path])
        size = _file_size(local_path)
        part_bytes = max(self.upload_chunk_bytes, _MIN_PART_BYTES, -(-size // _MAX_PARTS))
        etags: list[tuple[int, str]] = []
        try:
            # max(size, 1): an empty file still goes up, as one empty part.
            for part_number, offset in enumerate(range(0, max(size, 1), part_bytes), start=1):
                length = min(part_bytes, size - offset)
                etag = self._put_part(
                    task_id, chunk_n, upload, part_number, local_path, offset, length, tracker
                )
                etags.append((part_number, etag))
            self._platform.complete_multipart_upload(task_id, chunk_n, upload, etags)
        except (_PartFailed, PlatformRequestError) as exc:
            self._abort(task_id, chunk_n, upload)
            return UploadResult(chunk_n=chunk_n, success=False, error=str(exc))
        except Exception:
            self._abort(task_id, chunk_n, upload)
            raise
        return UploadResult(chunk_n=chunk_n, success=True, key=upload.key)

    def _put_part(
        self,
        task_id: str,
        chunk_n: int,
        upload: MultipartUpload,
        part_number: int,
        path: Path,
        offset: int,
        length: int,
        tracker: _UploadProgress,
    ) -> str:
        """PUT one part under :meth:`upload_chunk`'s retry policy; return its ETag."""
        part = (chunk_n, part_number)
        url = self._part_url(task_id, chunk_n, upload, part_number)
        refreshed = False
        while True:
            last: httpx.Response | None = None
            last_error = ""
            attempts = 0
            for attempts in range(1, _PUT_ATTEMPTS + 1):
                try:
                    # Stream the part straight off disk — a multi-GB chunk on
                    # a mini-PC must not be read whole into RAM (#56). Reopen
                    # per attempt so a retry starts from the part's first byte.
                    with path.open("rb") as fh:
                        fh.seek(offset)
                        tracker.restart(part)
                        last = self._http_put(
                            url,
                            content=_PartReader(fh, length, lambda n: tracker.add(part, n)),
                            headers={"Content-Length": str(length)},
                            timeout=_PUT_TIMEOUT,
                        )
                except httpx.HTTPError as exc:
                    # A transport error (ConnectError / ReadError / ReadTimeout
                    # from a Wi-Fi blip) counts against the same 3-attempt
                    # budget as a 5xx (issue #54).
                    last = None
                    last_error = f"transport error: {exc}"
                else:
                    if last.status_code < 500:
                        break
                if attempts < _PUT_ATTEMPTS:
                    self._sleep(_PUT_BACKOFFS[min(attempts, len(_PUT_BACKOFFS)) - 1])

            if last is None:
                raise _PartFailed(f"part {part_number}: {last_error} after {attempts} attempt(s)")
            if last.status_code == 403 and "SignatureDoesNotMatch" in last.text and not refreshed:
                url = self._part_url(task_id, chunk_n, upload, part_number)
                refreshed = True
                continue
            if 200 <= last.status_code < 300:
                etag = last.headers.get("etag")
                if not etag:
                    raise _PartFailed(
                        f"part {part_number}: R2 PUT returned {last.status_code} without an ETag"
                    )
                return etag
            raise _PartFailed(
                f"part {part_number}: R2 PUT returned {last.status_code}"
                f"{_r2_error_code(last)} after {attempts} attempt(s)"
            )

    def _part_url(
        self, task_id: str, chunk_n: int, upload: MultipartUpload, part_number: int
    ) -> str:
        try:
            return self._platform.get_upload_part_url(task_id, chunk_n, upload, part_number).url
        except PlatformRequestError as exc:
            raise _PartFailed(f"part {part_number}: {exc}") from exc

    def _abort(self, task_id: str, chunk_n: int, upload: MultipartUpload) -> None:
        try:
            self._platform.abort_multipart_upload(task_id, chunk_n, upload)
        except Exception:  # noqa: BLE001
            # Best effort: R2 drops an unfinished upload after 7 days anyway,
            # and the task's error must name the original failure.
            logger.warning("aborting the upload of task %s failed", task_id, exc_info=True)
