"""Presigned-URL upload manager for the client appliance (issue #28).

Sits between :class:`client_agent.poller.TaskPoller` (which drops a
trimmed chunk on local disk) and Cloudflare R2. Holds **no R2
credentials**: every PUT goes through a fresh presigned URL fetched
on demand from the platform. The platform-side issuer binds each URL
to ``tenants/{tid}/appliance-uploads/{task_id}/chunk_N.mp4``, so a
compromised appliance with a valid Bearer token still cannot scribble
outside its task scope (privacy boundary per DD-09).

Public surface is two methods:

* :meth:`PresignedUploader.upload_chunk` — one chunk, retry-aware,
  refresh-on-expiry, returns a :class:`UploadResult` (no exceptions
  bubble — the poller needs to mark the task ``failed`` cleanly).
* :meth:`PresignedUploader.upload_chunks` — many chunks in parallel
  via a :class:`ThreadPoolExecutor`. Results come back in the input
  order regardless of completion order so the poller's error-summary
  is stable.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from client_agent.platform import PlatformClient, PlatformRequestError

# Mirror :data:`client_agent.platform._DEFAULT_BACKOFFS`: 3 PUT attempts
# with 1s/2s sleeps between them. Kept as a module constant (not a class
# arg) because no caller has ever needed a different schedule and the
# parallel symmetry with the platform client is itself documentation.
_PUT_BACKOFFS: tuple[int, ...] = (1, 2)
_PUT_ATTEMPTS = 3


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
    """Upload chunks to R2 through platform-issued presigned PUT URLs."""

    def __init__(
        self,
        *,
        platform: PlatformClient,
        sleep: Callable[[float], None] = time.sleep,
        max_workers: int = 4,
    ) -> None:
        self._platform = platform
        self._sleep = sleep
        self._max_workers = max_workers

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

        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            return list(
                ex.map(
                    lambda pair: self.upload_chunk(task_id, pair[0], pair[1]),
                    list(enumerate(chunks)),
                )
            )

    def upload_chunk(self, task_id: str, chunk_n: int, local_path: Path) -> UploadResult:
        """Fetch a presigned URL, PUT the chunk with retries, return result.

        Retry policy: ``_PUT_ATTEMPTS`` total PUT attempts with
        ``_PUT_BACKOFFS`` (1s/2s) sleeps between them on any 5xx. A 5xx
        does *not* trigger a URL refresh — the URL is fine, the backend
        is sick. ``403 SignatureDoesNotMatch`` is special-cased: the URL
        has expired or been signed by a stale key, so we refetch once
        from the platform and try again with the fresh URL. A second
        SignatureDoesNotMatch on the refreshed URL is terminal (a
        configuration bug, not transient expiry)."""
        try:
            upload_url = self._platform.get_upload_url(task_id, chunk_n)
        except PlatformRequestError as exc:
            # Tenant isolation / unknown task / bad chunk_n — the platform
            # already decided not to issue a URL, so there is nothing to
            # PUT. Surface a failed result so the poller can mark the
            # task ``failed`` with a useful error.
            return UploadResult(
                chunk_n=chunk_n,
                success=False,
                error=f"platform refused upload-url ({exc.status_code})",
            )
        body = local_path.read_bytes()

        refreshed = False
        while True:
            last: httpx.Response | None = None
            for i in range(_PUT_ATTEMPTS):
                last = httpx.put(upload_url.url, content=body)
                if last.status_code < 500:
                    break
                if i + 1 >= _PUT_ATTEMPTS:
                    break
                self._sleep(_PUT_BACKOFFS[i] if i < len(_PUT_BACKOFFS) else _PUT_BACKOFFS[-1])

            assert last is not None
            if last.status_code == 403 and "SignatureDoesNotMatch" in last.text and not refreshed:
                # One-shot refresh: the URL expired between issuance and
                # this PUT. Most likely on the tail end of a slow parallel
                # batch. Re-fetch and reset the retry counter.
                upload_url = self._platform.get_upload_url(task_id, chunk_n)
                refreshed = True
                continue
            if 200 <= last.status_code < 300:
                return UploadResult(chunk_n=chunk_n, success=True, key=upload_url.key)
            return UploadResult(
                chunk_n=chunk_n,
                success=False,
                error=f"R2 PUT returned {last.status_code} after {_PUT_ATTEMPTS} attempt(s)",
            )
