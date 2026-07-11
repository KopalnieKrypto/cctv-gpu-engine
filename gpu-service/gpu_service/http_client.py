"""Presigned-URL HTTP client (issue #25 AC #2 / #3).

Thin urllib wrapper for the two operations the task runner performs:

* ``download(url, dest)`` — GET, stream the body into ``dest``.
* ``upload(url, body)`` — PUT, send ``body`` as the request data.

Both methods retry through :func:`gpu_service.http_retry.with_retry` (3
attempts, 1s/2s backoff). The HTTP transport is injected (``opener``) so
tests never open a real socket; production passes :func:`urllib.request.urlopen`.

We use stdlib urllib intentionally — no new dependency, and presigned URLs
need nothing more than a single GET/PUT with no auth headers.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gpu_service.http_retry import with_retry

# Exceptions we treat as retriable. URLError / HTTPError cover most
# transient network and 5xx conditions; ConnectionError catches socket-level
# drops; TimeoutError is for stuck reads.
RETRIABLE: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    urllib.error.URLError,
)

Opener = Callable[..., Any]


class HttpError(RuntimeError):
    """Raised on a non-2xx HTTP response (e.g. presigned URL expired)."""


class PresignedHttpClient:
    """Implements :class:`gpu_service.task_runner.HttpClientLike` over urllib."""

    def __init__(
        self,
        *,
        opener: Opener = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._opener = opener
        self._sleep = sleep

    def download(self, url: str, dest: Path) -> None:
        def _op() -> None:
            with self._opener(url) as response:
                body = response.read()
                _ensure_2xx(response, url)
            dest.write_bytes(body)

        with_retry(_op, sleep=self._sleep, retry_on=RETRIABLE)

    def upload(self, url: str, body: bytes) -> None:
        def _op() -> None:
            # REST mode PUTs the canonical result.json (#72). Tag the object
            # with application/json so R2 stores correct content-type metadata
            # instead of urllib's default application/x-www-form-urlencoded
            # (#75). The presigned PUT doesn't sign content-type, so this is
            # safe metadata-only cleanup.
            request = urllib.request.Request(
                url,
                data=body,
                method="PUT",
                headers={"Content-Type": "application/json"},
            )
            with self._opener(request) as response:
                # Drain the body so the connection can be reused / closed
                # cleanly on R2's side.
                response.read()
                _ensure_2xx(response, url)

        with_retry(_op, sleep=self._sleep, retry_on=RETRIABLE)


def _ensure_2xx(response: Any, url: str) -> None:
    status = getattr(response, "status", None)
    if status is None:
        # urllib's old-style response: status code via getcode()
        getter = getattr(response, "getcode", None)
        status = getter() if callable(getter) else 200
    if status < 200 or status >= 300:
        raise HttpError(f"HTTP {status} for {url}")
