"""Tests for the urllib-backed HTTP client used against presigned URLs.

We test:

* ``download`` streams to disk and re-tries through the retry decorator
  (AC #2 — fail 2× then success).
* ``upload`` PUTs bytes and retries identically (AC #3).
* ``upload`` raises a useful error on non-2xx responses.

The HTTP transport is injected (``opener``) so the suite never opens a real
socket — keeps the tests on macOS under 50 ms total.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpu_service.http_client import HttpError, PresignedHttpClient
from gpu_service.http_retry import RetryExhausted


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _no_sleep(_seconds: float) -> None:
    return None


class TestPresignedHttpClientDownload:
    def test_writes_response_body_to_destination(self, tmp_path) -> None:
        opener = MagicMock(return_value=_FakeResponse(b"VIDEO_BYTES"))
        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)
        dest = tmp_path / "chunk.mp4"

        client.download("https://r2.example.com/chunk.mp4", dest)

        assert dest.read_bytes() == b"VIDEO_BYTES"
        assert opener.call_count == 1

    def test_retries_through_exp_backoff_then_succeeds(self, tmp_path) -> None:
        opener = MagicMock(
            side_effect=[
                ConnectionError("flap"),
                ConnectionError("flap"),
                _FakeResponse(b"OK"),
            ]
        )
        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)
        dest = tmp_path / "chunk.mp4"

        client.download("https://r2.example.com/chunk.mp4", dest)

        assert dest.read_bytes() == b"OK"
        assert opener.call_count == 3

    def test_raises_retry_exhausted_after_three_failures(self, tmp_path) -> None:
        opener = MagicMock(side_effect=ConnectionError("dead"))
        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)
        dest = tmp_path / "chunk.mp4"

        with pytest.raises(RetryExhausted):
            client.download("https://r2.example.com/chunk.mp4", dest)

        assert opener.call_count == 3


class TestPresignedHttpClientUpload:
    def test_puts_bytes_to_presigned_url(self) -> None:
        captured: dict = {}

        def opener(request):
            # urllib.Request stores body in `data` and method in `method`.
            captured["url"] = request.full_url
            captured["data"] = request.data
            captured["method"] = request.get_method()
            return _FakeResponse(b"", status=200)

        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)

        client.upload("https://r2.example.com/put/report.html", b"<html>ok</html>")

        assert captured["url"] == "https://r2.example.com/put/report.html"
        assert captured["data"] == b"<html>ok</html>"
        assert captured["method"] == "PUT"

    def test_raises_on_non_2xx_response(self) -> None:
        opener = MagicMock(return_value=_FakeResponse(b"forbidden", status=403))
        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)

        with pytest.raises(HttpError):
            client.upload("https://r2.example.com/put/x.html", b"x")

    def test_retries_then_succeeds(self) -> None:
        opener = MagicMock(
            side_effect=[
                ConnectionError("flap"),
                ConnectionError("flap"),
                _FakeResponse(b"", status=200),
            ]
        )
        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)

        client.upload("https://r2.example.com/put/x.html", b"x")

        assert opener.call_count == 3
