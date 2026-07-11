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

import io
from unittest.mock import MagicMock

import pytest

from gpu_service.http_client import HttpError, PresignedHttpClient
from gpu_service.http_retry import RetryExhausted


class _FakeResponse:
    """BytesIO-backed fake: supports both a bare ``read()`` (upload drains
    the body) and a bounded ``read(n)`` (download streams in buffers)."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self._buf = io.BytesIO(body)
        self.status = status

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _StreamingFakeResponse:
    """Serves its body in small network-sized chunks and records how each
    ``read()`` was invoked, so a test can prove ``download`` copies in a
    bounded buffer rather than one unbounded ``read()`` that would pull a
    multi-GB chunk into RAM (#56)."""

    def __init__(self, total: int, *, serve: int = 64 * 1024, status: int = 200) -> None:
        pattern = b"0123456789abcdef"
        self._data = (pattern * (total // len(pattern) + 1))[:total]
        self._pos = 0
        self._serve = serve
        self.status = status
        self.max_read_arg = 0
        self.unbounded_reads = 0

    @property
    def payload(self) -> bytes:
        return self._data

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            self.unbounded_reads += 1
            chunk = self._data[self._pos :]
            self._pos = len(self._data)
            return chunk
        self.max_read_arg = max(self.max_read_arg, n)
        end = min(self._pos + min(n, self._serve), len(self._data))
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _StatusProbeResponse:
    """Records whether the body was read — lets a test assert the status
    is checked *before* any body consumption (#56)."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.read_calls = 0

    def read(self, n: int = -1) -> bytes:
        self.read_calls += 1
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FailMidStreamResponse:
    """Serves a few chunks then raises a retriable error, simulating a
    connection dropping partway through a download."""

    def __init__(self, fail_after: int = 3, *, chunk: int = 4096) -> None:
        self.status = 200
        self._reads = 0
        self._fail_after = fail_after
        self._chunk = chunk

    def read(self, n: int = -1) -> bytes:
        self._reads += 1
        if self._reads > self._fail_after:
            raise ConnectionError("stream dropped")
        size = self._chunk if not n or n < 0 else min(n, self._chunk)
        return b"x" * size

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

    def test_download_streams_in_bounded_chunks(self, tmp_path) -> None:
        # 5 MB body served in 64 KiB network chunks. The client must copy it
        # into dest with a bounded buffer and never a single unbounded read()
        # that materializes the whole (multi-GB in prod) chunk in RAM (#56).
        fake = _StreamingFakeResponse(5 * 1024 * 1024)
        opener = MagicMock(return_value=fake)
        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)
        dest = tmp_path / "chunk.mp4"

        client.download("https://r2.example.com/chunk.mp4", dest)

        assert dest.read_bytes() == fake.payload
        # Never a bare read(); every read is bounded by the copy buffer.
        assert fake.unbounded_reads == 0
        assert 0 < fake.max_read_arg <= 1024 * 1024

    def test_download_checks_status_before_body(self, tmp_path) -> None:
        # A non-2xx (e.g. expired presigned URL) must raise before the body is
        # consumed — status check ordered ahead of read() (#56).
        fake = _StatusProbeResponse(status=403)
        opener = MagicMock(return_value=fake)
        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)

        with pytest.raises(HttpError):
            client.download("https://r2.example.com/missing.mp4", tmp_path / "d.mp4")

        assert fake.read_calls == 0

    def test_download_is_atomic_across_a_failed_retry(self, tmp_path) -> None:
        # First attempt dies mid-stream; the retry must produce a *complete*
        # file at dest via an atomic rename and leave no partial artifact.
        partial = _FailMidStreamResponse(fail_after=3)
        full = _StreamingFakeResponse(256 * 1024)
        opener = MagicMock(side_effect=[partial, full])
        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)
        dest = tmp_path / "chunk.mp4"

        client.download("https://r2.example.com/chunk.mp4", dest)

        assert dest.read_bytes() == full.payload
        assert list(tmp_path.glob("*.part")) == []

    def test_failed_download_leaves_no_partial_dest(self, tmp_path) -> None:
        # Every attempt dies mid-stream → RetryExhausted, and neither dest nor
        # a .part temp survives (a partial file must never masquerade as done).
        opener = MagicMock(side_effect=lambda _url: _FailMidStreamResponse(fail_after=2))
        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)
        dest = tmp_path / "chunk.mp4"

        with pytest.raises(RetryExhausted):
            client.download("https://r2.example.com/chunk.mp4", dest)

        assert not dest.exists()
        assert list(tmp_path.glob("*.part")) == []


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

    def test_puts_json_content_type_header(self) -> None:
        # REST mode PUTs the canonical result.json (#72); the object should
        # carry application/json content-type metadata instead of urllib's
        # default application/x-www-form-urlencoded (#75).
        captured: dict = {}

        def opener(request):
            # urllib capitalizes only the first letter of a header key, so the
            # stored key is "Content-type".
            captured["content_type"] = request.get_header("Content-type")
            return _FakeResponse(b"", status=200)

        client = PresignedHttpClient(opener=opener, sleep=_no_sleep)

        client.upload("https://r2.example.com/put/result.json", b'{"ok":true}')

        assert captured["content_type"] == "application/json"

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
