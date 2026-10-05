"""Tests for the presigned-URL upload manager (issues #28, #128).

The uploader sits between :class:`TaskPoller` (which produces trimmed
chunks on local disk) and Cloudflare R2 (the durable store the
gpu-service worker pulls from). It deliberately holds **no R2
credentials**: every PUT goes through a fresh presigned URL issued by
the platform on demand, bound to
``tenants/{tid}/appliance-uploads/{task_id}/chunk_N.mp4``.

A chunk goes up as an R2 multipart upload (gpu-exchange#248): R2 takes
at most 5 GiB in one PUT. The platform opens and completes the upload;
each part travels straight to R2 through its own presigned URL.

Tests are hermetic — respx mocks the platform's multipart endpoints and
the presigned PUTs (intercepted by URL pattern). Sleep is injected so
retry tests run in microseconds, and the executor's parallelism is
exercised with a ``threading.Barrier`` rather than wallclock timing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import respx

from client_agent.platform import PlatformClient
from client_agent.uploader import PresignedUploader, UploadResult

PLATFORM = "https://platform.example"
MIB = 1024 * 1024
R2_PUT = r"https://r2\.example/.*"


def _key(chunk_n: int) -> str:
    return f"tenants/t-7/appliance-uploads/task-1/chunk_{chunk_n:03d}.mp4"


def _part_url(chunk_n: int, part_number: int, attempt: int = 1) -> str:
    return f"https://r2.example/c{chunk_n}/p{part_number}?sig={attempt}"


def _r2_ok(request: httpx.Request) -> httpx.Response:
    # R2 answers UploadPart with the part's MD5 as a quoted ETag header.
    return httpx.Response(200, headers={"ETag": f'"md5-{request.url.path.rsplit("/", 1)[-1]}"'})


def _mock_platform(mock: respx.MockRouter) -> dict[str, respx.Route]:
    """The platform's multipart endpoints for chunks of task ``task-1``:
    upload ``up-N`` under ``_key(N)``, part URLs from :func:`_part_url`."""

    def open_upload(request: httpx.Request) -> httpx.Response:
        chunk_n = json.loads(request.content)["chunk_n"]
        return httpx.Response(200, json={"upload_id": f"up-{chunk_n}", "key": _key(chunk_n)})

    def part_url(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        url = _part_url(int(params["chunk_n"]), int(params["part_number"]))
        return httpx.Response(200, json={"url": url, "expires_in": 1800})

    def complete(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"key": _key(json.loads(request.content)["chunk_n"])})

    return {
        "open": mock.post(f"{PLATFORM}/appliance/upload-multipart").mock(side_effect=open_upload),
        "part_url": mock.get(f"{PLATFORM}/appliance/upload-part-url").mock(side_effect=part_url),
        "complete": mock.post(f"{PLATFORM}/appliance/upload-multipart/complete").mock(
            side_effect=complete
        ),
        "abort": mock.post(f"{PLATFORM}/appliance/upload-multipart/abort").mock(
            return_value=httpx.Response(204)
        ),
    }


def _body(route: respx.Route, i: int = -1) -> dict:
    return json.loads(route.calls[i].request.content)


def _uploader(**kwargs: object) -> PresignedUploader:
    platform = PlatformClient(base_url=PLATFORM, token="tok-abc", sleep=lambda _s: None)
    kwargs.setdefault("sleep", lambda _s: None)
    return PresignedUploader(platform=platform, **kwargs)  # type: ignore[arg-type]


# ----- 1. tracer bullet: the chunk goes up in parts and R2 joins them -----


def test_chunk_goes_up_in_parts_that_r2_joins_under_one_key(tmp_path: Path) -> None:
    """11 MiB at 5 MiB a part is three PUTs of 5, 5 and 1 MiB, each to its
    own presigned URL with **no Bearer** (the URL carries its signature) and
    its own Content-Length (R2 refuses a chunked presigned PUT). The
    platform then completes the upload with every part's ETag, in order,
    and the result carries the key the GPU side will read."""
    chunk_path = tmp_path / "clip.mp4"
    body = bytes(range(256)) * (11 * MIB // 256)
    chunk_path.write_bytes(body)

    with respx.mock(assert_all_called=False) as mock:
        platform = _mock_platform(mock)
        put = mock.put(url__regex=R2_PUT).mock(side_effect=_r2_ok)
        uploader = _uploader()
        uploader.set_upload_chunk_bytes(5 * MIB)

        result = uploader.upload_chunk("task-1", 0, chunk_path)

    assert result == UploadResult(chunk_n=0, success=True, key=_key(0))
    assert _body(platform["open"]) == {"task_id": "task-1", "chunk_n": 0}
    assert platform["open"].calls.last.request.headers["authorization"] == "Bearer tok-abc"
    assert [call.request.url.params["part_number"] for call in platform["part_url"].calls] == [
        "1",
        "2",
        "3",
    ]
    assert {call.request.url.params["upload_id"] for call in platform["part_url"].calls} == {"up-0"}

    slices = [body[: 5 * MIB], body[5 * MIB : 10 * MIB], body[10 * MIB :]]
    assert [str(call.request.url) for call in put.calls] == [_part_url(0, n) for n in (1, 2, 3)]
    for call, expected in zip(put.calls, slices, strict=True):
        assert call.request.headers.get("authorization") is None
        assert call.request.headers["content-length"] == str(len(expected))
        assert "transfer-encoding" not in call.request.headers
        assert call.request.content == expected

    assert _body(platform["complete"]) == {
        "task_id": "task-1",
        "chunk_n": 0,
        "upload_id": "up-0",
        "parts": [
            {"part_number": 1, "etag": '"md5-p1"'},
            {"part_number": 2, "etag": '"md5-p2"'},
            {"part_number": 3, "etag": '"md5-p3"'},
        ],
    }
    assert not platform["abort"].called


# ----- 1b. parts stream off disk rather than into RAM -----


def test_part_streams_from_disk(tmp_path: Path) -> None:
    """A multi-GB chunk on a mini-PC must not be materialized in RAM before
    the PUT — the injected put-callable receives a readable object, not a
    ``bytes`` snapshot (#56)."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"m" * (2 * MIB))
    seen: dict = {}

    def capture_put(url: str, *, content: object, headers: dict, timeout: object) -> httpx.Response:
        seen["is_bytes"] = isinstance(content, (bytes, bytearray))
        seen["readable"] = hasattr(content, "read")
        return httpx.Response(200, headers={"ETag": '"x"'})

    with respx.mock(assert_all_called=False) as mock:
        _mock_platform(mock)
        result = _uploader(http_put=capture_put).upload_chunk("task-1", 0, chunk_path)

    assert result.success is True
    assert seen == {"is_bytes": False, "readable": True}


# ----- 1c. part size: upload_chunk_bytes, never under R2's 5 MiB floor -----


def test_part_size_never_drops_below_r2_minimum(tmp_path: Path) -> None:
    """R2 rejects any part but the last under 5 MiB (EntityTooSmall). A
    platform still configured for 1 MiB chunks (the pre-#248 lower bound)
    gets 5 MiB parts instead of a failed upload."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * (6 * MIB))

    with respx.mock(assert_all_called=False) as mock:
        _mock_platform(mock)
        put = mock.put(url__regex=R2_PUT).mock(side_effect=_r2_ok)
        uploader = _uploader()
        uploader.set_upload_chunk_bytes(1 * MIB)

        result = uploader.upload_chunk("task-1", 0, chunk_path)

    assert result.success is True
    assert [call.request.headers["content-length"] for call in put.calls] == [
        str(5 * MIB),
        str(1 * MIB),
    ]


# ----- 2. a 5xx is retried for the failing part only -----


def test_part_retries_on_5xx_without_resending_other_parts(tmp_path: Path) -> None:
    """A flaky R2 edge costs one part, not the whole recording: part 2 gets
    3 attempts with 1s/2s backoff while part 1 went up once."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * (6 * MIB))
    sleeps: list[float] = []
    part2 = iter([503, 503, 200])

    def flaky_part2(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/p2"):
            status = next(part2)
            if status != 200:
                return httpx.Response(status)
        return _r2_ok(request)

    with respx.mock(assert_all_called=False) as mock:
        platform = _mock_platform(mock)
        put = mock.put(url__regex=R2_PUT).mock(side_effect=flaky_part2)
        uploader = _uploader(sleep=sleeps.append)
        uploader.set_upload_chunk_bytes(5 * MIB)

        result = uploader.upload_chunk("task-1", 0, chunk_path)

    assert result.success is True
    assert [call.request.url.path for call in put.calls] == ["/c0/p1", "/c0/p2", "/c0/p2", "/c0/p2"]
    assert sleeps == [1, 2]
    assert [p["part_number"] for p in _body(platform["complete"])["parts"]] == [1, 2]


# ----- 3. 403 SignatureDoesNotMatch refetches the part's URL once -----


def test_part_refreshes_url_on_403_signature_mismatch(tmp_path: Path) -> None:
    """A URL signed by a stale key fails with ``403 SignatureDoesNotMatch``.
    The uploader asks the platform for that part's URL once more and PUTs
    again; a second mismatch would be a configuration bug, not expiry."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * 1024)
    fresh = _part_url(0, 1, attempt=2)

    with respx.mock(assert_all_called=False) as mock:
        platform = _mock_platform(mock)
        platform["part_url"].mock(
            side_effect=[
                httpx.Response(200, json={"url": _part_url(0, 1), "expires_in": 1800}),
                httpx.Response(200, json={"url": fresh, "expires_in": 1800}),
            ]
        )
        stale = mock.put(_part_url(0, 1)).mock(
            return_value=httpx.Response(403, text="<Error><Code>SignatureDoesNotMatch</Code>")
        )
        renewed = mock.put(fresh).mock(return_value=httpx.Response(200, headers={"ETag": '"e"'}))

        result = _uploader().upload_chunk("task-1", 0, chunk_path)

    assert result.success is True
    assert stale.call_count == 1
    assert renewed.call_count == 1
    assert platform["part_url"].call_count == 2


# ----- 4. a part that exhausts its retries fails the chunk and aborts the upload -----


def test_part_exhausting_retries_fails_and_aborts_upload(tmp_path: Path) -> None:
    """Three 503s on a part end the chunk as a failed result (data, not a
    raise — the poller marks the task ``failed``). The upload is aborted so
    R2 drops the parts already stored instead of keeping them for 7 days."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * 1024)

    with respx.mock(assert_all_called=False) as mock:
        platform = _mock_platform(mock)
        mock.put(url__regex=R2_PUT).mock(return_value=httpx.Response(503))

        result = _uploader().upload_chunk("task-1", 0, chunk_path)

    assert result.success is False
    assert result.key is None
    assert result.error == "part 1: R2 PUT returned 503 after 3 attempt(s)"
    assert _body(platform["abort"]) == {"task_id": "task-1", "chunk_n": 0, "upload_id": "up-0"}
    assert not platform["complete"].called


# ----- 5. a 4xx is terminal and the error says what R2 said -----


def test_r2_4xx_is_terminal_and_names_r2_code_and_real_attempts(tmp_path: Path) -> None:
    """A 4xx other than a signature mismatch will not improve on retry, so
    it ends after one attempt — and the error says so, with R2's code from
    the response body. Before #128 it read "after 3 attempt(s)" with no
    reason, which hid the 5 GiB limit behind the five failed tasks of
    2026-10-03."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * 1024)
    r2_error = (
        "<?xml version='1.0' encoding='UTF-8'?><Error><Code>EntityTooLarge</Code>"
        "<Message>Your proposed upload exceeds the maximum allowed object size.</Message></Error>"
    )

    with respx.mock(assert_all_called=False) as mock:
        platform = _mock_platform(mock)
        put = mock.put(url__regex=R2_PUT).mock(return_value=httpx.Response(400, text=r2_error))

        result = _uploader().upload_chunk("task-1", 0, chunk_path)

    assert result.error == "part 1: R2 PUT returned 400 EntityTooLarge after 1 attempt(s)"
    assert put.call_count == 1
    assert platform["abort"].called


def test_part_without_etag_fails_instead_of_completing_blind(tmp_path: Path) -> None:
    """Completing needs every part's ETag; a 200 without one cannot be
    completed, so it fails here rather than as a vaguer refusal later."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * 1024)

    with respx.mock(assert_all_called=False) as mock:
        platform = _mock_platform(mock)
        mock.put(url__regex=R2_PUT).mock(return_value=httpx.Response(200))

        result = _uploader().upload_chunk("task-1", 0, chunk_path)

    assert result.error == "part 1: R2 PUT returned 200 without an ETag"
    assert not platform["complete"].called


# ----- 6. platform refusals -----


def test_platform_refusing_to_open_upload_fails_without_any_put(tmp_path: Path) -> None:
    """Tenant isolation / unknown task: the platform opens nothing, so there
    is nothing to PUT and nothing to abort."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * 1024)

    with respx.mock(assert_all_called=False) as mock:
        platform = _mock_platform(mock)
        platform["open"].mock(return_value=httpx.Response(404, json={"error": "Task not found"}))
        put = mock.put(url__regex=R2_PUT)

        result = _uploader().upload_chunk("task-1", 0, chunk_path)

    assert result.success is False
    assert result.error == "platform refused upload-multipart (404): Task not found"
    assert not put.called
    assert not platform["abort"].called


def test_refused_completion_keeps_r2_reason_and_aborts(tmp_path: Path) -> None:
    """R2 refusing to join the parts reaches the appliance as a 409 with
    R2's reason, which goes into the task's error as it is."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * 1024)
    reason = "R2 refused the multipart upload: one of the specified parts could not be found."

    with respx.mock(assert_all_called=False) as mock:
        platform = _mock_platform(mock)
        mock.put(url__regex=R2_PUT).mock(side_effect=_r2_ok)
        platform["complete"].mock(
            return_value=httpx.Response(409, json={"error": reason, "code": "MULTIPART_REJECTED"})
        )

        result = _uploader().upload_chunk("task-1", 0, chunk_path)

    assert result.success is False
    assert result.error == f"platform refused upload-multipart/complete (409): {reason}"
    assert platform["abort"].called


# ----- 7. upload_chunks: chunks in parallel, results in input order -----


def test_upload_chunks_runs_chunks_in_parallel(tmp_path: Path) -> None:
    """Chunks go up concurrently. Real parallelism is checked with a
    ``threading.Barrier``: three PUTs on a 3-worker pool all reach it,
    while a sequential implementation would break it after 2 s.
    Results return in input order — the poller names failed chunks by it."""
    import threading

    chunks = []
    for i in range(3):
        p = tmp_path / f"chunk_{i}.mp4"
        p.write_bytes(f"chunk-{i}".encode())
        chunks.append(p)
    barrier = threading.Barrier(3, timeout=2.0)

    def put_after_barrier(request: httpx.Request) -> httpx.Response:
        barrier.wait()
        return _r2_ok(request)

    with respx.mock(assert_all_called=False) as mock:
        _mock_platform(mock)
        mock.put(url__regex=R2_PUT).mock(side_effect=put_after_barrier)

        results = _uploader(max_workers=3).upload_chunks("task-1", chunks)

    assert [r.chunk_n for r in results] == [0, 1, 2]
    assert all(r.success for r in results)
    assert [r.key for r in results] == [_key(0), _key(1), _key(2)]


def test_upload_chunks_returns_mixed_results_when_one_chunk_fails(tmp_path: Path) -> None:
    """Chunk 1 stuck on 5xx must not cost chunks 0 and 2 their uploads."""
    chunks = []
    for i in range(3):
        p = tmp_path / f"chunk_{i}.mp4"
        p.write_bytes(f"chunk-{i}".encode())
        chunks.append(p)

    def chunk1_down(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/c1/"):
            return httpx.Response(503)
        return _r2_ok(request)

    with respx.mock(assert_all_called=False) as mock:
        _mock_platform(mock)
        mock.put(url__regex=R2_PUT).mock(side_effect=chunk1_down)

        results = _uploader(max_workers=3).upload_chunks("task-1", chunks)

    assert [r.success for r in results] == [True, False, True]
    assert results[1].error == "part 1: R2 PUT returned 503 after 3 attempt(s)"


# ----- 8. transport error → failed result, no exception (issue #54) -----


def test_put_transport_error_returns_failed_result(tmp_path: Path) -> None:
    """A Wi-Fi blip mid-PUT surfaces as ``httpx.ConnectError`` — not a
    status. It counts against the same 3-attempt / 1s-2s budget as a 5xx,
    and when every attempt fails the chunk ends as a failed result instead
    of a raise that would wedge the task at ``uploading``."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"x" * 512)
    put_urls: list[str] = []
    sleeps: list[float] = []

    def flaky_put(url: str, *, content: object, headers: dict, timeout: object) -> httpx.Response:
        put_urls.append(url)
        raise httpx.ConnectError("[Errno 65] No route to host")

    with respx.mock(assert_all_called=False) as mock:
        platform = _mock_platform(mock)
        uploader = _uploader(sleep=sleeps.append, http_put=flaky_put)

        result = uploader.upload_chunk("task-1", 3, chunk_path)

    assert result.success is False
    assert result.chunk_n == 3
    assert result.error == (
        "part 1: transport error: [Errno 65] No route to host after 3 attempt(s)"
    )
    assert put_urls == [_part_url(3, 1)] * 3
    assert sleeps == [1, 2]
    assert platform["abort"].called


# ----- 9. upload_chunk_bytes: the part size, live-settable (#85) -----


def test_upload_chunk_bytes_defaults_and_is_settable() -> None:
    """The platform delivers ``upload_chunk_bytes`` in its runtime-config
    block (#85); it is the multipart part size. It defaults to 50 MiB and
    a runtime edit applies to the next upload."""
    uploader = _uploader()

    assert uploader.upload_chunk_bytes == 52_428_800

    uploader.set_upload_chunk_bytes(10_485_760)

    assert uploader.upload_chunk_bytes == 10_485_760


# ----- 10. upload progress reported to the platform (#127) -----

_BLOCK = 65_536


def _block_reading_put(
    clock: list[float], statuses: list[int], seconds_per_block: float = 4.0
) -> Callable[..., httpx.Response]:
    """Fake PUT that consumes ``content`` the way httpx does - ``read()`` in
    64 KiB blocks - with ``seconds_per_block`` of wall clock passing before
    every read. Each call answers with the next status from ``statuses``."""

    def put(url: str, *, content: object, headers: dict, timeout: object) -> httpx.Response:
        while True:
            clock[0] += seconds_per_block
            if not content.read(_BLOCK):  # type: ignore[attr-defined]
                break
        status = statuses.pop(0)
        return httpx.Response(status, headers={"ETag": '"e"'} if status == 200 else None)

    return put


def _reported(route: respx.Route) -> list[float]:
    return [json.loads(call.request.read())["progress_pct"] for call in route.calls]


def _mock_progress(mock: respx.MockRouter, response: httpx.Response) -> respx.Route:
    return mock.post(f"{PLATFORM}/appliance/tasks/task-1/progress").mock(return_value=response)


def test_upload_reports_growing_progress_at_most_once_per_interval(tmp_path: Path) -> None:
    """The panel draws its upload bar (gpu-exchange#246) from what the
    uploader reports while the PUT streams the file. Ten 64 KiB blocks read
    4 s apart give one report per 10 s interval. The first report waits a
    full interval: a 0% the moment the upload starts reads as a stalled task."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * (10 * _BLOCK))
    clock = [0.0]

    with respx.mock(assert_all_called=False) as mock:
        _mock_platform(mock)
        progress = _mock_progress(mock, httpx.Response(200, json={"ok": True, "applied": True}))
        uploader = _uploader(http_put=_block_reading_put(clock, [200]), clock=lambda: clock[0])

        results = uploader.upload_chunks("task-1", [chunk_path])

    assert results == [UploadResult(chunk_n=0, success=True, key=_key(0))]
    assert _reported(progress) == [30.0, 60.0, 90.0]


def test_failed_progress_report_keeps_upload_result_and_stops_reporting(tmp_path: Path) -> None:
    """The bar is cosmetic. A platform that cannot take the report (down, or
    older than the endpoint) must not change the upload's outcome, and after
    the first failure the uploader stops asking for the rest of the upload."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * (10 * _BLOCK))
    clock = [0.0]

    with respx.mock(assert_all_called=False) as mock:
        _mock_platform(mock)
        progress = _mock_progress(mock, httpx.Response(503))
        uploader = _uploader(http_put=_block_reading_put(clock, [200]), clock=lambda: clock[0])

        results = uploader.upload_chunks("task-1", [chunk_path])

    assert results == [UploadResult(chunk_n=0, success=True, key=_key(0))]
    assert progress.call_count == 1


def test_retried_part_counts_from_zero(tmp_path: Path) -> None:
    """A 5xx part is retried from its first byte, so the bytes the failed
    attempt read no longer count - otherwise the retry would pin the bar
    at 100% while the part goes up a second time."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * (10 * _BLOCK))
    clock = [0.0]

    with respx.mock(assert_all_called=False) as mock:
        _mock_platform(mock)
        progress = _mock_progress(mock, httpx.Response(200, json={"ok": True, "applied": True}))
        uploader = _uploader(http_put=_block_reading_put(clock, [503, 200]), clock=lambda: clock[0])

        results = uploader.upload_chunks("task-1", [chunk_path])

    assert results == [UploadResult(chunk_n=0, success=True, key=_key(0))]
    assert _reported(progress) == [30.0, 60.0, 90.0, 10.0, 40.0, 70.0, 100.0]


def test_progress_keeps_finished_parts_counted(tmp_path: Path) -> None:
    """The bar covers the whole chunk: starting part 2 must not drop the
    bytes part 1 already delivered."""
    chunk_path = tmp_path / "clip.mp4"
    chunk_path.write_bytes(b"v" * (10 * MIB))
    clock = [0.0]

    with respx.mock(assert_all_called=False) as mock:
        _mock_platform(mock)
        progress = _mock_progress(mock, httpx.Response(200, json={"ok": True, "applied": True}))
        # 160 blocks a second apart: a report every 10 blocks, not every 2.5.
        put = _block_reading_put(clock, [200, 200], seconds_per_block=1.0)
        uploader = _uploader(http_put=put, clock=lambda: clock[0])
        uploader.set_upload_chunk_bytes(5 * MIB)

        results = uploader.upload_chunks("task-1", [chunk_path])

    assert results == [UploadResult(chunk_n=0, success=True, key=_key(0))]
    reported = _reported(progress)
    assert reported == sorted(reported)
    assert reported[-1] > 90.0
