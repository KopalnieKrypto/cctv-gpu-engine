"""Tests for the presigned-URL upload manager (issue #28, Slice 1c.3).

The uploader sits between :class:`TaskPoller` (which produces trimmed
chunks on local disk) and Cloudflare R2 (the durable store the
gpu-service worker pulls from). It deliberately holds **no R2
credentials**: every upload is gated on a fresh presigned PUT URL
issued by the platform on demand. The platform-side check binds the
URL to ``tenants/{tid}/appliance-uploads/{task_id}/chunk_N.mp4`` so a
compromised appliance cannot scribble outside its task scope.

Tests are hermetic — respx mocks both the platform's
``/appliance/upload-url`` endpoint and the presigned-URL PUT itself
(intercepted by URL pattern). Sleep is injected so retry tests run in
microseconds, and the executor's parallelism is exercised with a
``threading.Barrier`` rather than wallclock timing.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import respx

from client_agent.platform import PlatformClient

# ----- 1. tracer bullet: upload_chunk fetches URL then PUTs the chunk -----


def test_upload_chunk_happy_path_gets_url_then_puts_chunk(tmp_path: Path) -> None:
    """One chunk, no retries, no expiry: GET ``/appliance/upload-url``
    with Bearer auth → platform returns a presigned PUT URL → uploader
    PUTs the chunk body to that URL with **no Bearer** (presigned URLs
    carry their own signature; sending a second auth header confuses
    some S3-compatible backends) → uploader returns ``UploadResult``
    flagged success with the platform-supplied key.

    This is the tracer bullet — it forces both ``PlatformClient.
    get_upload_url`` and ``PresignedUploader.upload_chunk`` to exist
    in their minimal shape. Edge cases (5xx retry, expiry refresh,
    parallelism) are layered on in later tests."""
    from client_agent.uploader import PresignedUploader

    chunk_path = tmp_path / "chunk_001.mp4"
    chunk_body = b"fake-mp4-bytes-roughly-7MB-or-so" * 100
    chunk_path.write_bytes(chunk_body)

    presigned_url = "https://r2.example/tenants/t-7/appliance-uploads/task-1/chunk_1.mp4?sig=abc"
    expected_key = "tenants/t-7/appliance-uploads/task-1/chunk_1.mp4"

    with respx.mock(assert_all_called=True) as mock:
        url_route = mock.get(
            "https://platform.example/appliance/upload-url",
            params={"task_id": "task-1", "chunk_n": "1"},
        ).mock(
            return_value=httpx.Response(
                200,
                json={"url": presigned_url, "key": expected_key, "expires_in": 1800},
            )
        )
        put_route = mock.put(presigned_url).mock(return_value=httpx.Response(200))

        platform = PlatformClient(base_url="https://platform.example", token="tok-abc")
        uploader = PresignedUploader(platform=platform)

        result = uploader.upload_chunk("task-1", 1, chunk_path)

    assert result.success is True
    assert result.chunk_n == 1
    assert result.key == expected_key
    assert result.error is None

    # Platform GET carried the Bearer; presigned PUT did NOT.
    assert url_route.calls.last.request.headers["authorization"] == "Bearer tok-abc"
    assert put_route.calls.last.request.headers.get("authorization") is None
    # PUT body is the chunk file's bytes — verifies the uploader actually
    # streamed the file rather than sending an empty body.
    assert put_route.calls.last.request.content == chunk_body


# ----- 1b. upload_chunk streams the file rather than reading it into RAM -----


def test_upload_streams_file_content(tmp_path: Path) -> None:
    """A multi-GB chunk on a mini-PC must not be materialized in RAM before
    the PUT — ``httpx`` accepts a file-like object as ``content`` and streams
    it off disk. Assert the injected put-callable receives a readable file
    object, not a ``bytes`` snapshot of the whole file (#56)."""
    from client_agent.uploader import PresignedUploader

    chunk_path = tmp_path / "chunk_777.mp4"
    chunk_path.write_bytes(b"m" * (2 * 1024 * 1024))

    seen: dict = {}

    def capture_put(url: str, *, content: object, timeout: object) -> httpx.Response:
        seen["is_bytes"] = isinstance(content, (bytes, bytearray))
        seen["readable"] = hasattr(content, "read")
        return httpx.Response(200)

    with respx.mock(assert_all_called=True) as mock:
        mock.get(
            "https://platform.example/appliance/upload-url",
            params={"task_id": "task-1", "chunk_n": "7"},
        ).mock(
            return_value=httpx.Response(
                200,
                json={"url": "https://r2.example/k?sig=ok", "key": "k", "expires_in": 1800},
            )
        )

        platform = PlatformClient(base_url="https://platform.example", token="tok")
        uploader = PresignedUploader(platform=platform, http_put=capture_put)

        result = uploader.upload_chunk("task-1", 7, chunk_path)

    assert result.success is True
    assert seen["is_bytes"] is False
    assert seen["readable"] is True


# ----- 2. upload_chunk retries on 5xx with exp backoff -----


def test_upload_chunk_retries_on_5xx_and_succeeds_on_third_attempt(tmp_path: Path) -> None:
    """A flaky R2 edge (or a transient CF outage) is the most likely
    real-world failure for an otherwise-valid presigned PUT. Spec
    matches :class:`PlatformClient`: 3 attempts, 1s/2s exp backoff
    between them, sleep is injected so the test runs in microseconds.

    The URL is *not* refreshed between 5xx retries — only on a
    SignatureDoesNotMatch (covered in T4). 5xx means the URL is fine,
    the backend is sick."""
    from client_agent.uploader import PresignedUploader

    chunk_path = tmp_path / "chunk_002.mp4"
    chunk_path.write_bytes(b"x" * 1024)

    presigned_url = "https://r2.example/k?sig=ok"
    sleeps: list[float] = []

    with respx.mock(assert_all_called=True) as mock:
        url_route = mock.get(
            "https://platform.example/appliance/upload-url",
            params={"task_id": "task-1", "chunk_n": "2"},
        ).mock(
            return_value=httpx.Response(
                200, json={"url": presigned_url, "key": "k", "expires_in": 1800}
            )
        )
        put_route = mock.put(presigned_url).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(503),
                httpx.Response(200),
            ]
        )

        platform = PlatformClient(base_url="https://platform.example", token="tok")
        uploader = PresignedUploader(platform=platform, sleep=sleeps.append)

        result = uploader.upload_chunk("task-1", 2, chunk_path)

    assert result.success is True
    assert result.chunk_n == 2
    # Three PUT attempts, two sleeps between them — same shape as platform retry.
    assert put_route.call_count == 3
    assert sleeps == [1, 2]
    # URL was fetched exactly once — 5xx does NOT trigger a refresh.
    assert url_route.call_count == 1


# ----- 3. upload_chunk refreshes presigned URL on 403 SignatureDoesNotMatch -----


def test_upload_chunk_refreshes_url_on_403_signature_mismatch(tmp_path: Path) -> None:
    """The 30-min default TTL on presigned URLs means a chunk PUT can
    land after expiry — especially the *last* chunk of a long parallel
    batch. R2 surfaces this as ``403 SignatureDoesNotMatch`` (S3-compat
    error code). On that specific signal the uploader gets one fresh
    URL and tries again; a second 403 is treated as terminal (likely a
    misconfigured server, not transient expiry).

    Other 403 bodies (e.g. tenant violation from the URL issuer) are
    not refresh-eligible — those land at :meth:`PlatformClient.
    get_upload_url`, never at the PUT, so we do not need to disambiguate
    here. The refresh path is narrowly scoped to the SignatureDoesNotMatch
    body."""
    from client_agent.uploader import PresignedUploader

    chunk_path = tmp_path / "chunk_005.mp4"
    chunk_path.write_bytes(b"y" * 512)

    stale_url = "https://r2.example/k?sig=stale"
    fresh_url = "https://r2.example/k?sig=fresh"

    with respx.mock(assert_all_called=True) as mock:
        url_route = mock.get(
            "https://platform.example/appliance/upload-url",
            params={"task_id": "task-1", "chunk_n": "5"},
        ).mock(
            side_effect=[
                httpx.Response(200, json={"url": stale_url, "key": "k", "expires_in": 1}),
                httpx.Response(200, json={"url": fresh_url, "key": "k", "expires_in": 1800}),
            ]
        )
        stale_put = mock.put(stale_url).mock(
            return_value=httpx.Response(
                403,
                text="<Error><Code>SignatureDoesNotMatch</Code></Error>",
            )
        )
        fresh_put = mock.put(fresh_url).mock(return_value=httpx.Response(200))

        platform = PlatformClient(base_url="https://platform.example", token="tok")
        uploader = PresignedUploader(platform=platform)

        result = uploader.upload_chunk("task-1", 5, chunk_path)

    assert result.success is True
    assert url_route.call_count == 2  # initial + 1 refresh
    assert stale_put.call_count == 1
    assert fresh_put.call_count == 1


# ----- 4. upload_chunk all retries exhausted → UploadResult(success=False) -----


def test_upload_chunk_all_retries_fail_returns_failed_result(tmp_path: Path) -> None:
    """Three 5xx in a row exhausts the retry budget. The contract is
    that the failure surfaces as ``UploadResult(success=False, error=...)``,
    **not** as an exception — the poller composes ``status=failed`` for
    the platform from the result objects and a mid-batch raise would
    abandon other chunks' results.

    The error message includes the final HTTP status so the operator
    has something concrete to diagnose ("R2 503 after 3 attempts" beats
    a generic "upload failed")."""
    from client_agent.uploader import PresignedUploader

    chunk_path = tmp_path / "chunk_009.mp4"
    chunk_path.write_bytes(b"z" * 256)

    presigned_url = "https://r2.example/k?sig=ok"

    with respx.mock(assert_all_called=True) as mock:
        mock.get(
            "https://platform.example/appliance/upload-url",
            params={"task_id": "task-1", "chunk_n": "9"},
        ).mock(
            return_value=httpx.Response(
                200, json={"url": presigned_url, "key": "k", "expires_in": 1800}
            )
        )
        put_route = mock.put(presigned_url).mock(
            side_effect=[httpx.Response(503), httpx.Response(502), httpx.Response(500)]
        )

        platform = PlatformClient(base_url="https://platform.example", token="tok")
        uploader = PresignedUploader(platform=platform, sleep=lambda _s: None)

        result = uploader.upload_chunk("task-1", 9, chunk_path)

    assert result.success is False
    assert result.chunk_n == 9
    assert result.key is None
    assert result.error is not None
    # Error mentions the final status code — surface for the operator,
    # not a structured field (callers only display it).
    assert "500" in result.error
    assert put_route.call_count == 3


# ----- 5. upload_chunk non-expiry 4xx → terminal failure, no retry, no refresh -----


def test_upload_chunk_4xx_other_than_expiry_is_terminal(tmp_path: Path) -> None:
    """A 404 on the presigned URL means the R2 object path doesn't
    exist in the way the URL signed for — typically a platform bug, not
    transient. Don't retry, don't refresh; return a failed result
    immediately so the operator/platform get a fast signal."""
    from client_agent.uploader import PresignedUploader

    chunk_path = tmp_path / "chunk_010.mp4"
    chunk_path.write_bytes(b"q" * 256)

    presigned_url = "https://r2.example/k?sig=ok"

    with respx.mock(assert_all_called=True) as mock:
        url_route = mock.get(
            "https://platform.example/appliance/upload-url",
            params={"task_id": "task-1", "chunk_n": "10"},
        ).mock(
            return_value=httpx.Response(
                200, json={"url": presigned_url, "key": "k", "expires_in": 1800}
            )
        )
        put_route = mock.put(presigned_url).mock(return_value=httpx.Response(404))

        platform = PlatformClient(base_url="https://platform.example", token="tok")
        uploader = PresignedUploader(platform=platform)

        result = uploader.upload_chunk("task-1", 10, chunk_path)

    assert result.success is False
    assert result.error is not None
    assert "404" in result.error
    assert put_route.call_count == 1  # no retry on 4xx
    assert url_route.call_count == 1  # no refresh on non-expiry 4xx


# ----- 6. upload_chunk tenant isolation: get_upload_url 403 propagates -----


def test_upload_chunk_propagates_get_upload_url_403_as_failure(tmp_path: Path) -> None:
    """The platform's URL issuer is the privacy gate: a Bearer token
    bound to tenant A asking for an upload URL on task B (which belongs
    to tenant Z) gets a 403 there, *not* at the PUT. The uploader must
    surface that 403 as a failed result without attempting any PUT (no
    URL was issued — there is nothing to PUT to).

    The error must be human-actionable: an operator seeing "tenant
    isolation: platform refused upload-url (403)" knows to check the
    APPLIANCE_TOKEN's tenant binding, not the R2 backend. We keep the
    message generic enough to also cover other 4xx returns from the
    URL issuer (the platform decides which 4xx to use)."""
    from client_agent.uploader import PresignedUploader

    chunk_path = tmp_path / "chunk_999.mp4"
    chunk_path.write_bytes(b"forbidden")

    with respx.mock(assert_all_called=True) as mock:
        url_route = mock.get(
            "https://platform.example/appliance/upload-url",
            params={"task_id": "task-cross-tenant", "chunk_n": "0"},
        ).mock(return_value=httpx.Response(403, json={"error": "tenant mismatch"}))
        # No PUT route registered — if upload_chunk tries to PUT despite
        # the 403, respx will raise an unmocked-request error.

        platform = PlatformClient(base_url="https://platform.example", token="tok")
        uploader = PresignedUploader(platform=platform)

        result = uploader.upload_chunk("task-cross-tenant", 0, chunk_path)

    assert result.success is False
    assert result.chunk_n == 0
    assert result.error is not None
    assert "403" in result.error
    assert url_route.call_count == 1  # no retry on 4xx from the URL issuer


# ----- 7. upload_chunks runs PUTs in parallel (barrier-verified) -----


def test_upload_chunks_runs_puts_in_parallel(tmp_path: Path) -> None:
    """A 30-minute task can produce a dozen chunks; uploading them
    sequentially would burn minutes that the platform's status timeline
    would mistake for an appliance hang. We use a ``ThreadPoolExecutor``
    to issue PUTs concurrently — bandwidth-bounded by R2, not by single-
    request RTT.

    We verify *real* parallelism via a ``threading.Barrier``: if 3 PUTs
    are submitted to a 3-worker pool, all three threads block on the
    barrier and then release together. A sequential implementation
    would deadlock at the first PUT (only one thread reaches the
    barrier; ``timeout=2.0`` makes the deadlock fail loudly instead of
    hanging the test suite).

    Results return in the input order regardless of which chunk's PUT
    finished first — the poller composes its ``status=failed`` error
    from this list and we don't want chunk-order ambiguity to leak."""
    import threading

    from client_agent.uploader import PresignedUploader

    chunks = []
    for i in range(3):
        p = tmp_path / f"chunk_{i}.mp4"
        p.write_bytes(f"chunk-{i}".encode())
        chunks.append(p)

    barrier = threading.Barrier(3, timeout=2.0)

    def _put_side_effect(request: httpx.Request) -> httpx.Response:
        # All three workers must reach the barrier; if any one is
        # serialized behind the others, barrier.wait() raises BrokenBarrierError.
        barrier.wait()
        return httpx.Response(200)

    with respx.mock(assert_all_called=True) as mock:
        # Bind responses by chunk_n so parallel races can't shuffle
        # which response goes to which chunk (a plain side_effect list
        # would assign in call order, not chunk_n order).
        for i in range(3):
            mock.get(
                "https://platform.example/appliance/upload-url",
                params={"task_id": "task-1", "chunk_n": str(i)},
            ).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "url": f"https://r2.example/k{i}?sig=ok",
                        "key": f"k{i}",
                        "expires_in": 1800,
                    },
                )
            )
            mock.put(f"https://r2.example/k{i}?sig=ok").mock(side_effect=_put_side_effect)

        platform = PlatformClient(base_url="https://platform.example", token="tok")
        uploader = PresignedUploader(platform=platform, max_workers=3)

        results = uploader.upload_chunks("task-1", chunks)

    assert [r.chunk_n for r in results] == [0, 1, 2]  # input-order preserved
    assert all(r.success for r in results)
    assert [r.key for r in results] == ["k0", "k1", "k2"]


# ----- 8. upload_chunks: one chunk's PUT fails, others succeed -----


def test_upload_chunks_returns_mixed_results_when_one_chunk_fails(tmp_path: Path) -> None:
    """If chunk 1's R2 PUT is stuck on 5xx after all retries, chunks 0
    and 2 must still complete and return success — otherwise the
    operator restarting the task uploads what already succeeded a
    second time, doubling bandwidth costs. The poller turns the mixed
    result list into a single ``status=failed`` payload that names
    *which* chunks failed (T10)."""
    from client_agent.uploader import PresignedUploader

    chunks = []
    for i in range(3):
        p = tmp_path / f"chunk_{i}.mp4"
        p.write_bytes(f"chunk-{i}".encode())
        chunks.append(p)

    with respx.mock(assert_all_called=True) as mock:
        for i in range(3):
            mock.get(
                "https://platform.example/appliance/upload-url",
                params={"task_id": "task-1", "chunk_n": str(i)},
            ).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "url": f"https://r2.example/k{i}?sig=ok",
                        "key": f"k{i}",
                        "expires_in": 1800,
                    },
                )
            )
        # Chunks 0 + 2 succeed, chunk 1 fails permanently (3× 503).
        mock.put("https://r2.example/k0?sig=ok").mock(return_value=httpx.Response(200))
        mock.put("https://r2.example/k1?sig=ok").mock(return_value=httpx.Response(503))
        mock.put("https://r2.example/k2?sig=ok").mock(return_value=httpx.Response(200))

        platform = PlatformClient(base_url="https://platform.example", token="tok")
        uploader = PresignedUploader(platform=platform, sleep=lambda _s: None, max_workers=3)

        results = uploader.upload_chunks("task-1", chunks)

    assert [r.chunk_n for r in results] == [0, 1, 2]
    assert [r.success for r in results] == [True, False, True]
    assert results[1].error is not None
    assert "503" in results[1].error


# ----- 9. upload_chunk transport error → failed result, no exception (issue #54) -----


def test_put_transport_error_returns_failed_result(tmp_path: Path) -> None:
    """A Wi-Fi blip mid-PUT surfaces as ``httpx.ConnectError`` /
    ``ReadError`` / ``ReadTimeout`` — not a 5xx status. The pre-#54 code
    only retried on 5xx *status codes*, so a raised transport error escaped
    ``upload_chunk`` and wedged the task at ``uploading`` (the class
    docstring promises "no exceptions bubble"). The uploader must catch it,
    count it against the same 3-attempt budget as a 5xx, and — when all
    attempts fail transport-wise — return ``UploadResult(success=False,
    error=...)`` instead of raising.

    ``http_put`` is injected here (the real default is ``httpx.put``) so we
    can simulate transport failures deterministically without a live socket;
    the GET for the presigned URL still goes through respx."""
    from client_agent.uploader import PresignedUploader

    chunk_path = tmp_path / "chunk_003.mp4"
    chunk_path.write_bytes(b"x" * 512)

    presigned_url = "https://r2.example/k?sig=ok"
    put_urls: list[str] = []
    sleeps: list[float] = []

    def flaky_put(url: str, *, content: bytes, timeout: object) -> httpx.Response:
        put_urls.append(url)
        raise httpx.ConnectError("[Errno 65] No route to host")

    with respx.mock(assert_all_called=True) as mock:
        mock.get(
            "https://platform.example/appliance/upload-url",
            params={"task_id": "task-1", "chunk_n": "3"},
        ).mock(
            return_value=httpx.Response(
                200, json={"url": presigned_url, "key": "k", "expires_in": 1800}
            )
        )

        platform = PlatformClient(base_url="https://platform.example", token="tok")
        uploader = PresignedUploader(platform=platform, sleep=sleeps.append, http_put=flaky_put)

        result = uploader.upload_chunk("task-1", 3, chunk_path)

    # No exception crossed the boundary — the failure is data, not a raise.
    assert result.success is False
    assert result.chunk_n == 3
    assert result.key is None
    assert result.error is not None
    # Transport errors count against the same 3-attempt / 1s-2s budget as 5xx.
    assert put_urls == [presigned_url, presigned_url, presigned_url]
    assert sleeps == [1, 2]


# ----- 10. upload_chunk_bytes: store-only runtime setter (#85) -----


def test_upload_chunk_bytes_defaults_and_is_settable() -> None:
    """The platform delivers ``upload_chunk_bytes`` in its runtime-config
    block (#85). The uploader currently streams a whole trimmed file per task
    — there is no byte-splitting path yet — so the value is stored, not acted
    on: it defaults to 50 MiB and a runtime edit re-points it in place for
    whenever a chunked uploader lands. Storing it keeps the four-setting
    apply contract honest without building an unused splitter."""
    from client_agent.uploader import PresignedUploader

    platform = PlatformClient(base_url="https://platform.example", token="tok")
    uploader = PresignedUploader(platform=platform)

    assert uploader.upload_chunk_bytes == 52_428_800

    uploader.set_upload_chunk_bytes(10_485_760)

    assert uploader.upload_chunk_bytes == 10_485_760
