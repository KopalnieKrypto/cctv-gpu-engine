"""Tests for the GPU Exchange platform-integration adapter (issue #26).

The adapter is a thin Bearer-authenticated HTTP client against the platform
side specified in
``KopalnieKrypto/gpu-exchange/docs/design/09-client-appliance.md``. We test
through respx — every assertion lands at the request boundary (method, URL,
headers, body) so the implementation is free to change without breaking
tests.

Mocks live only at the httpx transport boundary (respx). Sleep is injected
on the PlatformClient so retry tests run in <1ms instead of 3s real-time.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

# ----- 1. register: Bearer + body, parse RegisterResponse -----


def test_register_posts_to_appliance_register_with_bearer_and_body() -> None:
    """First contact with the platform: appliance announces itself, gets back
    its tenant binding. The Bearer token scopes the appliance to a tenant
    server-side; the appliance never sends a tenant_id of its own (defense
    against a compromised appliance trying to impersonate another tenant)."""
    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200, json={"appliance_id": "app-42", "tenant_id": "tenant-7"}
            )
        )
        client = PlatformClient(base_url="https://platform.example", token="tok-abc")

        response = client.register(hostname="cctv-mini-01", version="0.5.0")

    assert response.appliance_id == "app-42"
    assert response.tenant_id == "tenant-7"
    assert route.called
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer tok-abc"
    body = sent.read()
    import json as _json

    assert _json.loads(body) == {"hostname": "cctv-mini-01", "version": "0.5.0"}


# ----- 2. register: retry on 5xx with exp backoff -----


def test_register_retries_on_5xx_and_succeeds_on_third_attempt() -> None:
    """Platform 5xx during register is retriable: the platform may be
    rolling out, behind a brief LB hiccup, etc. Spec: 3 attempts total
    with 1s/2s exp backoff between them. Sleep is injected so the test
    does not actually wait 3 seconds."""
    from client_agent.platform import PlatformClient

    sleeps: list[float] = []

    with respx.mock(base_url="https://platform.example") as mock:
        # Two 503s, then a 200. respx side_effect cycles through responses.
        mock.post("/appliance/register").mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(503),
                httpx.Response(200, json={"appliance_id": "a", "tenant_id": "t"}),
            ]
        )
        client = PlatformClient(
            base_url="https://platform.example",
            token="tok",
            sleep=sleeps.append,
        )

        response = client.register(hostname="h", version="v")

    assert response.appliance_id == "a"
    # Two retries means two sleeps: 1s then 2s (the third would precede a
    # 4th attempt that never happens — matches gpu-service/http_retry.py).
    assert sleeps == [1, 2]


# ----- 3. register: 401 → PlatformAuthError, no retry -----


def test_register_401_raises_auth_error_without_retry() -> None:
    """A 401 means the operator's APPLIANCE_TOKEN is wrong (missing,
    revoked, mistyped). Retrying is pointless — the token is not going
    to become valid in 3 seconds — and the operator needs a clear,
    immediate failure to diagnose. ``PlatformAuthError`` triggers a
    non-zero exit at the appliance level."""
    from client_agent.platform import PlatformAuthError, PlatformClient

    sleeps: list[float] = []

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/register").mock(return_value=httpx.Response(401))
        client = PlatformClient(
            base_url="https://platform.example",
            token="bad-token",
            sleep=sleeps.append,
        )

        with pytest.raises(PlatformAuthError):
            client.register(hostname="h", version="v")

    # Single attempt — no retries — and no sleeps.
    assert route.call_count == 1
    assert sleeps == []


# ----- 4. register: 5xx exhausted → PlatformUnavailableError -----


def test_register_503_exhausted_raises_unavailable_after_three_attempts() -> None:
    """Persistent 5xx means the platform is genuinely down (rolling deploy
    that broke, infrastructure outage). After 3 attempts the appliance
    gives up; the operator gets a non-zero exit and journald carries the
    full sequence of attempt logs. The error is distinct from
    ``PlatformAuthError`` so init-script logic / monitoring can branch
    on the cause (rotate token vs. wait for platform recovery)."""
    from client_agent.platform import PlatformClient, PlatformUnavailableError

    sleeps: list[float] = []

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/register").mock(return_value=httpx.Response(503))
        client = PlatformClient(
            base_url="https://platform.example",
            token="tok",
            sleep=sleeps.append,
        )

        with pytest.raises(PlatformUnavailableError):
            client.register(hostname="h", version="v")

    assert route.call_count == 3
    assert sleeps == [1, 2]


# ----- 5. push_cameras: Bearer + body, returns None on 204 -----


def test_push_cameras_posts_camera_list_with_bearer() -> None:
    """After ONVIF discovery, the appliance pushes the full camera list to
    the platform with ``enabled=False``. The operator activates per camera
    in the platform UI; activation flows back through ``heartbeat`` config.

    The platform side returns 204 — no body needed because the appliance
    will see the activation state in the next heartbeat anyway. We pass
    each camera as a plain dict so adding optional fields (firmware,
    serial) later does not require a client-library bump."""
    from client_agent.platform import PlatformClient

    cameras = [
        {
            "onvif_uri": "http://192.168.50.2/onvif/device_service",
            "manufacturer": "Hikvision",
            "model": "DS-2CD2143G2-IS",
            "ip": "192.168.50.2",
            "rtsp_url": "rtsp://192.168.50.2:554/Streaming/Channels/101",
        },
        {
            "onvif_uri": "http://192.168.50.3/onvif/device_service",
            "manufacturer": "Dahua",
            "model": "IPC-HDBW2431R-ZS",
            "ip": "192.168.50.3",
            "rtsp_url": "rtsp://192.168.50.3:554/cam/realmonitor?channel=1&subtype=0",
        },
    ]

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/cameras").mock(return_value=httpx.Response(204))
        client = PlatformClient(base_url="https://platform.example", token="tok")

        result = client.push_cameras(cameras)

    assert result is None
    assert route.called
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer tok"
    import json as _json

    body = _json.loads(sent.read())
    assert body == {"cameras": cameras}


# ----- 6. heartbeat: status + recording_cameras → response.config -----


def test_heartbeat_posts_status_and_returns_config() -> None:
    """Heartbeat is the appliance's keepalive AND its way of pulling the
    desired camera config (which to record, which to leave idle). The
    request body carries appliance-side state (which cameras are currently
    recording so the platform can detect drift); the response carries the
    desired state, expressed as the per-camera ``enabled`` flag and the
    RTSP URL the appliance should hand to the recorder."""
    from client_agent.platform import PlatformClient

    desired_config = {
        "cameras": [
            {"id": "cam-1", "enabled": True, "rtsp_url": "rtsp://192.168.50.2:554/main"},
            {"id": "cam-2", "enabled": False, "rtsp_url": "rtsp://192.168.50.3:554/main"},
        ]
    }

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": desired_config})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        result = client.heartbeat(
            status={"uptime_s": 1234, "free_disk_gb": 42},
            recording_cameras=["cam-1"],
        )

    assert result.config == desired_config
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer tok"
    import json as _json

    body = _json.loads(sent.read())
    assert body == {
        "status": {"uptime_s": 1234, "free_disk_gb": 42},
        "recording_cameras": ["cam-1"],
    }


# ----- 7. tenant isolation: Bearer per client, never leaks -----


def test_two_clients_send_their_own_bearer_never_the_other() -> None:
    """Tenant isolation is enforced server-side via the Bearer token →
    tenant_id binding. The client's job is to make sure each
    ``PlatformClient`` sends its own token and never the other's. With a
    routing-by-token mock we verify that two different appliance instances
    only ever see their own slice of state — there is no shared global
    that could leak the previous client's token into the next request."""
    from client_agent.platform import PlatformClient

    seen_tokens: list[str] = []

    def _record_token(request: httpx.Request) -> httpx.Response:
        token = request.headers["authorization"].removeprefix("Bearer ")
        seen_tokens.append(token)
        return httpx.Response(204)

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/cameras").mock(side_effect=_record_token)

        tenant_a = PlatformClient(base_url="https://platform.example", token="tok-A")
        tenant_b = PlatformClient(base_url="https://platform.example", token="tok-B")

        tenant_a.push_cameras([{"ip": "10.0.0.1"}])
        tenant_b.push_cameras([{"ip": "10.0.0.2"}])
        tenant_a.push_cameras([{"ip": "10.0.0.3"}])

    assert seen_tokens == ["tok-A", "tok-B", "tok-A"]


# ----- 8. fetch_next_task: 204 → None (idle, no work) -----


def test_fetch_next_task_returns_none_on_204() -> None:
    """The platform returns 204 No Content when its queue is empty. The
    poller treats this as "idle — sleep and try again", so the client
    returns ``None`` rather than raising. The 204 path is the steady
    state of a healthy appliance, so it must not allocate or log noisily."""
    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.get("/appliance/tasks/next").mock(return_value=httpx.Response(204))
        client = PlatformClient(base_url="https://platform.example", token="tok")

        result = client.fetch_next_task()

    assert result is None
    assert route.called
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer tok"


# ----- 9. fetch_next_task: 200 with task body → Task dataclass -----


def test_fetch_next_task_returns_task_object_on_200() -> None:
    """The platform returns 200 with a task envelope ``{"task": {id,
    camera_id, start_time, end_time, ...}}``. We surface those four
    fields on a typed ``Task`` so the poller does not key-index into a
    raw dict (and miss a renamed field silently)."""
    from client_agent.platform import PlatformClient

    task_body = {
        "task": {
            "id": "task-abc",
            "camera_id": "cam-1",
            "start_time": "2026-05-15T10:15:00Z",
            "end_time": "2026-05-15T10:45:00Z",
        }
    }

    with respx.mock(base_url="https://platform.example") as mock:
        mock.get("/appliance/tasks/next").mock(return_value=httpx.Response(200, json=task_body))
        client = PlatformClient(base_url="https://platform.example", token="tok")

        task = client.fetch_next_task()

    assert task is not None
    assert task.id == "task-abc"
    assert task.camera_id == "cam-1"
    assert task.start_time == datetime(2026, 5, 15, 10, 15, 0, tzinfo=UTC)
    assert task.end_time == datetime(2026, 5, 15, 10, 45, 0, tzinfo=UTC)


# ----- 10. update_task_status: POST with status + optional error -----


def test_update_task_status_posts_status_and_optional_error() -> None:
    """The poller emits status transitions back to the platform: claimed
    → recording → uploading (happy path) or → failed (with error message).
    The ``error`` field is only present on the failed transition so the
    success path keeps the body minimal."""
    from client_agent.platform import PlatformClient

    seen_bodies: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen_bodies.append(_json.loads(request.read()))
        return httpx.Response(204)

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/tasks/task-abc/status").mock(side_effect=_capture)
        client = PlatformClient(base_url="https://platform.example", token="tok")

        client.update_task_status("task-abc", status="recording")
        client.update_task_status("task-abc", status="failed", error="buffer empty")

    assert seen_bodies == [
        {"status": "recording"},
        {"status": "failed", "error": "buffer empty"},
    ]


# ----- 11. get_upload_url: Bearer + query params, parses UploadUrl -----


def test_get_upload_url_sends_query_params_and_parses_response() -> None:
    """Per DD-09 (gpu-exchange) the appliance never holds R2 credentials —
    every upload is gated by a fresh presigned PUT URL. ``get_upload_url``
    is the only way the appliance can move bytes to R2; the platform binds
    the URL to ``tenants/{tid}/appliance-uploads/{task_id}/chunk_N.mp4``
    so cross-tenant scribbles are impossible by construction.

    This test pins the wire format: GET with Bearer, ``task_id`` and
    ``chunk_n`` as query params, response parsed into :class:`UploadUrl`."""
    from client_agent.platform import PlatformClient, UploadUrl

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.get(
            "/appliance/upload-url",
            params={"task_id": "task-9", "chunk_n": "3"},
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "url": "https://r2.example/tenants/t-1/appliance-uploads/task-9/chunk_3.mp4?sig=xyz",
                    "key": "tenants/t-1/appliance-uploads/task-9/chunk_3.mp4",
                    "expires_in": 1800,
                },
            )
        )
        client = PlatformClient(base_url="https://platform.example", token="tok-z")

        result = client.get_upload_url("task-9", 3)

    assert isinstance(result, UploadUrl)
    assert result.url.endswith("chunk_3.mp4?sig=xyz")
    assert result.key == "tenants/t-1/appliance-uploads/task-9/chunk_3.mp4"
    assert result.expires_in == 1800
    assert route.called
    assert route.calls.last.request.headers["authorization"] == "Bearer tok-z"


# ----- 12. get_upload_url: 5xx → retry budget shared with other GETs -----


def test_get_upload_url_retries_on_5xx_and_succeeds_on_third_attempt() -> None:
    """Presigned-URL generation is a normal idempotent GET — the existing
    3-attempt / 1s+2s backoff in :meth:`PlatformClient._get` covers it.
    This test pins that we did not accidentally bypass the shared retry
    path when wiring the new method."""
    from client_agent.platform import PlatformClient

    sleeps: list[float] = []

    with respx.mock(base_url="https://platform.example") as mock:
        mock.get(
            "/appliance/upload-url",
            params={"task_id": "task-1", "chunk_n": "0"},
        ).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(503),
                httpx.Response(
                    200, json={"url": "https://r2.example/u", "key": "k", "expires_in": 60}
                ),
            ]
        )
        client = PlatformClient(
            base_url="https://platform.example",
            token="tok",
            sleep=sleeps.append,
        )

        result = client.get_upload_url("task-1", 0)

    assert result.key == "k"
    assert sleeps == [1, 2]
