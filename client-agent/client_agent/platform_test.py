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
    against a compromised appliance trying to impersonate another tenant).

    Body matches DD-09 canon: ``agent_version`` + optional ``host_info``."""
    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200,
                json={
                    "appliance_id": "app-42",
                    "tenant_id": "tenant-7",
                    "installed_at": "2026-05-19T10:00:00Z",
                },
            )
        )
        client = PlatformClient(base_url="https://platform.example", token="tok-abc")

        response = client.register(
            agent_version="0.5.0", host_info={"platform": "linux", "arch": "aarch64"}
        )

    assert response.appliance_id == "app-42"
    assert response.tenant_id == "tenant-7"
    assert route.called
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer tok-abc"
    body = sent.read()
    import json as _json

    assert _json.loads(body) == {
        "agent_version": "0.5.0",
        "host_info": {"platform": "linux", "arch": "aarch64"},
    }


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

        response = client.register(agent_version="v")

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
            client.register(agent_version="v")

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
            client.register(agent_version="v")

    assert route.call_count == 3
    assert sleeps == [1, 2]


# ----- 4.5. register: parses the platform-delivered settings block (#85) -----


def test_register_parses_settings_block() -> None:
    """The register response now carries a snake_case ``settings`` block
    (gpu-exchange 9f4cbc2) so the box is correct from beat zero, not just
    after the first heartbeat. Parse it onto the response so the runtime-
    config applier can override the box's install-time env defaults (#85)."""
    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200,
                json={
                    "appliance_id": "app-1",
                    "tenant_id": "tenant-1",
                    "installed_at": "2026-07-16T10:00:00Z",
                    "settings": {
                        "buffer_hours": 8,
                        "polling_interval_seconds": 7,
                        "heartbeat_interval_seconds": 30,
                        "upload_chunk_bytes": 52428800,
                    },
                },
            )
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        response = client.register(agent_version="v")

    assert response.settings == {
        "buffer_hours": 8,
        "polling_interval_seconds": 7,
        "heartbeat_interval_seconds": 30,
        "upload_chunk_bytes": 52428800,
    }


def test_register_settings_absent_is_none() -> None:
    """A platform that predates the settings feature (or a slimmed response)
    omits ``settings`` entirely. The box must still boot: settings default
    to ``None`` and the applier falls back to env cold-start values (#85)."""
    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            return_value=httpx.Response(200, json={"appliance_id": "a", "tenant_id": "t"})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        response = client.register(agent_version="v")

    assert response.settings is None


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
            "rtsp_url": "rtsp://192.168.50.2:554/Streaming/Channels/101",
            "onvif_uuid": "urn:uuid:11111111-2222-3333-4444-555555555555",
            "name": "Hikvision DS-2CD2143G2-IS",
            "model_info": {"manufacturer": "Hikvision", "model": "DS-2CD2143G2-IS"},
        },
        {
            "rtsp_url": "rtsp://192.168.50.3:554/cam/realmonitor?channel=1&subtype=0",
            "name": "Dahua IPC-HDBW2431R-ZS",
            "model_info": {"manufacturer": "Dahua", "model": "IPC-HDBW2431R-ZS"},
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


# ----- 6.5. heartbeat: exposes the nested settings block (#85) -----


def test_heartbeat_exposes_settings_from_config() -> None:
    """Every heartbeat now carries ``config.settings`` alongside
    ``config.cameras`` (gpu-exchange 9f4cbc2). Surface it as ``.settings``
    so the applier reads the same shape from register and heartbeat without
    reaching into the raw config dict at every call site (#85)."""
    from client_agent.platform import PlatformClient

    config = {
        "cameras": [{"id": "cam-1", "enabled": True, "rtsp_url": "rtsp://h/main"}],
        "settings": {
            "buffer_hours": 12,
            "polling_interval_seconds": 5,
            "heartbeat_interval_seconds": 30,
            "upload_chunk_bytes": 52428800,
        },
    }

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": config})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        result = client.heartbeat(status={}, recording_cameras=[])

    assert result.settings == config["settings"]
    # The raw config is still intact — cameras reconciliation reads it.
    assert result.config == config


def test_heartbeat_settings_absent_is_none() -> None:
    """A config with no ``settings`` key (older platform, or a beat that
    only reconciles cameras) yields ``None`` — the applier no-ops and the
    box keeps its current values (#85)."""
    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": {"cameras": []}})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        result = client.heartbeat(status={}, recording_cameras=[])

    assert result.settings is None


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


# ----- 12.5. _post: retry on httpx.ReadTimeout (issue #42 tracer) -----


def test_post_retries_on_read_timeout_and_succeeds_on_second_attempt() -> None:
    """Transport-level read timeouts must be retried with the same backoff
    schedule as 5xx. Source incident (issue #42): a Cloudflare Worker cold
    start exceeded the httpx default 5 s read window during an
    ``update_task_status`` POST; the DB UPDATE succeeded server-side but
    the client saw ``httpx.ReadTimeout``, the existing retry loop only
    handled 5xx status codes, and the poller thread crashed mid-state-
    transition — leaving the task wedged at ``status=uploading`` with no
    upload ever attempted. Catching the timeout and retrying is what keeps
    a single transit blip from killing a task."""
    from client_agent.platform import PlatformClient

    sleeps: list[float] = []

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            side_effect=[
                httpx.ReadTimeout("simulated CF Worker cold start"),
                httpx.Response(200, json={"appliance_id": "a", "tenant_id": "t"}),
            ]
        )
        client = PlatformClient(
            base_url="https://platform.example",
            token="tok",
            sleep=sleeps.append,
        )

        response = client.register(agent_version="v")

    assert response.appliance_id == "a"
    # One retry → one sleep at backoff[0] (1 s).
    assert sleeps == [1]


# ----- 12.6. _post: ReadTimeout on every attempt → PlatformUnavailableError -----


def test_post_raises_unavailable_after_read_timeout_exhausts_retries() -> None:
    """If every attempt hits a transport-level timeout, the appliance must
    raise :class:`PlatformUnavailableError` — distinct from
    :class:`PlatformAuthError` so monitoring can branch on the cause. This
    is the persistent-outage twin of the single-blip happy path: 5xx and
    timeouts share one retry budget; exhausting it yields one error type
    either way."""
    from client_agent.platform import PlatformClient, PlatformUnavailableError

    sleeps: list[float] = []

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/register").mock(
            side_effect=[
                httpx.ReadTimeout("blip 1"),
                httpx.ReadTimeout("blip 2"),
                httpx.ReadTimeout("blip 3"),
            ]
        )
        client = PlatformClient(
            base_url="https://platform.example",
            token="tok",
            sleep=sleeps.append,
        )

        with pytest.raises(PlatformUnavailableError):
            client.register(agent_version="v")

    assert route.call_count == 3
    assert sleeps == [1, 2]


# ----- 12.7. _post: retry on httpx.ConnectError (issue #54) -----


def test_post_retries_on_connect_error() -> None:
    """A ``ConnectError`` (DNS blip, connection refused while the platform
    is mid-deploy) is transport-level just like a timeout, yet #42 only
    widened the retry whitelist to :class:`httpx.TimeoutException`. So a
    single connect blip during ``update_task_status`` propagated with zero
    retries and wedged the platform-side task (issue #54). Transport errors
    must share the same 3-attempt / 1s-2s budget as timeouts: first attempt
    raises ``ConnectError``, second returns 200, the call succeeds after one
    backoff sleep."""
    from client_agent.platform import PlatformClient

    sleeps: list[float] = []

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            side_effect=[
                httpx.ConnectError("[Errno 111] Connection refused"),
                httpx.Response(200, json={"appliance_id": "a", "tenant_id": "t"}),
            ]
        )
        client = PlatformClient(
            base_url="https://platform.example",
            token="tok",
            sleep=sleeps.append,
        )

        response = client.register(agent_version="v")

    assert response.appliance_id == "a"
    # One retry → one sleep at backoff[0] (1 s).
    assert sleeps == [1]


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


# ----- 13. _get: retry on httpx.ReadTimeout (issue #42 symmetry) -----


def test_get_retries_on_read_timeout_and_succeeds_on_second_attempt() -> None:
    """``fetch_next_task`` is hit on every poll tick — a single transit
    blip there would skip a poll cycle silently. Same fix as ``_post``:
    transport-level timeouts join the retry budget shared with 5xx."""
    from client_agent.platform import PlatformClient

    sleeps: list[float] = []

    with respx.mock(base_url="https://platform.example") as mock:
        mock.get("/appliance/tasks/next").mock(
            side_effect=[
                httpx.ReadTimeout("simulated blip"),
                httpx.Response(204),
            ]
        )
        client = PlatformClient(
            base_url="https://platform.example",
            token="tok",
            sleep=sleeps.append,
        )

        result = client.fetch_next_task()

    assert result is None
    assert sleeps == [1]


# ----- 14. _get: ReadTimeout on every attempt → PlatformUnavailableError -----


def test_get_raises_unavailable_after_read_timeout_exhausts_retries() -> None:
    """Persistent transport-level timeouts on a GET path (poll-tick or
    upload-url fetch) surface as :class:`PlatformUnavailableError` —
    distinct from :class:`PlatformAuthError` so monitoring / restart
    policy can branch on cause."""
    from client_agent.platform import PlatformClient, PlatformUnavailableError

    sleeps: list[float] = []

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.get("/appliance/tasks/next").mock(
            side_effect=[
                httpx.ReadTimeout("blip 1"),
                httpx.ReadTimeout("blip 2"),
                httpx.ReadTimeout("blip 3"),
            ]
        )
        client = PlatformClient(
            base_url="https://platform.example",
            token="tok",
            sleep=sleeps.append,
        )

        with pytest.raises(PlatformUnavailableError):
            client.fetch_next_task()

    assert route.call_count == 3
    assert sleeps == [1, 2]


# ----- 15. _post: explicit default 30s timeout passed to httpx -----


def test_post_passes_default_30s_timeout_to_httpx() -> None:
    """httpx's library default is 5 s read — too tight for a CF-Worker +
    Neon round-trip under occasional cold-start latency (source incident,
    issue #42). Without an explicit timeout, a single 6-second cold start
    crashes a poll cycle. The appliance must pass an explicit 30 s timeout
    by default; the value lands in ``Request.extensions['timeout']``,
    which httpx populates from the ``timeout`` kwarg."""
    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/register").mock(
            return_value=httpx.Response(200, json={"appliance_id": "a", "tenant_id": "t"})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        client.register(agent_version="v")

    timeout_ext = route.calls.last.request.extensions["timeout"]
    # httpx normalizes a scalar timeout into a per-phase dict; every phase
    # should reflect the 30 s default so a slow Neon HTTP cold connection
    # gets the same headroom as a slow CF Worker cold start.
    assert timeout_ext == {"connect": 30.0, "read": 30.0, "write": 30.0, "pool": 30.0}


# ----- 16. PLATFORM_HTTP_TIMEOUT_S env overrides the default -----


def test_platform_http_timeout_s_env_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator override knob: a customer LAN with a known-slow ISP can
    dial the timeout up via env var without a code change; a tight
    development loop can dial it down. Value is sourced per-call from
    ``PLATFORM_HTTP_TIMEOUT_S`` so an operator can tweak the env file
    and restart systemd without rebuilding."""
    from client_agent.platform import PlatformClient

    monkeypatch.setenv("PLATFORM_HTTP_TIMEOUT_S", "8.5")

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/register").mock(
            return_value=httpx.Response(200, json={"appliance_id": "a", "tenant_id": "t"})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        client.register(agent_version="v")

    timeout_ext = route.calls.last.request.extensions["timeout"]
    assert timeout_ext == {"connect": 8.5, "read": 8.5, "write": 8.5, "pool": 8.5}


# ----- 17. _get: explicit timeout passed (AC #1 symmetry) -----


def test_get_passes_default_30s_timeout_to_httpx() -> None:
    """``_get`` must mirror ``_post`` — same env-sourced timeout. Without
    this the upload-url fetch and the poll-tick still fall back to
    httpx's 5 s default."""
    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.get("/appliance/tasks/next").mock(return_value=httpx.Response(204))
        client = PlatformClient(base_url="https://platform.example", token="tok")

        client.fetch_next_task()

    timeout_ext = route.calls.last.request.extensions["timeout"]
    assert timeout_ext == {"connect": 30.0, "read": 30.0, "write": 30.0, "pool": 30.0}


# ----- 18. claim_next_snapshot: 204 idle path -----


def test_claim_next_snapshot_returns_none_on_204() -> None:
    """No pending snapshot request → 204 No Content. The poller treats
    this as "queue is empty, sleep and retry" just like ``fetch_next_task``.
    Steady-state path — must not allocate or log noisily."""
    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.get("/appliance/snapshot/next").mock(return_value=httpx.Response(204))
        client = PlatformClient(base_url="https://platform.example", token="tok")

        result = client.claim_next_snapshot()

    assert result is None
    assert route.called
    assert route.calls.last.request.headers["authorization"] == "Bearer tok"


# ----- 19. claim_next_snapshot: 200 parses SnapshotClaim -----


def test_claim_next_snapshot_returns_claim_on_200() -> None:
    """Successful claim: platform returns ``{request_id, camera_id,
    upload_url, key, expires_in, content_type}``. The appliance PUTs the
    JPEG to ``upload_url`` and then POSTs status with ``request_id`` —
    those two fields are load-bearing, the rest are diagnostic."""
    from client_agent.platform import PlatformClient

    body = {
        "request_id": "req-xyz",
        "camera_id": "cam-1",
        "upload_url": "https://r2.example/signed?X-Amz-Signature=abc",
        "key": "tenants/t/snapshots/c/latest.jpg",
        "expires_in": 300,
        "content_type": "image/jpeg",
    }
    with respx.mock(base_url="https://platform.example") as mock:
        mock.get("/appliance/snapshot/next").mock(return_value=httpx.Response(200, json=body))
        client = PlatformClient(base_url="https://platform.example", token="tok")

        claim = client.claim_next_snapshot()

    assert claim is not None
    assert claim.request_id == "req-xyz"
    assert claim.camera_id == "cam-1"
    assert claim.upload_url == "https://r2.example/signed?X-Amz-Signature=abc"
    assert claim.content_type == "image/jpeg"


# ----- 19b. claim_next_snapshot: variant (gpu-exchange #137) -----


def test_claim_next_snapshot_parses_detail_variant() -> None:
    """The platform tags each claim with the variant it wants captured:
    ``thumbnail`` (640px/q4 card image) or ``detail`` (native resolution,
    q2, on-demand for an open preview modal). The grabber picks its
    ffmpeg profile from this, so it must survive the parse."""
    from client_agent.platform import PlatformClient

    body = {
        "request_id": "req-xyz",
        "camera_id": "cam-1",
        "variant": "detail",
        "upload_url": "https://r2.example/signed?X-Amz-Signature=abc",
        "key": "tenants/t/snapshots/c/detail.jpg",
        "expires_in": 300,
        "content_type": "image/jpeg",
    }
    with respx.mock(base_url="https://platform.example") as mock:
        mock.get("/appliance/snapshot/next").mock(return_value=httpx.Response(200, json=body))
        client = PlatformClient(base_url="https://platform.example", token="tok")

        claim = client.claim_next_snapshot()

    assert claim is not None
    assert claim.variant == "detail"


def test_claim_next_snapshot_defaults_variant_when_platform_omits_it() -> None:
    """Rolling-deploy guard: an appliance updated ahead of the platform
    will see claims with no ``variant`` key. Every other field is parsed
    with a strict bracket lookup, so without a default this raises
    KeyError and the snapshot queue stalls until the Worker ships.
    Absent ``variant`` means the pre-#137 contract, i.e. thumbnail."""
    from client_agent.platform import PlatformClient

    body = {
        "request_id": "req-old",
        "camera_id": "cam-1",
        "upload_url": "https://r2.example/signed?X-Amz-Signature=abc",
        "key": "tenants/t/snapshots/c/latest.jpg",
        "expires_in": 300,
        "content_type": "image/jpeg",
    }
    with respx.mock(base_url="https://platform.example") as mock:
        mock.get("/appliance/snapshot/next").mock(return_value=httpx.Response(200, json=body))
        client = PlatformClient(base_url="https://platform.example", token="tok")

        claim = client.claim_next_snapshot()

    assert claim is not None
    assert claim.variant == "thumbnail"


def test_claim_next_snapshot_falls_back_to_thumbnail_on_unknown_variant() -> None:
    """An unrecognised variant from a newer platform must not crash the
    poller or silently capture the wrong profile. Degrade to the
    compatibility default — a 640px image is a poor answer, but a stalled
    snapshot queue is a worse one."""
    from client_agent.platform import PlatformClient

    body = {
        "request_id": "req-future",
        "camera_id": "cam-1",
        "variant": "8k-hdr",
        "upload_url": "https://r2.example/signed?X-Amz-Signature=abc",
        "key": "tenants/t/snapshots/c/latest.jpg",
        "expires_in": 300,
        "content_type": "image/jpeg",
    }
    with respx.mock(base_url="https://platform.example") as mock:
        mock.get("/appliance/snapshot/next").mock(return_value=httpx.Response(200, json=body))
        client = PlatformClient(base_url="https://platform.example", token="tok")

        claim = client.claim_next_snapshot()

    assert claim is not None
    assert claim.variant == "thumbnail"


# ----- 20. report_snapshot_status: uploaded vs failed body shapes -----


def test_report_snapshot_status_posts_uploaded_and_failed_bodies() -> None:
    """The platform's ``ApplianceSnapshotStatusRequestSchema`` is a
    discriminated union: ``uploaded`` carries no extra fields (the
    server already knows the r2_key from the row), ``failed`` accepts
    an optional ``error``. Missing required shape → 400 from validator."""
    from client_agent.platform import PlatformClient

    seen: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.append(_json.loads(request.read()))
        return httpx.Response(200, json={"ok": True})

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/snapshot/req-1/status").mock(side_effect=_capture)
        mock.post("/appliance/snapshot/req-2/status").mock(side_effect=_capture)
        client = PlatformClient(base_url="https://platform.example", token="tok")

        client.report_snapshot_status("req-1", status="uploaded")
        client.report_snapshot_status("req-2", status="failed", error="rtsp grab failed")

    assert seen == [
        {"status": "uploaded"},
        {"status": "failed", "error": "rtsp grab failed"},
    ]


# ----- 20. update_task_status: actual_start serialized on the uploaded call (#91) -----


def test_update_task_status_serializes_actual_start_iso8601() -> None:
    """The appliance reports where the delivered clip really begins so the
    platform can stamp ``recording_start`` from the artifact rather than the
    request (#91, gpu-exchange#154). Sent as an ISO-8601 string — a raw
    ``datetime`` is not JSON-serializable and would raise at the httpx
    boundary, not at the call site where it would be obvious.

    The key is omitted entirely when absent so bodies for the statuses that
    have no such notion (``recording`` / ``uploading`` / ``failed``) stay
    exactly as they were — the platform's schema is a discriminated union
    and every variant is validated separately."""
    from client_agent.platform import PlatformClient

    seen_bodies: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen_bodies.append(_json.loads(request.read()))
        return httpx.Response(204)

    delivered = datetime(2026, 7, 18, 5, 13, 17, tzinfo=UTC)

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/tasks/task-abc/status").mock(side_effect=_capture)
        client = PlatformClient(base_url="https://platform.example", token="tok")

        client.update_task_status(
            "task-abc",
            status="uploaded",
            chunk_r2_key="tenants/t/appliance-uploads/task-abc/chunk_0.mp4",
            actual_start=delivered,
        )
        # No actual_start → the key must not appear at all (not ``null``).
        client.update_task_status("task-abc", status="recording")

    assert seen_bodies == [
        {
            "status": "uploaded",
            "chunk_r2_key": "tenants/t/appliance-uploads/task-abc/chunk_0.mp4",
            "actual_start": "2026-07-18T05:13:17+00:00",
        },
        {"status": "recording"},
    ]


# ----- 6.6. heartbeat: disk + buffer depth + agent_version (#92) -----


def test_heartbeat_sends_disk_buffer_depth_and_agent_version() -> None:
    """Health telemetry rides as *top-level typed fields*, not nested in the
    free-form ``status`` dict. ``status`` is ``z.record(z.string(),
    z.unknown())`` platform-side — unvalidated, which is a large part of why
    it was easy to accept and then silently drop. Typed fields get Zod
    validation and force deliberate handling (gpu-exchange#157).

    ``buffer_depth`` values must carry a UTC offset: the platform validates
    with ``z.string().datetime({ offset: true })``, so Python's
    ``isoformat()`` (``+00:00``) passes where a naive datetime would fail the
    whole beat."""
    from datetime import UTC, datetime

    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": {"cameras": []}})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        client.heartbeat(
            status={},
            recording_cameras=["cam-1"],
            agent_version="ce95aad",
            disk_free_bytes=38_654_705_664,
            disk_total_bytes=123_480_309_760,
            buffer_depth={"cam-1": datetime(2026, 7, 21, 4, 52, 22, tzinfo=UTC)},
        )

    import json as _json

    body = _json.loads(route.calls.last.request.read())
    assert body == {
        "status": {},
        "recording_cameras": ["cam-1"],
        "agent_version": "ce95aad",
        "disk_free_bytes": 38_654_705_664,
        "disk_total_bytes": 123_480_309_760,
        "buffer_depth": {"cam-1": "2026-07-21T04:52:22+00:00"},
    }


def test_heartbeat_sends_buffer_newest_alongside_depth() -> None:
    """``buffer_newest`` is what lets the platform tell a recording camera
    from a stopped one (#94 / gpu-exchange#158).

    It rides beside ``buffer_depth``, not instead of it: depth answers "how
    far back can this box serve", newest answers "is it still writing". The
    platform warns once newest is more than ``BUFFER_STALE_SECONDS`` (300 s
    = 5 × the recorder's 60 s segment) behind now.

    Same offset requirement as ``buffer_depth`` — ``z.string().datetime({
    offset: true })`` rejects a naive or ``Z``-only value and fails the whole
    beat, so Python's ``isoformat()`` on a tz-aware datetime is the contract."""
    from datetime import UTC, datetime

    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": {"cameras": []}})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        client.heartbeat(
            status={},
            recording_cameras=["cam-1"],
            buffer_depth={"cam-1": datetime(2026, 7, 21, 4, 52, 22, tzinfo=UTC)},
            buffer_newest={"cam-1": datetime(2026, 7, 21, 9, 51, 3, tzinfo=UTC)},
        )

    import json as _json

    body = _json.loads(route.calls.last.request.read())
    assert body["buffer_depth"] == {"cam-1": "2026-07-21T04:52:22+00:00"}
    assert body["buffer_newest"] == {"cam-1": "2026-07-21T09:51:03+00:00"}


def test_heartbeat_sends_buffer_depth_alone_when_newest_absent() -> None:
    """The two maps are independent on the wire — the platform tolerates
    either alone. That independence is what let the platform half ship first
    (``buffer_newest`` optional in ``ApplianceHeartbeatRequestSchema``), and
    it must keep holding in reverse so a partial reading never suppresses the
    good one."""
    from datetime import UTC, datetime

    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": {"cameras": []}})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        client.heartbeat(
            status={},
            recording_cameras=["cam-1"],
            buffer_depth={"cam-1": datetime(2026, 7, 21, 4, 52, 22, tzinfo=UTC)},
        )

    import json as _json

    body = _json.loads(route.calls.last.request.read())
    assert body["buffer_depth"] == {"cam-1": "2026-07-21T04:52:22+00:00"}
    assert "buffer_newest" not in body


def test_heartbeat_omits_telemetry_keys_when_unavailable() -> None:
    """Absent, not ``null`` (#91's ``actual_start`` convention).

    The platform applies each field with an ``!== undefined`` guard, so an
    explicit null is not "no reading" — it either fails Zod (``buffer_depth``
    values are typed as datetime strings) or, for the numeric fields,
    overwrites a previously-good stored value with nothing. A box that cannot
    read its disk must leave the last known figure standing, not erase it.

    An *empty* ``buffer_depth`` / ``buffer_newest`` is omitted for the same
    reason: it carries no observation, only the fact that nothing is buffered
    yet."""
    from client_agent.platform import PlatformClient

    with respx.mock(base_url="https://platform.example") as mock:
        route = mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": {"cameras": []}})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        client.heartbeat(
            status={},
            recording_cameras=[],
            agent_version=None,
            disk_free_bytes=None,
            disk_total_bytes=None,
            buffer_depth={},
            buffer_newest={},
        )

    import json as _json

    body = _json.loads(route.calls.last.request.read())
    assert body == {"status": {}, "recording_cameras": []}
    for key in (
        "agent_version",
        "disk_free_bytes",
        "disk_total_bytes",
        "buffer_depth",
        "buffer_newest",
    ):
        assert key not in body
