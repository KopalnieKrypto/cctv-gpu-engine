"""GPU Exchange platform-integration adapter (issue #26).

Bearer-authenticated HTTP client against the platform endpoints specified
in ``KopalnieKrypto/gpu-exchange/docs/design/09-client-appliance.md``.

Three operations:

* :meth:`PlatformClient.register` — first contact at boot. Appliance announces
  itself; platform binds it to a tenant via the Bearer token.
* :meth:`PlatformClient.push_cameras` — after ONVIF discovery, push the
  camera list with ``enabled=False``. Operator activates per camera in the
  platform UI; activation flows back through ``heartbeat``.
* :meth:`PlatformClient.heartbeat` — periodic keepalive (30 s in production).
  Carries the appliance's current state; response carries the desired
  camera config (which to record, which to leave idle).

All HTTP goes outbound from the appliance — there is no inbound listener
from the platform side, so a NAT'd home network needs no port forwarding.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

import httpx

# Retry policy mirrors gpu-service/http_retry.py: 3 attempts total with
# 1s, 2s sleeps between them (the "4s" from the prose would precede a
# 4th attempt we never make). Suitable for a single appliance against a
# single platform — no thundering-herd jitter needed.
_DEFAULT_BACKOFFS: tuple[int, ...] = (1, 2)
_DEFAULT_ATTEMPTS = 3

# httpx's library default is 5 s read — too tight for a CF-Worker + Neon
# round-trip under occasional cold-start latency (issue #42). 30 s gives
# enough headroom that a single cold start does not cascade into a
# crashed poll thread, while still failing fast on genuine outages.
# Operator override via env var (no code change, no rebuild).
_DEFAULT_TIMEOUT_S = 30.0

# Snapshot capture profiles (gpu-exchange #137). ``thumbnail`` is the
# compatibility default: it is what every claim meant before the platform
# grew the field, so both an absent and an unrecognised value resolve here.
SnapshotVariant = Literal["thumbnail", "detail"]
_SNAPSHOT_VARIANTS: frozenset[str] = frozenset({"thumbnail", "detail"})
_SNAPSHOT_VARIANT_DEFAULT: SnapshotVariant = "thumbnail"


def _parse_snapshot_variant(raw: object) -> SnapshotVariant:
    """Resolve the claim's ``variant`` field, tolerantly.

    Deliberately the one non-strict field in :meth:`claim_next_snapshot`.
    The rest use bracket lookups so a malformed claim fails loudly, but
    ``variant`` must survive two rolling-deploy skews in either direction:
    an older platform that omits it, and a newer one that invents a
    profile this build has never heard of. Both degrade to a thumbnail —
    a lower-resolution image beats a stalled snapshot queue.
    """
    if isinstance(raw, str) and raw in _SNAPSHOT_VARIANTS:
        return cast(SnapshotVariant, raw)
    return _SNAPSHOT_VARIANT_DEFAULT


def _resolve_timeout() -> float:
    """Read the per-request timeout from the env each call.

    Lets the operator dial the timeout up (slow customer ISP) or down
    (tight dev loop) by editing ``/etc/cctv-client/platform.env`` and
    restarting systemd — no rebuild needed."""
    return float(os.environ.get("PLATFORM_HTTP_TIMEOUT_S", _DEFAULT_TIMEOUT_S))


class PlatformAuthError(RuntimeError):
    """Raised when the platform rejects the appliance's Bearer token (401).

    Token rotation is a manual operator action (regenerate in the platform
    admin UI, paste into ``/etc/cctv-client/platform.env``, restart the
    service). Retrying inside the request loop would mask the configuration
    error behind a stale connection. The appliance-level entrypoint
    converts this into a non-zero exit so systemd surfaces it loudly."""


class PlatformUnavailableError(RuntimeError):
    """Raised when the platform returns 5xx for every retry attempt.

    Distinct from :class:`PlatformAuthError` so monitoring / init scripts
    can branch on cause (rotate token vs. wait for platform recovery).
    The appliance-level entrypoint converts this into a non-zero exit so
    systemd's restart policy can back off and try again later."""


class PlatformRequestError(RuntimeError):
    """Raised when the platform returns a non-401 4xx for a request that
    expects a parseable JSON body (currently only ``get_upload_url``).

    Carries the HTTP status so callers can build a user-facing message
    (e.g. ``PresignedUploader`` surfaces "platform refused upload-url
    (403)" without further dispatch on body content). 401 is handled
    separately by :class:`PlatformAuthError` — token rotation is an
    operator concern, not a request-level error."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class HeartbeatResponse:
    """Response of POST ``/appliance/heartbeat``.

    Carries the desired camera config (``{"cameras": [{id, enabled,
    rtsp_url}, ...]}``). The appliance compares this against its current
    recorder threads and spawns / stops as needed.

    Held as a free-form ``dict`` (rather than a nested dataclass) so the
    server can extend the schema (preset hints, retention policy) without
    a client-library bump."""

    config: dict

    @property
    def settings(self) -> dict | None:
        """The nested ``config.settings`` block, or ``None`` if absent (#85).

        The platform ships the runtime-config block under ``config`` on every
        heartbeat (alongside ``config.cameras``). Surfacing it here gives the
        applier the same ``.settings`` accessor register exposes, so the two
        deliver paths converge without each call site reaching into the raw
        config dict. ``None`` (older platform / cameras-only beat) means the
        box keeps its current values."""
        return self.config.get("settings")


@dataclass(frozen=True)
class RegisterResponse:
    """Response of POST ``/appliance/register``.

    Carries the platform-issued ``appliance_id`` (used in subsequent
    callbacks for human-readable logs) and the ``tenant_id`` the Bearer
    token resolved to. The appliance does not act on ``tenant_id`` —
    it's surfaced only so the operator can confirm the right token was
    installed without a separate platform lookup."""

    appliance_id: str
    tenant_id: str
    settings: dict | None = None
    """Platform-delivered runtime config (issue #85).

    Snake-case block (``buffer_hours``, ``polling_interval_seconds``,
    ``heartbeat_interval_seconds``, ``upload_chunk_bytes``) the platform now
    ships at register time so the box overrides its install-time env defaults
    from beat zero, not just after the first heartbeat. ``None`` when the
    platform predates the feature — the applier then keeps env cold-start
    values (see :class:`client_agent.runtime_config.RuntimeConfig`)."""


class PlatformClient:
    """Bearer-authenticated client against the GPU Exchange platform."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._sleep = sleep

    def _post(self, path: str, *, json: dict) -> httpx.Response:
        """POST ``path`` with Bearer auth, 5xx retry, and typed errors.

        Retries up to ``_DEFAULT_ATTEMPTS`` times with 1s/2s backoff on
        5xx. Raises :class:`PlatformAuthError` on 401 (fail-fast — token
        rotation is an operator action) and :class:`PlatformUnavailableError`
        if the retry budget is exhausted. Other 4xx pass through; the
        caller can branch on the response."""
        last: httpx.Response | None = None
        for i in range(_DEFAULT_ATTEMPTS):
            try:
                last = httpx.post(
                    f"{self._base_url}{path}",
                    headers={"Authorization": f"Bearer {self._token}"},
                    json=json,
                    timeout=_resolve_timeout(),
                )
            except httpx.TransportError:
                # Any transport-level failure (timeout, ConnectError from a
                # DNS blip / connection refused mid-deploy, ReadError) shares
                # the 5xx retry budget — #42 only caught timeouts, leaving a
                # single ConnectError to wedge the task (issue #54).
                last = None
            if last is not None:
                if last.status_code == 401:
                    raise PlatformAuthError("platform rejected APPLIANCE_TOKEN (401)")
                if last.status_code < 500:
                    return last
            if i + 1 >= _DEFAULT_ATTEMPTS:
                break
            self._sleep(
                _DEFAULT_BACKOFFS[i] if i < len(_DEFAULT_BACKOFFS) else _DEFAULT_BACKOFFS[-1]
            )
        if last is None:
            raise PlatformUnavailableError("platform timed out after retries")
        raise PlatformUnavailableError(f"platform returned {last.status_code} after retries")

    def register(
        self,
        *,
        agent_version: str,
        host_info: dict | None = None,
    ) -> RegisterResponse:
        """POST ``/appliance/register`` — boot-time announcement.

        Body matches DD-09 (gpu-exchange) canonical shape: ``agent_version``
        (semver string the platform stores on the appliance row for the
        admin UI) and an optional ``host_info`` blob (platform/arch/kernel
        diagnostics, persisted as jsonb)."""
        body: dict = {"agent_version": agent_version}
        if host_info is not None:
            body["host_info"] = host_info
        response = self._post("/appliance/register", json=body)
        data = response.json()
        return RegisterResponse(
            appliance_id=data["appliance_id"],
            tenant_id=data["tenant_id"],
            settings=data.get("settings"),
        )

    def push_cameras(self, cameras: list[dict]) -> None:
        """POST ``/appliance/cameras`` — push the discovered camera list.

        Cameras arrive at the platform with ``enabled=False`` (the operator
        opts them in from the platform UI). The server response is 204 —
        activation state is fetched on the next heartbeat, not returned
        here."""
        self._post("/appliance/cameras", json={"cameras": cameras})

    def heartbeat(self, *, status: dict, recording_cameras: list[str]) -> HeartbeatResponse:
        """POST ``/appliance/heartbeat`` — keepalive + config pull.

        Body carries the appliance's current state (``status``) and which
        cameras it is currently recording (``recording_cameras``) so the
        platform can detect drift between requested and actual state. The
        response carries the desired config; the appliance reconciles by
        spawning / stopping recorder threads to match."""
        response = self._post(
            "/appliance/heartbeat",
            json={"status": status, "recording_cameras": recording_cameras},
        )
        data = response.json()
        return HeartbeatResponse(config=data["config"])

    def fetch_next_task(self) -> Task | None:
        """GET ``/appliance/tasks/next`` — claim the next pending task.

        Returns a :class:`Task` if the platform has work, ``None`` on 204
        (idle steady state — appliance sleeps and retries on the next
        poll tick). Server-side claiming is atomic: a single GET both
        reserves and returns the task, so two appliances racing for the
        same task is platform's problem to solve."""
        response = self._get("/appliance/tasks/next")
        if response.status_code == 204:
            return None
        data = response.json()["task"]
        return Task(
            id=data["id"],
            camera_id=data["camera_id"],
            start_time=_parse_iso(data["start_time"]),
            end_time=_parse_iso(data["end_time"]),
        )

    def update_task_status(
        self,
        task_id: str,
        *,
        status: str,
        error: str | None = None,
        chunk_r2_key: str | None = None,
    ) -> None:
        """POST ``/appliance/tasks/{task_id}/status`` — status transition.

        The platform's ``ApplianceTaskStatusRequestSchema`` is a discriminated
        union: ``uploaded`` requires ``chunk_r2_key`` (the R2 key the
        appliance just PUT into), ``failed`` accepts an optional ``error``.
        Missing fields → 400 from the validator, hence the per-status
        argument list rather than a single opaque dict."""
        body: dict = {"status": status}
        if error is not None:
            body["error"] = error
        if chunk_r2_key is not None:
            body["chunk_r2_key"] = chunk_r2_key
        self._post(f"/appliance/tasks/{task_id}/status", json=body)

    def get_upload_url(self, task_id: str, chunk_n: int) -> UploadUrl:
        """GET ``/appliance/upload-url`` — fetch a fresh presigned PUT URL.

        The platform binds the URL to ``tenants/{tid}/appliance-uploads/
        {task_id}/chunk_N.mp4`` based on the Bearer token's tenant. A
        cross-tenant ``task_id`` produces a 403 at this hop, **not** at
        the PUT, so a compromised appliance can never even learn a
        sibling tenant's R2 key. The 30-min default TTL means
        :class:`PresignedUploader` may need a refresh mid-upload if the
        chunk PUT lands after expiry — the response's ``expires_in`` is
        surfaced for diagnostics but not enforced client-side (the R2
        edge does that for us)."""
        response = self._get(
            "/appliance/upload-url", params={"task_id": task_id, "chunk_n": chunk_n}
        )
        if response.status_code >= 400:
            # 4xx here is meaningful: tenant mismatch (403), unknown
            # task_id (404), bad chunk_n (400). The uploader converts
            # this to a failed UploadResult; we surface only the status
            # because parsed-body error fields vary across deployments.
            raise PlatformRequestError(
                response.status_code,
                f"platform refused upload-url ({response.status_code})",
            )
        data = response.json()
        return UploadUrl(url=data["url"], key=data["key"], expires_in=int(data["expires_in"]))

    def claim_next_snapshot(self) -> SnapshotClaim | None:
        """GET ``/appliance/snapshot/next`` — claim the oldest pending
        snapshot request for this appliance (gpu-exchange #91).

        Returns a :class:`SnapshotClaim` on 200 (the platform has work,
        URL is ready to PUT into) or ``None`` on 204 (idle steady state).
        Atomic claim is platform-side: the same GET reserves the row and
        hands back the presigned URL in one round-trip."""
        response = self._get("/appliance/snapshot/next")
        if response.status_code == 204:
            return None
        data = response.json()
        return SnapshotClaim(
            request_id=data["request_id"],
            camera_id=data["camera_id"],
            upload_url=data["upload_url"],
            key=data["key"],
            expires_in=int(data["expires_in"]),
            content_type=data["content_type"],
            variant=_parse_snapshot_variant(data.get("variant")),
        )

    def report_snapshot_status(
        self,
        request_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        """POST ``/appliance/snapshot/{request_id}/status`` — terminal
        transition for one snapshot request (gpu-exchange #91).

        Discriminated-union body matching the platform's
        ``ApplianceSnapshotStatusRequestSchema``: ``uploaded`` carries
        no extra fields (the server derives the R2 key from the row),
        ``failed`` accepts an optional ``error`` string capped at 500
        chars server-side."""
        body: dict = {"status": status}
        if error is not None:
            body["error"] = error
        self._post(f"/appliance/snapshot/{request_id}/status", json=body)

    def _get(self, path: str, *, params: dict | None = None) -> httpx.Response:
        """GET ``path`` with Bearer auth, 5xx retry, and typed errors.

        Mirrors :meth:`_post` — split into its own helper because httpx's
        sync API has separate ``get`` / ``post`` functions and combining
        them through a method dispatch would obscure the call site."""
        last: httpx.Response | None = None
        for i in range(_DEFAULT_ATTEMPTS):
            try:
                last = httpx.get(
                    f"{self._base_url}{path}",
                    headers={"Authorization": f"Bearer {self._token}"},
                    params=params,
                    timeout=_resolve_timeout(),
                )
            except httpx.TransportError:
                # See _post: transport errors (timeout / ConnectError /
                # ReadError) share the 5xx retry budget (issue #54).
                last = None
            if last is not None:
                if last.status_code == 401:
                    raise PlatformAuthError("platform rejected APPLIANCE_TOKEN (401)")
                if last.status_code < 500:
                    return last
            if i + 1 >= _DEFAULT_ATTEMPTS:
                break
            self._sleep(
                _DEFAULT_BACKOFFS[i] if i < len(_DEFAULT_BACKOFFS) else _DEFAULT_BACKOFFS[-1]
            )
        if last is None:
            raise PlatformUnavailableError("platform timed out after retries")
        raise PlatformUnavailableError(f"platform returned {last.status_code} after retries")


@dataclass(frozen=True)
class SnapshotClaim:
    """One claimed snapshot request (gpu-exchange #91).

    The appliance PUTs the JPEG to ``upload_url`` (presigned, no Bearer
    needed) and then POSTs status with ``request_id`` so the platform
    can flip the row from ``claimed`` to ``uploaded`` / ``failed``.

    ``key`` and ``expires_in`` are diagnostics only — the URL carries
    its own signature, the R2 edge enforces expiry. ``content_type`` is
    always ``"image/jpeg"`` per the platform contract but surfaced so
    the PUT can attach the same header the URL was signed with.

    ``variant`` (gpu-exchange #137) selects the capture profile. It is
    advisory to the grab only — the platform already fixed the R2 key
    when it signed ``upload_url``, so getting this wrong yields the wrong
    pixels, never bytes in the wrong place."""

    request_id: str
    camera_id: str
    upload_url: str
    key: str
    expires_in: int
    content_type: str
    variant: SnapshotVariant = "thumbnail"


@dataclass(frozen=True)
class UploadUrl:
    """One single-use presigned PUT URL for a chunk upload (issue #28).

    ``url`` is the full presigned URL (carries its own signature in the
    query string — sending a Bearer header alongside it confuses some
    S3-compatible backends). ``key`` is the R2 key the platform bound
    the URL to (``tenants/{tid}/appliance-uploads/{task_id}/chunk_N.mp4``)
    and is surfaced so the uploader can log / return it for audit.
    ``expires_in`` is operator-readable diagnostics only — the R2 edge
    enforces expiry, the client just refreshes on 403."""

    url: str
    key: str
    expires_in: int


@dataclass(frozen=True)
class Task:
    """One task pulled off the platform's queue.

    The poller hands these fields to :class:`RollingBuffer.chunks_in_range`
    and :func:`ffmpeg_trim.trim_and_concat`. ISO-8601 strings get parsed
    into aware datetimes at this boundary so downstream code never has
    to think about timezone parsing again."""

    id: str
    camera_id: str
    start_time: datetime
    end_time: datetime


def _parse_iso(s: str) -> datetime:
    """Parse a platform-emitted ISO-8601 timestamp into an aware datetime.

    The platform emits ``Z`` for UTC (per DD-09); Python 3.11's
    ``fromisoformat`` accepts that natively, but we keep the explicit
    replace as belt-and-suspenders for older platform code paths."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
