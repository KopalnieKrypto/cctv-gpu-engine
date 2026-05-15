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

import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

# Retry policy mirrors gpu-service/http_retry.py: 3 attempts total with
# 1s, 2s sleeps between them (the "4s" from the prose would precede a
# 4th attempt we never make). Suitable for a single appliance against a
# single platform — no thundering-herd jitter needed.
_DEFAULT_BACKOFFS: tuple[int, ...] = (1, 2)
_DEFAULT_ATTEMPTS = 3


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
            last = httpx.post(
                f"{self._base_url}{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                json=json,
            )
            if last.status_code == 401:
                raise PlatformAuthError("platform rejected APPLIANCE_TOKEN (401)")
            if last.status_code < 500:
                return last
            if i + 1 >= _DEFAULT_ATTEMPTS:
                break
            self._sleep(
                _DEFAULT_BACKOFFS[i] if i < len(_DEFAULT_BACKOFFS) else _DEFAULT_BACKOFFS[-1]
            )
        assert last is not None
        raise PlatformUnavailableError(f"platform returned {last.status_code} after retries")

    def register(self, *, hostname: str, version: str) -> RegisterResponse:
        """POST ``/appliance/register`` — boot-time announcement."""
        response = self._post(
            "/appliance/register",
            json={"hostname": hostname, "version": version},
        )
        data = response.json()
        return RegisterResponse(
            appliance_id=data["appliance_id"],
            tenant_id=data["tenant_id"],
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
