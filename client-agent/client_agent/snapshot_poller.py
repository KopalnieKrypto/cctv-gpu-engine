"""Snapshot poller for the client appliance (gpu-exchange issue #91).

The platform's ``/cameras`` UI in gpu-exchange enqueues a
``snapshot_requests`` row whenever a tab opens or polls the per-camera
thumbnail proxy. The appliance side polls ``GET /appliance/snapshot/next``
for those rows, grabs a JPEG locally (same path the on-site Flask
``/cameras/<id>/snapshot`` route from #41 uses), and PUTs it to the
platform-supplied presigned R2 URL — closing the loop:

    [browser tab] → /api/data/appliances/.../cameras/.../snapshot (gpu-exchange)
                       → 404 SNAPSHOT_PENDING (first call enqueues row)
    [this poller]   ←  GET /appliance/snapshot/next  (200 with presigned URL)
                    →  RTSP/HTTP grab → JPEG
                    →  PUT presigned URL with image/jpeg body
                    →  POST /appliance/snapshot/:id/status {uploaded}
    [browser tab]   →  next 30 s refetch → 200 image/jpeg from R2

Mirrors :class:`client_agent.poller.TaskPoller` in shape: single-flight,
inject every system boundary (platform / resolver / grabber / http put),
``run_once``+``run`` split so unit tests cover the former and the daemon
thread wraps the latter with the same broad-except as the task poller.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

from client_agent.discovery import scrub_url_credentials
from client_agent.platform import SnapshotClaim
from client_agent.snapshot import SnapshotGrabberFn
from client_agent.web import CameraSnapshotSource

logger = logging.getLogger(__name__)

# Matches the Flask snapshot route's per-grab budget (web.py constant of
# the same name). The 30 s appliance-side TTL absorbs frequent polls;
# the grab itself must fail fast so a dead camera does not back up the
# poll loop. 5 s covers a clean RTSP handshake + first-frame on a
# healthy LAN with significant headroom.
_GRAB_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class PutResult:
    """Outcome of one PUT to a presigned R2 URL.

    Held as a value object (mirroring :class:`client_agent.uploader.UploadResult`)
    so the poller can branch on success / failure without an exception
    dance across the network boundary. ``status_code`` is None when the
    failure happened pre-response (DNS / TCP / TLS) — same convention
    as the uploader."""

    success: bool
    status_code: int | None = None
    error: str | None = None


HttpPutFn = Callable[[str, bytes, str], PutResult]
"""Injection point for the presigned-URL PUT. Args are (url, body, content_type)."""

CameraResolver = Callable[[str], CameraSnapshotSource | None]
"""Camera-id → snapshot source. Same Protocol-shape as the Flask route's
``camera_resolver`` so the appliance can wire one shared registry into
both call sites (the local on-site preview AND this platform-facing
poller). Returning ``None`` means "no source for this camera_id" — the
poller treats that as a terminal failure for the request."""


class _PlatformLike(Protocol):
    """Narrow surface of :class:`client_agent.platform.PlatformClient`
    this poller needs. Declared as a Protocol so the tests can pass
    in-memory fakes without touching httpx or its retry semantics."""

    def claim_next_snapshot(self) -> SnapshotClaim | None: ...
    def report_snapshot_status(
        self, request_id: str, *, status: str, error: str | None = None
    ) -> None: ...


class SnapshotPoller:
    """Single-flight snapshot pump against the platform queue."""

    def __init__(
        self,
        *,
        platform: _PlatformLike,
        camera_resolver: CameraResolver,
        snapshot_grabber: SnapshotGrabberFn,
        http_put: HttpPutFn,
        poll_interval_s: float = 5.0,
        grab_timeout_s: float = _GRAB_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._platform = platform
        self._resolve_camera = camera_resolver
        self._grab = snapshot_grabber
        self._put = http_put
        self._poll_interval_s = poll_interval_s
        self._grab_timeout_s = grab_timeout_s
        self._sleep = sleep

    def run_once(self) -> bool:
        """One claim-grab-put-report cycle.

        Returns ``True`` if a request was handled (success or failure
        — either way the row moves out of ``claimed``), ``False`` if
        the platform was idle (no row, no work done). The blocking
        :meth:`run` loop uses the boolean to decide its inter-poll sleep."""
        claim = self._platform.claim_next_snapshot()
        if claim is None:
            return False

        # 1. Resolve camera → source URL. A missing source means the platform
        #    holds a snapshot_requests row for a camera the appliance no longer
        #    knows (recently removed from heartbeat config). Report failed
        #    immediately — never attempt an RTSP open on a guess URL.
        source = self._resolve_camera(claim.camera_id)
        if source is None:
            self._report_failed(
                claim.request_id, f"camera_id not in heartbeat config: {claim.camera_id}"
            )
            return True

        # 2. Grab JPEG locally. Prefer the vendor's native HTTP snapshot
        #    endpoint (single GET, no RTSP handshake) when discovery
        #    surfaced one — mirrors the on-site web.py preference order.
        url = source.snapshot_url or source.rtsp_url
        try:
            jpeg = self._grab(url, self._grab_timeout_s)
        except Exception as exc:  # noqa: BLE001 — cv2/ffmpeg/urllib failures
            # Log server-side (with full detail incl. URL → camera_id)
            # but DON'T echo the URL into the failure message we report
            # to the platform: the RTSP-scan code path embeds
            # ``user:pass@`` in the url, and the failure column on the
            # platform might be displayed to admins of a different
            # tenant. Keep the platform-facing message scrubbed.
            logger.warning("snapshot grab failed for camera_id=%s: %s", claim.camera_id, exc)
            # Scrub before the message crosses to the platform's error column
            # (issue #53): the grabber's exception embeds the url, which now
            # carries injected RTSP userinfo. The local log above stays full.
            self._report_failed(
                claim.request_id,
                scrub_url_credentials(f"grab failed for camera {claim.camera_id}: {exc}"),
            )
            return True

        # 3. PUT to presigned URL. The URL carries its own SigV4
        #    signature in the query string — no Bearer header.
        put = self._put(claim.upload_url, jpeg, claim.content_type)
        if not put.success:
            error = put.error or f"R2 PUT failed with status {put.status_code}"
            logger.warning(
                "snapshot PUT failed for request_id=%s (camera_id=%s): %s",
                claim.request_id,
                claim.camera_id,
                error,
            )
            # A transport-error PUT failure can embed the presigned R2 url
            # (SigV4 query secrets) — scrub before it reaches the platform (#53).
            self._report_failed(claim.request_id, scrub_url_credentials(error))
            return True

        # 4. Success — flip the row to uploaded so the next browser
        #    refetch (30 s tick) gets a 200 image/jpeg from R2.
        self._platform.report_snapshot_status(claim.request_id, status="uploaded")
        return True

    def _report_failed(self, request_id: str, error: str) -> None:
        """Best-effort failure callback. Swallows its own exceptions so a
        platform 5xx on the status post doesn't take down the poll loop —
        the row stays in ``claimed`` and the platform's stale-claim
        sweeper (or the next operator-triggered re-enqueue) recovers."""
        try:
            self._platform.report_snapshot_status(request_id, status="failed", error=error)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to report status=failed for request_id=%s: %s", request_id, exc)

    @staticmethod
    def default_http_put(url: str, body: bytes, content_type: str) -> PutResult:
        """Production PUT against a presigned R2 URL.

        Distinct from :class:`client_agent.uploader.PresignedUploader` —
        the uploader is task-chunk-shaped (fetches a fresh URL per chunk
        from the platform, retries on signature expiry). Snapshot PUTs
        are one-shot (the URL comes embedded in the claim) and the
        thumbnail is regenerated on the next poll if anything goes
        wrong, so retry-and-refresh would be wasted effort. Single
        attempt, generous read/write timeout, classify the outcome."""
        try:
            response = httpx.put(
                url,
                content=body,
                headers={"content-type": content_type},
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
            )
        except httpx.HTTPError as exc:
            return PutResult(success=False, error=f"transport error: {exc}")
        if 200 <= response.status_code < 300:
            return PutResult(success=True, status_code=response.status_code)
        return PutResult(
            success=False,
            status_code=response.status_code,
            error=f"R2 PUT returned {response.status_code}",
        )

    def run(self) -> None:
        """Blocking poll loop — production entrypoint (daemon thread).

        Iterates :meth:`run_once` forever, sleeping ``poll_interval_s``
        only when the queue was idle (so a burst of pending requests
        drains as fast as the appliance can grab + PUT). Broad-except
        mirrors :meth:`client_agent.poller.TaskPoller.run` — a transient
        ``httpx.ConnectError`` from a Wi-Fi blip must not kill the daemon
        thread or the appliance heartbeats normally but never serves
        thumbnails again until restart."""
        while True:
            try:
                handled = self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("snapshot poller iteration failed: %s", exc)
                handled = False
            if not handled:
                self._sleep(self._poll_interval_s)
