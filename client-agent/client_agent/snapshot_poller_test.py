"""Tests for the snapshot poller (gpu-exchange issue #91).

The poller is the bridge between the platform's snapshot_requests queue
(rows the gpu-exchange web UI enqueues when a tab opens /cameras) and
the appliance's local JPEG grabber. It does NOT call cv2 / urllib /
httpx directly — every system boundary is injected so these tests stay
hermetic.

Hermetic policy:
- Fake :class:`_PlatformLike` records claim / status calls.
- Fake camera resolver returns a canned :class:`CameraSnapshotSource`.
- Fake snapshot grabber returns canned bytes.
- Fake http put records the (url, body, headers) it received.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from client_agent.platform import SnapshotClaim
from client_agent.snapshot_poller import PutResult, SnapshotPoller
from client_agent.web import CameraSnapshotSource


@dataclass
class FakePlatform:
    """In-memory platform fake for the snapshot poller tests."""

    next_claims: list[SnapshotClaim | None] = field(default_factory=list)
    status_calls: list[tuple[str, str, str | None]] = field(default_factory=list)

    def claim_next_snapshot(self) -> SnapshotClaim | None:
        if not self.next_claims:
            return None
        return self.next_claims.pop(0)

    def report_snapshot_status(
        self, request_id: str, *, status: str, error: str | None = None
    ) -> None:
        self.status_calls.append((request_id, status, error))


def _claim(camera_id: str = "cam-1", request_id: str = "req-1") -> SnapshotClaim:
    return SnapshotClaim(
        request_id=request_id,
        camera_id=camera_id,
        upload_url=f"https://r2.example/signed/{request_id}?X-Amz-Signature=abc",
        key=f"tenants/t/snapshots/{camera_id}/latest.jpg",
        expires_in=300,
        content_type="image/jpeg",
    )


def test_run_once_returns_false_when_platform_is_idle() -> None:
    """No pending requests → ``run_once`` returns ``False`` and does not
    touch the grabber, resolver, or http put. The ``run()`` loop uses
    this to decide to sleep before the next poll."""
    platform = FakePlatform(next_claims=[])
    grab_calls: list[str] = []
    put_calls: list[tuple[str, bytes, str]] = []

    poller = SnapshotPoller(
        platform=platform,
        camera_resolver=lambda _id: None,
        snapshot_grabber=lambda url, _t: grab_calls.append(url) or b"",
        http_put=lambda url, body, content_type: (
            put_calls.append((url, body, content_type)) or PutResult(success=True)
        ),
    )

    handled = poller.run_once()

    assert handled is False
    assert grab_calls == []
    assert put_calls == []
    assert platform.status_calls == []


def test_run_once_happy_path_claim_grab_put_report_uploaded() -> None:
    """The full successful round-trip: claim → resolve URL → grab JPEG
    → PUT to presigned URL → report ``uploaded``. The grabber must
    receive the camera's snapshot_url (preferred) or rtsp_url; the
    PUT must carry the presigned URL and the image/jpeg content-type."""
    platform = FakePlatform(next_claims=[_claim()])
    grab_calls: list[tuple[str, float]] = []
    put_calls: list[tuple[str, bytes, str]] = []
    canned_jpeg = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    cam_source = CameraSnapshotSource(
        rtsp_url="rtsp://192.168.1.198:554/V_ENC_000",
        snapshot_url="http://192.168.1.198/snapshot.jpg",
    )

    def fake_grab(url: str, timeout_s: float) -> bytes:
        grab_calls.append((url, timeout_s))
        return canned_jpeg

    def fake_put(url: str, body: bytes, content_type: str) -> PutResult:
        put_calls.append((url, body, content_type))
        return PutResult(success=True)

    poller = SnapshotPoller(
        platform=platform,
        camera_resolver=lambda _id: cam_source,
        snapshot_grabber=fake_grab,
        http_put=fake_put,
    )

    handled = poller.run_once()

    assert handled is True
    # Grabber received the camera's HTTP snapshot URL (preferred over RTSP).
    assert grab_calls == [("http://192.168.1.198/snapshot.jpg", 5.0)]
    # PUT landed at the presigned URL with the JPEG body + image/jpeg type.
    assert put_calls == [
        (
            "https://r2.example/signed/req-1?X-Amz-Signature=abc",
            canned_jpeg,
            "image/jpeg",
        )
    ]
    # Status reported uploaded; no error field.
    assert platform.status_calls == [("req-1", "uploaded", None)]


def test_run_once_rtsp_fallback_when_no_snapshot_url() -> None:
    """When the camera's source has no ``snapshot_url`` (Tuya / Setti+
    cameras, RTSP-scan discovery), the grabber must fall back to the
    ``rtsp_url``. Mirrors the web.py snapshot route's preference order
    so platform thumbnails behave the same as the local on-site panel."""
    platform = FakePlatform(next_claims=[_claim()])
    grab_calls: list[str] = []
    cam_source = CameraSnapshotSource(rtsp_url="rtsp://cam/stream", snapshot_url=None)

    def fake_grab(url: str, _t: float) -> bytes:
        grab_calls.append(url)
        return b"jpeg"

    poller = SnapshotPoller(
        platform=platform,
        camera_resolver=lambda _id: cam_source,
        snapshot_grabber=fake_grab,
        http_put=lambda _u, _b, _c: PutResult(success=True),
    )

    poller.run_once()

    assert grab_calls == ["rtsp://cam/stream"]


def test_run_once_camera_unknown_reports_failed_without_grabbing() -> None:
    """If the camera_id has no heartbeat-supplied source (the platform
    enqueued a snapshot for a camera the appliance no longer knows about
    — e.g. just removed from the config), the poller must report ``failed``
    immediately, NOT hang the request or attempt an RTSP open."""
    platform = FakePlatform(next_claims=[_claim(camera_id="ghost-cam")])
    grab_calls: list[str] = []

    poller = SnapshotPoller(
        platform=platform,
        camera_resolver=lambda _id: None,
        snapshot_grabber=lambda url, _t: grab_calls.append(url) or b"",
        http_put=lambda _u, _b, _c: PutResult(success=True),
    )

    handled = poller.run_once()

    assert handled is True
    assert grab_calls == []
    assert len(platform.status_calls) == 1
    request_id, status, error = platform.status_calls[0]
    assert (request_id, status) == ("req-1", "failed")
    assert error is not None and "ghost-cam" in error


def test_run_once_grabber_raises_reports_failed_without_put() -> None:
    """RTSP grab can fail (camera offline, network blip, cv2 timeout).
    The poller must catch, report ``failed``, and NOT attempt the PUT —
    a 0-byte upload to the presigned URL would persist a broken thumb."""
    platform = FakePlatform(next_claims=[_claim()])
    put_calls: list[str] = []

    def fake_grab(_url: str, _t: float) -> bytes:
        raise RuntimeError("cv2.VideoCapture failed to open rtsp://cam")

    poller = SnapshotPoller(
        platform=platform,
        camera_resolver=lambda _id: CameraSnapshotSource(rtsp_url="rtsp://cam/x"),
        snapshot_grabber=fake_grab,
        http_put=lambda url, _b, _c: put_calls.append(url) or PutResult(success=True),
    )

    handled = poller.run_once()

    assert handled is True
    assert put_calls == []
    assert len(platform.status_calls) == 1
    request_id, status, error = platform.status_calls[0]
    assert (request_id, status) == ("req-1", "failed")
    assert error is not None and "cv2" in error


def test_run_once_put_failure_reports_failed_with_status_code() -> None:
    """The PUT itself can fail (R2 5xx, signature expiry, network).
    Must surface the failure as a ``failed`` status with a diagnostic
    error message — the platform UI shows operators which leg broke."""
    platform = FakePlatform(next_claims=[_claim()])

    poller = SnapshotPoller(
        platform=platform,
        camera_resolver=lambda _id: CameraSnapshotSource(rtsp_url="rtsp://cam/x"),
        snapshot_grabber=lambda _u, _t: b"jpeg",
        http_put=lambda _u, _b, _c: PutResult(success=False, status_code=503),
    )

    handled = poller.run_once()

    assert handled is True
    assert len(platform.status_calls) == 1
    request_id, status, error = platform.status_calls[0]
    assert (request_id, status) == ("req-1", "failed")
    assert error is not None and "503" in error


def test_run_once_reports_failed_does_not_propagate_grabber_exception() -> None:
    """A grabber exception must be caught — letting it propagate would
    kill the daemon thread and leave the appliance heartbeating fine
    but never serving thumbnails. The poller's loop already catches at
    ``run()``, but ``run_once`` must also return cleanly so a unit-test
    or single-shot invocation gets a deterministic ``True`` back."""
    platform = FakePlatform(next_claims=[_claim()])

    def boom(_u: str, _t: float) -> bytes:
        raise RuntimeError("boom")

    poller = SnapshotPoller(
        platform=platform,
        camera_resolver=lambda _id: CameraSnapshotSource(rtsp_url="rtsp://cam/x"),
        snapshot_grabber=boom,
        http_put=lambda _u, _b, _c: PutResult(success=True),
    )

    # Should NOT raise.
    handled = poller.run_once()
    assert handled is True


def test_run_once_grab_failure_report_scrubs_credentials() -> None:
    """Issue #53: the credentialed RTSP url the snapshot path now injects must
    NOT reach the platform's error column (visible to other tenants' admins).
    The grab-failure exception embeds the full url — report it scrubbed."""
    platform = FakePlatform(next_claims=[_claim(camera_id="cam-9")])

    def fake_grab(_url: str, _t: float) -> bytes:
        raise RuntimeError(
            "cv2.VideoCapture failed to open 'rtsp://admin:s3cret@10.0.0.5:554/h264'"
        )

    poller = SnapshotPoller(
        platform=platform,
        camera_resolver=lambda _id: CameraSnapshotSource(
            rtsp_url="rtsp://admin:s3cret@10.0.0.5:554/h264"
        ),
        snapshot_grabber=fake_grab,
        http_put=lambda _u, _b, _c: PutResult(success=True),
    )

    poller.run_once()

    _req, status, error = platform.status_calls[0]
    assert status == "failed"
    assert error is not None
    # No credentials leaked...
    assert "s3cret" not in error
    assert "admin:" not in error
    # ...but still actionable: names the camera + the failing host.
    assert "cam-9" in error
    assert "10.0.0.5" in error


def test_run_once_grab_failure_local_log_keeps_full_detail(caplog) -> None:
    """Acceptance 2: the operator's own journald keeps the full url (creds
    included) for diagnosis — only the platform-facing report is scrubbed."""
    import logging

    platform = FakePlatform(next_claims=[_claim(camera_id="cam-9")])

    def fake_grab(_url: str, _t: float) -> bytes:
        raise RuntimeError(
            "cv2.VideoCapture failed to open 'rtsp://admin:s3cret@10.0.0.5:554/h264'"
        )

    poller = SnapshotPoller(
        platform=platform,
        camera_resolver=lambda _id: CameraSnapshotSource(
            rtsp_url="rtsp://admin:s3cret@10.0.0.5:554/h264"
        ),
        snapshot_grabber=fake_grab,
        http_put=lambda _u, _b, _c: PutResult(success=True),
    )

    with caplog.at_level(logging.WARNING):
        poller.run_once()

    # Operator-side log keeps the creds; platform-facing report does not.
    assert "s3cret" in caplog.text
    _req, _status, error = platform.status_calls[0]
    assert error is not None and "s3cret" not in error
