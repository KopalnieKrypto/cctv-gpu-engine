"""Flask-app factory for the client-agent (issues #7, #8, #23).

Builds the client-agent Flask UI — camera discovery, per-camera snapshots,
the managed-cameras panel, and the RTSP recorder — via :func:`build_app`.

Historically this module also had a Docker container entrypoint (``main``)
that read R2 credentials from the env and served the app on ``0.0.0.0:8080``
through the Werkzeug dev server. That entrypoint and the R2 credential path
were removed in #29 when the Docker deployment was retired: the appliance
(``client_agent.appliance``) is now the only entrypoint, uploads go through
presigned URLs (:class:`client_agent.uploader.PresignedUploader`), and the
recorder writes to a local rolling buffer. ``build_app`` stays here as the
shared Flask-app factory the appliance imports.

Env vars:

* ``RECORDINGS_DIR`` — host dir for in-flight recordings (defaults to
  ``$TMPDIR/cctv-recordings``); the appliance overrides this with the XDG
  state dir.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from flask import Flask

from client_agent.discovery import (
    discover_cameras,
    make_real_rtsp_scan,
    make_real_tuya_scan,
    resolve_camera_credentials,
)
from client_agent.recorder import BackgroundRecorder, Recorder
from client_agent.snapshot import build_snapshot_grabber
from client_agent.web import CameraResolverFn, ManagedCamerasFn, create_app

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltApp:
    """Bundle returned by :func:`build_app`.

    Carries the ready-to-serve Flask app plus values the appliance
    entrypoint logs at startup so the operator can confirm wiring without
    grepping env. ``recorder`` is exposed so the appliance entrypoint
    (issue #26) can drive it from the platform heartbeat without going
    through the Flask request layer. Frozen so a downstream caller can't
    accidentally rebind ``app`` after construction."""

    app: Flask
    recordings_root: Path
    recorder: BackgroundRecorder


def build_app(
    environ: Mapping[str, str],
    *,
    recordings_root: Path | None = None,
    camera_resolver: CameraResolverFn | None = None,
    managed_cameras_lister: ManagedCamerasFn | None = None,
) -> BuiltApp:
    """Construct the full client-agent Flask app from the given environment.

    The appliance entrypoint (``client_agent.appliance.main``) runs the
    returned app under waitress. The Flask UI has no R2 backend since #29 —
    ``create_app`` is wired with ``client=None`` so the legacy R2-backed
    routes (/upload, /start, /jobs, /report) return 503; real uploads go
    through the platform's presigned URLs instead.

    ``recordings_root`` is injectable so the appliance can override the
    default (``$TMPDIR/cctv-recordings``) with the XDG state dir. When
    ``None``, the historical default is preserved.
    """
    if recordings_root is None:
        recordings_root = (
            Path(environ.get("RECORDINGS_DIR", tempfile.gettempdir())) / "cctv-recordings"
        )
    recordings_root.mkdir(parents=True, exist_ok=True)

    # Build the production recorder (#8). Each recording lands in its own
    # subdir under ``recordings_root``. The recorder writes to a local
    # rolling buffer and never uploads (buffer-only since #29 — the poller
    # ships chunks via presigned URLs). The synchronous Recorder is wrapped
    # by BackgroundRecorder so the Flask request handler returns immediately
    # while ffmpeg runs for hours.
    sync_recorder = Recorder(
        runner=subprocess.run,
        output_dir_factory=lambda job_id: str(recordings_root / job_id),
    )
    recorder = BackgroundRecorder(sync_recorder)

    # Discovery (issue #21): Stage 1 ONVIF + Stage 2 RTSP-scan, both wired
    # to the same env-driven credentials resolver. Operators populate
    # ``RTSP_DEFAULT_USER/PASS`` in ``cameras.env`` (with optional per-IP
    # overrides ``RTSP_CAM_<sanitized_ip>_USER/PASS``); both stages use
    # those creds — Stage 1 hands them to ``ONVIFCamera``, Stage 2 embeds
    # them in the RTSP URL templates. The resolver closes over a snapshot
    # of ``environ`` so a long-running process always sees the same creds
    # it started with (no surprise reloads on /cameras/discover).
    env_snapshot = dict(environ)
    creds_resolver = lambda ip: resolve_camera_credentials(ip, env_snapshot)  # noqa: E731
    rtsp_scan = make_real_rtsp_scan(creds_resolver)
    # Stage 3 (issue #38): Tuya local broadcast — catches Setti+/Tapo/Tuya
    # IPCs that don't expose ONVIF and ship with RTSP disabled by default.
    # No creds: purely passive UDP listening.
    tuya_scan = make_real_tuya_scan()

    def _discover():
        return discover_cameras(
            credentials_resolver=creds_resolver,
            rtsp_scan_fn=rtsp_scan,
            tuya_scan_fn=tuya_scan,
        )

    # Issue #41: per-camera snapshot endpoint. The production grabber
    # dispatches HTTP (vendor ONVIF GetSnapshotUri) vs RTSP (cv2 frame
    # grab) on URL scheme. ``camera_resolver`` is None in legacy mode —
    # last_discovery (keyed by IP) carries every camera the operator
    # could ask for. The appliance entrypoint overrides this with a
    # closure over the platform-supplied camera registry.
    #
    # ``client=None``: the R2-backed routes are retired (#29); the appliance
    # uploads via presigned URLs, not a client-agent R2 client.
    app = create_app(
        None,
        recorder=recorder,
        discover_fn=_discover,
        credentials_resolver=creds_resolver,
        camera_resolver=camera_resolver,
        snapshot_grabber=build_snapshot_grabber(),
        managed_cameras_lister=managed_cameras_lister,
    )

    return BuiltApp(app=app, recordings_root=recordings_root, recorder=recorder)
