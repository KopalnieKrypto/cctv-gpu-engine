"""Container entrypoint for the client-agent (issues #7, #8).

Reads R2 credentials from the env, constructs an :class:`R2Client`, builds
the Flask web UI via :func:`client_agent.web.create_app`, and serves it on
``0.0.0.0:8080`` — the port both the Dockerfile and
``docker-compose.client.yml`` expose.

Env vars (SPEC §10.1):

* ``R2_ENDPOINT``          — e.g. ``https://<acct>.r2.cloudflarestorage.com``
* ``R2_ACCESS_KEY_ID``     — scoped R2 API token
* ``R2_SECRET_ACCESS_KEY`` — paired secret
* ``R2_BUCKET``            — defaults to ``surveillance-data`` (CLAUDE.md rule)
* ``RECORDINGS_DIR``       — host dir for in-flight recordings (defaults to
  ``$TMPDIR/cctv-recordings``); the docker-compose volume mount lands here.

The previous placeholder body (``signal.pause()`` until SIGTERM, issue #16)
is gone now that real Flask UI lives in :mod:`client_agent.web`. PID-1
signal handling is no longer our concern: Werkzeug's dev server installs
its own SIGTERM/SIGINT handlers and exits cleanly on ``docker stop``.
"""

from __future__ import annotations

import logging
import os
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
from client_agent.r2_client import R2Client
from client_agent.recorder import BackgroundRecorder, Recorder
from client_agent.web import create_app

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltApp:
    """Bundle returned by :func:`build_app`.

    Carries the ready-to-serve Flask app plus a couple of values the
    entrypoint logs at startup so the operator can confirm wiring without
    grepping env. ``recorder`` is exposed so the appliance entrypoint
    (issue #26) can drive it from the platform heartbeat without going
    through the Flask request layer. Frozen so a downstream caller can't
    accidentally rebind ``app`` after construction."""

    app: Flask
    bucket: str
    recordings_root: Path
    recorder: BackgroundRecorder


def build_app(environ: Mapping[str, str], *, recordings_root: Path | None = None) -> BuiltApp:
    """Construct the full client-agent Flask app from the given environment.

    Shared between the Docker entrypoint (:func:`main`, runs Werkzeug dev
    server) and the standalone appliance entrypoint
    (``client_agent.appliance.main``, runs waitress). Both must produce
    byte-identical behaviour, so all wiring lives here.

    ``recordings_root`` is injectable so the appliance can override the
    Docker default (``$TMPDIR/cctv-recordings``) with the XDG state dir.
    When ``None``, the historical Docker behaviour is preserved.
    """
    # R2 credentials are required for the Docker container entrypoint (where
    # the Flask UI is the upload path) but unused in platform-mode bare-metal
    # appliance (where the platform mints presigned URLs and the recorder
    # writes to local buffer). All-three-missing → ``client = None`` and the
    # Flask handlers that need R2 return 503 ("R2 backend disabled in
    # platform mode"). Partial config (one or two of three) still fails loud
    # because that's almost certainly a typo in ``.env.client``.
    endpoint = environ.get("R2_ENDPOINT")
    access_key = environ.get("R2_ACCESS_KEY_ID")
    secret_key = environ.get("R2_SECRET_ACCESS_KEY")
    bucket = environ.get("R2_BUCKET", "surveillance-data")

    r2_set = [v for v in (endpoint, access_key, secret_key) if v]
    if r2_set and len(r2_set) < 3:
        raise ValueError(
            "R2 credentials misconfigured: R2_ENDPOINT, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY must all be set together (or all absent for "
            "platform-mode appliance)"
        )

    client: R2Client | None = None
    if endpoint and access_key and secret_key:
        client = R2Client(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
        )

    # Build the production recorder (#8). Each recording lands in its
    # own subdir under ``recordings_root`` so the cleanup step in
    # Recorder.start can shutil.rmtree it without touching siblings.
    # The synchronous Recorder is wrapped by BackgroundRecorder so the
    # Flask request handler returns immediately while ffmpeg runs for
    # hours.
    if recordings_root is None:
        recordings_root = (
            Path(environ.get("RECORDINGS_DIR", tempfile.gettempdir())) / "cctv-recordings"
        )
    recordings_root.mkdir(parents=True, exist_ok=True)

    sync_recorder = Recorder(
        uploader=client,
        runner=subprocess.run,
        output_dir_factory=lambda job_id: str(recordings_root / job_id),
    )
    recorder = BackgroundRecorder(sync_recorder)

    # Discovery (issue #21): Stage 1 ONVIF + Stage 2 RTSP-scan, both wired
    # to the same env-driven credentials resolver. Operators populate
    # ``RTSP_DEFAULT_USER/PASS`` in ``.env.client`` (with optional per-IP
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

    app = create_app(
        client,
        recorder=recorder,
        discover_fn=_discover,
        credentials_resolver=creds_resolver,
    )

    return BuiltApp(app=app, bucket=bucket, recordings_root=recordings_root, recorder=recorder)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    built = build_app(os.environ)

    logger.info(
        "client-agent web UI starting on http://0.0.0.0:8080 (bucket=%s, recordings=%s)",
        built.bucket,
        built.recordings_root,
    )
    built.app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
