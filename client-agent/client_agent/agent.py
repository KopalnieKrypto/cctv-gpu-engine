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
from pathlib import Path

from gpu_service.r2_client import R2Client

from client_agent.recorder import BackgroundRecorder, Recorder
from client_agent.web import create_app

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Required env vars — fail loud at startup so a misconfigured deploy
    # surfaces in `docker logs` immediately instead of as a 500 on first
    # upload. Bucket has a default because CLAUDE.md pins it project-wide.
    endpoint = os.environ["R2_ENDPOINT"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    bucket = os.environ.get("R2_BUCKET", "surveillance-data")

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
    recordings_root = (
        Path(os.environ.get("RECORDINGS_DIR", tempfile.gettempdir())) / "cctv-recordings"
    )
    recordings_root.mkdir(parents=True, exist_ok=True)

    sync_recorder = Recorder(
        uploader=client,
        runner=subprocess.run,
        output_dir_factory=lambda job_id: str(recordings_root / job_id),
    )
    recorder = BackgroundRecorder(sync_recorder)

    app = create_app(client, recorder=recorder)

    logger.info(
        "client-agent web UI starting on http://0.0.0.0:8080 (bucket=%s, recordings=%s)",
        bucket,
        recordings_root,
    )
    app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
