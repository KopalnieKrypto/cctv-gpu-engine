"""Placeholder entrypoint for the client-agent container (issue #16).

Real behaviour (Flask UI on :8080, ffmpeg RTSP capture, boto3 upload to R2)
lands in issues #7 and #8. Until then this module just logs a banner and
blocks forever so:

* `docker compose -f docker-compose.client.yml up` shows a healthy
  long-running process instead of a crash-loop, and
* a CI image build (the issue #16 acceptance criterion) has something
  legitimate to ENTRYPOINT into.

Block strategy: ``signal.pause()`` on POSIX (zero CPU, wakes on SIGTERM so
``docker stop`` exits cleanly). Falls back to ``threading.Event().wait()``
on platforms without ``signal.pause`` (Windows dev boxes), which is also
SIGTERM-friendly via the default handler.
"""

from __future__ import annotations

import logging
import signal

logger = logging.getLogger(__name__)


def _block_forever() -> None:
    """Sleep until the process receives a termination signal."""
    if hasattr(signal, "pause"):
        signal.pause()
        return
    import threading

    threading.Event().wait()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "client-agent placeholder running — real Flask UI + RTSP + R2 upload "
        "will land in issues #7 and #8. Container will block until SIGTERM."
    )
    _block_forever()


if __name__ == "__main__":
    main()
