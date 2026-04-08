"""Placeholder entrypoint for the client-agent container (issue #16).

Real behaviour (Flask UI on :8080, ffmpeg RTSP capture, boto3 upload to R2)
lands in issues #7 and #8. Until then this module just logs a banner and
blocks forever so:

* `docker compose -f docker-compose.client.yml up` shows a healthy
  long-running process instead of a crash-loop, and
* a CI image build (the issue #16 acceptance criterion) has something
  legitimate to ENTRYPOINT into.

Block strategy: ``signal.pause()`` on POSIX (zero CPU). We MUST install
explicit SIGTERM/SIGINT handlers because this process runs as PID 1
inside the container and the Linux kernel drops default signal handlers
for PID 1 — without an explicit handler, ``docker stop`` would time out
and escalate to SIGKILL (exit 137). Falls back to
``threading.Event().wait()`` on platforms without ``signal.pause``
(Windows dev boxes).
"""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

logger = logging.getLogger(__name__)


def _handle_termination(signum: int, _frame: FrameType | None) -> None:
    logger.info("received signal %d, shutting down", signum)
    sys.exit(0)


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
    # Explicit handlers are mandatory because PID 1 in a container does not
    # inherit kernel-default signal dispositions — see module docstring.
    signal.signal(signal.SIGTERM, _handle_termination)
    signal.signal(signal.SIGINT, _handle_termination)
    logger.info(
        "client-agent placeholder running — real Flask UI + RTSP + R2 upload "
        "will land in issues #7 and #8. Container will block until SIGTERM."
    )
    _block_forever()


if __name__ == "__main__":
    main()
