"""Appliance self-telemetry for the heartbeat (issue #92).

The box is the only thing that can see its own disk. Until this module there
was no ``shutil.disk_usage`` call anywhere in ``client_agent``: DD-09 §7.6
specified disk health and nothing implemented it, so a production appliance
ran free space from 100 % to 17.4 % over seven days while every heartbeat
reported healthy. It was caught by a manual ``df -h`` during an unrelated
audit (§9.9).

What is deliberately *not* here: the band. The appliance reports raw bytes and
the platform derives OK / LOW / CRITICAL, so the thresholds live in one place
and can be retuned without redeploying every box. DD-09 §7.6's local CRITICAL
"stop all recorders" behaviour is a separate concern that must stay box-side —
it cannot depend on the network being up.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def sample_disk_bytes(path: Path) -> tuple[int | None, int | None]:
    """``(free, total)`` bytes for the volume holding ``path``.

    Sampled at the buffer directory rather than ``/``: the buffer is what
    actually grows, and on an appliance it can sit on a different filesystem
    than the root.

    Returns ``(None, None)`` if the volume cannot be read. This runs inside
    the heartbeat, which is also how the box pulls its config and reconciles
    recorders — letting a cosmetic telemetry read abort that would turn "I
    can't measure the disk" into "the appliance stopped taking instructions".
    """
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        logger.warning("disk telemetry unavailable for %s: %s", path, exc)
        return (None, None)
    return (usage.free, usage.total)
