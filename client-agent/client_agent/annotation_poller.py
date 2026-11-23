"""Annotation-window poller for the client appliance (gpu-exchange#302).

The panel asks for a window of one camera's buffer for a zone-annotator
project. This box glues it from its own segments, without re-encoding and
below the recorder's priority, and leaves it where zone-annotator behind the
proxy is allowed to read it: ``<windows>/<project>/<window>.mp4``. The client
then draws zones on it in the panel, and zone-annotator cuts the crops.

The clip is a copy of footage, so it lives only until zone-annotator has
ingested it: once the project's manifest lists the file, this deletes it.

Mirrors :class:`client_agent.snapshot_poller.SnapshotPoller`: every system
boundary is injected and ``run_once`` is what the tests drive.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

from client_agent.buffer import BufferChunk
from client_agent.discovery import scrub_url_credentials
from client_agent.ffmpeg_trim import trim_and_concat
from client_agent.platform import AnnotationJobClaim

logger = logging.getLogger(__name__)

#: Free space the windows disk must keep before a window is glued: a 60 min
#: 4K window is about 2.4 GB, and zone-annotator's crops land on the same disk.
MIN_FREE_BYTES = 10 * 1024**3

#: Below the recorder in CPU and in I/O, so gluing never costs a segment.
#: Best-effort class at its lowest level rather than idle, which a recorder
#: writing all the time could starve indefinitely.
_LOWER_PRIORITY = ["nice", "-n", "10", "ionice", "-c2", "-n7"]


class _PlatformLike(Protocol):
    def claim_next_annotation_job(self) -> AnnotationJobClaim | None: ...
    def report_annotation_job(
        self,
        job_id: str,
        *,
        status: str,
        clip_path: str | None = None,
        error: str | None = None,
    ) -> None: ...


class _BufferLike(Protocol):
    def chunks_in_range(
        self, camera_id: str, *, start: datetime, end: datetime
    ) -> list[BufferChunk]: ...


def _free_bytes(path: Path) -> int | None:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


class AnnotationPoller:
    """Single-flight window pump against the platform's annotation queue."""

    def __init__(
        self,
        *,
        platform: _PlatformLike,
        buffer: _BufferLike,
        windows_dir: Path,
        projects_dir: Path,
        trim_fn: Callable[..., datetime | None] = trim_and_concat,
        runner: Callable[..., object] = subprocess.run,
        free_bytes: Callable[[Path], int | None] = _free_bytes,
        min_free_bytes: int = MIN_FREE_BYTES,
        poll_interval_s: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._platform = platform
        self._buffer = buffer
        self._windows_dir = windows_dir
        self._projects_dir = projects_dir
        self._trim = trim_fn
        self._runner = runner
        self._free_bytes = free_bytes
        self._min_free_bytes = min_free_bytes
        self._poll_interval_s = poll_interval_s
        self._sleep = sleep

    def run_once(self) -> bool:
        """Glue one claimed window. ``False`` when the platform had nothing."""
        self._delete_ingested_windows()
        claim = self._platform.claim_next_annotation_job()
        if claim is None:
            return False

        self._windows_dir.mkdir(parents=True, exist_ok=True)
        free = self._free_bytes(self._windows_dir)
        if free is None or free < self._min_free_bytes:
            have = "nieznane" if free is None else f"{free / 1024**3:.1f} GiB"
            self._fail(
                claim,
                f"Za mało miejsca na dysku komputera z nagraniami: wolne {have}, "
                f"potrzeba co najmniej {self._min_free_bytes / 1024**3:.1f} GiB.",
            )
            return True

        chunks = self._buffer.chunks_in_range(claim.camera_id, start=claim.start, end=claim.end)
        if not chunks:
            self._fail(
                claim,
                "Brak nagrań tej kamery w buforze między "
                f"{claim.start:%Y-%m-%d %H:%M} a {claim.end:%H:%M} UTC.",
            )
            return True

        output = self._windows_dir / claim.project_id / f"{claim.window_id}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._trim(
                chunks=chunks,
                start=claim.start,
                end=claim.end,
                output=output,
                runner=self._lowered,
            )
        except Exception as exc:  # noqa: BLE001 — ffmpeg's sentence is the report
            output.unlink(missing_ok=True)
            self._fail(claim, scrub_url_credentials(f"Sklejanie okna nie powiodło się: {exc}"))
            return True
        self._platform.report_annotation_job(
            claim.job_id, status="awaiting_zones", clip_path=str(output.resolve())
        )
        return True

    def run(self) -> None:
        """Blocking loop for the daemon thread; sleeps only when idle."""
        while True:
            try:
                handled = self.run_once()
            except Exception as exc:  # noqa: BLE001 — a Wi-Fi blip must not kill the thread
                logger.warning("annotation poller iteration failed: %s", exc)
                handled = False
            if not handled:
                self._sleep(self._poll_interval_s)

    def _fail(self, claim: AnnotationJobClaim, error: str) -> None:
        try:
            self._platform.report_annotation_job(claim.job_id, status="failed", error=error)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to report annotation job %s as failed: %s", claim.job_id, exc)

    def _delete_ingested_windows(self) -> None:
        """Delete every glued window its project's manifest already lists.

        The manifest is zone-annotator's record that the crops are cut; it is
        written whole (gpu-exchange#310), so a file listed there is done with.
        Only files under the windows directory are ever deleted.
        """
        if not self._windows_dir.is_dir():
            return
        for window in self._windows_dir.glob("*/*.mp4"):
            manifest = self._projects_dir / window.parent.name / "manifest.json"
            try:
                clips = json.loads(manifest.read_text(encoding="utf-8")).get("clips", [])
            except (OSError, ValueError):
                continue
            ingested = {str(Path(c.get("clip", {}).get("path", "")).resolve()) for c in clips}
            if str(window.resolve()) in ingested:
                window.unlink(missing_ok=True)

    def _lowered(self, cmd: list[str], **kwargs: object) -> object:
        return self._runner([*_LOWER_PRIORITY, *cmd], **kwargs)
