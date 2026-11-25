"""Annotation-window poller for the client appliance (gpu-exchange#302).

The panel asks for a window of one camera's buffer for a zone-annotator
project. This box glues it from its own segments, without re-encoding and
below the recorder's priority, and leaves it where zone-annotator behind the
proxy is allowed to read it: ``<windows>/<project>/<window>.mp4``. The client
then draws zones on it in the panel, and zone-annotator cuts the crops.

The clip is a copy of footage, so it lives only until zone-annotator has
ingested it: once the project's manifest lists the file, this deletes it, and
a clip no manifest lists goes after a week.

A later window of the same project (gpu-exchange#305) needs no drawing: this
box ingests it itself with the first window's zones and stride, under its own
shift and lighting, and deletes the glued file straight after.

An export (gpu-exchange#306) leaves the box only as zone-annotator's checked
package, over the tailnet to this box's write-only directory on cctv-vps. When
cctv-vps cannot be reached the package waits here and is sent again later.

A split (gpu-exchange#308) is zone-annotator's ``split declare`` or ``amend``
over the windows' clips; the panel is told the tool's ledger as it then stands.

A head trained on cctv-vps from one of the project's exports (gpu-exchange#314)
is registered with ``split register-run``, so the split it was trained on is
frozen; the ledger is reported the same way.

Mirrors :class:`client_agent.snapshot_poller.SnapshotPoller`: every system
boundary is injected and ``run_once`` is what the tests drive.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol

from client_agent.buffer import BufferChunk
from client_agent.discovery import scrub_url_credentials
from client_agent.ffmpeg_trim import trim_and_concat
from client_agent.platform import (
    AnnotationDeleteProjectClaim,
    AnnotationExportClaim,
    AnnotationJobClaim,
    AnnotationRegisterRunClaim,
    AnnotationSplitClaim,
    PlatformRequestError,
)

try:
    import fcntl
except ImportError:  # shortcut: Windows dev machines take no lock; the box is Linux
    fcntl = None

logger = logging.getLogger(__name__)

#: Free space the windows disk must keep before a window is glued: a 60 min
#: 4K window is about 2.4 GB, and zone-annotator's crops land on the same disk.
MIN_FREE_BYTES = 10 * 1024**3

#: Below the recorder in CPU and in I/O, so gluing never costs a segment.
#: Best-effort class at its lowest level rather than idle, which a recorder
#: writing all the time could starve indefinitely.
_LOWER_PRIORITY = ["nice", "-n", "10", "ionice", "-c2", "-n7"]

#: The release install-annotator.sh points ``current`` at (gpu-exchange#298). Its
#: tools run as the agent's user, who owns the projects directory the service writes to.
ZONE_ANNOTATOR_PYTHON = "/opt/zone-annotator/current/.venv/bin/python"

#: install-annotator.sh's shipping key (gpu-exchange#300), and the account on
#: cctv-vps whose rrsync -wo takes it only into this box's own directory.
DROP_KEY = Path.home() / ".ssh" / "annotation-drop"
DROP_TARGET = "annotation-drop@cctv-vps"

#: How long a checked package waits on the box before cctv-vps is tried again.
RETRY_AFTER_S = 300.0

#: Longest an ingest may run, its wait for a cut already running in serve included: a
#: 60-minute window with 3 zones is about 3 x 24 min of decoding (6-8 min per zone of a
#: 20-minute window, zone-annotator's README), one such cut ahead of it doubles that to
#: ~2.4 h. Under the platform's 6 h sweeper, so the box's answer lands first.
CUT_LIMIT_S = 4 * 3600.0

#: How often a wait for zone-annotator's cutting lock tries again.
LOCK_POLL_S = 5.0

#: A copy made for annotation stays on the box at most this long: a glued window no manifest
#: lists, a package cctv-vps has not taken. Footage of people at work, so nothing waits forever.
COPY_LIMIT_S = 7 * 86400.0


class _PlatformLike(Protocol):
    def claim_next_annotation_job(
        self,
    ) -> (
        AnnotationJobClaim
        | AnnotationExportClaim
        | AnnotationSplitClaim
        | AnnotationRegisterRunClaim
        | AnnotationDeleteProjectClaim
        | None
    ): ...
    def report_annotation_delete(
        self, job_id: str, *, status: str, error: str | None = None
    ) -> None: ...
    def report_annotation_split(
        self, job_id: str, *, status: str, ledger: dict, error: str | None = None
    ) -> None: ...
    def report_annotation_register_run(
        self, job_id: str, *, status: str, ledger: dict, error: str | None = None
    ) -> None: ...
    def report_annotation_job(
        self,
        job_id: str,
        *,
        status: str,
        clip_path: str | None = None,
        error: str | None = None,
    ) -> None: ...
    def report_annotation_export(
        self,
        job_id: str,
        *,
        status: str,
        samples: int | None = None,
        destination: str | None = None,
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
        zone_annotator_python: str = ZONE_ANNOTATOR_PYTHON,
        exports_dir: Path | None = None,
        drop_target: str = DROP_TARGET,
        drop_key: Path = DROP_KEY,
        # Wall time: a marker's age must survive a restart.
        clock: Callable[[], float] = time.time,
        retry_after_s: float = RETRY_AFTER_S,
        cut_limit_s: float = CUT_LIMIT_S,
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
        self._python = zone_annotator_python
        # Beside the windows, on the agent's own disk; a package is a copy of crops.
        self._exports_dir = exports_dir or windows_dir.parent / "exports"
        self._drop_target = drop_target
        self._drop_key = drop_key
        self._clock = clock
        self._retry_after_s = retry_after_s
        self._cut_limit_s = cut_limit_s
        self._retry_at: dict[str, float] = {}

    def run_once(self) -> bool:
        """Handle one claimed job. ``False`` when the platform had nothing."""
        try:
            self._delete_spent_windows()
        except Exception as exc:  # noqa: BLE001 — housekeeping never holds up a claim
            logger.warning("deleting spent annotation windows failed: %s", exc)
        self._retry_waiting_exports()
        claim = self._platform.claim_next_annotation_job()
        if claim is None:
            return False
        if isinstance(claim, AnnotationExportClaim):
            self._export(claim)
            return True
        if isinstance(claim, AnnotationSplitClaim):
            self._split(claim)
            return True
        if isinstance(claim, AnnotationRegisterRunClaim):
            self._register_run(claim)
            return True
        if isinstance(claim, AnnotationDeleteProjectClaim):
            self._delete_project(claim)
            return True

        project_dir = self._projects_dir / claim.project_id
        # zone-annotator's own test for "a project is here", before 2 GB are glued for nothing.
        if claim.kind == "next" and not (project_dir / "labels.sqlite").is_file():
            self._fail(
                claim,
                "Projekt nie ma jeszcze stref. Narysuj je na pierwszym oknie, "
                "a potem dodaj kolejne okno.",
            )
            return True

        refusal = self._no_room(self._windows_dir)
        if refusal:
            self._fail(claim, refusal)
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
            began = self._trim(
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
        if claim.kind == "next":
            # Where the clip really begins: the buffer may have rotated past the window's start.
            self._ingest(claim, output, project_dir, began or claim.start)
            return True
        # serve behind the proxy answers 404 until the project has its directory.
        project_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._platform.report_annotation_job(
                claim.job_id, status="awaiting_zones", clip_path=str(output.resolve())
            )
        except PlatformRequestError as exc:
            if exc.status_code != 409:
                raise
            logger.warning("annotation job %s settled without this box (%s)", claim.job_id, exc)
            output.unlink(missing_ok=True)
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

    def _ingest(
        self, claim: AnnotationJobClaim, clip: Path, project_dir: Path, began: datetime
    ) -> None:
        """Add a later window to the project with its first window's zones (gpu-exchange#305).

        The CLI waits on zone-annotator's cutting lock by itself, so a cut that
        is running on the box only delays this one. The glued copy is deleted
        either way: once ingested the crops hold it, and a failed window's
        footage has no reason to stay on the disk.
        """
        cmd = [
            # The CLI waits on the cutting lock for ever, and a cut hung in serve holds it.
            # GNU timeout signals its whole process group, so the tool's ffmpeg stops too.
            "timeout",
            "--kill-after=60",
            f"{self._cut_limit_s:g}",
            self._python,
            "-m",
            "zone_annotator.ingest",
            str(clip.resolve()),
            "--project",
            str(project_dir),
            "--same-as-first",
            "--recorded-at",
            began.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ]
        # Declared together with the pick: a project knows only the values its windows brought.
        if claim.shift:
            cmd += ["--shifts", claim.shift, "--shift", claim.shift]
        if claim.lighting:
            cmd += ["--lightings", claim.lighting, "--lighting", claim.lighting]
        try:
            result = self._lowered(cmd, capture_output=True, text=True)
        except Exception as exc:  # noqa: BLE001 — a missing tool is the client's answer too
            self._fail(claim, f"Wczytanie okna do zone-annotatora nie powiodło się: {exc}")
            return
        finally:
            clip.unlink(missing_ok=True)
        if result.returncode == 124:  # GNU timeout: the limit ran out
            self._fail(
                claim,
                "Wczytanie okna do zone-annotatora trwało dłużej niż "
                f"{self._cut_limit_s / 3600:g} h i zostało przerwane; na komputerze z "
                "nagraniami mogło zawiesić się wycinanie. Dodaj to okno ponownie.",
            )
            return
        if result.returncode != 0:
            self._fail(claim, (result.stderr or "").strip() or f"ingest exited {result.returncode}")
            return
        self._platform.report_annotation_job(claim.job_id, status="ready")

    def _split(self, claim: AnnotationSplitClaim) -> None:
        """Apply the panel's folds with zone-annotator's ``split`` (gpu-exchange#308).

        ``declare`` while the project has no split, ``amend`` once it has one: the
        tool's own ledger decides, not the panel's last view of it. The name and
        reason go to an amendment only, as ``--by=`` so no value reads as a flag.
        Either way the ledger is reported as it stands afterwards, so the panel
        shows what the tool holds.
        """
        project = self._projects_dir / claim.project_id
        declared = any(e["kind"] != "model_run" for e in self._ledger(project)["entries"])
        # Each window's clip is named after it: the box glues `<windows>/<project>/<window>.mp4`.
        args = [str(project), "amend" if declared else "declare"]
        args += [
            f"--fold={name}=" + ",".join(f"{window}.mp4" for window in windows)
            for name, windows in claim.folds.items()
        ]
        if declared and claim.by:
            args.append(f"--by={claim.by}")
        if declared and claim.reason:
            args.append(f"--reason={claim.reason}")
        result = self._zone_annotator("split", *args)
        ledger = self._ledger(project)
        if result.returncode != 0:
            self._platform.report_annotation_split(
                claim.job_id, status="failed", ledger=ledger, error=_said(result, "split")
            )
            return
        self._platform.report_annotation_split(claim.job_id, status="applied", ledger=ledger)

    def _register_run(self, claim: AnnotationRegisterRunClaim) -> None:
        """Register the published head with zone-annotator (gpu-exchange#314, D7).

        From here on the tool asks who and why before the split changes. The
        name goes as ``--model=`` so no head name reads as a flag; the ledger is
        reported as it stands afterwards, refused or not.
        """
        project = self._projects_dir / claim.project_id
        result = self._zone_annotator(
            "split", str(project), "register-run", f"--model={claim.model}"
        )
        ledger = self._ledger(project)
        if result.returncode != 0:
            self._platform.report_annotation_register_run(
                claim.job_id, status="failed", ledger=ledger, error=_said(result, "register-run")
            )
            return
        self._platform.report_annotation_register_run(
            claim.job_id, status="registered", ledger=ledger
        )

    def _delete_project(self, claim: AnnotationDeleteProjectClaim) -> None:
        """Everything of a project deleted in the panel: crops and labels, glued windows,
        packages not yet on cctv-vps. Under zone-annotator's cutting lock, so no cut
        writes into the project while it goes; a link is removed, never followed."""
        doomed = [self._projects_dir / claim.project_id, self._windows_dir / claim.project_id]
        for export_id in claim.export_ids:
            doomed += [self._exports_dir / export_id, self._exports_dir / f"{export_id}.waiting"]
            self._retry_at.pop(export_id, None)
        try:
            with self._cutting_lock(wait=True):
                for path in doomed:
                    _remove(path)
        except BlockingIOError:
            self._platform.report_annotation_delete(
                claim.job_id,
                status="failed",
                error="Na komputerze z nagraniami od ponad "
                f"{self._cut_limit_s / 3600:g} h trwa wycinanie w zone-annotatorze, więc "
                "projekt nie został usunięty. Usuń go ponownie później.",
            )
            return
        except OSError as exc:
            self._platform.report_annotation_delete(
                claim.job_id, status="failed", error=str(exc)[-2000:]
            )
            return
        self._platform.report_annotation_delete(claim.job_id, status="deleted")

    def _ledger(self, project: Path) -> dict:
        """zone-annotator's ``split.json``, its clip ids read as clip names from the manifest."""
        entries = _json_or_empty(project / "split.json").get("entries") or []
        clips = _json_or_empty(project / "manifest.json").get("clips") or []
        names = {c.get("id"): Path(c.get("clip", {}).get("path", "")).name for c in clips}
        out = []
        for entry in entries:
            if entry.get("kind") == "model_run":
                out.append({"kind": "model_run", "at": entry["at"], "model": entry["model"]})
                continue
            read = {
                "kind": entry["kind"],
                "at": entry["at"],
                "folds": {
                    fold: [names.get(clip) or f"clip {clip}" for clip in clip_ids]
                    for fold, clip_ids in entry.get("folds", {}).items()
                },
            }
            if entry["kind"] == "amendment":
                read |= {"by": entry.get("by"), "reason": entry.get("reason")}
            out.append(read)
        return {"entries": out}

    def _export(self, claim: AnnotationExportClaim) -> None:
        """Export the project's answers, check the package and send it (gpu-exchange#306).

        Nothing leaves the box before zone-annotator's own ``check_export`` passes,
        and nothing of it goes anywhere but cctv-vps: no R2, no signed URL. A
        refused export or check is the client's answer in the tool's own words.
        """
        package = self._exports_dir / claim.export_id
        refusal = self._no_room(self._exports_dir)
        if refusal:
            self._export_failed(claim.job_id, refusal)
            return
        exported = self._zone_annotator(
            "export",
            str(self._projects_dir / claim.project_id),
            "--out",
            str(package),
            "--bound-to",
            claim.bound_to,
        )
        if exported.returncode != 0:
            shutil.rmtree(package, ignore_errors=True)
            self._export_failed(claim.job_id, _said(exported, "export"))
            return
        self._platform.report_annotation_export(claim.job_id, status="checking")
        checked = self._zone_annotator("check_export", str(package))
        if checked.returncode != 0:
            shutil.rmtree(package, ignore_errors=True)
            self._export_failed(claim.job_id, _said(checked, "check_export"))
            return
        self._mark_waiting(claim.job_id, claim.export_id)
        self._send(claim.job_id, claim.export_id)

    def _mark_waiting(self, job_id: str, export_id: str) -> None:
        """The ``.waiting`` marker, whole or not at all, before the first report or rsync.

        From here until ``on_vps`` is taken the package is retried on its own:
        after ``retry_after_s``, after a restart, after any report that did not
        get through.
        """
        marker = self._exports_dir / f"{export_id}.waiting"
        tmp = marker.with_name(f"{marker.name}.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"job_id": job_id, "created": self._clock()}, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(marker)

    def _send(self, job_id: str, export_id: str) -> None:
        """rsync a marked package to this box's directory on cctv-vps, or leave it waiting.

        The backoff is set before the try, so neither a refused rsync nor a
        report that does not get through brings the package back before
        ``retry_after_s``. The marker goes only once ``on_vps`` is taken, or
        when the platform answers 409: it settled this export without the box.
        """
        package = self._exports_dir / export_id
        destination = f"{self._drop_target}:{export_id}/"
        self._retry_at[export_id] = self._clock() + self._retry_after_s
        ssh = (
            f"ssh -i {self._drop_key} -o IdentitiesOnly=yes -o BatchMode=yes "
            # Only the key install-annotator.sh pinned from the repo: "cctv-vps" may resolve in
            # the client's LAN, so a key learned on first use could be anybody's.
            "-o StrictHostKeyChecking=yes "
            f"-o UserKnownHostsFile={self._drop_key}.known_hosts -o ConnectTimeout=15 "
            # Errors only: the client reads this sentence, not ssh's notes.
            "-o LogLevel=ERROR"
        )
        try:
            self._platform.report_annotation_export(job_id, status="sending")
            sent = self._run(["rsync", "-rt", "-e", ssh, f"{package}/", destination])
            if sent.returncode != 0:
                self._platform.report_annotation_export(
                    job_id, status="waiting_for_vps", error=_said(sent, "rsync")
                )
                return
            document = json.loads((package / "export.json").read_text(encoding="utf-8"))
            self._platform.report_annotation_export(
                job_id, status="on_vps", samples=len(document["samples"]), destination=destination
            )
        except PlatformRequestError as exc:
            if exc.status_code != 409:
                raise
            logger.warning("annotation export %s settled without this box (%s)", job_id, exc)
        self._drop_package(export_id)

    def _retry_waiting_exports(self) -> None:
        """Every waiting package on its own, so none of them ever holds up the next claim.

        Between iterations only a package with its marker belongs here; anything
        else was left by an export the agent did not finish, and goes.
        """
        if not self._exports_dir.is_dir():
            return
        for entry in sorted(self._exports_dir.iterdir()):
            try:
                if entry.suffix == ".waiting":
                    self._retry(entry)
                elif not entry.with_name(f"{entry.name}.waiting").exists():
                    logger.warning("deleting %s, left by an interrupted export", entry)
                    _remove(entry)
            except Exception as exc:  # noqa: BLE001 — retried after its backoff
                logger.warning("annotation package %s: %s", entry.name, exc)

    def _retry(self, marker: Path) -> None:
        export_id = marker.stem
        if self._clock() < self._retry_at.get(export_id, float("-inf")):
            return
        try:
            waiting = json.loads(marker.read_text(encoding="utf-8"))
            job_id, created = waiting["job_id"], float(waiting["created"])
            if not (self._exports_dir / export_id).is_dir():
                raise ValueError("no package beside it")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "annotation package %s has a broken marker (%s); deleted", export_id, exc
            )
            self._drop_package(export_id)
            return
        if self._clock() - created >= COPY_LIMIT_S:
            self._export_failed(job_id, "cctv-vps nieosiągalny przez 7 dni, paczka usunięta z boxa")
            self._drop_package(export_id)
            return
        self._send(job_id, export_id)

    def _drop_package(self, export_id: str) -> None:
        shutil.rmtree(self._exports_dir / export_id, ignore_errors=True)
        (self._exports_dir / f"{export_id}.waiting").unlink(missing_ok=True)
        self._retry_at.pop(export_id, None)

    def _zone_annotator(self, tool: str, *args: str) -> subprocess.CompletedProcess:
        return self._run([self._python, "-m", f"zone_annotator.{tool}", *args])

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        """A tool under lowered priority, its output kept; a missing binary is an answer too."""
        try:
            return self._lowered(cmd, capture_output=True, text=True)
        except OSError as exc:
            return subprocess.CompletedProcess(cmd, 127, "", str(exc))

    def _no_room(self, directory: Path) -> str | None:
        """The client's sentence when ``directory``'s disk is below the floor, else ``None``."""
        directory.mkdir(parents=True, exist_ok=True)
        free = self._free_bytes(directory)
        if free is not None and free >= self._min_free_bytes:
            return None
        have = "nieznane" if free is None else f"{free / 1024**3:.1f} GiB"
        return (
            f"Za mało miejsca na dysku komputera z nagraniami: wolne {have}, "
            f"potrzeba co najmniej {self._min_free_bytes / 1024**3:.1f} GiB."
        )

    def _export_failed(self, job_id: str, error: str) -> None:
        try:
            self._platform.report_annotation_export(job_id, status="failed", error=error)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to report annotation export %s as failed: %s", job_id, exc)

    def _fail(self, claim: AnnotationJobClaim, error: str) -> None:
        # The platform takes at most 2000 characters; the end is where the reason is.
        try:
            self._platform.report_annotation_job(claim.job_id, status="failed", error=error[-2000:])
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to report annotation job %s as failed: %s", claim.job_id, exc)

    def _delete_spent_windows(self) -> None:
        """Delete every glued window its project's manifest lists, and the rest after a week.

        The manifest is zone-annotator's record that the crops are cut; it is
        written whole (gpu-exchange#310), so a file listed there is done with.
        Any other file there (a first window nobody drew zones on, a clip whose
        report was lost, a half-glued one) goes after ``COPY_LIMIT_S``, under
        the cutting lock so no cut loses its clip midway. Only files under the
        windows directory are ever deleted.
        """
        if not self._windows_dir.is_dir():
            return
        stale = []
        for window in self._windows_dir.glob("*/*"):
            if not window.is_file():
                continue
            clips = _json_or_empty(self._projects_dir / window.parent.name / "manifest.json")
            ingested = {
                str(Path(c.get("clip", {}).get("path", "")).resolve())
                for c in clips.get("clips") or []
            }
            if str(window.resolve()) in ingested:
                window.unlink(missing_ok=True)
            elif self._clock() - window.stat().st_mtime >= COPY_LIMIT_S:
                stale.append(window)
        if not stale:
            return
        try:
            with self._cutting_lock(wait=False):
                for window in stale:
                    logger.info("deleting %s, kept the longest a glued window may be", window)
                    window.unlink(missing_ok=True)
        except BlockingIOError:
            logger.debug("a cut is running; %d spent window(s) wait for it", len(stale))

    @contextmanager
    def _cutting_lock(self, *, wait: bool) -> Iterator[None]:
        """zone-annotator's own lock (``ingest.CuttingLock``): no cut runs while its files go.

        A cut hung in serve would hold it for ever, so this waits at most
        ``cut_limit_s`` (``wait=False``: not at all), then raises BlockingIOError.
        """
        self._projects_dir.mkdir(parents=True, exist_ok=True)
        with open(self._projects_dir / ".cutting.lock", "a+b") as handle:
            deadline = self._clock() + (self._cut_limit_s if wait else 0.0)
            while fcntl is not None:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if self._clock() >= deadline:
                        raise
                    self._sleep(LOCK_POLL_S)
            yield

    def _lowered(self, cmd: list[str], **kwargs: object) -> object:
        return self._runner([*_LOWER_PRIORITY, *cmd], **kwargs)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _said(result: subprocess.CompletedProcess, tool: str) -> str:
    """What a tool said, ending where its reason is: the platform keeps 2000 characters."""
    said = f"{result.stdout or ''}{result.stderr or ''}".strip()
    return (said or f"{tool} exited {result.returncode}")[-2000:]


def _json_or_empty(path: Path) -> dict:
    """A file zone-annotator wrote whole, or ``{}`` while it has not written one."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
