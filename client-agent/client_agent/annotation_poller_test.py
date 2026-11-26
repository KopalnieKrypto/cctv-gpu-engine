"""Annotation-window poller (gpu-exchange#302).

What these tests encode:

* **Input**: a claimed job from the platform (project, window, camera, UTC
  start and end), the camera's buffer, the free bytes on the windows disk,
  and an injected trim and runner.
* **Output**: the glued window at ``<windows>/<project>/<window>.mp4`` and a
  report ``awaiting_zones`` carrying its absolute path; or ``failed`` with a
  sentence the client reads in the panel.
* **Boundaries**: disk below the floor, no segments in the range, a trim that
  raises, a window zone-annotator has already ingested.
* **Not tested here**: the HTTP client (``platform_test.py``) and ffmpeg
  itself (``ffmpeg_trim_test.py``).
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from client_agent import annotation_poller
from client_agent.annotation_poller import AnnotationPoller
from client_agent.buffer import BufferChunk
from client_agent.platform import (
    AnnotationDeleteProjectClaim,
    AnnotationExportClaim,
    AnnotationJobClaim,
    AnnotationRegisterRunClaim,
    AnnotationSplitClaim,
    PlatformRequestError,
    PlatformUnavailableError,
)

START = datetime(2026, 11, 23, 9, 0, tzinfo=UTC)
END = datetime(2026, 11, 23, 9, 20, tzinfo=UTC)
GIB = 1024**3


@dataclass
class FakePlatform:
    claims: list[AnnotationJobClaim] = field(default_factory=list)
    reports: list[dict] = field(default_factory=list)
    #: status -> what its next reports raise, one each, after the report is recorded.
    errors: dict[str, list[Exception]] = field(default_factory=dict)

    def claim_next_annotation_job(self) -> AnnotationJobClaim | None:
        return self.claims.pop(0) if self.claims else None

    def _report(self, job_id: str, body: dict) -> None:
        self.reports.append({"job_id": job_id, **body})
        if self.errors.get(body["status"]):
            raise self.errors[body["status"]].pop(0)

    def report_annotation_job(self, job_id: str, **body: object) -> None:
        self._report(job_id, body)

    def report_annotation_export(self, job_id: str, **body: object) -> None:
        self._report(job_id, body)

    def report_annotation_split(self, job_id: str, **body: object) -> None:
        self._report(job_id, body)

    def report_annotation_register_run(self, job_id: str, **body: object) -> None:
        self._report(job_id, body)

    def report_annotation_delete(self, job_id: str, **body: object) -> None:
        self._report(job_id, body)


@dataclass
class FakeBuffer:
    chunks: list[BufferChunk] = field(default_factory=list)

    def chunks_in_range(self, camera_id: str, *, start: datetime, end: datetime):
        return list(self.chunks)


def a_claim() -> AnnotationJobClaim:
    return AnnotationJobClaim(
        job_id="job-1",
        window_id="window-1",
        project_id="project-1",
        camera_id="camera-1",
        start=START,
        end=END,
    )


def a_chunk(tmp_path: Path) -> BufferChunk:
    path = tmp_path / "buffer" / "chunk.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return BufferChunk(path=path, start=START, end=END)


@dataclass
class Trims:
    calls: list[dict] = field(default_factory=list)

    def __call__(self, *, chunks, start, end, output: Path, runner):
        self.calls.append({"chunks": chunks, "start": start, "end": end, "output": output})
        runner(["ffmpeg", "-i", "x", str(output)], capture_output=True, text=True)
        output.write_bytes(b"glued")
        # As trim_and_concat: the clip starts at the buffer's oldest moment when that is later.
        return max(start, chunks[0].start)


def a_poller(
    tmp_path: Path,
    *,
    platform,
    buffer,
    trim=None,
    free=60 * GIB,
    runner=None,
    clock=None,
    **extra,
):
    return AnnotationPoller(
        platform=platform,
        buffer=buffer,
        windows_dir=tmp_path / "windows",
        projects_dir=tmp_path / "projects",
        trim_fn=trim or Trims(),
        runner=runner or (lambda cmd, **kw: None),
        free_bytes=lambda _path: free,
        clock=clock or (lambda: 0.0),
        **extra,
    )


def test_a_claimed_window_is_glued_into_the_projects_windows_dir_and_reported(tmp_path):
    platform = FakePlatform(claims=[a_claim()])
    trims = Trims()
    commands: list[list[str]] = []
    poller = a_poller(
        tmp_path,
        platform=platform,
        buffer=FakeBuffer([a_chunk(tmp_path)]),
        trim=trims,
        runner=lambda cmd, **kw: commands.append(cmd),
    )

    assert poller.run_once() is True

    window = (tmp_path / "windows" / "project-1" / "window-1.mp4").resolve()
    assert window.read_bytes() == b"glued"
    assert trims.calls[0]["start"] == START and trims.calls[0]["end"] == END
    assert commands[0][:5] == ["nice", "-n", "10", "ionice", "-c2"], commands[0]
    assert platform.reports == [
        {"job_id": "job-1", "status": "awaiting_zones", "clip_path": str(window)}
    ]
    # zone-annotator behind the proxy answers 404 for a project with no directory.
    assert (tmp_path / "projects" / "project-1").is_dir()


def test_a_first_window_is_reported_ready_once_its_zones_are_drawn(tmp_path):
    """The panel kept saying "czeka na strefy" on a window labelled long ago: nothing told
    the platform. The manifest listing the clip is zone-annotator's word that the crops are cut."""
    platform = FakePlatform(claims=[a_claim()])
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]))
    assert poller.run_once() is True
    window = (tmp_path / "windows" / "project-1" / "window-1.mp4").resolve()

    # Nobody has drawn yet: nothing more is said.
    poller.run_once()
    assert [r["status"] for r in platform.reports] == ["awaiting_zones"]

    manifest = {"clips": [{"clip": {"path": str(window)}}]}
    (tmp_path / "projects" / "project-1" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    poller.run_once()
    poller.run_once()

    assert platform.reports[1:] == [{"job_id": "job-1", "status": "ready"}]
    assert list(window.parent.iterdir()) == []


def test_a_first_window_the_platform_has_moved_on_from_is_deleted(tmp_path, caplog):
    platform = FakePlatform(claims=[a_claim()], errors={"awaiting_zones": [GONE]})
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]))

    assert poller.run_once() is True

    assert not (tmp_path / "windows" / "project-1" / "window-1.mp4").exists()
    assert "job-1" in caplog.text


def test_a_glued_file_no_manifest_lists_is_deleted_after_seven_days(tmp_path):
    """Retention: a first window nobody drew on, a lost report, a crash mid-glue."""
    windows = tmp_path / "windows" / "project-1"
    windows.mkdir(parents=True)
    undrawn, stray = windows / "window-1.mp4", windows / "tmpab12.concat.txt"
    for path in (undrawn, stray):
        path.write_bytes(b"x")
        os.utime(path, (1000.0, 1000.0))

    a_poller(
        tmp_path, platform=FakePlatform(), buffer=FakeBuffer(), clock=lambda: 1000.0 + 7 * 86400 - 1
    ).run_once()
    assert undrawn.exists() and stray.exists()

    a_poller(
        tmp_path, platform=FakePlatform(), buffer=FakeBuffer(), clock=lambda: 1000.0 + 7 * 86400
    ).run_once()
    assert not undrawn.exists() and not stray.exists()


def test_a_disk_below_the_floor_refuses_in_words_before_gluing(tmp_path):
    platform = FakePlatform(claims=[a_claim()])
    trims = Trims()
    poller = a_poller(
        tmp_path,
        platform=platform,
        buffer=FakeBuffer([a_chunk(tmp_path)]),
        trim=trims,
        free=3 * GIB,
    )

    poller.run_once()

    assert trims.calls == []
    [report] = platform.reports
    assert report["status"] == "failed"
    assert "3.0 GiB" in report["error"] and "10.0 GiB" in report["error"], report["error"]


def test_no_segments_in_the_range_fails_naming_the_range(tmp_path):
    platform = FakePlatform(claims=[a_claim()])
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer([]))

    poller.run_once()

    [report] = platform.reports
    assert report["status"] == "failed"
    assert "2026-11-23 09:00" in report["error"] and "09:20" in report["error"], report["error"]
    assert not (tmp_path / "windows" / "project-1" / "window-1.mp4").exists()


def test_a_trim_that_fails_is_reported_and_leaves_no_partial_window(tmp_path):
    platform = FakePlatform(claims=[a_claim()])

    def broken(*, output: Path, **_kw):
        output.write_bytes(b"half")
        raise RuntimeError("ffmpeg exited 1: Invalid data found when processing input")

    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]), trim=broken
    )

    poller.run_once()

    [report] = platform.reports
    assert report["status"] == "failed"
    assert "Invalid data" in report["error"]
    assert not (tmp_path / "windows" / "project-1" / "window-1.mp4").exists()


def test_a_trim_failure_longer_than_the_platform_keeps_is_cut_to_its_end(tmp_path):
    """The platform takes at most 2000 characters; a longer report was refused and lost."""
    platform = FakePlatform(claims=[a_claim()])

    def broken(**_kw):
        raise RuntimeError("ffmpeg trim exited 1: " + "configuration: --enable-x " * 200 + "EOF")

    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]), trim=broken
    )

    poller.run_once()

    [report] = platform.reports
    assert len(report["error"]) == 2000 and report["error"].endswith("EOF"), report["error"]


def test_a_window_zone_annotator_has_ingested_is_deleted(tmp_path):
    """The glued file is a copy of the client's footage; once the project's
    manifest lists it, zone-annotator holds its crops and the copy goes."""
    windows = tmp_path / "windows" / "project-1"
    windows.mkdir(parents=True)
    ingested = windows / "window-1.mp4"
    waiting = windows / "window-2.mp4"
    ingested.write_bytes(b"glued")
    waiting.write_bytes(b"glued")
    project = tmp_path / "projects" / "project-1"
    project.mkdir(parents=True)
    # zone-annotator's own shape: `Clip.as_dict()` keeps the probe's facts under "clip".
    manifest = {"clips": [{"clip": {"path": str(ingested.resolve())}}]}
    (project / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    poller = a_poller(tmp_path, platform=FakePlatform(), buffer=FakeBuffer())

    assert poller.run_once() is False

    assert not ingested.exists()
    assert waiting.exists()


# ----- later windows (gpu-exchange#305) -----


@dataclass
class Ingests:
    """The runner: records every command, and answers zone-annotator's ingest as told."""

    returncode: int = 0
    stderr: str = ""
    commands: list[list[str]] = field(default_factory=list)

    def __call__(self, cmd: list[str], **_kw):
        self.commands.append(cmd)
        return subprocess.CompletedProcess(cmd, self.returncode, stdout="", stderr=self.stderr)

    @property
    def ingest(self) -> list[str]:
        [ingest] = [c for c in self.commands if "zone_annotator.ingest" in c]
        return ingest


def a_later_claim(**overrides) -> AnnotationJobClaim:
    return replace(a_claim(), job_id="job-2", window_id="window-2", kind="next", **overrides)


def a_project(tmp_path: Path) -> Path:
    project = tmp_path / "projects" / "project-1"
    project.mkdir(parents=True)
    (project / "labels.sqlite").write_bytes(b"")
    return project


def test_a_later_window_is_ingested_with_the_first_windows_zones_and_its_conditions(tmp_path):
    project = a_project(tmp_path)
    platform = FakePlatform(claims=[a_later_claim(shift="popoludnie", lighting="sztuczne")])
    runner = Ingests()
    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]), runner=runner
    )

    assert poller.run_once() is True

    window = (tmp_path / "windows" / "project-1" / "window-2.mp4").resolve()
    assert runner.ingest[:6] == ["nice", "-n", "10", "ionice", "-c2", "-n7"]
    assert runner.ingest[6:] == [
        "timeout",
        "--kill-after=60",
        "14400",
        "/opt/zone-annotator/current/.venv/bin/python",
        "-m",
        "zone_annotator.ingest",
        str(window),
        "--project",
        str(project),
        "--same-as-first",
        "--recorded-at",
        "2026-11-23T09:00:00Z",
        "--shifts",
        "popoludnie",
        "--shift",
        "popoludnie",
        "--lightings",
        "sztuczne",
        "--lighting",
        "sztuczne",
    ]
    assert not window.exists(), "the glued copy goes once zone-annotator has the crops"
    assert platform.reports == [{"job_id": "job-2", "status": "ready"}]


def test_a_later_window_the_buffer_partly_rotated_out_is_recorded_from_its_real_start(tmp_path):
    """The clip begins where the buffer does; the window's start would shift every sample (#91)."""
    a_project(tmp_path)
    runner = Ingests()
    late = BufferChunk(path=a_chunk(tmp_path).path, start=START + timedelta(minutes=5), end=END)
    poller = a_poller(
        tmp_path,
        platform=FakePlatform(claims=[a_later_claim()]),
        buffer=FakeBuffer([late]),
        runner=runner,
    )

    poller.run_once()

    assert runner.ingest[runner.ingest.index("--recorded-at") + 1] == "2026-11-23T09:05:00Z"


def test_a_later_window_without_conditions_declares_none(tmp_path):
    a_project(tmp_path)
    runner = Ingests()
    poller = a_poller(
        tmp_path,
        platform=FakePlatform(claims=[a_later_claim()]),
        buffer=FakeBuffer([a_chunk(tmp_path)]),
        runner=runner,
    )

    poller.run_once()

    assert not {"--shift", "--shifts", "--lighting", "--lightings"} & set(runner.ingest)


def test_a_later_window_waits_out_a_running_cut_and_is_ready(tmp_path):
    """The CLI waits on the cutting lock by itself and says so on stderr; that is no failure."""
    a_project(tmp_path)
    platform = FakePlatform(claims=[a_later_claim()])
    runner = Ingests(
        stderr="ingest: another cut is running on this box; waiting for it to finish\n"
    )
    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]), runner=runner
    )

    poller.run_once()

    assert platform.reports == [{"job_id": "job-2", "status": "ready"}]


def test_a_failed_ingest_reports_the_tools_own_words_and_deletes_the_window(tmp_path):
    a_project(tmp_path)
    platform = FakePlatform(claims=[a_later_claim()])
    said = "ingest failed: 'Rano' is not one of this project's shifts (rano)"
    runner = Ingests(returncode=1, stderr=f"{said}\n")
    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]), runner=runner
    )

    poller.run_once()

    assert platform.reports == [{"job_id": "job-2", "status": "failed", "error": said}]
    assert not (tmp_path / "windows" / "project-1" / "window-2.mp4").exists()


def test_a_later_window_before_the_project_has_zones_fails_before_gluing(tmp_path):
    platform = FakePlatform(claims=[a_later_claim()])
    trims = Trims()
    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]), trim=trims
    )

    poller.run_once()

    assert trims.calls == []
    [report] = platform.reports
    assert report["status"] == "failed"
    assert "stref" in report["error"], report["error"]


# ----- exports to cctv-vps (gpu-exchange#306) -----

UNREACHABLE = "ssh: connect to host cctv-vps port 22: Connection timed out"


@dataclass
class Tools:
    """The runner for zone-annotator's export and check_export and for rsync:
    records every command and answers each as told."""

    export_code: int = 0
    export_err: str = ""
    check_code: int = 0
    check_out: str = ""
    #: One exit code per rsync call, 0 once they run out.
    rsync_codes: list[int] = field(default_factory=list)
    samples: int = 3
    commands: list[list[str]] = field(default_factory=list)

    def __call__(self, cmd: list[str], **_kw):
        self.commands.append(cmd)
        if "zone_annotator.export" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            (out / "images").mkdir(parents=True)
            if self.export_code == 0:
                doc = {"samples": [{"id": i} for i in range(self.samples)]}
                (out / "export.json").write_text(json.dumps(doc), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, self.export_code, "", self.export_err)
        if "zone_annotator.check_export" in cmd:
            return subprocess.CompletedProcess(cmd, self.check_code, self.check_out, "")
        if "rsync" in cmd:
            code = self.rsync_codes.pop(0) if self.rsync_codes else 0
            return subprocess.CompletedProcess(cmd, code, "", UNREACHABLE if code else "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def named(self, tool: str) -> list[list[str]]:
        return [c for c in self.commands if tool in c]


def an_export_claim() -> AnnotationExportClaim:
    return AnnotationExportClaim(
        job_id="job-3", export_id="export-1", project_id="project-1", bound_to="hala/v1.0.0"
    )


def test_an_export_is_exported_checked_and_sent_to_cctv_vps(tmp_path):
    project = a_project(tmp_path)
    platform = FakePlatform(claims=[an_export_claim()])
    tools = Tools()
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer(), runner=tools)

    assert poller.run_once() is True

    package = tmp_path / "exports" / "export-1"
    [export] = tools.named("zone_annotator.export")
    assert export[:6] == ["nice", "-n", "10", "ionice", "-c2", "-n7"]
    assert export[6:] == [
        "/opt/zone-annotator/current/.venv/bin/python",
        "-m",
        "zone_annotator.export",
        str(project),
        "--out",
        str(package),
        "--bound-to",
        "hala/v1.0.0",
    ]
    [check] = tools.named("zone_annotator.check_export")
    assert check[-1] == str(package)
    [rsync] = tools.named("rsync")
    ssh = rsync[rsync.index("-e") + 1]
    key = str(Path.home() / ".ssh" / "annotation-drop")
    assert f"-i {key}" in ssh and "BatchMode=yes" in ssh
    # install-annotator.sh pins cctv-vps's key; a first-use key from the client LAN is refused.
    assert "StrictHostKeyChecking=yes" in ssh
    assert f"UserKnownHostsFile={key}.known_hosts" in ssh
    # A link that dies mid-transfer ends the try instead of holding the poller's one thread.
    assert "--timeout=120" in rsync
    assert "-o ServerAliveInterval=30 -o ServerAliveCountMax=4" in ssh
    assert rsync[-2:] == [f"{package}/", "annotation-drop@cctv-vps:export-1/"]
    assert platform.reports == [
        {"job_id": "job-3", "status": "checking"},
        {"job_id": "job-3", "status": "sending"},
        {
            "job_id": "job-3",
            "status": "on_vps",
            "samples": 3,
            "destination": "annotation-drop@cctv-vps:export-1/",
        },
    ]
    assert not package.exists(), "nothing of the package stays on the box once it is on cctv-vps"


def test_an_export_zone_annotator_refuses_is_reported_in_its_words_and_never_sent(tmp_path):
    a_project(tmp_path)
    platform = FakePlatform(claims=[an_export_claim()])
    said = "export failed: no labelled sample in this project, so there is nothing to export"
    tools = Tools(export_code=1, export_err=f"{said}\n")
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer(), runner=tools)

    poller.run_once()

    assert platform.reports == [{"job_id": "job-3", "status": "failed", "error": said}]
    assert tools.named("zone_annotator.check_export") == [] and tools.named("rsync") == []
    assert not (tmp_path / "exports" / "export-1").exists()


def test_a_package_check_export_refuses_is_reported_in_its_words_and_never_sent(tmp_path):
    a_project(tmp_path)
    platform = FakePlatform(claims=[an_export_claim()])
    said = "  ✗ images/W1-d20fccbd/stanowisko-1/t00001.jpg has sha256 13815f9a…\n1 problem(s)"
    tools = Tools(check_code=1, check_out=said)
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer(), runner=tools)

    poller.run_once()

    assert platform.reports == [
        {"job_id": "job-3", "status": "checking"},
        {"job_id": "job-3", "status": "failed", "error": said.strip()},
    ]
    assert tools.named("rsync") == []
    assert not (tmp_path / "exports" / "export-1").exists()


def test_an_unreachable_cctv_vps_keeps_the_checked_package_and_retries_it_later(tmp_path):
    a_project(tmp_path)
    platform = FakePlatform(claims=[an_export_claim()])
    tools = Tools(rsync_codes=[255])
    now = [1000.0]
    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer(), runner=tools, clock=lambda: now[0]
    )

    poller.run_once()
    package = tmp_path / "exports" / "export-1"
    assert package.is_dir(), "the checked package waits on the box"
    assert platform.reports[-1] == {
        "job_id": "job-3",
        "status": "waiting_for_vps",
        "error": UNREACHABLE,
    }

    now[0] += 60
    assert poller.run_once() is False
    assert len(tools.named("rsync")) == 1, "no retry before the interval"

    now[0] += 300
    poller.run_once()
    assert len(tools.named("zone_annotator.export")) == 1, "the package is sent again, not re-cut"
    assert [r["status"] for r in platform.reports] == [
        "checking",
        "sending",
        "waiting_for_vps",
        "sending",
        "on_vps",
    ]
    assert not package.exists()
    assert not list((tmp_path / "exports").iterdir())


def test_a_package_waiting_from_before_a_restart_is_sent_by_the_next_poller(tmp_path):
    a_project(tmp_path)
    first = FakePlatform(claims=[an_export_claim()])
    a_poller(
        tmp_path, platform=first, buffer=FakeBuffer(), runner=Tools(rsync_codes=[255])
    ).run_once()

    platform = FakePlatform()
    tools = Tools()
    a_poller(tmp_path, platform=platform, buffer=FakeBuffer(), runner=tools).run_once()

    assert tools.named("zone_annotator.export") == []
    assert [r["status"] for r in platform.reports] == ["sending", "on_vps"]
    assert platform.reports[-1]["job_id"] == "job-3"


DOWN = PlatformUnavailableError("platform timed out after retries")
GONE = PlatformRequestError(409, "platform refused annotation export status (409): failed")


def a_waiting_package(tmp_path: Path, *, at: float = 0.0) -> None:
    """An export whose rsync failed at ``at``: its checked package waits on the box."""
    a_project(tmp_path)
    a_poller(
        tmp_path,
        platform=FakePlatform(claims=[an_export_claim()]),
        buffer=FakeBuffer(),
        runner=Tools(rsync_codes=[255]),
        clock=lambda: at,
    ).run_once()


def test_a_waiting_package_the_platform_cannot_take_never_holds_up_new_claims(tmp_path):
    a_waiting_package(tmp_path)
    platform = FakePlatform(claims=[a_claim()], errors={"sending": [DOWN]})
    now = [1000.0]
    poller = a_poller(
        tmp_path,
        platform=platform,
        buffer=FakeBuffer([a_chunk(tmp_path)]),
        runner=Tools(),
        clock=lambda: now[0],
    )

    assert poller.run_once() is True
    assert platform.reports[-1]["status"] == "awaiting_zones"

    now[0] += 60
    poller.run_once()
    assert [r["status"] for r in platform.reports].count("sending") == 1, "backoff before the try"
    assert (tmp_path / "exports" / "export-1.waiting").is_file()


def test_a_waiting_package_the_platform_has_moved_on_from_is_deleted(tmp_path, caplog):
    """A 409 means the export is settled elsewhere (e.g. the sweeper failed it): drop it here."""
    a_waiting_package(tmp_path)
    platform = FakePlatform(claims=[a_claim()], errors={"sending": [GONE]})
    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]), runner=Tools()
    )

    assert poller.run_once() is True

    assert not list((tmp_path / "exports").iterdir())
    assert "job-3" in caplog.text
    assert platform.reports[-1]["status"] == "awaiting_zones"


def test_a_package_sent_whose_on_vps_report_was_lost_is_reported_again(tmp_path):
    a_project(tmp_path)
    platform = FakePlatform(claims=[an_export_claim()], errors={"on_vps": [DOWN]})
    tools = Tools()
    marker = tmp_path / "exports" / "export-1.waiting"
    marked_at_rsync: list[bool] = []

    def runner(cmd, **kw):
        if "rsync" in cmd:
            marked_at_rsync.append(marker.is_file())
        return tools(cmd, **kw)

    now = [1000.0]
    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer(), runner=runner, clock=lambda: now[0]
    )

    with pytest.raises(PlatformUnavailableError):
        poller.run_once()
    assert marked_at_rsync == [True]
    assert json.loads(marker.read_text(encoding="utf-8")) == {"job_id": "job-3", "created": 1000.0}

    now[0] += 300
    poller.run_once()
    assert [r["status"] for r in platform.reports] == [
        "checking",
        "sending",
        "on_vps",
        "sending",
        "on_vps",
    ]
    assert not list((tmp_path / "exports").iterdir())


def test_a_broken_marker_is_deleted_with_its_package_and_the_queue_goes_on(tmp_path, caplog):
    a_waiting_package(tmp_path)
    (tmp_path / "exports" / "export-1.waiting").write_text('{"job_', encoding="utf-8")
    platform = FakePlatform(claims=[a_claim()])
    tools = Tools()
    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]), runner=tools
    )

    assert poller.run_once() is True

    assert not list((tmp_path / "exports").iterdir())
    assert tools.named("rsync") == []
    assert "export-1" in caplog.text
    assert platform.reports[-1]["status"] == "awaiting_zones"


def test_a_package_cctv_vps_has_not_taken_for_seven_days_is_failed_and_deleted(tmp_path):
    a_waiting_package(tmp_path, at=1000.0)
    platform = FakePlatform()
    tools = Tools()

    a_poller(
        tmp_path,
        platform=platform,
        buffer=FakeBuffer(),
        runner=tools,
        clock=lambda: 1000.0 + 7 * 86400,
    ).run_once()

    assert platform.reports == [
        {
            "job_id": "job-3",
            "status": "failed",
            "error": "cctv-vps nieosiągalny przez 7 dni, paczka usunięta z boxa",
        }
    ]
    assert tools.named("rsync") == []
    assert not list((tmp_path / "exports").iterdir())


@pytest.mark.parametrize(
    ("age_s", "statuses"),
    [(60, ["sending", "on_vps"]), (7 * 86400, ["failed"])],
    ids=["young", "a-week-old"],
)
def test_a_marker_from_before_its_created_field_is_aged_by_its_mtime(tmp_path, age_s, statuses):
    """A box that waited under the first version of the marker keeps its package."""
    a_waiting_package(tmp_path)
    marker = tmp_path / "exports" / "export-1.waiting"
    marker.write_text(json.dumps({"job_id": "job-3"}), encoding="utf-8")
    os.utime(marker, (1000.0, 1000.0))
    platform = FakePlatform()

    a_poller(
        tmp_path,
        platform=platform,
        buffer=FakeBuffer(),
        runner=Tools(),
        clock=lambda: 1000.0 + age_s,
    ).run_once()

    assert [r["status"] for r in platform.reports] == statuses
    assert not list((tmp_path / "exports").iterdir())


def test_a_package_left_by_an_interrupted_export_is_deleted_on_the_next_round(tmp_path):
    a_project(tmp_path)
    platform = FakePlatform(claims=[an_export_claim()], errors={"checking": [DOWN]})
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer(), runner=Tools())

    with pytest.raises(PlatformUnavailableError):
        poller.run_once()
    assert (tmp_path / "exports" / "export-1").is_dir()

    poller.run_once()
    assert not list((tmp_path / "exports").iterdir())


def test_a_disk_below_the_floor_refuses_an_export_before_anything_is_written(tmp_path):
    a_project(tmp_path)
    platform = FakePlatform(claims=[an_export_claim()])
    tools = Tools()
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer(), runner=tools, free=2 * GIB)

    poller.run_once()

    [report] = platform.reports
    assert report["status"] == "failed" and "2.0 GiB" in report["error"], report
    assert tools.commands == []


def test_the_export_path_reaches_neither_r2_nor_a_signed_url():
    """The package is the client's know-how: it goes to cctv-vps by the tailnet, nowhere else."""
    tree = ast.parse(Path(annotation_poller.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for alias in n.names
    }
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    found = [n for n in names if re.search(r"r2|presign|signed|upload|boto", n, re.IGNORECASE)]
    assert found == []


# ----- splits into folds (gpu-exchange#308) -----

DECLARED = {
    "kind": "declaration",
    "at": "2026-11-24T10:00:00+00:00",
    "declared_before_any_model_run": True,
    "folds": {"train": [1], "heldout": [2]},
}
RUN = {"kind": "model_run", "at": "2026-11-24T11:00:00+00:00", "model": "hala/v1.0.0"}


def a_project_with_clips(tmp_path: Path, *entries: dict) -> Path:
    """A project whose clips are the glued windows `w1.mp4` and `w2.mp4`, as on the box."""
    project = a_project(tmp_path)
    clips = [{"id": i, "clip": {"path": f"/state/windows/project-1/w{i}.mp4"}} for i in (1, 2)]
    (project / "manifest.json").write_text(json.dumps({"clips": clips}), encoding="utf-8")
    if entries:
        ledger = {"schema": "zone-annotator/split/1", "entries": list(entries)}
        (project / "split.json").write_text(json.dumps(ledger), encoding="utf-8")
    return project


@dataclass
class SplitTool:
    """zone-annotator's `split`: writes the declaration it was given, or refuses."""

    project: Path
    code: int = 0
    err: str = ""
    commands: list[list[str]] = field(default_factory=list)

    def __call__(self, cmd: list[str], **_kw):
        self.commands.append(cmd)
        if self.code == 0 and "declare" in cmd:
            ledger = {"schema": "zone-annotator/split/1", "entries": [DECLARED]}
            (self.project / "split.json").write_text(json.dumps(ledger), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, self.code, "", self.err)


def a_split_claim(**overrides) -> AnnotationSplitClaim:
    claim = AnnotationSplitClaim(
        job_id="job-4",
        split_id="split-1",
        project_id="project-1",
        folds={"train": ["w1"], "heldout": ["w2"]},
    )
    return replace(claim, **overrides)


def test_a_first_split_is_declared_over_the_windows_clips_and_reports_the_ledger(tmp_path):
    project = a_project_with_clips(tmp_path)
    platform = FakePlatform(claims=[a_split_claim()])
    tool = SplitTool(project)
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer(), runner=tool)

    assert poller.run_once() is True

    [split] = tool.commands
    assert split[6:] == [
        "/opt/zone-annotator/current/.venv/bin/python",
        "-m",
        "zone_annotator.split",
        str(project),
        "declare",
        "--fold=train=w1.mp4",
        "--fold=heldout=w2.mp4",
    ]
    assert platform.reports == [
        {
            "job_id": "job-4",
            "status": "applied",
            "ledger": {
                "entries": [
                    {
                        "kind": "declaration",
                        "at": "2026-11-24T10:00:00+00:00",
                        "folds": {"train": ["w1.mp4"], "heldout": ["w2.mp4"]},
                    }
                ]
            },
        }
    ]


def test_a_split_after_a_registered_run_is_an_amendment_the_tool_may_refuse(tmp_path):
    project = a_project_with_clips(tmp_path, DECLARED, RUN)
    platform = FakePlatform(claims=[a_split_claim(folds={"train": ["w1", "w2"]})])
    said = "split failed: a model run has been registered, so amending needs --by"
    tool = SplitTool(project, code=1, err=f"{said}\n")
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer(), runner=tool)

    poller.run_once()

    [split] = tool.commands
    assert split[10:] == ["amend", "--fold=train=w1.mp4,w2.mp4"]
    [report] = platform.reports
    assert report["status"] == "failed" and report["error"] == said
    assert [e["kind"] for e in report["ledger"]["entries"]] == ["declaration", "model_run"]
    assert report["ledger"]["entries"][1] == {
        "kind": "model_run",
        "at": "2026-11-24T11:00:00+00:00",
        "model": "hala/v1.0.0",
    }


def test_an_amendment_carries_the_name_and_reason_it_was_given_as_values(tmp_path):
    project = a_project_with_clips(tmp_path, DECLARED, RUN)
    claim = a_split_claim(by="-Anna Kowalska", reason="doszło trzecie okno")
    tool = SplitTool(project)
    poller = a_poller(
        tmp_path, platform=FakePlatform(claims=[claim]), buffer=FakeBuffer(), runner=tool
    )

    poller.run_once()

    [split] = tool.commands
    assert split[-2:] == ["--by=-Anna Kowalska", "--reason=doszło trzecie okno"]


@dataclass
class RunTool:
    """zone-annotator's `split register-run`: appends the run to the ledger, or refuses."""

    project: Path
    code: int = 0
    err: str = ""
    commands: list[list[str]] = field(default_factory=list)

    def __call__(self, cmd: list[str], **_kw):
        self.commands.append(cmd)
        if self.code == 0:
            ledger = {"schema": "zone-annotator/split/1", "entries": [DECLARED, RUN]}
            (self.project / "split.json").write_text(json.dumps(ledger), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, self.code, "", self.err)


def a_run_claim() -> AnnotationRegisterRunClaim:
    return AnnotationRegisterRunClaim(
        job_id="job-5", export_id="export-1", project_id="project-1", model="hala/v1.0.0"
    )


def test_a_published_head_is_registered_by_its_name_freezing_the_split(tmp_path):
    """gpu-exchange#314, D7: the run is registered on the box, and the ledger after it reported."""
    project = a_project_with_clips(tmp_path, DECLARED)
    platform = FakePlatform(claims=[a_run_claim()])
    tool = RunTool(project)
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer(), runner=tool)

    assert poller.run_once() is True

    [run] = tool.commands
    assert run[-3:] == [str(project), "register-run", "--model=hala/v1.0.0"]
    [report] = platform.reports
    assert report["job_id"] == "job-5" and report["status"] == "registered"
    assert [e["kind"] for e in report["ledger"]["entries"]] == ["declaration", "model_run"]


def test_a_run_the_tool_refuses_is_reported_in_its_words_with_the_ledger(tmp_path):
    project = a_project_with_clips(tmp_path)
    platform = FakePlatform(claims=[a_run_claim()])
    said = "register-run failed: no split has been declared"
    tool = RunTool(project, code=1, err=f"{said}\n")
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer(), runner=tool)

    poller.run_once()

    [report] = platform.reports
    assert report == {
        "job_id": "job-5",
        "status": "failed",
        "error": said,
        "ledger": {"entries": []},
    }


# ----- a project deleted in the panel -----


def a_project_on_the_box(tmp_path: Path) -> list[Path]:
    """All of project-1 on the box: labels, a glued window, a package waiting for cctv-vps."""
    project = a_project(tmp_path)
    # zone-annotator's lock around manifest writes (15ddc8d): it goes with the project.
    (project / ".manifest.lock").write_bytes(b"")
    window = tmp_path / "windows" / "project-1" / "window-1.mp4"
    window.parent.mkdir(parents=True)
    window.write_bytes(b"glued")
    package = tmp_path / "exports" / "export-1"
    (package / "images").mkdir(parents=True)
    marker = tmp_path / "exports" / "export-1.waiting"
    marker.write_text(json.dumps({"job_id": "job-3", "created": 0.0}), encoding="utf-8")
    return [project, window.parent, package, marker]


def a_delete_claim() -> AnnotationDeleteProjectClaim:
    return AnnotationDeleteProjectClaim(
        job_id="job-6", project_id="project-1", export_ids=["export-1"]
    )


def test_a_deleted_project_leaves_nothing_of_it_on_the_box_and_nothing_else_goes(tmp_path):
    doomed = a_project_on_the_box(tmp_path)
    other = tmp_path / "projects" / "project-2"
    other.mkdir()
    platform = FakePlatform(claims=[a_delete_claim()])
    tools = Tools()
    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer(), runner=tools, clock=lambda: 1.0
    )

    assert poller.run_once() is True

    assert [p for p in doomed if p.exists()] == []
    assert other.is_dir()
    assert platform.reports[-1] == {"job_id": "job-6", "status": "deleted"}


@pytest.mark.skipif(sys.platform == "win32", reason="flock: the box is Linux")
def test_a_project_is_deleted_only_while_no_cut_holds_zone_annotators_lock(tmp_path, monkeypatch):
    import fcntl

    a_project_on_the_box(tmp_path)
    held: list[bool] = []
    rmtree = shutil.rmtree

    def spy(path, *args, **kwargs):
        with open(tmp_path / "projects" / ".cutting.lock", "a+b") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(False)
            except BlockingIOError:
                held.append(True)
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(annotation_poller.shutil, "rmtree", spy)
    poller = a_poller(
        tmp_path, platform=FakePlatform(claims=[a_delete_claim()]), buffer=FakeBuffer()
    )

    poller.run_once()

    assert held and all(held), held


def test_a_project_the_box_cannot_delete_is_reported_failed_in_the_systems_words(
    tmp_path, monkeypatch
):
    a_project_on_the_box(tmp_path)

    def refused(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(annotation_poller.shutil, "rmtree", refused)
    platform = FakePlatform(claims=[a_delete_claim()])
    poller = a_poller(tmp_path, platform=platform, buffer=FakeBuffer())

    poller.run_once()

    report = platform.reports[-1]
    assert report["job_id"] == "job-6" and report["status"] == "failed", report
    assert "Permission denied" in report["error"], report


# ----- a cut that never ends -----


def test_a_later_window_whose_ingest_outlives_the_limit_is_failed_and_its_copy_deleted(
    tmp_path,
):
    """GNU timeout exits 124 when it had to stop the command."""
    a_project(tmp_path)
    platform = FakePlatform(claims=[a_later_claim()])
    runner = Ingests(returncode=124)
    poller = a_poller(
        tmp_path, platform=platform, buffer=FakeBuffer([a_chunk(tmp_path)]), runner=runner
    )

    poller.run_once()

    [report] = platform.reports
    assert report["status"] == "failed" and "4 h" in report["error"], report
    assert not (tmp_path / "windows" / "project-1" / "window-2.mp4").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="GNU timeout and process groups: Linux")
def test_the_limit_stops_the_whole_ingest_including_what_it_started(tmp_path):
    """zone-annotator cuts with ffmpeg children; killing only Python would leave them cutting."""
    a_project(tmp_path)
    grandchild = tmp_path / "grandchild.pid"
    hung = tmp_path / "hung-ingest"
    hung.write_text(f"#!/bin/sh\nsleep 60 &\necho $! >{grandchild}\nwait\n")
    hung.chmod(0o755)
    platform = FakePlatform(claims=[a_later_claim()])
    poller = AnnotationPoller(
        platform=platform,
        buffer=FakeBuffer([a_chunk(tmp_path)]),
        windows_dir=tmp_path / "windows",
        projects_dir=tmp_path / "projects",
        trim_fn=Trims(),
        free_bytes=lambda _path: 60 * GIB,
        zone_annotator_python=str(hung),
        cut_limit_s=1,
    )

    poller.run_once()

    assert platform.reports[-1]["status"] == "failed", platform.reports
    stat = Path(f"/proc/{grandchild.read_text().strip()}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z", "ffmpeg's stand-in still runs"


@pytest.mark.skipif(sys.platform == "win32", reason="flock: the box is Linux")
def test_a_project_whose_cut_outlives_the_limit_is_not_deleted_and_says_why(tmp_path):
    import fcntl

    doomed = a_project_on_the_box(tmp_path)
    now = [0.0]
    platform = FakePlatform(claims=[a_delete_claim()])
    poller = a_poller(
        tmp_path,
        platform=platform,
        buffer=FakeBuffer(),
        clock=lambda: now[0],
        sleep=lambda s: now.__setitem__(0, now[0] + s),
    )

    with open(tmp_path / "projects" / ".cutting.lock", "a+b") as cut:
        fcntl.flock(cut, fcntl.LOCK_EX)
        poller.run_once()

    report = platform.reports[-1]
    assert report["status"] == "failed" and "4 h" in report["error"], report
    assert all(p.exists() for p in doomed)


@pytest.mark.skipif(sys.platform == "win32", reason="flock: the box is Linux")
def test_a_week_old_window_waits_for_the_next_round_while_a_cut_runs(tmp_path):
    import fcntl

    window = tmp_path / "windows" / "project-1" / "window-1.mp4"
    window.parent.mkdir(parents=True)
    window.write_bytes(b"glued")
    os.utime(window, (1000.0, 1000.0))
    (tmp_path / "projects").mkdir()
    poller = a_poller(
        tmp_path, platform=FakePlatform(), buffer=FakeBuffer(), clock=lambda: 1000.0 + 8 * 86400
    )

    with open(tmp_path / "projects" / ".cutting.lock", "a+b") as cut:
        fcntl.flock(cut, fcntl.LOCK_EX)
        poller.run_once()
        assert window.exists()

    poller.run_once()
    assert not window.exists()


# ----- a job's last report, kept until the platform has it -----


def test_a_report_that_does_not_get_through_is_kept_and_sent_again(tmp_path):
    """A lost `ready` left a cut window to the platform's sweeper, which fails it after 6 h."""
    a_project(tmp_path)
    platform = FakePlatform(claims=[a_later_claim(), a_split_claim()], errors={"ready": [DOWN]})
    now = [1000.0]
    poller = a_poller(
        tmp_path,
        platform=platform,
        buffer=FakeBuffer([a_chunk(tmp_path)]),
        runner=Ingests(),
        clock=lambda: now[0],
    )

    assert poller.run_once() is True
    assert poller.run_once() is True, "the next claim is not held up by the lost report"
    now[0] += 300
    poller.run_once()

    assert [(r["job_id"], r["status"]) for r in platform.reports] == [
        ("job-2", "ready"),
        ("job-4", "applied"),
        ("job-2", "ready"),
    ]
    assert not list((tmp_path / "reports").iterdir())


def test_a_kept_report_is_sent_by_the_next_poller_after_a_restart(tmp_path):
    first = FakePlatform(claims=[a_claim()], errors={"awaiting_zones": [DOWN]})
    a_poller(tmp_path, platform=first, buffer=FakeBuffer([a_chunk(tmp_path)])).run_once()

    platform = FakePlatform()
    a_poller(tmp_path, platform=platform, buffer=FakeBuffer()).run_once()

    window = (tmp_path / "windows" / "project-1" / "window-1.mp4").resolve()
    assert platform.reports == [
        {"job_id": "job-1", "status": "awaiting_zones", "clip_path": str(window)}
    ]
    assert not list((tmp_path / "reports").iterdir())


@pytest.mark.parametrize("code", [404, 409])
def test_a_report_the_platform_answers_404_or_409_is_not_sent_again(tmp_path, code):
    """The job is gone, or settled without this box: a repeat would get the same answer."""
    a_project_with_clips(tmp_path)
    refused = PlatformRequestError(code, f"platform refused annotation split status ({code})")
    platform = FakePlatform(claims=[a_split_claim()], errors={"applied": [refused]})
    now = [1000.0]
    poller = a_poller(
        tmp_path,
        platform=platform,
        buffer=FakeBuffer(),
        runner=SplitTool(tmp_path / "projects" / "project-1"),
        clock=lambda: now[0],
    )

    poller.run_once()
    now[0] += 3600
    poller.run_once()

    assert [r["status"] for r in platform.reports] == ["applied"]
    assert not list((tmp_path / "reports").iterdir())
