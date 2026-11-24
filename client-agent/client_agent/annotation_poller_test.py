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
import re
import subprocess
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from client_agent import annotation_poller
from client_agent.annotation_poller import AnnotationPoller
from client_agent.buffer import BufferChunk
from client_agent.platform import AnnotationExportClaim, AnnotationJobClaim

START = datetime(2026, 11, 23, 9, 0, tzinfo=UTC)
END = datetime(2026, 11, 23, 9, 20, tzinfo=UTC)
GIB = 1024**3


@dataclass
class FakePlatform:
    claims: list[AnnotationJobClaim] = field(default_factory=list)
    reports: list[dict] = field(default_factory=list)

    def claim_next_annotation_job(self) -> AnnotationJobClaim | None:
        return self.claims.pop(0) if self.claims else None

    def report_annotation_job(self, job_id: str, **body: object) -> None:
        self.reports.append({"job_id": job_id, **body})

    def report_annotation_export(self, job_id: str, **body: object) -> None:
        self.reports.append({"job_id": job_id, **body})


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
        return start


def a_poller(
    tmp_path: Path, *, platform, buffer, trim=None, free=60 * GIB, runner=None, clock=None
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
    assert "StrictHostKeyChecking=accept-new" in ssh
    assert f"UserKnownHostsFile={key}.known_hosts" in ssh
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
