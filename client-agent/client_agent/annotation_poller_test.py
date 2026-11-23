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

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from client_agent.annotation_poller import AnnotationPoller
from client_agent.buffer import BufferChunk
from client_agent.platform import AnnotationJobClaim

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


def a_poller(tmp_path: Path, *, platform, buffer, trim=None, free=60 * GIB, runner=None):
    return AnnotationPoller(
        platform=platform,
        buffer=buffer,
        windows_dir=tmp_path / "windows",
        projects_dir=tmp_path / "projects",
        trim_fn=trim or Trims(),
        runner=runner or (lambda cmd, **kw: None),
        free_bytes=lambda _path: free,
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
