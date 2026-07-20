"""End-to-end integration test: mediamtx → recorder → buffer → TaskPoller.

Real ffmpeg, real RTSP server (bluenviron/mediamtx via Docker), real
:class:`RollingBuffer`, real :func:`trim_and_concat`. Only the platform
client and the uploader stay faked — everything else is the production
code path. Marked ``integration`` so it skips by default; opt-in with
``pytest -m integration``.

Auto-skips on hosts without Docker or ffmpeg so contributors on macOS
dev boxes can run ``pytest -m integration`` without a confusing failure.

The runbook this test encodes is the carry-over from #27 AC: "mediamtx
fixture, recorder consumes fake RTSP, buffer trims old, TaskPoller
(mock platform) claim → trim → uploader (mock R2) → status uploaded".
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _docker_running() -> bool:
    """``docker info`` exits non-zero (or times out) when the daemon
    isn't listening on the conventional socket. We check that before
    every test rather than at import time so transient daemon outages
    surface as test skips, not collection errors."""
    if not shutil.which("docker"):
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    return True


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


_REASON = "mediamtx integration test requires docker daemon + ffmpeg on host"


@pytest.fixture(scope="module")
def mediamtx_url() -> str:
    """Spin up bluenviron/mediamtx for the module and tear it down after.

    Skips at fixture acquisition time if docker is unreachable so each
    test still gets a clean ``SKIPPED`` line rather than an ERROR."""
    if not _docker_running() or not _ffmpeg_available():
        pytest.skip(_REASON)

    container_name = f"cctv-test-mediamtx-{uuid.uuid4().hex[:8]}"
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "-p",
            "8554:8554",
            "bluenviron/mediamtx:latest",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        pytest.skip(f"docker run mediamtx failed: {proc.stderr.strip()}")

    # Mediamtx binds in <1s on warm machines but a fresh image pull can
    # take longer. Poll the published port instead of sleeping a fixed
    # amount so warm runs stay fast.
    import socket

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 8554), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)
    else:
        subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=10)
        pytest.skip("mediamtx never bound :8554 within 10s")

    try:
        yield "rtsp://127.0.0.1:8554"
    finally:
        subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=10)


@pytest.fixture
def rtsp_stream(mediamtx_url: str):
    """Push a 30s ffmpeg testsrc to a fresh mediamtx path; yield the URL.

    Each test gets its own UUID path so a leaked publisher from the
    previous test cannot interfere with reads."""
    stream_path = f"/test-{uuid.uuid4().hex[:8]}"
    pushed_url = f"{mediamtx_url}{stream_path}"
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "warning",
            "-re",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=30:size=320x240:rate=10",
            "-c:v",
            "libx264",
            "-tune",
            "zerolatency",
            "-preset",
            "ultrafast",
            "-f",
            "rtsp",
            pushed_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Let mediamtx accept the publisher before any reader connects.
    # 2s is comfortably above the typical bind-and-handshake window.
    time.sleep(2)
    try:
        yield pushed_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def _buffer_chunks(cam_dir: Path) -> list[Path]:
    """The chunks a buffer-mode recording left for :class:`RollingBuffer`.

    Passing ``camera_id`` selects buffer mode, which *always* uses the segment
    muxer, so the recorder already writes ``chunk_*.mp4`` regardless of how
    short the recording is — the same glob the buffer scans for. This used to
    rename ``recording.mp4`` into place on the belief that segmenting only
    kicked in above ``SEGMENT_SECONDS``; that only holds for the non-buffer
    branch, so the rename never actually fired.

    Sorting lexically is sorting chronologically — see ``BUFFER_CHUNK_TEMPLATE``
    (issue #90)."""
    return sorted(cam_dir.glob("chunk_*.mp4"))


# ----- 1. Recorder writes at least one usable chunk -----


def test_recorder_against_mediamtx_writes_chunks(rtsp_stream: str, tmp_path: Path) -> None:
    """Real :class:`Recorder` in buffer mode reads mediamtx for 5s and
    deposits at least one >0 byte mp4 in the per-camera buffer dir.

    A regression in ``build_ffmpeg_cmd`` (e.g. dropping ``-c copy`` and
    silently transcoding) or in the buffer-mode branch of
    :meth:`Recorder.start` (e.g. cleaning up the dir post-write) would
    surface here as an empty buffer."""
    from client_agent.recorder import Recorder

    buffer_dir = tmp_path / "buf"
    rec = Recorder(
        runner=subprocess.run,
        output_dir_factory=lambda cam_id: str(buffer_dir / cam_id),
    )
    rec.start(url=rtsp_stream, duration_s=5, camera_id="cam-integration")

    cam_files = list((buffer_dir / "cam-integration").glob("*.mp4"))
    assert cam_files, f"recorder produced no chunks in {buffer_dir / 'cam-integration'}"
    assert all(f.stat().st_size > 0 for f in cam_files), "recorder left 0-byte chunks"


# ----- 2. Full pipeline: recorder → buffer → poller → status uploaded -----


@dataclass
class _FakePlatform:
    """Single-task scripted fake matching the poller's platform Protocol."""

    next_task: object | None
    status_calls: list[tuple[str, str, str | None]] = field(default_factory=list)
    _drained: bool = False

    def fetch_next_task(self):
        if self._drained:
            return None
        self._drained = True
        return self.next_task

    def update_task_status(self, task_id: str, *, status: str, error: str | None = None) -> None:
        self.status_calls.append((task_id, status, error))


@dataclass
class _FakeUploader:
    """Always-success uploader so the test focuses on the recorder ↔
    buffer ↔ poller boundary, not on R2 retry semantics (those are
    covered exhaustively in :file:`uploader_test.py`)."""

    upload_calls: list[tuple[str, list[Path]]] = field(default_factory=list)

    def upload_chunks(self, task_id: str, chunks):
        from client_agent.uploader import UploadResult

        self.upload_calls.append((task_id, list(chunks)))
        return [
            UploadResult(
                chunk_n=i,
                success=True,
                key=f"tenants/t/results/t/{task_id}/chunk_{i}.mp4",
            )
            for i in range(len(chunks))
        ]


def test_full_pipeline_mediamtx_to_status_uploaded(rtsp_stream: str, tmp_path: Path) -> None:
    """Production pipeline end-to-end: real recorder records → real
    buffer locates → real trim/concat produces → fake uploader confirms
    → poller transitions claim → recording → uploading → uploaded.

    The transitions list is the load-bearing assertion: anything other
    than the four-state happy path is a regression in either the poller
    state machine or in the buffer's range scan."""
    from client_agent.buffer import RollingBuffer
    from client_agent.ffmpeg_trim import trim_and_concat
    from client_agent.platform import Task
    from client_agent.poller import TaskPoller
    from client_agent.recorder import Recorder

    buffer_dir = tmp_path / "buf"
    trim_output_dir = tmp_path / "trim"
    cam_id = "cam-1"

    # 1. Record 5s into the buffer.
    rec = Recorder(
        runner=subprocess.run,
        output_dir_factory=lambda c: str(buffer_dir / c),
    )
    rec.start(url=rtsp_stream, duration_s=5, camera_id=cam_id)
    chunks = _buffer_chunks(buffer_dir / cam_id)
    assert chunks, "no usable chunks after recorder finished"

    # 2. Build a Task whose window overlaps the chunks' inferred range
    # (RollingBuffer infers a chunk's [start, end] from its mtime and
    # ``segment_seconds``; with a 5s recording we use segment_seconds=5
    # so the window straddles the just-written file's mtime exactly).
    chunk_mtime = datetime.fromtimestamp(chunks[-1].stat().st_mtime, tz=UTC)
    buffer = RollingBuffer(
        base_dir=buffer_dir,
        buffer_hours=1,
        segment_seconds=5,
    )
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    task = Task(
        id=task_id,
        camera_id=cam_id,
        start_time=chunk_mtime - timedelta(seconds=4),
        end_time=chunk_mtime,
    )

    platform = _FakePlatform(next_task=task)
    uploader = _FakeUploader()

    # 3. Drive the poller through one cycle.
    poller = TaskPoller(
        platform=platform,
        buffer=buffer,
        trim_fn=trim_and_concat,
        output_dir=trim_output_dir,
        uploader=uploader,
    )
    handled = poller.run_once()

    assert handled is True
    statuses = [s for _, s, _ in platform.status_calls]
    assert statuses == ["recording", "uploading", "uploaded"], (
        f"expected happy-path transitions, got {platform.status_calls!r}"
    )
    assert uploader.upload_calls, "uploader was never invoked"
    (uploaded_task_id, uploaded_chunks) = uploader.upload_calls[0]
    assert uploaded_task_id == task_id
    assert len(uploaded_chunks) == 1
    assert uploaded_chunks[0].exists(), "trimmed mp4 vanished before assertion"
    assert uploaded_chunks[0].stat().st_size > 0
