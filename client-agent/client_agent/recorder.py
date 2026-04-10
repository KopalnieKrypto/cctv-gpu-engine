"""RTSP recorder for the client-agent (issue #8).

Three concerns live here:

1. :func:`probe_rtsp` — short ffmpeg invocation that gives the operator a
   yes/no answer for "does this RTSP URL work?" without ever leaving a
   hung process behind.
2. :func:`build_ffmpeg_cmd` — pure helper that builds the ffmpeg argv for
   stream-copy recording. Short recordings (≤1h) use ``-t``; long ones
   use ``-f segment -segment_time 3600`` so files cap at 1h each.
3. :class:`Recorder` — single-recording state machine that runs ffmpeg,
   then uploads the produced chunks to R2 using the same ``status.json``
   handshake the MP4-upload path uses.

The subprocess and R2 surfaces are both injectable so the unit tests in
``recorder_test.py`` never fork ffmpeg or hit the network.
"""

from __future__ import annotations

import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of an RTSP probe.

    ``ok`` is the operator's go/no-go signal; ``message`` carries the
    underlying ffmpeg/ffprobe stderr (or our own ``"timeout"`` marker)
    so the UI can show *why* it failed.
    """

    ok: bool
    message: str = ""


def probe_rtsp(url: str, *, timeout: float, runner) -> ProbeResult:
    """Probe an RTSP URL via ``runner`` (a ``subprocess.run``-shaped callable).

    Production wires ``runner=subprocess.run``; tests pass a fake. The
    ``subprocess.TimeoutExpired`` exception is the contract for "ffmpeg
    hung past the deadline" — we catch it here so the operator sees a
    clean ProbeResult instead of a Flask 500 (#8 testing focus).
    """
    import subprocess as _sp

    try:
        result = runner(
            [
                "ffmpeg",
                "-rtsp_transport",
                "tcp",
                "-i",
                url,
                "-t",
                "1",
                "-f",
                "null",
                "-",
            ],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except _sp.TimeoutExpired:
        return ProbeResult(ok=False, message=f"timeout after {timeout}s")
    if result.returncode == 0:
        return ProbeResult(ok=True)
    return ProbeResult(ok=False, message=(result.stderr or "").strip() or "ffmpeg failed")


SEGMENT_SECONDS = 3600
"""Segment boundary for long recordings (SPEC §7.3 — 1h chunks)."""


def build_ffmpeg_cmd(*, url: str, duration_s: int, output_dir: str) -> list[str]:
    """Build the ffmpeg argv for an RTSP stream-copy recording.

    Short recordings (≤1h) land in a single ``recording.mp4``. Recordings
    longer than ``SEGMENT_SECONDS`` use the segment muxer so each file
    caps at one hour and is independently uploadable. ``-t`` still bounds
    the total duration in both branches — without it a stuck camera would
    record forever.

    ``-an`` drops the audio track: the pipeline only analyses video
    (YOLO-pose), and many IP cameras (Hikvision in particular) emit
    ``pcm_mulaw`` audio which is not muxable into MP4, causing ffmpeg to
    fail with "Could not find tag for codec pcm_mulaw in stream #1".
    """
    base = [
        "ffmpeg",
        "-rtsp_transport",
        "tcp",
        "-i",
        url,
        "-c",
        "copy",
        "-an",
        "-t",
        str(duration_s),
    ]
    if duration_s > SEGMENT_SECONDS:
        return [
            *base,
            "-f",
            "segment",
            "-segment_time",
            str(SEGMENT_SECONDS),
            "-reset_timestamps",
            "1",
            f"{output_dir}/chunk_%03d.mp4",
        ]
    return [*base, f"{output_dir}/recording.mp4"]


class _UploaderLike(Protocol):
    """Narrow R2 surface the recorder needs.

    Mirrors the relevant slice of :class:`client_agent.web.ClientR2Like`
    — no overlap with the upload-MP4 form, no list/get methods, just the
    two writes a recording produces."""

    def upload_input_chunk(self, job_id: str, fileobj: Any) -> str: ...
    def put_status(self, job_id: str, status: dict[str, Any]) -> None: ...


class RecorderBusy(RuntimeError):
    """Raised when ``Recorder.start`` is called while a recording is in
    flight. Acceptance criterion (#8): "second recording while one
    active → rejected"."""


@dataclass
class RecorderStatus:
    """Snapshot of the recorder's state machine.

    Held behind a lock; the dataclass is value-semantics so callers can
    inspect a snapshot without racing the worker thread.
    """

    state: str = "idle"  # idle | recording | uploading | done | failed
    job_id: str | None = None
    message: str = ""
    chunks_uploaded: int = 0


def _default_job_id() -> str:
    """Same shape as the upload form's job_id factory so /jobs rows from
    recordings sit alongside upload rows without visual surprise."""
    return f"job-{uuid.uuid4().hex[:12]}"


@dataclass
class Recorder:
    """Single-recording state machine.

    Holds at most one recording at a time — concurrent ``start`` calls
    raise :class:`RecorderBusy`. After ffmpeg exits, all chunk files in
    the output dir are uploaded to R2 in lexical order, then a single
    ``status.json`` (status=pending) is written so the gpu-service worker
    picks up the recording exactly the same way it picks up an MP4 from
    the upload form.

    All side-effecting collaborators (subprocess runner, R2 client, dir
    factory, job_id factory, clock) are injected so the unit tests stay
    hermetic.
    """

    uploader: _UploaderLike
    runner: Callable[..., Any]
    output_dir_factory: Callable[[str], str]
    job_id_factory: Callable[[], str] = _default_job_id
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))

    _status: RecorderStatus = field(default_factory=RecorderStatus)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def status(self) -> RecorderStatus:
        """Return a snapshot of the current state. Cheap, lock-protected."""
        with self._lock:
            return RecorderStatus(
                state=self._status.state,
                job_id=self._status.job_id,
                message=self._status.message,
                chunks_uploaded=self._status.chunks_uploaded,
            )

    def start(self, *, url: str, duration_s: int) -> str:
        """Run ffmpeg synchronously, then upload the produced chunks.

        Returns the ``job_id`` of the recording. Raises
        :class:`RecorderBusy` if another recording is already in flight.
        """
        with self._lock:
            if self._status.state in ("recording", "uploading"):
                raise RecorderBusy(f"recorder already busy in state {self._status.state!r}")
            job_id = self.job_id_factory()
            self._status = RecorderStatus(state="recording", job_id=job_id)

        out_dir = Path(self.output_dir_factory(job_id))
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_ffmpeg_cmd(url=url, duration_s=duration_s, output_dir=str(out_dir))
        # We don't catch subprocess errors here — partial output (the
        # camera-offline-mid-recording case) is reported by ffmpeg as a
        # non-zero exit but still leaves chunks on disk, so we always
        # walk the dir afterwards instead of branching on returncode.
        result = self.runner(cmd, capture_output=True, text=True)

        chunks = sorted(out_dir.glob("*.mp4"))
        if not chunks:
            # Hard failure — invalid URL, immediate connection refused.
            # Don't write status.json: the worker would pick up an empty job.
            with self._lock:
                self._status = RecorderStatus(
                    state="failed",
                    job_id=job_id,
                    message=(getattr(result, "stderr", "") or "").strip()
                    or f"ffmpeg exited {getattr(result, 'returncode', '?')} with no output",
                )
            shutil.rmtree(out_dir, ignore_errors=True)
            return job_id

        with self._lock:
            self._status = RecorderStatus(state="uploading", job_id=job_id, chunks_uploaded=0)

        for chunk in chunks:
            with chunk.open("rb") as fh:
                self.uploader.upload_input_chunk(job_id, fh)
            with self._lock:
                self._status.chunks_uploaded += 1

        now = self.clock().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.uploader.put_status(
            job_id,
            {
                "job_id": job_id,
                "status": "pending",
                "updated_at": now,
                "submitted_at": now,
            },
        )
        # Free the disk space — an 8h recording is ~32GB (#8 testing focus).
        shutil.rmtree(out_dir, ignore_errors=True)

        with self._lock:
            # Preserve chunks_uploaded across the idle transition so the
            # UI can show "last recording: N chunks" without re-walking R2.
            self._status = RecorderStatus(state="idle", job_id=job_id, chunks_uploaded=len(chunks))
        return job_id

    def probe(self, url: str, *, timeout: float = 10.0) -> ProbeResult:
        """Run an RTSP probe using the same runner as the recording.

        Lives on the Recorder so the web layer only needs to know about
        one collaborator. Production wires ``runner=subprocess.run``;
        tests pass a fake on ``Recorder`` construction (the FakeRecorder
        in the web tests overrides ``probe`` directly so this method is
        never invoked there)."""
        return probe_rtsp(url, timeout=timeout, runner=self.runner)

    def stop(self) -> None:
        """Mark the current recording as cancelled.

        The synchronous ffmpeg run can't be interrupted from here without
        a real subprocess handle, but flipping state to ``idle`` lets the
        web layer return a clean 200 and the next recording proceed. The
        production wiring (web route → background thread → real
        ``subprocess.run``) hands the actual SIGTERM problem to the
        thread layer, which is out of scope for the unit tests."""
        with self._lock:
            self._status = RecorderStatus(state="idle")


class BackgroundRecorder:
    """Thread wrapper around :class:`Recorder` for production use.

    A real recording runs for hours; the Flask request handler must
    return immediately. ``BackgroundRecorder.start`` spawns a daemon
    thread that calls the wrapped Recorder synchronously, leaving the
    busy-guard semantics intact (the wrapped Recorder still rejects
    concurrent ``start`` calls). All other methods delegate.

    Kept out of the unit tests because threading is the exact thing the
    injectable runner is meant to avoid — verifying it requires a real
    join + sleep loop, which the SPEC §6.2 e2e test on ``cctv-vps``
    will exercise once issue #8 lands.
    """

    def __init__(self, inner: Recorder) -> None:
        self._inner = inner

    def start(self, *, url: str, duration_s: int) -> str:
        # Pre-flight the busy check on the main thread so /start can
        # return 409 *before* the operator's request returns. Otherwise
        # the rejection would be invisible to the HTTP layer.
        snapshot = self._inner.status()
        if snapshot.state in ("recording", "uploading"):
            raise RecorderBusy(f"recorder already busy in state {snapshot.state!r}")

        thread = threading.Thread(
            target=self._inner.start,
            kwargs={"url": url, "duration_s": duration_s},
            daemon=True,
            name=f"recorder-{url[:32]}",
        )
        thread.start()
        return "scheduled"

    def stop(self) -> None:
        self._inner.stop()

    def status(self) -> RecorderStatus:
        return self._inner.status()

    def probe(self, url: str, *, timeout: float = 10.0) -> ProbeResult:
        return self._inner.probe(url, timeout=timeout)
