"""RTSP recorder for the client-agent (issue #8).

Three concerns live here:

1. :func:`probe_rtsp` — short ffmpeg invocation that gives the operator a
   yes/no answer for "does this RTSP URL work?" without ever leaving a
   hung process behind.
2. :func:`build_ffmpeg_cmd` — pure helper that builds the ffmpeg argv for
   stream-copy recording. Short recordings (≤1h) use ``-t``; long ones
   use ``-f segment -segment_time 3600`` so files cap at 1h each.
3. :class:`Recorder` — single-recording state machine that runs ffmpeg
   and leaves the produced chunks on disk for the task poller to pick up
   (buffer mode, issue #27). The historical one-shot R2 upload path was
   removed together with the legacy Docker UI flow (issue #29): the
   appliance uploads exclusively via presigned URLs, so the recorder no
   longer touches R2 at all.

The subprocess boundary is injectable so the unit tests in
``recorder_test.py`` never fork ffmpeg.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    except subprocess.TimeoutExpired:
        return ProbeResult(ok=False, message=f"timeout after {timeout}s")
    if result.returncode == 0:
        return ProbeResult(ok=True)
    return ProbeResult(ok=False, message=(result.stderr or "").strip() or "ffmpeg failed")


SEGMENT_SECONDS = 3600
"""Segment boundary for legacy long recordings (SPEC §7.3 — 1h chunks)."""

BUFFER_SEGMENT_SECONDS = 60
"""Segment boundary for buffer-mode recordings. Smaller than ``SEGMENT_SECONDS``
because the task poller needs *finalized* chunks to trim from (mid-segment
trim fails — the moov atom is only flushed at segment close). 60s strikes a
balance: enough time for ffmpeg to amortize segment-open overhead, short
enough that recent footage (last ~minute) lands in trimmable chunks within
a heartbeat cycle. Must stay in sync with ``RollingBuffer(segment_seconds=)``
in :func:`client_agent.appliance.start_poller_thread`."""

BUFFER_CHUNK_TEMPLATE = "chunk_%Y%m%d-%H%M%S.mp4"
"""ffmpeg segment filename template for buffer mode — a ``strftime`` pattern,
so the command MUST also carry ``-strftime 1`` or ffmpeg writes the literal
``%Y…`` string as a filename.

Time-derived, not counter-derived, and that is the whole point (issue #90).
The recorder respawns about hourly (``-t`` is a liveness cadence, not a
retention knob — #85), and the previous ``chunk_%03d.mp4`` restarted its
counter at 000 on every respawn, silently **overwriting** chunks the
retention cron intended to keep. An appliance set to ``buffer_hours=5``
therefore held ~1 h of footage. Timestamps never repeat, so a respawn now
appends to history instead of destroying it.

Two properties downstream code leans on:

* Still matches :class:`~client_agent.buffer.RollingBuffer`'s ``chunk_*.mp4``
  glob, so ``chunks_in_range`` / ``trim_old_chunks`` needed no change — both
  derive time from ``st_mtime``, never from the filename.
* Zero-padded, fixed-width, big-endian fields, so lexical sort == chronological
  sort (the property ``%03d`` used to provide).

One caveat worth knowing before reusing this template elsewhere: ffmpeg stamps
the name with **wallclock** time at segment open, not media time. Uniqueness
therefore relies on the source being realtime, so that consecutive segments
close in different seconds. A live RTSP camera always is — and it is the only
input this recorder ever has — which leaves a 60 s margin at
``BUFFER_SEGMENT_SECONDS``. Feed the same flags a *file* instead and ffmpeg
races through it, closing every segment inside one wallclock second so they
all overwrite each other (verified against ffmpeg 8.1.2: a 6 s clip at
``segment_time 1`` yields 5 files with ``-re`` and 1 without).

Removing the wrap also removes the accidental ~1 h disk cap that overwriting
provided: ``RollingBuffer.trim_all_cameras`` (ticked by the appliance's
maintenance thread every 60 s, issue #51) is now the ONLY thing bounding
buffer size."""


def build_ffmpeg_cmd(
    *, url: str, duration_s: int, output_dir: str, buffer_mode: bool = False
) -> list[str]:
    """Build the ffmpeg argv for an RTSP stream-copy recording.

    Three branches:

    * **buffer_mode=True** — segment muxer with ``BUFFER_SEGMENT_SECONDS``
      (default 60s) so the task poller has finalized chunks to trim. This
      is the appliance's per-camera recorder (issue #27) and the only path
      wired in production.
    * **duration_s > SEGMENT_SECONDS** — segment muxer with 1h chunks.
    * **otherwise** — single ``recording.mp4``.

    ``-t`` still bounds the total duration in all branches — without it a
    stuck camera would record forever.

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
    if buffer_mode:
        return [
            *base,
            "-f",
            "segment",
            "-segment_time",
            str(BUFFER_SEGMENT_SECONDS),
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            f"{output_dir}/{BUFFER_CHUNK_TEMPLATE}",
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

    state: str = "idle"  # idle | recording | failed
    job_id: str | None = None
    message: str = ""
    chunks_uploaded: int = 0


def _default_job_id() -> str:
    """Generate a job_id for a recording that isn't keyed by a camera_id.

    ``job-<12 hex>`` — the historical shape from when recordings shared the
    /jobs table with manual uploads."""
    return f"job-{uuid.uuid4().hex[:12]}"


@dataclass(kw_only=True)
class Recorder:
    """Single-recording state machine.

    Holds at most one recording at a time — concurrent ``start`` calls
    raise :class:`RecorderBusy`. After ffmpeg exits, the produced chunk
    files are left on disk under ``output_dir_factory(job_id)`` for the
    task poller to pick up (buffer mode, issue #27). The recorder never
    uploads to R2 — that was the legacy Docker UI flow, retired in #29.

    All side-effecting collaborators (subprocess runner, dir factory,
    job_id factory, clock) are injected so the unit tests stay hermetic."""

    # ``runner`` is the ``subprocess.run``-shaped boundary used ONLY by
    # :meth:`probe` (a bounded, blocking call that must surface
    # ``TimeoutExpired``). ``popen_factory`` is the ``subprocess.Popen``-shaped
    # boundary used by the recording itself: we keep the handle so
    # :meth:`stop` can actually SIGTERM the ffmpeg child (issue #52) instead
    # of only flipping in-memory state.
    runner: Callable[..., Any] = subprocess.run
    popen_factory: Callable[..., Any] = subprocess.Popen
    output_dir_factory: Callable[[str], str]
    job_id_factory: Callable[[], str] = _default_job_id
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    # Seconds to wait for ffmpeg to honour SIGTERM before escalating to
    # SIGKILL (issue #52). ffmpeg flushes the current segment's moov atom on
    # SIGTERM, so a few seconds keeps the last chunk playable; past that we
    # stop waiting so an operator "stop" can't hang on a wedged encoder.
    stop_grace_s: float = 5.0

    _status: RecorderStatus = field(default_factory=RecorderStatus)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # Handle on the in-flight ffmpeg child (``None`` when idle). Held under
    # ``_lock`` so :meth:`stop` on the web thread can read it while the
    # recording thread is blocked in ``communicate()``.
    _proc: Any = field(default=None)
    # Set by :meth:`stop` so the recording thread, once ffmpeg is terminated
    # and ``communicate`` returns, knows the exit was operator-requested and
    # returns without re-flipping the state machine.
    _cancelled: bool = field(default=False)

    def status(self) -> RecorderStatus:
        """Return a snapshot of the current state. Cheap, lock-protected."""
        with self._lock:
            return RecorderStatus(
                state=self._status.state,
                job_id=self._status.job_id,
                message=self._status.message,
                chunks_uploaded=self._status.chunks_uploaded,
            )

    def start(self, *, url: str, duration_s: int, camera_id: str | None = None) -> str:
        """Run ffmpeg synchronously; leave the produced chunks on disk.

        Returns the ``job_id`` of the recording. Raises
        :class:`RecorderBusy` if another recording is already in flight.

        ``camera_id`` selects **buffer mode** (issue #27): chunks land in
        ``output_dir_factory(camera_id)`` (keyed by camera, not a generated
        job_id) with 60s segments so the task poller has finalized chunks to
        trim. Either way the chunks are **left on disk** — the recorder no
        longer uploads anywhere (the poller owns upload via presigned URLs,
        issue #29).

        Split into :meth:`_reserve` (atomic slot claim) + :meth:`_run`
        (the ffmpeg work) so :class:`BackgroundRecorder` can claim the slot
        synchronously on the HTTP thread and only defer ``_run`` to the
        daemon thread — closing the concurrent-start TOCTOU (issue #52)."""
        job_id = self._reserve(camera_id)
        return self._run(job_id, url=url, duration_s=duration_s, camera_id=camera_id)

    def _reserve(self, camera_id: str | None) -> str:
        """Atomically claim the single recording slot, returning the ``job_id``.

        Raises :class:`RecorderBusy` if a recording is already in flight. This
        is the ONLY place the ``idle → recording`` transition happens, and it
        runs under ``_lock``, so two callers racing to start can never both
        win — exactly one gets the slot, the other gets ``RecorderBusy`` on
        its own (HTTP) thread."""
        buffer_mode = camera_id is not None
        with self._lock:
            if self._status.state == "recording":
                raise RecorderBusy(f"recorder already busy in state {self._status.state!r}")
            # In buffer mode the camera_id stands in for the job_id on the
            # status snapshot — operators and metrics key on it the same
            # way (one identifier per active recording).
            job_id = camera_id if buffer_mode else self.job_id_factory()
            self._status = RecorderStatus(state="recording", job_id=job_id)
            self._cancelled = False
            self._proc = None
        return job_id

    def _run(self, job_id: str, *, url: str, duration_s: int, camera_id: str | None) -> str:
        """Run ffmpeg for an already-reserved slot and leave its chunks on disk.

        Assumes :meth:`_reserve` has already claimed the slot for ``job_id``;
        never re-checks busy state. Safe to run on a background thread. The
        produced chunks stay on disk for the task poller — the recorder does
        not upload (issue #29)."""
        buffer_mode = camera_id is not None
        out_dir = Path(self.output_dir_factory(job_id))
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_ffmpeg_cmd(
            url=url, duration_s=duration_s, output_dir=str(out_dir), buffer_mode=buffer_mode
        )
        # Snapshot the dir BEFORE ffmpeg runs so the post-run walk can tell
        # this run's output from history. In buffer mode ``out_dir`` is keyed
        # by camera and persists across respawns, so it is routinely full of
        # earlier chunks; counting them would report a run that wrote nothing
        # as a healthy one (and, since #90 removed the ``%03d`` wrap that
        # capped the dir at 60 files, by an ever-growing margin). Safe for the
        # legacy job_id-keyed path too — that dir is freshly created per run,
        # so the snapshot is empty and behaviour is unchanged.
        pre_existing = set(out_dir.glob("*.mp4"))
        # Spawn ffmpeg and KEEP THE HANDLE so ``stop`` can terminate it
        # (issue #52). ``communicate`` blocks until ffmpeg exits — naturally
        # (``-t`` elapsed), on a crash, or because ``stop`` sent it a signal.
        # We don't catch subprocess errors here — partial output (the
        # camera-offline-mid-recording case) is reported by ffmpeg as a
        # non-zero exit but still leaves chunks on disk, so we always
        # walk the dir afterwards instead of branching on returncode.
        proc = self.popen_factory(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with self._lock:
            self._proc = proc
            # Close the spawn-window race: if ``stop`` fired between the
            # reservation and here (proc was still ``None`` when it looked),
            # it set ``_cancelled`` without a handle to kill. Honour it now so
            # a stop-right-after-start can never leave ffmpeg running.
            cancel_now = self._cancelled
        if cancel_now:
            self._terminate_proc(proc)
        _, stderr = proc.communicate()
        returncode = proc.returncode
        with self._lock:
            self._proc = None
            cancelled = self._cancelled

        if cancelled:
            # Operator/platform requested the stop. Any chunks ffmpeg already
            # flushed stay on disk for the poller. ``stop`` already set
            # state=idle; leave it (a fresh start may already own the state
            # machine, so we must not clobber it).
            return job_id

        chunks = sorted(set(out_dir.glob("*.mp4")) - pre_existing)
        # Filter out 0-byte files — ffmpeg creates the output file before
        # opening the input, so a mux failure (e.g. pcm_mulaw codec
        # unsupported by MP4) leaves an empty file on disk. Treat that the
        # same as "no chunks" → failed so the poller never trims an empty
        # segment.
        chunks = [c for c in chunks if c.stat().st_size > 0]
        if not chunks:
            with self._lock:
                self._status = RecorderStatus(
                    state="failed",
                    job_id=job_id,
                    message=(stderr or "").strip() or f"ffmpeg exited {returncode} with no output",
                )
            return job_id

        # Chunks stay on disk for the task poller to find later — no upload,
        # no cleanup (the buffer layout IS the contract). Flip straight back
        # to idle so the next restart cycle (e.g. per-day -t expiry) proceeds
        # cleanly. ``chunks_uploaded`` carries the produced-chunk count so the
        # UI can show "last recording: N chunks".
        with self._lock:
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
        """Terminate the in-flight ffmpeg child and reset to ``idle`` (issue #52).

        For a CCTV product "stop recording" must mean recording stops *now*,
        not "within ``duration_s``". We hold the ffmpeg handle (``_proc``), so
        this actually SIGTERMs it: the recording thread's ``communicate``
        returns, sees ``_cancelled``, and returns without re-flipping state.

        Ordering is race-safe against a near-simultaneous ``start``: we set
        ``_cancelled`` and read ``_proc`` under the same lock ``start`` uses to
        publish the handle. If the handle isn't up yet (stop beat the spawn),
        ``start`` observes ``_cancelled`` right after it publishes ``_proc``
        and terminates the child itself.

        Idempotent: calling ``stop`` while idle just re-asserts idle."""
        with self._lock:
            self._cancelled = True
            proc = self._proc
            self._status = RecorderStatus(state="idle")
        if proc is not None:
            self._terminate_proc(proc)

    def _terminate_proc(self, proc: Any) -> None:
        """SIGTERM the ffmpeg child and wait up to ``stop_grace_s`` for it to
        flush and exit. Safe to call from the web thread while the recording
        thread is blocked in ``communicate`` — modern CPython guards
        concurrent ``wait``/``communicate`` on one ``Popen`` internally."""
        proc.terminate()
        try:
            proc.wait(timeout=self.stop_grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()


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
        self._thread: threading.Thread | None = None

    def start(self, *, url: str, duration_s: int, camera_id: str | None = None) -> str:
        # Claim the recording slot SYNCHRONOUSLY on the caller's (HTTP)
        # thread. ``_reserve`` does the atomic idle→recording flip under the
        # inner lock and raises ``RecorderBusy`` if the slot is taken — so a
        # second concurrent ``/start`` is rejected here, on the request
        # thread, and the web layer turns it into a 409. The old code only
        # pre-checked a stale ``status()`` then spawned the thread, so two
        # near-simultaneous callers both passed and the loser raised
        # ``RecorderBusy`` invisibly inside its daemon thread (issue #52 #4).
        job_id = self._inner._reserve(camera_id)

        def _target() -> None:
            # Wrap ``_run`` so an exception inside ffmpeg / chunk-walk does
            # NOT silently kill the daemon thread with no trace. Source
            # incident: Wi-Fi blip on a macOS dev box dropped
            # the RTSP TCP, ffmpeg exited with an error, the exception
            # propagated out of the thread and ``reconcile_recorders`` then
            # refused to respawn because the dead handle was still in
            # ``active`` (#66-shaped silent fail for the recorder lane). The
            # warning + downstream is_running() check together make this
            # self-healing on the next heartbeat.
            try:
                self._inner._run(job_id, url=url, duration_s=duration_s, camera_id=camera_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "recorder thread for camera_id=%s url=%s exited with exception: %s",
                    camera_id,
                    url[:64],
                    exc,
                )

        thread = threading.Thread(
            target=_target,
            daemon=True,
            name=f"recorder-{url[:32]}",
        )
        self._thread = thread
        thread.start()
        return "scheduled"

    def stop(self) -> None:
        self._inner.stop()

    def status(self) -> RecorderStatus:
        return self._inner.status()

    def probe(self, url: str, *, timeout: float = 10.0) -> ProbeResult:
        return self._inner.probe(url, timeout=timeout)

    def is_running(self) -> bool:
        """``True`` iff the recorder is actively producing footage.

        ``reconcile_recorders`` calls this each 30s heartbeat: a ``False``
        return triggers a respawn so a recorder that exited (ffmpeg crash,
        RTSP drop, duration_s elapsed) doesn't leave the camera silently
        offline forever. Returns ``False`` when the thread is dead OR the
        inner state machine has fallen back to ``idle`` even if the thread
        is technically alive — both shapes mean "stop producing chunks"."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return False
        return self._inner.status().state == "recording"
