"""Standalone appliance entrypoint for the client-agent (issue #23).

Run on bare metal (mini-PC, Raspberry Pi 5) without Docker:

    python -m client_agent.appliance --env-dir /etc/cctv-client

Reads camera + R2 credentials from disk (``cameras.env`` and ``r2.env`` in
the env-dir), serves the same Flask app the Docker entrypoint serves, but
through waitress (multithreaded WSGI) so concurrent UI requests do not
serialize through a single Werkzeug dev-server thread.

Issue #26 adds the platform-integration adapter: when ``PLATFORM_URL`` and
``APPLIANCE_TOKEN`` are present in the env, :func:`run_platform_session`
registers, pushes the discovered camera list, and runs one heartbeat
iteration (production wraps it in ``while True: sleep(30); ...``).
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from waitress import serve

from client_agent.agent import build_app
from client_agent.buffer import RollingBuffer
from client_agent.build_info import build_payload
from client_agent.discovery import (
    DiscoveredCamera,
    inject_credentials,
    resolve_camera_credentials,
)
from client_agent.ffmpeg_trim import trim_and_concat
from client_agent.platform import HeartbeatResponse, PlatformClient
from client_agent.poller import TaskPoller
from client_agent.runtime_config import RuntimeConfig
from client_agent.snapshot import build_snapshot_grabber
from client_agent.snapshot_poller import SnapshotPoller
from client_agent.telemetry import sample_disk_bytes
from client_agent.uploader import PresignedUploader
from client_agent.web import CameraSnapshotSource

logger = logging.getLogger(__name__)


# Default per-recording duration when the platform marks a camera enabled
# but doesn't specify a length. One hour matches the segment boundary in
# ``recorder.SEGMENT_SECONDS`` so a short camera reactivation doesn't
# cross a chunk boundary unnecessarily.
DEFAULT_RECORDING_DURATION_S = 3600


def authenticated_rtsp_url(url: str, environ: Mapping[str, str]) -> str:
    """Re-attach RTSP credentials to a platform-stored (credential-free) URL so
    the buffer recorder's ffmpeg can authenticate.

    The platform strips userinfo from every stored ``rtsp_url`` (issue #22), so a
    recorder started straight off ``response.config.cameras`` gets ``401`` on any
    ONVIF-discovered camera — ``GetStreamUri`` returns a bare
    ``rtsp://host:port/path`` — whose creds live in ``cameras.env``. The recorder
    then exits and respawns forever, buffering nothing. Resolve the per-host creds
    from ``environ`` and inject them. No-op when the URL already carries userinfo
    or no creds are configured for the host, so it is safe to apply to every
    camera the platform hands back.
    """
    parsed = urlparse(url)
    if parsed.username or not parsed.hostname:
        return url
    return inject_credentials(url, resolve_camera_credentials(parsed.hostname, environ))


def _camera_to_push_dict(cam: DiscoveredCamera) -> dict[str, Any]:
    """Project a :class:`DiscoveredCamera` into the canonical
    ``POST /appliance/cameras`` payload (DD-09 gpu-exchange).

    The wire shape is ``{rtsp_url, needs_manual_url, name?, model_info?}``.
    ``needs_manual_url`` is emitted as an explicit bool so the platform can
    surface URL-less devices (Tuya #38 / Unknown-vendor #37, ``rtsp_url=""``)
    on the /cameras page for the operator to fill in, instead of us dropping
    them (#70). Vendor / model / discovery metadata land inside ``model_info``
    as a free-form jsonb so the platform can persist them without a schema
    bump every time the discovery code learns a new field. Snapshot URL is
    included there too — the operator's preview lives in the same metadata
    blob."""
    body: dict[str, Any] = {
        "rtsp_url": cam.rtsp_url,
        "needs_manual_url": cam.needs_manual_url,
    }
    name = f"{cam.vendor} {cam.model}".strip()
    if name:
        body["name"] = name
    model_info: dict[str, Any] = {}
    if cam.vendor:
        model_info["manufacturer"] = cam.vendor
    if cam.model:
        model_info["model"] = cam.model
    if cam.snapshot_url:
        model_info["snapshot_url"] = cam.snapshot_url
    if cam.discovery_method:
        model_info["discovery_method"] = cam.discovery_method
    if cam.ip:
        model_info["ip"] = cam.ip
    if cam.port:
        model_info["port"] = cam.port
    if model_info:
        body["model_info"] = model_info
    return body


def build_camera_registry(
    response: HeartbeatResponse,
    environ: Mapping[str, str] | None = None,
) -> dict[str, CameraSnapshotSource]:
    """Project the heartbeat's camera config into the snapshot resolver
    registry (issue #44, refactored out of #41).

    Skips rows missing ``id`` or ``rtsp_url`` — same defensive guard the
    closure version had. Pulls operator-facing labels from both shapes
    the platform might send: top-level ``name``/``vendor``/``model`` and
    nested ``model_info.{manufacturer,model}`` (the appliance-pushed
    shape via :func:`_camera_to_push_dict`). Either way the panel gets a
    label without a second platform round-trip.

    When ``environ`` is supplied, each RTSP url is re-credentialed from
    ``cameras.env`` (the same resolver the buffer recorder uses): the platform
    stores stream urls credential-free (#22), so a bare ``rtsp://host/path``
    makes the snapshot poller's cv2/ffmpeg open 401 — leaving the /cameras
    preview a placeholder. HTTP snapshot urls are left untouched (their auth is
    the vendor endpoint's concern)."""
    out: dict[str, CameraSnapshotSource] = {}
    for cam in response.config.get("cameras", []):
        cam_id = cam.get("id")
        rtsp_url = cam.get("rtsp_url")
        if not cam_id or not rtsp_url:
            continue
        if environ is not None:
            rtsp_url = authenticated_rtsp_url(rtsp_url, environ)
        model_info = cam.get("model_info") or {}
        out[cam_id] = CameraSnapshotSource(
            rtsp_url=rtsp_url,
            snapshot_url=cam.get("snapshot_url"),
            name=cam.get("name"),
            vendor=cam.get("vendor") or model_info.get("manufacturer"),
            model=cam.get("model") or model_info.get("model"),
        )
    return out


def list_managed_cameras(
    registry: Mapping[str, CameraSnapshotSource],
    active_recorders: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join the platform-supplied camera registry with the live recorder
    handles to produce the payload for ``GET /cameras/managed`` (issue #44).

    Cameras in the registry but with no active recorder report
    ``recording_state="idle"`` — registry membership means "platform
    knows about this camera", not "we're actively recording it". The
    active map is the source of truth for ``recording``/``failed``."""
    rows: list[dict[str, Any]] = []
    for cam_id, src in registry.items():
        handle = active_recorders.get(cam_id)
        if handle is not None:
            state = handle.status().state
        else:
            state = "idle"
        rows.append(
            {
                "id": cam_id,
                "name": src.name or "",
                "vendor": src.vendor or "",
                "model": src.model or "",
                "recording_state": state,
            }
        )
    return rows


# Respawn backoff for crash-looping recorders (issue #71). The first crash
# respawns immediately (transient blip self-heals on the next heartbeat); each
# further consecutive crash opens an exponentially larger window during which a
# still-dead recorder is left alone, so a persistently-broken camera stops
# churning ffmpeg every 30s.
_RESPAWN_BACKOFF_BASE_S = 30.0
_RESPAWN_BACKOFF_CAP_S = 300.0


@dataclass
class _RecorderBackoff:
    """Per-camera respawn state carried across heartbeats in ``supervision``.

    ``failures`` counts consecutive crashes handled so far; ``next_spawn_at``
    is the monotonic deadline before which a dead recorder is not respawned."""

    failures: int = 0
    next_spawn_at: float = 0.0


def _respawn_window_s(failures: int) -> float:
    """Backoff window opened after the ``failures``-th consecutive crash.

    ``failures`` is the pre-increment count, so the first crash (0) opens a
    30s window, then 60/120/240, capped at 300s. The window gates only the
    *next* respawn — the current one already fired — so a one-off blip never
    waits while a broken camera's respawns spread further and further apart."""
    return min(_RESPAWN_BACKOFF_BASE_S * (2**failures), _RESPAWN_BACKOFF_CAP_S)


def _describe_recorder_death(handle: Any) -> str:
    """Best-effort root cause for a dead recorder handle (issue #71).

    ``Recorder._run`` stashes the ffmpeg/RTSP failure in ``status().message``
    (e.g. an RTSP DESCRIBE error) and sets ``state="failed"`` — but nothing
    read it, so the respawn log was unactionable. Pull it out here so the
    warning says *why* the recorder died. Guarded so opaque test fakes and
    handles without ``status()`` degrade to a readable placeholder instead
    of raising inside the reconcile loop."""
    status = getattr(handle, "status", None)
    if not callable(status):
        return "cause unknown (handle exposes no status)"
    try:
        snapshot = status()
    except Exception as exc:  # noqa: BLE001
        return f"cause unknown (status() raised: {exc})"
    state = getattr(snapshot, "state", "?")
    message = (getattr(snapshot, "message", "") or "").strip()
    return f"state={state}: {message}" if message else f"state={state}"


def reconcile_recorders(
    config_cameras: list[dict],
    *,
    active: dict[str, Any],
    spawn: Callable[[dict], Any],
    stop: Callable[[Any], None],
    supervision: dict[str, _RecorderBackoff] | None = None,
    now: Callable[[], float] = time.monotonic,
) -> None:
    """Apply the heartbeat config to the live recorder set (issue #27).

    Camera approval lock semantics:

    * ``enabled=True`` and not active → ``spawn``; record the handle.
    * ``enabled=True`` and active but DEAD (thread exited / state=idle) →
      ``spawn`` a replacement and overwrite the stale handle. This is the
      self-healing path: if ffmpeg crashes (RTSP drop, Wi-Fi blip on a
      macOS dev box, codec hiccup), the recorder thread can exit without
      crashing the appliance — the very next heartbeat respawns it
      instead of leaving the camera silently offline forever.
    * ``enabled=False`` (or absent from config) and active → ``stop``;
      drop the handle.
    * ``enabled=True`` and already active AND alive → no-op (steady
      state must be idempotent across heartbeats).

    The ``active`` dict is mutated in place; the appliance loop reuses
    the same dict across iterations so handle lifetime is bounded by
    the appliance process, not by a per-heartbeat scope.

    Liveness is detected via ``handle.is_running()`` when present
    (BackgroundRecorder exposes it). Handles without the method are
    treated as always-alive — preserves backwards compatibility with
    test fakes that don't model thread lifecycle.
    """
    if supervision is None:
        supervision = {}

    desired_ids: set[str] = set()
    for cam in config_cameras:
        cam_id = cam["id"]
        if cam.get("enabled"):
            desired_ids.add(cam_id)
            existing = active.get(cam_id)
            if existing is None:
                active[cam_id] = spawn(cam)
                continue
            is_running = getattr(existing, "is_running", None)
            if callable(is_running) and not is_running():
                # Stale handle — respawn, unless we're inside the crash-loop
                # backoff window (issue #71). We don't ``stop`` here because
                # the thread is already dead; stopping again would be a no-op
                # at best and an error at worst, and stop()/spawn() in the same
                # cycle would race the new ffmpeg process against the dead
                # one's status.json.
                backoff = supervision.setdefault(cam_id, _RecorderBackoff())
                now_s = now()
                if now_s < backoff.next_spawn_at:
                    # Persistently-broken camera — leave it dead until the
                    # window elapses instead of respawning every heartbeat.
                    logger.debug(
                        "reconcile_recorders: camera_id=%s dead but within "
                        "respawn backoff (%.0fs remaining); skipping",
                        cam_id,
                        backoff.next_spawn_at - now_s,
                    )
                    continue
                logger.warning(
                    "reconcile_recorders: recorder for camera_id=%s died (%s); "
                    "respawning (failure #%d)",
                    cam_id,
                    _describe_recorder_death(existing),
                    backoff.failures + 1,
                )
                active[cam_id] = spawn(cam)
                backoff.next_spawn_at = now_s + _respawn_window_s(backoff.failures)
                backoff.failures += 1
            else:
                # Alive (or an opaque handle treated as alive) — the recorder
                # is healthy, so drop any crash-loop backoff so a future death
                # starts fresh (immediate respawn) instead of inheriting the
                # penalty from an earlier outage (issue #71).
                supervision.pop(cam_id, None)

    # Anything currently active that the heartbeat doesn't list as
    # ``enabled=True`` (either ``enabled=False`` or removed) must stop.
    for cam_id in list(active.keys()):
        if cam_id not in desired_ids:
            stop(active.pop(cam_id))
            # Drop any backoff state so a re-approved camera starts fresh.
            supervision.pop(cam_id, None)


def run_platform_session(
    *,
    platform_client: PlatformClient,
    discover_fn: Callable[[], list[DiscoveredCamera]],
    recorder_factory: Callable[[], Any],
    environ: Mapping[str, str],
    active_recorders: dict[str, Any] | None = None,
    supervision: dict[str, _RecorderBackoff] | None = None,
    hostname: str | None = None,
    version: str = "0.5.0",
    runtime_config: RuntimeConfig | None = None,
    buffer: RollingBuffer | None = None,
) -> HeartbeatResponse:
    """One iteration of the platform-mode loop.

    Order:

    1. ``register`` (boot announcement; safe to repeat — platform treats
       it as upsert).
    2. ``discover`` cameras on the LAN.
    3. ``push_cameras`` the discovered list (platform tags new cameras
       with ``enabled=False``; the operator opts in via the platform UI).
    4. ``heartbeat`` once. The response carries the desired config.
    5. Reconcile recorders against the desired config: spawn one fresh
       recorder per newly-``enabled=True`` camera (via ``recorder_factory``)
       and stop any active recorder whose camera flipped to disabled or
       was removed (camera approval lock, issue #27).

    ``active_recorders`` carries the active set across heartbeat
    iterations. The caller (``main()``) owns the dict so its lifetime is
    bound to the appliance process; reconcile mutates it in place. When
    omitted (typical test fixture) a fresh empty dict is used per call
    and previously-spawned recorders cannot be stopped — fine for a
    single iteration, wrong for a multi-iteration loop.

    ``supervision`` is the sibling of ``active_recorders`` for respawn
    backoff (issue #71): it carries each camera's crash-loop state across
    heartbeats so a persistently-broken recorder backs off instead of
    respawning every tick. Same ownership contract — ``main()`` owns it for
    the process lifetime; omitting it (single-iteration test) disables
    cross-tick backoff, which is harmless for one call.

    Each spawn calls ``recorder.start(url=..., duration_s=..., camera_id=...)``
    where ``camera_id`` puts the recorder into buffer mode (issue #27):
    chunks land in the per-camera rolling buffer for the task poller to
    pick up, not in R2."""
    resolved_hostname = hostname or environ.get("HOSTNAME") or "cctv-appliance"
    # `build` carries the real identity of the installed code (git SHA, whether
    # the worktree was clean, and whether site-packages has changed since the
    # install). `agent_version` is a hand-maintained literal that tracks
    # nothing — kept only so an older platform keeps rendering something.
    register_response = platform_client.register(
        agent_version=version,
        host_info={"hostname": resolved_hostname, "build": build_payload()},
    )
    # Apply the register-delivered settings before anything else so the box is
    # correct from beat zero (issue #85), not just after the first heartbeat.
    # On-change dedup means this is a no-op when the values match the env
    # cold-start seeds. A single-iteration / legacy caller passes no
    # runtime_config and the settings are simply ignored.
    if runtime_config is not None:
        applied = runtime_config.apply(register_response.settings)
        if applied:
            logger.info("runtime config applied from register: %s", applied)

    cameras = discover_fn()
    if not cameras:
        # ONVIF empty (multicast dropped, no compliant firmware). Push a
        # synthetic entry from RTSP_DEFAULT_URL so the operator still has
        # one camera they can opt-in via the platform UI. Skipping the
        # fallback when the env var is also unset is intentional — empty
        # push is a valid signal "appliance is alive but found nothing"
        # rather than a crash.
        fallback_url = environ.get("RTSP_DEFAULT_URL")
        if fallback_url:
            cameras = [
                DiscoveredCamera(
                    ip="fallback",
                    port=554,
                    vendor="",
                    model="",
                    rtsp_url=fallback_url,
                    discovery_method="env-fallback",
                )
            ]

    # Stage 2 Unknown-vendor (#37) and Stage 3 Tuya (#38) discovery can emit a
    # DiscoveredCamera with rtsp_url="" + needs_manual_url=True — the device
    # exists on the LAN but the streaming URI is per-device and only obtainable
    # via the vendor app. The platform companion (#122) accepts these URL-less
    # rows carrying needs_manual_url, so we push the whole discovered set and
    # let the platform surface them on the operator's /cameras page (#70).
    # (Before #122 the platform's z.string().min(1) rejected the whole batch on
    # the first empty row — do not deploy this ahead of that companion.)
    manual = [c for c in cameras if not c.rtsp_url]
    if manual:
        logger.info(
            "push_cameras: surfacing %d needs_manual_url camera(s) to platform: %s",
            len(manual),
            ", ".join(f"{c.ip}:{c.port} ({c.vendor})" for c in manual),
        )
    payload = [_camera_to_push_dict(cam) for cam in cameras]
    platform_client.push_cameras(payload)

    if active_recorders is None:
        active_recorders = {}

    # Health telemetry (#92). Sampled per-iteration rather than at boot: a disk
    # that fills at 03:00 has to surface on the next beat, not the next restart
    # — which is exactly how a box ran to 17.4 % free over seven days unnoticed.
    # `buffer` is absent for legacy / single-iteration callers; the fields are
    # then omitted entirely rather than sent as zeros, which the platform would
    # band as a full disk.
    disk_free_bytes, disk_total_bytes = (
        sample_disk_bytes(buffer.base_dir) if buffer is not None else (None, None)
    )
    response = platform_client.heartbeat(
        status={},
        recording_cameras=list(active_recorders.keys()),
        agent_version=version,
        disk_free_bytes=disk_free_bytes,
        disk_total_bytes=disk_total_bytes,
        buffer_depth=buffer.buffer_depths() if buffer is not None else None,
    )
    # Every heartbeat carries the current settings block; apply on-change so an
    # admin edit in the platform UI lands on the next beat without an appliance
    # restart (issue #85).
    if runtime_config is not None:
        applied = runtime_config.apply(response.settings)
        if applied:
            logger.info("runtime config applied from heartbeat: %s", applied)

    def _spawn(cam: dict) -> Any:
        rec = recorder_factory()
        rec.start(
            url=authenticated_rtsp_url(cam["rtsp_url"], environ),
            duration_s=int(cam.get("duration_s", DEFAULT_RECORDING_DURATION_S)),
            camera_id=cam["id"],
        )
        return rec

    def _stop(handle: Any) -> None:
        handle.stop()

    reconcile_recorders(
        response.config.get("cameras", []),
        active=active_recorders,
        spawn=_spawn,
        stop=_stop,
        supervision=supervision,
    )

    return response


def load_env_files(env_dir: Path, environ: MutableMapping[str, str]) -> None:
    """Read ``cameras.env`` from ``env_dir`` into ``environ``.

    Format follows systemd's ``EnvironmentFile=``: ``KEY=VALUE`` per line.
    ``r2.env`` was retired in #29 — the appliance no longer uses R2
    credentials (uploads go through presigned URLs). ``platform.env`` is
    loaded separately by :func:`_load_platform_env`.
    """
    for name in ("cameras.env",):
        path = env_dir / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, sep, value = stripped.partition("=")
            if not sep:
                continue
            if key in environ:
                continue
            environ[key] = value


def default_recordings_dir(environ: MutableMapping[str, str]) -> Path:
    """Resolve the appliance default recordings directory per the XDG spec.

    Recordings are persistent state (kept across reboots until upload to R2
    succeeds), so XDG_STATE_HOME is the right base — not XDG_CACHE_HOME and
    not ``/tmp`` (PrivateTmp=yes in systemd would isolate it from operator
    inspection)."""
    base = environ.get("XDG_STATE_HOME") or f"{environ['HOME']}/.local/state"
    return Path(base) / "cctv-client" / "recordings"


def parse_buffer_hours(environ: Mapping[str, str]) -> int:
    """Resolve and validate ``BUFFER_HOURS`` at appliance boot (issue #30).

    The rolling buffer's retention window. Defaults to 1 (dev/MVP) so the
    appliance still boots when the operator hasn't seeded ``platform.env``.
    Bad values fail fast at boot — surfacing through systemd as a unit-
    failed state — rather than 30 min later when the first trim cycle
    casts to int and raises an opaque ``ValueError`` inside the poller
    thread."""
    raw = environ.get("BUFFER_HOURS")
    if raw is None or raw == "":
        return 1
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"BUFFER_HOURS must be a positive integer (got {raw!r})") from exc
    if value <= 0:
        raise ValueError(
            f"BUFFER_HOURS must be > 0 (got {value}); set to 1 or higher in platform.env"
        )
    return value


def parse_positive_int_env(environ: Mapping[str, str], key: str, *, default: int) -> int:
    """Resolve a positive-int env var, or ``default`` when unset/empty (#85).

    The cold-start fallback for the three runtime settings that gained env
    configurability with #85 (``POLLING_INTERVAL_SECONDS``,
    ``HEARTBEAT_INTERVAL_SECONDS``, ``UPLOAD_CHUNK_BYTES``). Mirrors
    :func:`parse_buffer_hours`: a malformed or non-positive value fails fast at
    boot — surfacing as a systemd unit failure — rather than casting to int
    inside a daemon-thread loop 30 min later where the traceback is buried in
    journald. Platform-delivered values override these once received."""
    raw = environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a positive integer (got {raw!r})") from exc
    if value <= 0:
        raise ValueError(f"{key} must be > 0 (got {value})")
    return value


def start_poller_thread(
    *,
    platform_client: PlatformClient,
    buffer_dir: Path,
    trim_output_dir: Path,
    environ: Mapping[str, str],
    runtime_config: RuntimeConfig | None = None,
) -> threading.Thread:
    """Build a :class:`TaskPoller` and run it on a daemon background thread.

    Called once at appliance boot in platform mode (after the initial
    ``run_platform_session`` has registered + heartbeated). The poller
    then drives the ``claim → trim → upload`` cycle indefinitely; the
    appliance's main thread carries on into ``waitress.serve(...)`` so
    the Flask UI keeps serving on :8080 in parallel.

    ``daemon=True`` is intentional: on SIGTERM systemd wants the process
    to exit promptly. A non-daemon thread blocking on
    ``platform.fetch_next_task`` would force a stop-timeout SIGKILL.

    When ``runtime_config`` is supplied (platform mode), the poll interval and
    the uploader's chunk size are seeded from its current values (the boot-time
    apply may already have overridden the env defaults) and their setters are
    wired so a later platform edit re-points them in place (issue #85).
    """
    buffer = _build_rolling_buffer(buffer_dir, environ)
    # In platform mode seed the poll interval + chunk size from the (already
    # boot-applied) runtime config; without it, fall back to the constructor
    # defaults so legacy / test callers are unaffected.
    uploader_kwargs: dict[str, Any] = {}
    poller_kwargs: dict[str, Any] = {}
    if runtime_config is not None:
        uploader_kwargs["upload_chunk_bytes"] = runtime_config.upload_chunk_bytes
        poller_kwargs["poll_interval_s"] = runtime_config.polling_interval_seconds
    uploader = PresignedUploader(platform=platform_client, **uploader_kwargs)
    poller = TaskPoller(
        platform=platform_client,
        buffer=buffer,
        trim_fn=trim_and_concat,
        output_dir=trim_output_dir,
        uploader=uploader,
        **poller_kwargs,
    )
    if runtime_config is not None:
        runtime_config.wire(
            set_polling_interval_seconds=poller.set_poll_interval_s,
            set_upload_chunk_bytes=uploader.set_upload_chunk_bytes,
        )
    thread = threading.Thread(
        target=poller.run,
        name="cctv-task-poller",
        daemon=True,
    )
    thread.start()
    return thread


def _build_rolling_buffer(buffer_dir: Path, environ: Mapping[str, str]) -> RollingBuffer:
    """The one :class:`RollingBuffer` the appliance uses in platform mode.

    Both the task poller (reads chunks) and the maintenance thread (trims
    chunks) go through here so they can never disagree on the two settings
    that must match: ``buffer_hours`` (retention window) and
    ``segment_seconds``. The latter must equal what the recorder writes
    (issue #27 buffer mode emits 60s chunks via
    ``build_ffmpeg_cmd(buffer_mode=True)``); otherwise ``chunks_in_range``'s
    ``chunk_start = mtime - segment_seconds`` would mis-estimate a chunk's
    coverage window and miss overlap matches."""
    from client_agent.recorder import BUFFER_SEGMENT_SECONDS

    return RollingBuffer(
        base_dir=buffer_dir,
        buffer_hours=parse_buffer_hours(environ),
        segment_seconds=BUFFER_SEGMENT_SECONDS,
    )


# How often the maintenance thread trims stale chunks. Matches the buffer
# segment length (``recorder.BUFFER_SEGMENT_SECONDS``): trimming once per
# segment bounds the overshoot to at most one extra 60s chunk per camera
# beyond ``BUFFER_HOURS``, while keeping the per-tick glob+stat cost trivial.
MAINTENANCE_INTERVAL_S = 60


def _maintenance_loop(
    buffer: RollingBuffer,
    *,
    interval_s: int,
    now_fn: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> None:
    """Trim every camera's stale chunks, sleep, repeat — forever.

    This is the production caller that finally enforces ``BUFFER_HOURS``
    (issue #51): before it, retention was dead code and a recording
    appliance filled its disk until ffmpeg failed and the camera went dark.

    A trim failure (transient FS error, recorder racing a delete) is logged
    and swallowed so the daemon thread survives — letting it propagate would
    silently stop retention and reintroduce the disk-fill bug. ``sleep`` sits
    outside the guard so a test can end the loop cleanly via a sentinel."""
    while True:
        try:
            deleted = buffer.trim_all_cameras(now=now_fn())
            if deleted:
                logger.info("buffer maintenance: trimmed %d stale chunk(s)", deleted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("buffer maintenance iteration failed: %s", exc)
        sleep(interval_s)


def start_maintenance_thread(
    *,
    buffer_dir: Path,
    environ: Mapping[str, str],
    interval_s: int = MAINTENANCE_INTERVAL_S,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
    runtime_config: RuntimeConfig | None = None,
) -> threading.Thread:
    """Build a :class:`RollingBuffer` and run :func:`_maintenance_loop` on a
    daemon thread. Called once at platform-mode boot alongside the task
    poller so retention runs continuously without operator action.

    ``daemon=True`` mirrors ``start_poller_thread`` — systemd wants a prompt
    SIGTERM exit, and a non-daemon loop thread would force a stop-timeout
    SIGKILL (and leak a thread per ``--update`` redeploy).

    When ``runtime_config`` is supplied (platform mode), the trim buffer is
    seeded to its current ``buffer_hours`` (which the boot-time apply may have
    already overridden past the env default) and ``set_buffer_hours`` is wired
    so a later admin edit re-times the trim window in place — no ffmpeg
    restart (issue #85)."""
    buffer = _build_rolling_buffer(buffer_dir, environ)
    if runtime_config is not None:
        buffer.set_buffer_hours(runtime_config.buffer_hours)
        runtime_config.wire(set_buffer_hours=buffer.set_buffer_hours)
    thread = threading.Thread(
        target=_maintenance_loop,
        args=(buffer,),
        kwargs={"interval_s": interval_s, "now_fn": now_fn, "sleep": sleep},
        name="cctv-buffer-maintenance",
        daemon=True,
    )
    thread.start()
    return thread


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the appliance.

    Defaults align with the install layout of the planned systemd unit:
    env-dir at ``/etc/cctv-client`` (matches ``EnvironmentFile=``), and
    foreground execution (matches ``Type=simple`` — systemd manages the
    daemon side; the process itself just runs in the foreground)."""
    parser = argparse.ArgumentParser(
        prog="client_agent.appliance",
        description="Standalone client-agent for bare-metal (non-Docker) operation.",
    )
    parser.add_argument(
        "--env-dir",
        type=Path,
        default=Path("/etc/cctv-client"),
        help="Directory containing cameras.env and r2.env (default: /etc/cctv-client)",
    )
    parser.add_argument(
        "--foreground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run in foreground (default; matches systemd Type=simple)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    args = parse_args(argv)
    # platform.env joins cameras.env / r2.env in the env-dir (issue #26).
    # The loader is generic but the file list is hardcoded; extend the
    # tuple in load_env_files instead of looping here.
    load_env_files(args.env_dir, os.environ)
    _load_platform_env(args.env_dir, os.environ)

    # Validate BUFFER_HOURS at boot (issue #30) — fails fast on typos
    # before network or filesystem work so systemd's Restart=on-failure
    # surfaces the misconfig to the operator instead of looping with a
    # broken trim cycle every poll interval.
    parse_buffer_hours(os.environ)

    recordings_root = default_recordings_dir(os.environ)
    # Issue #41: platform-supplied camera registry. The map is mutated by
    # ``run_platform_session`` after each heartbeat (see ``_refresh_camera_registry``
    # below) so the Flask snapshot endpoint can resolve platform-UUID
    # camera_ids to RTSP URLs without going through last_discovery
    # (which is keyed by IP and lives behind a /cameras/discover call
    # the appliance never makes in platform mode).
    platform_camera_registry: dict[str, CameraSnapshotSource] = {}
    # Issue #44: active recorder handles, joined against the registry by
    # the Managed cameras panel's lister. Created here (above ``build_app``)
    # so the lister closure captures the same dict instance the heartbeat
    # loop mutates via ``reconcile_recorders``. Empty in legacy / Docker
    # mode — the lister returns [] and the panel stays hidden.
    active_recorders: dict[str, Any] = {}
    # Sibling of ``active_recorders`` (issue #71): per-camera respawn backoff
    # state, owned here so it persists across every heartbeat tick and a
    # crash-looping camera actually backs off instead of respawning each 30s.
    recorder_supervision: dict[str, _RecorderBackoff] = {}
    built = build_app(
        os.environ,
        recordings_root=recordings_root,
        camera_resolver=platform_camera_registry.get,
        managed_cameras_lister=(
            lambda: list_managed_cameras(platform_camera_registry, active_recorders)
        )
        if _is_platform_mode(os.environ)
        else None,
    )

    def _refresh_camera_registry(response: HeartbeatResponse) -> None:
        """Replace the registry with the camera set from the latest
        heartbeat config. ``clear() + update()`` (not assignment) so the
        resolver closure keeps pointing at the same dict instance."""
        platform_camera_registry.clear()
        platform_camera_registry.update(build_camera_registry(response, os.environ))

    if _is_platform_mode(os.environ):
        # Run one platform session up-front so a bad token / unreachable
        # platform fails fast at boot — before waitress binds the port and
        # systemd treats the unit as healthy. The production heartbeat
        # loop wraps run_platform_session in ``while True: sleep(30); ...``
        # but is out of scope for #26 (deferred to the slice that adds the
        # task poller, gpu-exchange#21).
        platform_client = PlatformClient(
            base_url=os.environ["PLATFORM_URL"],
            token=os.environ["APPLIANCE_TOKEN"],
        )
        from client_agent.discovery import (
            discover_cameras,
            make_ffmpeg_path_prober,
            make_real_rtsp_scan,
            make_real_tuya_scan,
            rtsp_probe_paths,
        )

        env_snapshot = dict(os.environ)
        creds_resolver = lambda ip: resolve_camera_credentials(ip, env_snapshot)  # noqa: E731
        # Stage 2.5 (issue #74): candidate RTSP paths for probing fingerprint
        # dead-ends. Seeded from RTSP_KNOWN_URLS (operator's confirmed cameras)
        # + RTSP_PROBE_PATHS + built-in defaults. Built once from the boot env
        # snapshot; the prober forks ffmpeg to confirm each candidate opens.
        rtsp_path_prober = make_ffmpeg_path_prober(rtsp_probe_paths(env_snapshot))

        # Discovery timeout bumped to 15s (default is 5s): a /24 RTSP port
        # scan of 254 IPs × 4 ports = 1016 TCP connects needs more headroom
        # than 5s, especially on macOS dev machines where mDNS / Tailscale
        # interference makes ONVIF multicast unreliable so Stage 2 carries
        # the discovery on its own. On a bare-metal Linux mini-PC at the
        # customer Stage 1 typically returns all cameras and 15s is upper
        # bound either way (boot-time call, not hot path).
        def _discover() -> list[DiscoveredCamera]:
            return discover_cameras(
                timeout=15.0,
                credentials_resolver=creds_resolver,
                rtsp_scan_fn=make_real_rtsp_scan(creds_resolver, path_prober=rtsp_path_prober),
                # Stage 3 (issue #38): Tuya local broadcast — catches
                # Setti+/Tapo/Tuya IPCs that don't expose ONVIF and ship
                # with RTSP disabled by default. Purely passive listening,
                # no creds needed.
                tuya_scan_fn=make_real_tuya_scan(),
            )

        # Buffer-mode recorder factory (issue #27): each platform-approved
        # camera gets its own BackgroundRecorder writing into the per-camera
        # rolling buffer at ``BUFFER_DIR/{camera_id}/chunk_<UTC timestamp>.mp4``
        # (timestamp-named so hourly respawns append instead of overwriting —
        # issue #90; retention is what bounds the dir, not the filenames). The
        # uploader is None — buffer mode skips R2; the task poller picks
        # up chunks from the buffer on demand.
        import subprocess as _sp

        from client_agent.recorder import BackgroundRecorder, Recorder

        buffer_dir = Path(
            os.environ.get("BUFFER_DIR") or built.recordings_root.parent / "cctv-buffer"
        )
        buffer_dir.mkdir(parents=True, exist_ok=True)

        # Read-only view of the buffer for heartbeat telemetry (#92). A separate
        # instance from the maintenance thread's on purpose: that one owns the
        # mutable retention window (``set_buffer_hours``), while this one only
        # ever stats. ``buffer_hours`` is irrelevant here — neither
        # ``buffer_depths`` nor ``base_dir`` consults it.
        telemetry_buffer = _build_rolling_buffer(buffer_dir, os.environ)

        def _recorder_factory() -> BackgroundRecorder:
            return BackgroundRecorder(
                Recorder(
                    runner=_sp.run,
                    output_dir_factory=lambda cam_id: str(buffer_dir / cam_id),
                )
            )

        # Active map already created above ``build_app`` (issue #44) so the
        # Managed cameras lister closure can read it. Heartbeat iterations
        # mutate this same dict via ``reconcile_recorders``.

        # Runtime config (issue #85): the four platform-editable settings.
        # Seeded from env as a cold-start fallback (BUFFER_HOURS et al.); the
        # boot-time register/heartbeat apply below overrides them, then the
        # poller / maintenance threads read the post-apply values as their
        # construction seed and wire their setters so later admin edits land
        # on the next heartbeat without a restart.
        runtime_config = RuntimeConfig(
            buffer_hours=parse_buffer_hours(os.environ),
            polling_interval_seconds=parse_positive_int_env(
                os.environ, "POLLING_INTERVAL_SECONDS", default=5
            ),
            heartbeat_interval_seconds=parse_positive_int_env(
                os.environ, "HEARTBEAT_INTERVAL_SECONDS", default=30
            ),
            upload_chunk_bytes=parse_positive_int_env(
                os.environ, "UPLOAD_CHUNK_BYTES", default=52_428_800
            ),
        )

        # Boot-time session: register, discover, push, heartbeat-once,
        # reconcile. Surfaces fatal config errors (bad token, unreachable
        # platform) early — before waitress binds and systemd treats the
        # unit as healthy. Passing runtime_config applies the register +
        # first-heartbeat settings so the box is correct from beat zero (#85).
        boot_response = run_platform_session(
            platform_client=platform_client,
            discover_fn=_discover,
            recorder_factory=_recorder_factory,
            active_recorders=active_recorders,
            supervision=recorder_supervision,
            environ=os.environ,
            runtime_config=runtime_config,
            buffer=telemetry_buffer,
        )
        # Issue #41: seed the snapshot resolver registry from the boot
        # heartbeat config so a curl right after appliance start (before
        # the 30-s loop ticks) can resolve camera_ids.
        _refresh_camera_registry(boot_response)

        # Heartbeat loop: every 30s re-run the session so cameras the
        # operator approves via the platform UI get a recorder spawned
        # without requiring an appliance restart. Daemon thread so process
        # exit doesn't hang on this loop. Discovery is re-run each tick —
        # cheap on Linux (multicast WS-Discovery returns quickly), 15s on
        # macOS dev where ONVIF probe is silent and Stage 2 RTSP scan
        # carries it; acceptable since heartbeat cadence is 30s.
        import threading as _threading
        import time as _time

        def _heartbeat_loop() -> None:
            while True:
                try:
                    # Sleep the live interval, re-read each tick so a platform
                    # edit to heartbeat_interval_seconds re-times the cadence
                    # without a restart (issue #85).
                    _time.sleep(runtime_config.heartbeat_interval_seconds)
                    response = run_platform_session(
                        platform_client=platform_client,
                        discover_fn=_discover,
                        recorder_factory=_recorder_factory,
                        active_recorders=active_recorders,
                        supervision=recorder_supervision,
                        environ=os.environ,
                        runtime_config=runtime_config,
                        buffer=telemetry_buffer,
                    )
                    # Keep the snapshot resolver registry in lockstep with
                    # the heartbeat config — a camera the operator just
                    # enabled in the platform UI starts answering snapshot
                    # requests on the very next tick (issue #41).
                    _refresh_camera_registry(response)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("heartbeat iteration failed: %s", exc)

        _threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat-loop").start()

        # Spawn the task poller (issue #30). Drives claim → trim → upload
        # indefinitely on a daemon thread; main() continues into waitress
        # so the Flask UI on :8080 keeps serving concurrently. The trim
        # output dir lives next to the buffer to keep rename(2) on one fs.
        trim_output_dir = buffer_dir.parent / "cctv-trim"
        trim_output_dir.mkdir(parents=True, exist_ok=True)
        start_poller_thread(
            platform_client=platform_client,
            buffer_dir=buffer_dir,
            trim_output_dir=trim_output_dir,
            environ=os.environ,
            runtime_config=runtime_config,
        )

        # Enforce BUFFER_HOURS (issue #51). Without this daemon the rolling
        # buffer only ever grows: recorders write 60s chunks continuously and
        # nothing deletes the old ones, so the appliance disk fills in days.
        start_maintenance_thread(
            buffer_dir=buffer_dir,
            environ=os.environ,
            runtime_config=runtime_config,
        )

        # Spawn the snapshot poller (gpu-exchange #91). Drives
        # /appliance/snapshot/next claims → JPEG grab → PUT to presigned
        # R2 URL. Shares the same camera registry the on-site Flask
        # snapshot route uses, so platform-side thumbnails come from the
        # same source the on-site managed-cameras panel does. Daemon
        # thread so process exit doesn't hang on the poll loop.
        snapshot_poller = SnapshotPoller(
            platform=platform_client,
            camera_resolver=platform_camera_registry.get,
            snapshot_grabber=build_snapshot_grabber(),
            http_put=SnapshotPoller.default_http_put,
        )
        _threading.Thread(target=snapshot_poller.run, daemon=True, name="snapshot-poller").start()

    logger.info(
        "client-agent appliance starting on http://0.0.0.0:8080 "
        "(recordings=%s, env_dir=%s, platform=%s)",
        built.recordings_root,
        args.env_dir,
        os.environ.get("PLATFORM_URL", "off"),
    )
    serve(built.app, host="0.0.0.0", port=8080)


def _is_platform_mode(environ: Mapping[str, str]) -> bool:
    """Platform mode is opt-in via env presence (auto-fallback semantics).

    Both ``PLATFORM_URL`` and ``APPLIANCE_TOKEN`` must be set; either
    alone is treated as a misconfiguration but tolerated (legacy
    standalone mode is the safe default while operators migrate)."""
    return bool(environ.get("PLATFORM_URL") and environ.get("APPLIANCE_TOKEN"))


def _load_platform_env(env_dir: Path, environ: MutableMapping[str, str]) -> None:
    """Read ``platform.env`` (issue #26) into ``environ``.

    Same KEY=VALUE format as ``cameras.env`` / ``r2.env``. Kept separate
    on disk because the platform token has tighter rotation semantics
    than camera or R2 credentials and lives under a distinct chmod 600
    file owned by ``cctv:cctv``."""
    path = env_dir / "platform.env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition("=")
        if not sep:
            continue
        if key in environ:
            continue
        environ[key] = value


if __name__ == "__main__":
    main()
