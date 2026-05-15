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
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from waitress import serve

from client_agent.agent import build_app
from client_agent.buffer import RollingBuffer
from client_agent.discovery import DiscoveredCamera
from client_agent.ffmpeg_trim import trim_and_concat
from client_agent.platform import HeartbeatResponse, PlatformClient
from client_agent.poller import TaskPoller
from client_agent.uploader import PresignedUploader

logger = logging.getLogger(__name__)


# Default per-recording duration when the platform marks a camera enabled
# but doesn't specify a length. One hour matches the segment boundary in
# ``recorder.SEGMENT_SECONDS`` so a short camera reactivation doesn't
# cross a chunk boundary unnecessarily.
DEFAULT_RECORDING_DURATION_S = 3600


def _camera_to_push_dict(cam: DiscoveredCamera) -> dict[str, Any]:
    """Project a :class:`DiscoveredCamera` into the platform push payload.

    The platform side stores cameras keyed by ``ip`` (per DD-09 §3.2);
    other fields are informational. Snapshot URL is included when ONVIF
    enrichment produced one — it lets the operator preview the camera in
    the platform UI before activating it."""
    return {
        "ip": cam.ip,
        "port": cam.port,
        "manufacturer": cam.vendor,
        "model": cam.model,
        "rtsp_url": cam.rtsp_url,
        "snapshot_url": cam.snapshot_url,
        "discovery_method": cam.discovery_method,
    }


def reconcile_recorders(
    config_cameras: list[dict],
    *,
    active: dict[str, Any],
    spawn: Callable[[dict], Any],
    stop: Callable[[Any], None],
) -> None:
    """Apply the heartbeat config to the live recorder set (issue #27).

    Camera approval lock semantics:

    * ``enabled=True`` and not active → ``spawn``; record the handle.
    * ``enabled=False`` (or absent from config) and active → ``stop``;
      drop the handle.
    * ``enabled=True`` and already active → no-op (steady state must be
      idempotent across heartbeats).

    The ``active`` dict is mutated in place; the appliance loop reuses
    the same dict across iterations so handle lifetime is bounded by
    the appliance process, not by a per-heartbeat scope.
    """
    desired_ids: set[str] = set()
    for cam in config_cameras:
        cam_id = cam["id"]
        if cam.get("enabled"):
            desired_ids.add(cam_id)
            if cam_id not in active:
                active[cam_id] = spawn(cam)

    # Anything currently active that the heartbeat doesn't list as
    # ``enabled=True`` (either ``enabled=False`` or removed) must stop.
    for cam_id in list(active.keys()):
        if cam_id not in desired_ids:
            stop(active.pop(cam_id))


def run_platform_session(
    *,
    platform_client: PlatformClient,
    discover_fn: Callable[[], list[DiscoveredCamera]],
    recorder_factory: Callable[[], Any],
    environ: Mapping[str, str],
    active_recorders: dict[str, Any] | None = None,
    hostname: str | None = None,
    version: str = "0.5.0",
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

    Each spawn calls ``recorder.start(url=..., duration_s=..., camera_id=...)``
    where ``camera_id`` puts the recorder into buffer mode (issue #27):
    chunks land in the per-camera rolling buffer for the task poller to
    pick up, not in R2."""
    resolved_hostname = hostname or environ.get("HOSTNAME") or "cctv-appliance"
    platform_client.register(hostname=resolved_hostname, version=version)

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

    payload = [_camera_to_push_dict(cam) for cam in cameras]
    platform_client.push_cameras(payload)

    if active_recorders is None:
        active_recorders = {}

    response = platform_client.heartbeat(
        status={},
        recording_cameras=list(active_recorders.keys()),
    )

    def _spawn(cam: dict) -> Any:
        rec = recorder_factory()
        rec.start(
            url=cam["rtsp_url"],
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
    )

    return response


def load_env_files(env_dir: Path, environ: MutableMapping[str, str]) -> None:
    """Read ``cameras.env`` and ``r2.env`` from ``env_dir`` into ``environ``.

    Format follows systemd's ``EnvironmentFile=``: ``KEY=VALUE`` per line.
    """
    for name in ("cameras.env", "r2.env"):
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


def start_poller_thread(
    *,
    platform_client: PlatformClient,
    buffer_dir: Path,
    trim_output_dir: Path,
    environ: Mapping[str, str],
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
    """
    buffer = RollingBuffer(
        base_dir=buffer_dir,
        buffer_hours=parse_buffer_hours(environ),
    )
    uploader = PresignedUploader(platform=platform_client)
    poller = TaskPoller(
        platform=platform_client,
        buffer=buffer,
        trim_fn=trim_and_concat,
        output_dir=trim_output_dir,
        uploader=uploader,
    )
    thread = threading.Thread(
        target=poller.run,
        name="cctv-task-poller",
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
    built = build_app(os.environ, recordings_root=recordings_root)

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
            make_real_rtsp_scan,
            resolve_camera_credentials,
        )

        env_snapshot = dict(os.environ)
        creds_resolver = lambda ip: resolve_camera_credentials(ip, env_snapshot)  # noqa: E731

        def _discover() -> list[DiscoveredCamera]:
            return discover_cameras(
                credentials_resolver=creds_resolver,
                rtsp_scan_fn=make_real_rtsp_scan(creds_resolver),
            )

        # Buffer-mode recorder factory (issue #27): each platform-approved
        # camera gets its own BackgroundRecorder writing into the per-camera
        # rolling buffer at ``BUFFER_DIR/{camera_id}/chunk_NNN.mp4``. The
        # uploader is None — buffer mode skips R2; the task poller picks
        # up chunks from the buffer on demand.
        import subprocess as _sp

        from client_agent.recorder import BackgroundRecorder, Recorder

        buffer_dir = Path(
            os.environ.get("BUFFER_DIR") or built.recordings_root.parent / "cctv-buffer"
        )
        buffer_dir.mkdir(parents=True, exist_ok=True)

        def _recorder_factory() -> BackgroundRecorder:
            return BackgroundRecorder(
                Recorder(
                    runner=_sp.run,
                    output_dir_factory=lambda cam_id: str(buffer_dir / cam_id),
                )
            )

        # Active map lives in main()'s scope so a future heartbeat loop
        # wrapping run_platform_session in ``while True: sleep(30); ...``
        # can re-use the dict across iterations and reconcile correctly.
        active_recorders: dict[str, Any] = {}

        run_platform_session(
            platform_client=platform_client,
            discover_fn=_discover,
            recorder_factory=_recorder_factory,
            active_recorders=active_recorders,
            environ=os.environ,
        )

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
        )

    logger.info(
        "client-agent appliance starting on http://0.0.0.0:8080 "
        "(bucket=%s, recordings=%s, env_dir=%s, platform=%s)",
        built.bucket,
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
