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
from client_agent.snapshot import build_snapshot_grabber
from client_agent.snapshot_poller import SnapshotPoller
from client_agent.uploader import PresignedUploader
from client_agent.web import CameraSnapshotSource

logger = logging.getLogger(__name__)


# Default per-recording duration when the platform marks a camera enabled
# but doesn't specify a length. One hour matches the segment boundary in
# ``recorder.SEGMENT_SECONDS`` so a short camera reactivation doesn't
# cross a chunk boundary unnecessarily.
DEFAULT_RECORDING_DURATION_S = 3600


def _camera_to_push_dict(cam: DiscoveredCamera) -> dict[str, Any]:
    """Project a :class:`DiscoveredCamera` into the canonical
    ``POST /appliance/cameras`` payload (DD-09 gpu-exchange).

    The wire shape is ``{rtsp_url, onvif_uuid?, name?, model_info?}``.
    Vendor / model / discovery metadata land inside ``model_info`` as a
    free-form jsonb so the platform can persist them without a schema
    bump every time the discovery code learns a new field. Snapshot URL
    is included there too — the operator's preview lives in the same
    metadata blob."""
    body: dict[str, Any] = {"rtsp_url": cam.rtsp_url}
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
) -> dict[str, CameraSnapshotSource]:
    """Project the heartbeat's camera config into the snapshot resolver
    registry (issue #44, refactored out of #41).

    Skips rows missing ``id`` or ``rtsp_url`` — same defensive guard the
    closure version had. Pulls operator-facing labels from both shapes
    the platform might send: top-level ``name``/``vendor``/``model`` and
    nested ``model_info.{manufacturer,model}`` (the appliance-pushed
    shape via :func:`_camera_to_push_dict`). Either way the panel gets a
    label without a second platform round-trip."""
    out: dict[str, CameraSnapshotSource] = {}
    for cam in response.config.get("cameras", []):
        cam_id = cam.get("id")
        rtsp_url = cam.get("rtsp_url")
        if not cam_id or not rtsp_url:
            continue
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
                # Stale handle — respawn. We don't ``stop`` here because the
                # thread is already dead; stopping again would be a no-op at
                # best and an error at worst, and stop()/spawn() in the same
                # cycle would race the new ffmpeg process against the dead
                # one's status.json.
                logger.warning(
                    "reconcile_recorders: recorder for camera_id=%s exited; respawning",
                    cam_id,
                )
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
    platform_client.register(
        agent_version=version,
        host_info={"hostname": resolved_hostname},
    )

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
    # via the vendor app. The platform schema requires rtsp_url to be non-empty
    # (z.string().min(1)) and rejects the whole batch on the first empty row,
    # which previously caused every heartbeat cycle after a Tuya broadcast
    # landed to fail with 400 Bad Request. Filter those rows out here; the
    # devices stay in local discovery state and can surface in a future UI flow
    # that lets the operator paste the manual URL.
    skipped = [c for c in cameras if not c.rtsp_url]
    if skipped:
        logger.info(
            "push_cameras: skipping %d camera(s) with empty rtsp_url (needs_manual_url): %s",
            len(skipped),
            ", ".join(f"{c.ip}:{c.port} ({c.vendor})" for c in skipped),
        )
    publishable = [c for c in cameras if c.rtsp_url]
    payload = [_camera_to_push_dict(cam) for cam in publishable]
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
    # segment_seconds must match what the recorder writes (issue #27 buffer
    # mode emits 60s chunks via build_ffmpeg_cmd(buffer_mode=True)); otherwise
    # chunks_in_range's chunk_start = mtime - segment_seconds would over- or
    # under-estimate the chunk's coverage window and miss overlap matches.
    from client_agent.recorder import BUFFER_SEGMENT_SECONDS

    buffer = RollingBuffer(
        base_dir=buffer_dir,
        buffer_hours=parse_buffer_hours(environ),
        segment_seconds=BUFFER_SEGMENT_SECONDS,
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
        platform_camera_registry.update(build_camera_registry(response))

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
            make_real_tuya_scan,
            resolve_camera_credentials,
        )

        env_snapshot = dict(os.environ)
        creds_resolver = lambda ip: resolve_camera_credentials(ip, env_snapshot)  # noqa: E731

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
                rtsp_scan_fn=make_real_rtsp_scan(creds_resolver),
                # Stage 3 (issue #38): Tuya local broadcast — catches
                # Setti+/Tapo/Tuya IPCs that don't expose ONVIF and ship
                # with RTSP disabled by default. Purely passive listening,
                # no creds needed.
                tuya_scan_fn=make_real_tuya_scan(),
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

        # Active map already created above ``build_app`` (issue #44) so the
        # Managed cameras lister closure can read it. Heartbeat iterations
        # mutate this same dict via ``reconcile_recorders``.

        # Boot-time session: register, discover, push, heartbeat-once,
        # reconcile. Surfaces fatal config errors (bad token, unreachable
        # platform) early — before waitress binds and systemd treats the
        # unit as healthy.
        boot_response = run_platform_session(
            platform_client=platform_client,
            discover_fn=_discover,
            recorder_factory=_recorder_factory,
            active_recorders=active_recorders,
            environ=os.environ,
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
                    _time.sleep(30)
                    response = run_platform_session(
                        platform_client=platform_client,
                        discover_fn=_discover,
                        recorder_factory=_recorder_factory,
                        active_recorders=active_recorders,
                        environ=os.environ,
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
