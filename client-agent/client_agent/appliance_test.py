"""Tests for the standalone appliance entrypoint (issue #23).

The appliance is the non-Docker entrypoint: a CLI that reads env from disk
(``/etc/cctv-client/{cameras,r2}.env`` by default), defaults
``RECORDINGS_DIR`` to the XDG state dir, and serves the same Flask app the
Docker entrypoint serves — but through waitress (multithreaded WSGI),
not Werkzeug's dev server.

Mocks live only at boundaries: filesystem (via ``tmp_path``), ``waitress.serve``,
the R2 client. The Flask app and ``build_app`` are real — refactoring
should not break these tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ----- 1. load_env_files: KEY=VALUE pairs are written into environ -----


def test_load_env_files_reads_key_value_pairs_from_cameras_env(tmp_path: Path) -> None:
    """``cameras.env`` is the operator's camera credentials file. Format
    matches ``EnvironmentFile=`` in systemd: ``KEY=VALUE`` per line.

    The function writes parsed pairs into the supplied ``environ`` mapping
    (kept injectable so tests don't mutate ``os.environ``).
    """
    from client_agent.appliance import load_env_files

    (tmp_path / "cameras.env").write_text("RTSP_DEFAULT_USER=admin\nRTSP_DEFAULT_PASS=secret\n")
    environ: dict[str, str] = {}

    load_env_files(tmp_path, environ)

    assert environ["RTSP_DEFAULT_USER"] == "admin"
    assert environ["RTSP_DEFAULT_PASS"] == "secret"


def test_load_env_files_also_reads_r2_env(tmp_path: Path) -> None:
    """The two-file split mirrors systemd's design (one ``EnvironmentFile=``
    per concern). ``r2.env`` carries cloud credentials; ``cameras.env``
    carries on-prem RTSP credentials. The loader must read both."""
    from client_agent.appliance import load_env_files

    (tmp_path / "cameras.env").write_text("RTSP_DEFAULT_USER=admin\n")
    (tmp_path / "r2.env").write_text(
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\nR2_BUCKET=surveillance-data\n"
    )
    environ: dict[str, str] = {}

    load_env_files(tmp_path, environ)

    assert environ["RTSP_DEFAULT_USER"] == "admin"
    assert environ["R2_ENDPOINT"] == "https://acct.r2.cloudflarestorage.com"
    assert environ["R2_BUCKET"] == "surveillance-data"


def test_load_env_files_is_noop_when_env_dir_missing(tmp_path: Path) -> None:
    """Operator may not have run install.sh yet, or the directory may be
    empty on first boot. The loader must not crash — the appliance still
    starts and reads env from ``os.environ`` (e.g. systemd EnvironmentFile=
    or shell ``export``)."""
    from client_agent.appliance import load_env_files

    environ: dict[str, str] = {}
    load_env_files(tmp_path / "does-not-exist", environ)

    assert environ == {}


def test_load_env_files_does_not_overwrite_existing_environ(tmp_path: Path) -> None:
    """Mirrors systemd's ``EnvironmentFile=`` semantics: variables already
    set in the process env take precedence over the file. The Docker stack
    relies on this — env from compose must not be silently overridden by a
    stale ``r2.env`` mounted into the container."""
    from client_agent.appliance import load_env_files

    (tmp_path / "r2.env").write_text("R2_BUCKET=from-file\n")
    environ = {"R2_BUCKET": "from-process-env"}

    load_env_files(tmp_path, environ)

    assert environ["R2_BUCKET"] == "from-process-env"


def test_load_env_files_skips_blank_lines_and_comments(tmp_path: Path) -> None:
    """Operators copy-paste env files from documentation; comments (``#``)
    and blank lines must be tolerated, not parsed as ``KEY=VALUE``."""
    from client_agent.appliance import load_env_files

    (tmp_path / "cameras.env").write_text(
        "# Camera credentials\n\nRTSP_DEFAULT_USER=admin\n  \n# trailing comment\n"
    )
    environ: dict[str, str] = {}

    load_env_files(tmp_path, environ)

    assert environ == {"RTSP_DEFAULT_USER": "admin"}


# ----- 2. default_recordings_dir: XDG state dir -----


def test_default_recordings_dir_uses_xdg_state_home_when_set() -> None:
    """``$XDG_STATE_HOME/cctv-client/recordings`` per the XDG Base Directory
    spec — recordings are persistent state, not cache, so they belong in
    state-dir not cache-dir. Operator can override with ``RECORDINGS_DIR``
    in env (handled later, not here)."""
    from client_agent.appliance import default_recordings_dir

    environ = {"XDG_STATE_HOME": "/var/lib/xdg-state", "HOME": "/home/operator"}

    result = default_recordings_dir(environ)

    assert result == Path("/var/lib/xdg-state/cctv-client/recordings")


def test_default_recordings_dir_falls_back_to_home_local_state() -> None:
    """When ``XDG_STATE_HOME`` is unset (the typical case on a vanilla
    distro), the spec mandates ``$HOME/.local/state``."""
    from client_agent.appliance import default_recordings_dir

    environ = {"HOME": "/home/operator"}

    result = default_recordings_dir(environ)

    assert result == Path("/home/operator/.local/state/cctv-client/recordings")


# ----- 2b. BUFFER_HOURS boot validation (issue #30) -----


def test_parse_buffer_hours_defaults_to_1_when_unset() -> None:
    """1 hour is the dev/MVP default. Production overrides to 8+ in
    platform.env. Default lives in the parser so missing env doesn't crash
    appliance boot — the operator gets a working rolling buffer either way."""
    from client_agent.appliance import parse_buffer_hours

    assert parse_buffer_hours({}) == 1


def test_parse_buffer_hours_accepts_positive_integer() -> None:
    """Numeric env values parse cleanly. Operator on production hardware
    sets ``BUFFER_HOURS=8`` (or higher) to keep more history available
    for forensic task requests."""
    from client_agent.appliance import parse_buffer_hours

    assert parse_buffer_hours({"BUFFER_HOURS": "8"}) == 8


def test_parse_buffer_hours_rejects_non_numeric() -> None:
    """Operator typo (e.g. ``BUFFER_HOURS=eight``) must fail fast at boot,
    not 30 minutes later when the first trim cycle hits an int(...) cast.
    The error message has to name the var and the bad value so
    ``journalctl -u cctv-client`` points at the fix."""
    import pytest

    from client_agent.appliance import parse_buffer_hours

    with pytest.raises(ValueError, match="BUFFER_HOURS"):
        parse_buffer_hours({"BUFFER_HOURS": "eight"})


def test_parse_buffer_hours_rejects_zero_and_negative() -> None:
    """``BUFFER_HOURS=0`` would mean "delete every chunk immediately",
    which is logically a misconfig (no task could ever find footage).
    Negative is also a typo. Both fail at boot rather than producing an
    appliance that silently drops every chunk."""
    import pytest

    from client_agent.appliance import parse_buffer_hours

    with pytest.raises(ValueError, match="BUFFER_HOURS"):
        parse_buffer_hours({"BUFFER_HOURS": "0"})
    with pytest.raises(ValueError, match="BUFFER_HOURS"):
        parse_buffer_hours({"BUFFER_HOURS": "-3"})


# ----- 3. CLI parser -----


def test_parse_args_defaults_match_systemd_install_layout() -> None:
    """No CLI args → defaults align with the install layout described in
    the plan: ``/etc/cctv-client`` for env files (matches the systemd
    ``EnvironmentFile=`` path) and foreground mode (matches
    ``Type=simple``)."""
    from client_agent.appliance import parse_args

    args = parse_args([])

    assert args.env_dir == Path("/etc/cctv-client")
    assert args.foreground is True


def test_parse_args_accepts_env_dir_override() -> None:
    """Operator running on dev box without root needs to point at a local
    env-dir."""
    from client_agent.appliance import parse_args

    args = parse_args(["--env-dir", "/home/dev/cctv-config"])

    assert args.env_dir == Path("/home/dev/cctv-config")


# ----- 4. main(): full wire-up serves on :8080 via waitress -----


def test_main_serves_built_app_on_8080_via_waitress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end wire-up of the appliance entrypoint:

    1. Reads env files from ``--env-dir``.
    2. Resolves a recordings dir (defaulting to XDG state dir; here we
       point HOME at a tmp_path so the test never writes to the user's
       real ``~/.local/state``).
    3. Builds the Flask app via the shared ``build_app`` helper — the
       same one Docker uses, so a regression that broke it would fail
       both entrypoints.
    4. Hands the app to ``waitress.serve(host=0.0.0.0, port=8080)``.

    We mock ``waitress.serve`` (boundary — would otherwise bind a real
    socket) and ``R2Client`` (boundary — would otherwise hit the network
    inside ``build_app``).
    """
    env_dir = tmp_path / "etc-cctv-client"
    env_dir.mkdir()
    (env_dir / "r2.env").write_text(
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
        "R2_ACCESS_KEY_ID=AK-test\n"
        "R2_SECRET_ACCESS_KEY=SK-test\n"
        "R2_BUCKET=surveillance-data\n"
    )
    (env_dir / "cameras.env").write_text("RTSP_DEFAULT_USER=admin\nRTSP_DEFAULT_PASS=secret\n")

    # Point HOME at tmp_path so the XDG fallback writes the recordings
    # dir under the test sandbox, not the user's real home.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("RECORDINGS_DIR", raising=False)
    # Strip pre-existing R2_* from the test env so the test asserts the
    # appliance loaded creds *from the file*, not from inherited env.
    for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)

    fake_client = MagicMock()

    with (
        patch("client_agent.agent.R2Client", return_value=fake_client),
        patch("client_agent.appliance.serve") as serve,
    ):
        from client_agent.appliance import main

        main(["--env-dir", str(env_dir)])

    # waitress.serve called once with host/port the Dockerfile exposes.
    serve.assert_called_once()
    args, kwargs = serve.call_args
    served_app = args[0] if args else kwargs.get("app")
    # Sanity: it's a Flask app (real, not a Mock — build_app wasn't faked)
    from flask import Flask

    assert isinstance(served_app, Flask)
    assert kwargs.get("host") == "0.0.0.0"
    assert kwargs.get("port") == 8080

    # Recordings root was created idempotently under XDG state dir.
    assert (tmp_path / ".local" / "state" / "cctv-client" / "recordings").is_dir()


# ----- 5. platform integration adapter (issue #26) -----


def _make_camera(ip: str = "192.168.50.2") -> object:
    """Build a real DiscoveredCamera so the test exercises the adapter's
    real serialization path (not a MagicMock that always returns truthy)."""
    from client_agent.discovery import DiscoveredCamera

    return DiscoveredCamera(
        ip=ip,
        port=554,
        vendor="Hikvision",
        model="DS-2CD2143G2-IS",
        rtsp_url=f"rtsp://{ip}:554/Streaming/Channels/101",
    )


def test_run_platform_session_registers_pushes_and_heartbeats() -> None:
    """Single iteration of the platform loop hits all three endpoints in
    the right order. ``run_platform_session`` is the testable unit; the
    production heartbeat loop wraps this in ``while True: sleep(30); ...``
    (covered separately, not in this unit test)."""
    import httpx
    import respx

    from client_agent.appliance import run_platform_session
    from client_agent.platform import PlatformClient

    discover_fn = lambda: [_make_camera()]  # noqa: E731
    recorder_factory = lambda: MagicMock()  # noqa: E731

    with respx.mock(base_url="https://platform.example") as mock:
        register_route = mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200, json={"appliance_id": "app-1", "tenant_id": "tenant-1"}
            )
        )
        push_route = mock.post("/appliance/cameras").mock(return_value=httpx.Response(204))
        heartbeat_route = mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": {"cameras": []}})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        run_platform_session(
            platform_client=client,
            discover_fn=discover_fn,
            recorder_factory=recorder_factory,
            environ={"HOSTNAME": "cctv-mini-01"},
        )

    assert register_route.called
    assert push_route.called
    assert heartbeat_route.called


def test_run_platform_session_falls_back_to_rtsp_default_url_when_discovery_empty() -> None:
    """ONVIF multicast can fail for many reasons (managed switch dropping
    multicast, camera firmware non-compliant, appliance on a subnet that
    doesn't share L2 with the cameras). The operator can still configure
    a single fallback camera by hand: ``RTSP_DEFAULT_URL`` in the env. The
    appliance must not crash on empty discovery — it must push that single
    synthetic entry so the operator can at least record from one camera."""
    import json as _json

    import httpx
    import respx

    from client_agent.appliance import run_platform_session
    from client_agent.platform import PlatformClient

    discover_fn = lambda: []  # noqa: E731
    recorder_factory = lambda: MagicMock()  # noqa: E731
    fallback_url = "rtsp://operator-supplied-camera.local:554/stream"

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200, json={"appliance_id": "app-1", "tenant_id": "tenant-1"}
            )
        )
        push_route = mock.post("/appliance/cameras").mock(return_value=httpx.Response(204))
        mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": {"cameras": []}})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        run_platform_session(
            platform_client=client,
            discover_fn=discover_fn,
            recorder_factory=recorder_factory,
            environ={"RTSP_DEFAULT_URL": fallback_url},
        )

    body = _json.loads(push_route.calls.last.request.read())
    cameras = body["cameras"]
    assert len(cameras) == 1
    assert cameras[0]["rtsp_url"] == fallback_url


def test_run_platform_session_filters_cameras_with_empty_rtsp_url() -> None:
    """Stage 2 Unknown-vendor (#37) and Stage 3 Tuya (#38) discovery emit
    ``DiscoveredCamera(rtsp_url="", needs_manual_url=True)`` when the device
    exists on the LAN but its streaming URI is per-device. The platform
    validator rejects empty rtsp_url (``z.string().min(1)``) and 400s the
    whole batch on the first invalid item, so heartbeat cycles after a Tuya
    broadcast landed previously failed end-to-end. The appliance must filter
    those rows out before the push and surface the count via the logger."""
    import json as _json

    import httpx
    import respx

    from client_agent.appliance import run_platform_session
    from client_agent.discovery import DiscoveredCamera
    from client_agent.platform import PlatformClient

    real_cam = _make_camera()
    tuya_cam = DiscoveredCamera(
        ip="192.168.50.99",
        port=6668,
        vendor="Tuya (Setti+/Tapo/Vstarcam/…)",
        model="abcd1234",
        rtsp_url="",
        discovery_method="tuya-local",
        needs_manual_url=True,
    )
    unknown_cam = DiscoveredCamera(
        ip="192.168.50.42",
        port=554,
        vendor="Unknown (nginx-RTSP / per-device URI)",
        model="",
        rtsp_url="",
        discovery_method="rtsp-scan",
        needs_manual_url=True,
    )
    discover_fn = lambda: [real_cam, tuya_cam, unknown_cam]  # noqa: E731
    recorder_factory = lambda: MagicMock()  # noqa: E731

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200, json={"appliance_id": "app-1", "tenant_id": "tenant-1"}
            )
        )
        push_route = mock.post("/appliance/cameras").mock(return_value=httpx.Response(204))
        mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": {"cameras": []}})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        run_platform_session(
            platform_client=client,
            discover_fn=discover_fn,
            recorder_factory=recorder_factory,
            environ={"HOSTNAME": "cctv-mini-01"},
        )

    body = _json.loads(push_route.calls.last.request.read())
    cameras = body["cameras"]
    assert len(cameras) == 1
    assert cameras[0]["rtsp_url"] == real_cam.rtsp_url
    # Belt-and-braces: no row in the payload should ever have empty rtsp_url.
    assert all(c["rtsp_url"] for c in cameras)


def test_run_platform_session_starts_recorder_for_each_enabled_camera() -> None:
    """The platform's heartbeat response is the source of truth for which
    cameras should be recording. The appliance reconciles by spawning a
    fresh recorder (via ``recorder_factory``) per ``enabled=True`` camera
    with the RTSP URL the platform provides (which may differ from what
    was discovered if the operator overrode it in the platform UI).
    ``enabled=False`` cameras must not spawn — they are present in the
    config so the appliance knows they exist (e.g. for surfacing a
    per-camera "disabled" state in logs)."""
    import httpx
    import respx

    from client_agent.appliance import run_platform_session
    from client_agent.platform import PlatformClient

    discover_fn = lambda: [_make_camera()]  # noqa: E731
    recorders_made: list[MagicMock] = []

    def factory():
        rec = MagicMock()
        recorders_made.append(rec)
        return rec

    config = {
        "cameras": [
            {
                "id": "cam-1",
                "enabled": True,
                "rtsp_url": "rtsp://192.168.50.2:554/live",
                "duration_s": 1800,
            },
            {
                "id": "cam-2",
                "enabled": False,
                "rtsp_url": "rtsp://192.168.50.3:554/live",
            },
        ]
    }

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200, json={"appliance_id": "app-1", "tenant_id": "tenant-1"}
            )
        )
        mock.post("/appliance/cameras").mock(return_value=httpx.Response(204))
        mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": config})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        run_platform_session(
            platform_client=client,
            discover_fn=discover_fn,
            recorder_factory=factory,
            environ={},
        )

    # Exactly one factory call — for the enabled camera, with the URL,
    # duration, and camera_id from the platform (camera_id puts the
    # recorder into buffer mode per issue #27).
    assert len(recorders_made) == 1
    recorders_made[0].start.assert_called_once_with(
        url="rtsp://192.168.50.2:554/live",
        duration_s=1800,
        camera_id="cam-1",
    )


# ----- 5b. reconcile-integrated wiring (#27): factory per camera + camera_id -----


def test_run_platform_session_factory_spawns_buffer_mode_recorder_per_camera() -> None:
    """The reconcile-integrated wiring (issue #27) builds a fresh recorder
    per ``enabled=True`` camera via ``recorder_factory`` and starts it in
    **buffer mode** (passing ``camera_id`` so chunks land in the per-camera
    rolling buffer, not in R2). The handle goes into ``active_recorders``
    so the next heartbeat can spot the camera as already running and skip
    the spawn (idempotency) or stop it if the operator flipped enabled to
    false."""
    import httpx
    import respx

    from client_agent.appliance import run_platform_session
    from client_agent.platform import PlatformClient

    discover_fn = lambda: [_make_camera()]  # noqa: E731
    recorders_made: list[MagicMock] = []

    def factory():
        rec = MagicMock()
        recorders_made.append(rec)
        return rec

    config = {
        "cameras": [
            {
                "id": "cam-1",
                "enabled": True,
                "rtsp_url": "rtsp://192.168.50.2:554/live",
                "duration_s": 86400,
            },
            {
                "id": "cam-2",
                "enabled": False,
                "rtsp_url": "rtsp://192.168.50.3:554/live",
            },
        ]
    }
    active_recorders: dict[str, MagicMock] = {}

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200, json={"appliance_id": "app-1", "tenant_id": "tenant-1"}
            )
        )
        mock.post("/appliance/cameras").mock(return_value=httpx.Response(204))
        mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": config})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        run_platform_session(
            platform_client=client,
            discover_fn=discover_fn,
            recorder_factory=factory,
            active_recorders=active_recorders,
            environ={},
        )

    # Exactly one factory call (for the one enabled camera).
    assert len(recorders_made) == 1
    # Start uses the URL/duration from the platform AND passes camera_id
    # so the recorder writes into the rolling buffer instead of uploading
    # to R2 (issue #27 buffer mode).
    recorders_made[0].start.assert_called_once_with(
        url="rtsp://192.168.50.2:554/live",
        duration_s=86400,
        camera_id="cam-1",
    )
    # Handle is captured so the next iteration sees the active recorder.
    assert active_recorders == {"cam-1": recorders_made[0]}


def test_run_platform_session_stops_recorder_when_camera_flips_to_disabled() -> None:
    """Second heartbeat after the operator revokes approval for cam-1:
    the recorder handle in ``active_recorders`` from the previous iteration
    gets ``.stop()`` called on it and is dropped from the active map. No
    new factory call (the camera is being torn down, not respawned)."""
    import httpx
    import respx

    from client_agent.appliance import run_platform_session
    from client_agent.platform import PlatformClient

    discover_fn = lambda: [_make_camera()]  # noqa: E731
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return MagicMock()

    existing_handle = MagicMock()
    active_recorders: dict[str, MagicMock] = {"cam-1": existing_handle}

    # Platform now says cam-1 is disabled.
    config = {
        "cameras": [
            {"id": "cam-1", "enabled": False, "rtsp_url": "rtsp://192.168.50.2:554/live"},
        ]
    }

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200, json={"appliance_id": "app-1", "tenant_id": "tenant-1"}
            )
        )
        mock.post("/appliance/cameras").mock(return_value=httpx.Response(204))
        mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": config})
        )
        client = PlatformClient(base_url="https://platform.example", token="tok")

        run_platform_session(
            platform_client=client,
            discover_fn=discover_fn,
            recorder_factory=factory,
            active_recorders=active_recorders,
            environ={},
        )

    existing_handle.stop.assert_called_once_with()
    assert active_recorders == {}
    # No new spawn — disabled means tear-down only.
    assert factory_calls == 0


# ----- 6. reconcile_recorders: camera approval lock (issue #27, Slice 1c.2) -----


def test_reconcile_recorders_spawns_new_enabled_camera() -> None:
    """Camera that is newly ``enabled=True`` in the heartbeat config and
    has no active recorder → ``spawn`` is invoked with the camera dict and
    the returned handle goes into the active map. This is the "operator
    approved a camera in the platform UI" transition."""
    from client_agent.appliance import reconcile_recorders

    active: dict[str, object] = {}
    spawn_calls: list[dict] = []
    stop_calls: list[object] = []

    def _spawn(cam: dict) -> object:
        spawn_calls.append(cam)
        return f"handle-{cam['id']}"

    reconcile_recorders(
        [
            {"id": "cam-1", "enabled": True, "rtsp_url": "rtsp://x/1"},
            {"id": "cam-2", "enabled": False, "rtsp_url": "rtsp://x/2"},
        ],
        active=active,
        spawn=_spawn,
        stop=stop_calls.append,
    )

    assert spawn_calls == [{"id": "cam-1", "enabled": True, "rtsp_url": "rtsp://x/1"}]
    assert stop_calls == []
    assert active == {"cam-1": "handle-cam-1"}


def test_reconcile_recorders_stops_previously_enabled_now_disabled() -> None:
    """Camera that was recording but is now ``enabled=False`` in the
    heartbeat config → ``stop`` is invoked with the handle and the active
    map loses the entry. This is the "operator revoked approval" path —
    the recorder thread must terminate (no zombie recording continues
    against operator intent)."""
    from client_agent.appliance import reconcile_recorders

    active: dict[str, object] = {"cam-1": "handle-cam-1"}
    spawn_calls: list[dict] = []
    stop_calls: list[object] = []

    reconcile_recorders(
        [{"id": "cam-1", "enabled": False, "rtsp_url": "rtsp://x/1"}],
        active=active,
        spawn=spawn_calls.append,  # type: ignore[arg-type]
        stop=stop_calls.append,
    )

    assert stop_calls == ["handle-cam-1"]
    assert spawn_calls == []
    assert active == {}


def test_reconcile_recorders_stops_camera_dropped_from_config() -> None:
    """Camera that vanished entirely from the config (operator deleted it)
    → also stop. The platform UI may signal revocation via either
    ``enabled=False`` *or* removing the camera; the appliance treats both
    the same way."""
    from client_agent.appliance import reconcile_recorders

    active: dict[str, object] = {"cam-1": "handle-cam-1", "cam-2": "handle-cam-2"}
    stop_calls: list[object] = []

    reconcile_recorders(
        [{"id": "cam-1", "enabled": True, "rtsp_url": "rtsp://x/1"}],
        active=active,
        spawn=lambda cam: f"handle-{cam['id']}",
        stop=stop_calls.append,
    )

    # cam-2 is gone from the config → stopped.
    assert stop_calls == ["handle-cam-2"]
    assert active == {"cam-1": "handle-cam-1"}


def test_reconcile_recorders_does_not_respawn_already_active() -> None:
    """Idempotency: a camera that is ``enabled=True`` and already has an
    active recorder must not be respawned. Without this guard every
    heartbeat would spawn a duplicate thread; with it the steady-state
    case is a no-op."""
    from client_agent.appliance import reconcile_recorders

    class AliveHandle:
        def is_running(self) -> bool:
            return True

    active: dict[str, object] = {"cam-1": AliveHandle()}
    spawn_calls: list[dict] = []
    stop_calls: list[object] = []

    reconcile_recorders(
        [{"id": "cam-1", "enabled": True, "rtsp_url": "rtsp://x/1"}],
        active=active,
        spawn=spawn_calls.append,  # type: ignore[arg-type]
        stop=stop_calls.append,
    )

    assert spawn_calls == []
    assert stop_calls == []


def test_reconcile_recorders_respawns_dead_recorder() -> None:
    """If a recorder thread silently exited (ffmpeg crash, RTSP drop on a
    Wi-Fi blip), the dead handle stays in ``active``. Without this branch
    every subsequent heartbeat would see ``cam_id in active`` and skip
    respawn — the camera would be silently offline until appliance
    restart. ``is_running()`` is the contract: ``False`` means respawn,
    ``True`` means leave alone.

    Mirrors the same self-healing pattern as the heartbeat loop's
    try/except in ``appliance.py`` and the task poller's exception guard
    in ``poller.run``. Three threads, three independent guards — none
    can take the appliance offline silently."""
    from client_agent.appliance import reconcile_recorders

    class DeadHandle:
        def is_running(self) -> bool:
            return False

    active: dict[str, object] = {"cam-1": DeadHandle()}
    spawn_calls: list[dict] = []
    stop_calls: list[object] = []

    fresh_handle = object()
    reconcile_recorders(
        [{"id": "cam-1", "enabled": True, "rtsp_url": "rtsp://x/1"}],
        active=active,
        spawn=lambda cam: spawn_calls.append(cam) or fresh_handle,  # type: ignore[arg-type,return-value]
        stop=stop_calls.append,
    )

    # The dead handle was replaced by the freshly-spawned one. We do NOT
    # call stop() on the dead handle — the thread is already gone, and
    # stop()/spawn() racing the new ffmpeg against the dead one's
    # status.json would corrupt the state machine.
    assert len(spawn_calls) == 1
    assert spawn_calls[0]["id"] == "cam-1"
    assert stop_calls == []
    assert active["cam-1"] is fresh_handle


def test_reconcile_recorders_treats_handle_without_is_running_as_alive() -> None:
    """Backwards compat: handles that don't expose ``is_running`` (test
    fakes, future variants) must be treated as alive — silent respawn
    would double-spawn ffmpeg per heartbeat against fakes."""
    from client_agent.appliance import reconcile_recorders

    active: dict[str, object] = {"cam-1": "opaque-handle"}
    spawn_calls: list[dict] = []
    stop_calls: list[object] = []

    reconcile_recorders(
        [{"id": "cam-1", "enabled": True, "rtsp_url": "rtsp://x/1"}],
        active=active,
        spawn=spawn_calls.append,  # type: ignore[arg-type]
        stop=stop_calls.append,
    )

    assert spawn_calls == []
    assert stop_calls == []
    assert active == {"cam-1": "opaque-handle"}


# ----- 5c. TaskPoller wired into appliance boot (issue #30) -----


def test_start_poller_thread_returns_started_daemon_thread(tmp_path: Path) -> None:
    """The poller runs in the background so waitress can still serve the
    Flask UI on the main thread. ``daemon=True`` is non-negotiable —
    without it the appliance can't exit on SIGTERM (systemd would have to
    SIGKILL it after the stop timeout), and an ``--update`` flow would
    leak a poller thread per redeploy.

    :class:`TaskPoller` is patched so the daemon thread returns
    immediately — we want to assert the *threading* contract, not drive
    the loop here (the loop is covered in :file:`poller_test.py`).
    """
    from unittest.mock import MagicMock

    from client_agent.appliance import start_poller_thread

    fake_poller = MagicMock()
    fake_poller.run = MagicMock(return_value=None)
    with patch("client_agent.appliance.TaskPoller", return_value=fake_poller):
        thread = start_poller_thread(
            platform_client=MagicMock(),
            buffer_dir=tmp_path / "buf",
            trim_output_dir=tmp_path / "trim",
            environ={"BUFFER_HOURS": "1"},
        )

    thread.join(timeout=2)
    assert thread.daemon is True
    fake_poller.run.assert_called_once_with()


def test_start_poller_thread_uses_buffer_hours_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``BUFFER_HOURS`` from env flows into the :class:`RollingBuffer`
    the poller queries. A wrong wiring (e.g. hardcoded 1) would silently
    lose every hour past the first on a production 8-hour appliance —
    silent because the trim-failure path looks the same to the operator
    as a not-yet-recorded camera."""
    from unittest.mock import MagicMock, patch

    from client_agent.appliance import start_poller_thread

    captured: dict[str, object] = {}

    def fake_rolling_buffer(*, base_dir, buffer_hours, **kw):
        captured["buffer_hours"] = buffer_hours
        captured["base_dir"] = base_dir
        return MagicMock()

    fake_poller = MagicMock()
    fake_poller.run = MagicMock(return_value=None)
    with (
        patch("client_agent.appliance.RollingBuffer", side_effect=fake_rolling_buffer),
        patch("client_agent.appliance.TaskPoller", return_value=fake_poller),
    ):
        thread = start_poller_thread(
            platform_client=MagicMock(),
            buffer_dir=tmp_path / "buf",
            trim_output_dir=tmp_path / "trim",
            environ={"BUFFER_HOURS": "8"},
        )

    thread.join(timeout=2)
    assert captured["buffer_hours"] == 8
    assert captured["base_dir"] == tmp_path / "buf"
    assert thread.daemon is True


def test_main_starts_poller_thread_in_platform_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end wiring guard: when platform.env is present, ``main()``
    calls ``start_poller_thread`` (or an equivalent helper) after the
    first ``run_platform_session``. Without this, the appliance would
    register + heartbeat once but never actually consume any task from
    the platform queue — Phase 4 demo blocker."""
    import httpx
    import respx

    from client_agent.appliance import main

    env_dir = tmp_path / "etc-cctv-client"
    env_dir.mkdir()
    (env_dir / "r2.env").write_text(
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
        "R2_ACCESS_KEY_ID=AK-test\n"
        "R2_SECRET_ACCESS_KEY=SK-test\n"
        "R2_BUCKET=surveillance-data\n"
    )
    (env_dir / "platform.env").write_text(
        "PLATFORM_URL=https://platform.example\nAPPLIANCE_TOKEN=tok\nBUFFER_HOURS=1\n"
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    for k in (
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "PLATFORM_URL",
        "APPLIANCE_TOKEN",
        "BUFFER_HOURS",
    ):
        monkeypatch.delenv(k, raising=False)

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200, json={"appliance_id": "app-1", "tenant_id": "tenant-1"}
            )
        )
        mock.post("/appliance/cameras").mock(return_value=httpx.Response(204))
        mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": {"cameras": []}})
        )

        with (
            patch("client_agent.agent.R2Client", return_value=MagicMock()),
            patch("client_agent.appliance.serve"),
            patch("client_agent.appliance.start_poller_thread") as fake_start,
            patch("client_agent.appliance.start_maintenance_thread"),
        ):
            main(["--env-dir", str(env_dir)])

    fake_start.assert_called_once()
    call_kwargs = fake_start.call_args.kwargs
    # buffer_dir defaults to <recordings_parent>/cctv-buffer; trim_output_dir
    # lives under the same parent so the poller can write trimmed mp4s
    # adjacent to the buffer without crossing fs boundaries (would slow
    # the rename + slow the upload).
    assert "platform_client" in call_kwargs
    assert "buffer_dir" in call_kwargs
    assert "trim_output_dir" in call_kwargs


def test_main_does_not_start_poller_thread_in_legacy_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-fallback: without platform.env the appliance runs the legacy
    Phase 1-3 flow and the poller stays asleep. A regression that started
    the poller anyway would crash on first ``fetch_next_task`` (no
    PlatformClient configured)."""
    env_dir = tmp_path / "etc-cctv-client"
    env_dir.mkdir()
    (env_dir / "r2.env").write_text(
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
        "R2_ACCESS_KEY_ID=AK-test\n"
        "R2_SECRET_ACCESS_KEY=SK-test\n"
        "R2_BUCKET=surveillance-data\n"
    )
    # No platform.env — legacy mode.

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    for k in (
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "PLATFORM_URL",
        "APPLIANCE_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)

    with (
        patch("client_agent.agent.R2Client", return_value=MagicMock()),
        patch("client_agent.appliance.serve"),
        patch("client_agent.appliance.start_poller_thread") as fake_start,
    ):
        from client_agent.appliance import main

        main(["--env-dir", str(env_dir)])

    fake_start.assert_not_called()


def test_main_exits_nonzero_on_buffer_hours_misconfig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``BUFFER_HOURS=eight`` in platform.env must abort boot with a
    non-zero exit — systemd's ``Restart=on-failure`` then surfaces the
    crash to the operator via ``systemctl status``. Booting with a bad
    value and silently defaulting to 1 would mask the typo until the
    operator wonders why their 8-hour retention vanished."""
    env_dir = tmp_path / "etc-cctv-client"
    env_dir.mkdir()
    (env_dir / "r2.env").write_text(
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
        "R2_ACCESS_KEY_ID=AK-test\n"
        "R2_SECRET_ACCESS_KEY=SK-test\n"
        "R2_BUCKET=surveillance-data\n"
    )
    (env_dir / "platform.env").write_text(
        "PLATFORM_URL=https://platform.example\nAPPLIANCE_TOKEN=tok\nBUFFER_HOURS=eight\n"
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    for k in (
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "PLATFORM_URL",
        "APPLIANCE_TOKEN",
        "BUFFER_HOURS",
    ):
        monkeypatch.delenv(k, raising=False)

    with (
        patch("client_agent.agent.R2Client", return_value=MagicMock()),
        patch("client_agent.appliance.serve"),
        pytest.raises((SystemExit, ValueError)) as exc_info,
    ):
        from client_agent.appliance import main

        main(["--env-dir", str(env_dir)])

    if isinstance(exc_info.value, SystemExit):
        assert exc_info.value.code != 0
    else:
        assert "BUFFER_HOURS" in str(exc_info.value)


def test_main_exits_nonzero_on_platform_auth_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad APPLIANCE_TOKEN must surface as a non-zero exit so systemd
    flags the unit as failed (and the operator gets paged via
    ``OnFailure=`` if configured). Letting the process die silently or
    re-loop on a known-bad token would mask a configuration problem the
    operator must fix manually."""
    import httpx
    import respx

    from client_agent.appliance import main
    from client_agent.platform import PlatformAuthError

    env_dir = tmp_path / "etc-cctv-client"
    env_dir.mkdir()
    (env_dir / "r2.env").write_text(
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
        "R2_ACCESS_KEY_ID=AK-test\n"
        "R2_SECRET_ACCESS_KEY=SK-test\n"
        "R2_BUCKET=surveillance-data\n"
    )
    (env_dir / "platform.env").write_text(
        "PLATFORM_URL=https://platform.example\nAPPLIANCE_TOKEN=bad-token\n"
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    for k in (
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "PLATFORM_URL",
        "APPLIANCE_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(return_value=httpx.Response(401))

        with (
            patch("client_agent.agent.R2Client", return_value=MagicMock()),
            patch("client_agent.appliance.serve"),
            pytest.raises((SystemExit, PlatformAuthError)) as exc_info,
        ):
            main(["--env-dir", str(env_dir)])

    # Either SystemExit with non-zero code OR PlatformAuthError propagated
    # to the caller — both end the process with a non-zero exit. We accept
    # either to give the implementation a small amount of latitude (catch
    # and sys.exit(1), or let the exception propagate).
    if isinstance(exc_info.value, SystemExit):
        assert exc_info.value.code != 0


def test_main_exits_nonzero_on_platform_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent platform 5xx (after retries) must also surface as a
    non-zero exit so systemd's ``Restart=on-failure`` policy backs off and
    retries with delay rather than busy-looping."""
    import httpx
    import respx

    from client_agent.appliance import main
    from client_agent.platform import PlatformUnavailableError

    env_dir = tmp_path / "etc-cctv-client"
    env_dir.mkdir()
    (env_dir / "r2.env").write_text(
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
        "R2_ACCESS_KEY_ID=AK-test\n"
        "R2_SECRET_ACCESS_KEY=SK-test\n"
        "R2_BUCKET=surveillance-data\n"
    )
    (env_dir / "platform.env").write_text(
        "PLATFORM_URL=https://platform.example\nAPPLIANCE_TOKEN=tok\n"
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    for k in (
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "PLATFORM_URL",
        "APPLIANCE_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(return_value=httpx.Response(503))

        with (
            patch("client_agent.agent.R2Client", return_value=MagicMock()),
            patch("client_agent.appliance.serve"),
            patch("client_agent.platform.time.sleep"),
            pytest.raises((SystemExit, PlatformUnavailableError)) as exc_info,
        ):
            main(["--env-dir", str(env_dir)])

    if isinstance(exc_info.value, SystemExit):
        assert exc_info.value.code != 0


def test_main_skips_platform_mode_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-fallback: without PLATFORM_URL+APPLIANCE_TOKEN the appliance
    runs the legacy standalone flow (Phase 1-3). This test guards against
    a regression where introducing platform mode silently breaks
    operators who haven't onboarded onto the platform yet."""
    env_dir = tmp_path / "etc-cctv-client"
    env_dir.mkdir()
    (env_dir / "r2.env").write_text(
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
        "R2_ACCESS_KEY_ID=AK-test\n"
        "R2_SECRET_ACCESS_KEY=SK-test\n"
        "R2_BUCKET=surveillance-data\n"
    )
    # Note: no platform.env at all — the legacy standalone path.

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    for k in (
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "PLATFORM_URL",
        "APPLIANCE_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)

    with (
        patch("client_agent.agent.R2Client", return_value=MagicMock()),
        patch("client_agent.appliance.serve") as fake_serve,
        patch("client_agent.appliance.run_platform_session") as fake_session,
    ):
        from client_agent.appliance import main

        main(["--env-dir", str(env_dir)])

    fake_serve.assert_called_once()
    fake_session.assert_not_called()


# ----- Issue #44: managed cameras lister (registry × active_recorders) -----


def test_list_managed_cameras_joins_registry_with_active_recorders() -> None:
    """Issue #44: the appliance owns the join between
    ``platform_camera_registry`` (heartbeat-supplied id/name/vendor/model)
    and ``active_recorders`` (live recording state). The pure function is
    a module-level helper so the wiring is testable without spinning up
    waitress or the heartbeat loop.

    Order of the returned list mirrors registry insertion order — the
    operator sees the same row order across heartbeats unless the
    platform reshuffles the config."""
    from client_agent.appliance import list_managed_cameras
    from client_agent.recorder import RecorderStatus
    from client_agent.web import CameraSnapshotSource

    registry = {
        "cam-uuid-1": CameraSnapshotSource(
            rtsp_url="rtsp://192.168.1.10/stream",
            name="Front door",
            vendor="Hikvision",
            model="DS-2CD2042",
        ),
        "cam-uuid-2": CameraSnapshotSource(
            rtsp_url="rtsp://192.168.1.11/stream",
            name="Backyard",
            vendor="Dahua",
            model="IPC-HFW",
        ),
        "cam-uuid-3": CameraSnapshotSource(
            rtsp_url="rtsp://192.168.1.12/stream",
            name="Garage",
            vendor="",
            model="",
        ),
    }

    class _FakeRec:
        def __init__(self, state: str) -> None:
            self._s = RecorderStatus(state=state)

        def status(self) -> RecorderStatus:
            return self._s

    active = {
        "cam-uuid-1": _FakeRec("recording"),
        "cam-uuid-2": _FakeRec("failed"),
        # cam-uuid-3: not in active_recorders → recording_state == "idle"
    }

    rows = list_managed_cameras(registry, active)

    assert rows == [
        {
            "id": "cam-uuid-1",
            "name": "Front door",
            "vendor": "Hikvision",
            "model": "DS-2CD2042",
            "recording_state": "recording",
        },
        {
            "id": "cam-uuid-2",
            "name": "Backyard",
            "vendor": "Dahua",
            "model": "IPC-HFW",
            "recording_state": "failed",
        },
        {
            "id": "cam-uuid-3",
            "name": "Garage",
            "vendor": "",
            "model": "",
            "recording_state": "idle",
        },
    ]


def test_build_camera_registry_extracts_name_vendor_model_from_heartbeat() -> None:
    """Issue #44: ``_refresh_camera_registry`` now also pulls the
    operator-facing metadata (name, vendor, model) from the heartbeat
    config so the Managed cameras panel can label each row without a
    second platform round-trip.

    The platform stores ``name`` at the top level (from
    ``_camera_to_push_dict`` — ``"{vendor} {model}".strip()``) and the
    vendor/model under ``model_info.{manufacturer,model}``; the registry
    builder must accept both shapes (top-level vendor/model too, so a
    schema-bump on the platform side doesn't silently empty the panel)."""
    from client_agent.appliance import build_camera_registry
    from client_agent.platform import HeartbeatResponse

    response = HeartbeatResponse(
        config={
            "cameras": [
                {
                    "id": "cam-uuid-1",
                    "rtsp_url": "rtsp://192.168.1.10/stream",
                    "snapshot_url": "http://192.168.1.10/snap.jpg",
                    "name": "Front door",
                    "model_info": {
                        "manufacturer": "Hikvision",
                        "model": "DS-2CD2042",
                    },
                },
                {
                    "id": "cam-uuid-2",
                    "rtsp_url": "rtsp://192.168.1.11/stream",
                    # Top-level vendor/model (defensive against schema drift).
                    "vendor": "Dahua",
                    "model": "IPC-HFW",
                },
                # Rejected: missing rtsp_url (matches existing skip logic).
                {"id": "cam-uuid-3"},
                # Rejected: missing id.
                {"rtsp_url": "rtsp://x/y"},
            ]
        }
    )

    registry = build_camera_registry(response)

    assert set(registry.keys()) == {"cam-uuid-1", "cam-uuid-2"}
    assert registry["cam-uuid-1"].rtsp_url == "rtsp://192.168.1.10/stream"
    assert registry["cam-uuid-1"].snapshot_url == "http://192.168.1.10/snap.jpg"
    assert registry["cam-uuid-1"].name == "Front door"
    assert registry["cam-uuid-1"].vendor == "Hikvision"
    assert registry["cam-uuid-1"].model == "DS-2CD2042"
    assert registry["cam-uuid-2"].vendor == "Dahua"
    assert registry["cam-uuid-2"].model == "IPC-HFW"


# ----- 5d. Buffer maintenance loop enforces BUFFER_HOURS (issue #51) -----


def test_maintenance_loop_trims_each_tick_and_survives_errors() -> None:
    """The maintenance loop is the production caller that finally makes
    ``BUFFER_HOURS`` mean something: each tick it trims every camera dir,
    then sleeps ``interval_s``. A transient filesystem error on one tick
    (recorder racing a delete, momentary EACCES) must NOT kill the daemon
    thread — otherwise trimming silently stops and the disk fills, the
    exact regression #51 is about. Drives the loop directly with an
    injected sleep that trips a sentinel, mirroring the poller's
    run()-loop resilience test."""
    from datetime import UTC, datetime

    from client_agent.appliance import _maintenance_loop

    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    trims: list[object] = []

    class Stop(Exception):
        pass

    class FlakyBuffer:
        def trim_all_cameras(self, *, now: object) -> int:
            trims.append(now)
            if len(trims) == 1:
                raise OSError("transient FS error")
            return 0

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            raise Stop()

    try:
        _maintenance_loop(
            FlakyBuffer(),
            interval_s=60,
            now_fn=lambda: now,
            sleep=fake_sleep,
        )
    except Stop:
        pass

    # First tick raised inside trim, but the loop kept going — three ticks
    # happened before the sentinel, each trimming with the injected clock
    # and sleeping the configured interval.
    assert len(trims) >= 3
    assert all(t == now for t in trims)
    assert sleeps[:3] == [60, 60, 60]


def test_start_maintenance_thread_returns_started_daemon_thread(
    tmp_path: Path,
) -> None:
    """The maintenance sweep runs on its own daemon thread so waitress and
    the poller keep running. ``daemon=True`` is non-negotiable (same reason
    as the poller thread — clean SIGTERM exit under systemd). The loop is
    patched so the thread returns immediately; we assert the *threading*
    contract here, the loop body is covered above."""
    from client_agent.appliance import start_maintenance_thread

    with patch("client_agent.appliance._maintenance_loop") as fake_loop:
        thread = start_maintenance_thread(
            buffer_dir=tmp_path / "buf",
            environ={"BUFFER_HOURS": "1"},
        )

    thread.join(timeout=2)
    assert thread.daemon is True
    fake_loop.assert_called_once()


def test_start_maintenance_thread_builds_buffer_with_env_hours(
    tmp_path: Path,
) -> None:
    """``BUFFER_HOURS`` from env flows into the :class:`RollingBuffer` the
    maintenance loop trims against — a hardcoded default would enforce the
    wrong retention window (e.g. trim an 8-hour appliance down to 1h)."""
    from client_agent.appliance import start_maintenance_thread

    captured: dict[str, object] = {}

    def fake_rolling_buffer(*, base_dir, buffer_hours, **kw):
        captured["buffer_hours"] = buffer_hours
        captured["base_dir"] = base_dir
        return MagicMock()

    with (
        patch("client_agent.appliance.RollingBuffer", side_effect=fake_rolling_buffer),
        patch("client_agent.appliance._maintenance_loop"),
    ):
        thread = start_maintenance_thread(
            buffer_dir=tmp_path / "buf",
            environ={"BUFFER_HOURS": "8"},
        )

    thread.join(timeout=2)
    assert captured["buffer_hours"] == 8
    assert captured["base_dir"] == tmp_path / "buf"
    assert thread.daemon is True


def test_main_starts_maintenance_thread_in_platform_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wiring guard: platform-mode ``main()`` must start the maintenance
    thread, else ``BUFFER_HOURS`` stays dead code and the box fills up
    (issue #51). The poller thread is patched too so the test doesn't
    spawn real background loops."""
    import httpx
    import respx

    from client_agent.appliance import main

    env_dir = tmp_path / "etc-cctv-client"
    env_dir.mkdir()
    (env_dir / "r2.env").write_text(
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
        "R2_ACCESS_KEY_ID=AK-test\n"
        "R2_SECRET_ACCESS_KEY=SK-test\n"
        "R2_BUCKET=surveillance-data\n"
    )
    (env_dir / "platform.env").write_text(
        "PLATFORM_URL=https://platform.example\nAPPLIANCE_TOKEN=tok\nBUFFER_HOURS=1\n"
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    for k in (
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "PLATFORM_URL",
        "APPLIANCE_TOKEN",
        "BUFFER_HOURS",
    ):
        monkeypatch.delenv(k, raising=False)

    with respx.mock(base_url="https://platform.example") as mock:
        mock.post("/appliance/register").mock(
            return_value=httpx.Response(
                200, json={"appliance_id": "app-1", "tenant_id": "tenant-1"}
            )
        )
        mock.post("/appliance/cameras").mock(return_value=httpx.Response(204))
        mock.post("/appliance/heartbeat").mock(
            return_value=httpx.Response(200, json={"config": {"cameras": []}})
        )

        with (
            patch("client_agent.agent.R2Client", return_value=MagicMock()),
            patch("client_agent.appliance.serve"),
            patch("client_agent.appliance.start_poller_thread"),
            patch("client_agent.appliance.start_maintenance_thread") as fake_maint,
        ):
            main(["--env-dir", str(env_dir)])

    fake_maint.assert_called_once()
    call_kwargs = fake_maint.call_args.kwargs
    assert "buffer_dir" in call_kwargs
    assert "environ" in call_kwargs


def test_main_does_not_start_maintenance_thread_in_legacy_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without platform.env the appliance runs the legacy flow — no rolling
    buffer, so no maintenance thread. Starting it anyway would trim a
    directory that legacy mode never populates (harmless) but signals a
    mode-detection regression."""
    env_dir = tmp_path / "etc-cctv-client"
    env_dir.mkdir()
    (env_dir / "r2.env").write_text(
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
        "R2_ACCESS_KEY_ID=AK-test\n"
        "R2_SECRET_ACCESS_KEY=SK-test\n"
        "R2_BUCKET=surveillance-data\n"
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    for k in (
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "PLATFORM_URL",
        "APPLIANCE_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)

    with (
        patch("client_agent.agent.R2Client", return_value=MagicMock()),
        patch("client_agent.appliance.serve"),
        patch("client_agent.appliance.start_maintenance_thread") as fake_maint,
    ):
        from client_agent.appliance import main

        main(["--env-dir", str(env_dir)])

    fake_maint.assert_not_called()
