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

    active: dict[str, object] = {"cam-1": "handle-cam-1"}
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
    assert active == {"cam-1": "handle-cam-1"}


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
