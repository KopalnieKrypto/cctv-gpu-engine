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
