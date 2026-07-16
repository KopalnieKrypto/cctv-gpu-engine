"""Tests for the systemd unit file shipped with the appliance package.

The unit file is what makes the appliance survive a reboot — every
operator-visible promise (`systemctl status` after reboot, journald logs,
EnvironmentFile= permissions) hinges on directives we set here. We parse
the file with configparser (strict=False so duplicate keys like multiple
``EnvironmentFile=`` entries do not raise) and assert the directives the
plan requires.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest
from systemd_unit import directive_values, parse_unit

UNIT_FILE = Path(__file__).resolve().parent.parent / "cctv-client.service"


@pytest.fixture(scope="module")
def unit() -> configparser.ConfigParser:
    return parse_unit(UNIT_FILE)


def test_unit_file_exists() -> None:
    assert UNIT_FILE.is_file(), f"missing unit file: {UNIT_FILE}"


def test_service_type_simple(unit: configparser.ConfigParser) -> None:
    assert unit["Service"]["Type"] == "simple"


def test_install_wantedby_multiuser(unit: configparser.ConfigParser) -> None:
    """Without ``[Install] WantedBy=`` the unit cannot be enabled and would
    not survive reboot — which is the headline acceptance criterion."""
    assert unit["Install"]["WantedBy"] == "multi-user.target"


def test_starts_after_network_online(unit: configparser.ConfigParser) -> None:
    """Recorder hits R2 + cameras, ONVIF discovery does multicast — both fail
    fast if the network is not yet up. ``After=`` orders the start; ``Wants=``
    pulls network-online.target into the boot graph (it is not active by
    default on minimal Ubuntu/RPi installs)."""
    assert unit["Unit"]["After"] == "network-online.target"
    assert unit["Unit"]["Wants"] == "network-online.target"


def test_restart_policy_resilient(unit: configparser.ConfigParser) -> None:
    """``on-failure`` (not ``always``) so an explicit ``systemctl stop`` stays
    stopped during operator maintenance. ``RestartSec`` keeps a tight crash
    loop from hammering the network if RTSP/R2 is flapping."""
    assert unit["Service"]["Restart"] == "on-failure"
    assert int(unit["Service"]["RestartSec"]) >= 5


def test_logs_to_journald(unit: configparser.ConfigParser) -> None:
    """journald is the default in modern systemd, but the AC explicitly calls
    it out — we make the contract explicit so a future edit cannot silently
    redirect logs to a file the operator does not know about."""
    assert unit["Service"]["StandardOutput"] == "journal"
    assert unit["Service"]["StandardError"] == "journal"


def test_runs_as_unprivileged_user(unit: configparser.ConfigParser) -> None:
    """Running as ``cctv`` (system user created by install.sh) keeps Flask,
    waitress, and ffmpeg out of root — a recorder bug or RTSP parser flaw
    cannot then escalate. Group=cctv keeps recordings dir writes consistent."""
    assert unit["Service"]["User"] == "cctv"
    assert unit["Service"]["Group"] == "cctv"


def test_environment_files_loaded_from_etc(unit: configparser.ConfigParser) -> None:
    """The env files must be referenced. Leading ``-`` makes systemd treat a
    missing file as non-fatal — the unit installs *before* the operator fills
    in real creds, so without the dash the first ``systemctl start`` after
    install would fail with ENOENT. (``r2.env`` was retired in #29 — the
    appliance no longer uses R2 credentials.)"""
    files = directive_values(UNIT_FILE, "Service", "EnvironmentFile")
    assert "-/etc/cctv-client/cameras.env" in files
    assert "-/etc/cctv-client/platform.env" in files


def test_exec_start_uses_appliance_venv(unit: configparser.ConfigParser) -> None:
    """ExecStart must launch the appliance entrypoint from the install venv.

    The install layout (install.sh) creates the venv at /opt/cctv-client and
    installs the client_agent package into it; the entrypoint module is
    client_agent.appliance (Phase 3, issue #23). systemd does not expand $PATH
    or shell aliases in ExecStart, so the path has to be absolute."""
    exec_start = unit["Service"]["ExecStart"]
    assert exec_start.startswith("/opt/cctv-client/bin/python"), exec_start
    assert "-m client_agent.appliance" in exec_start, exec_start
