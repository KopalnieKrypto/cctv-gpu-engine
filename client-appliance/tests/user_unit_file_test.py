"""Tests for the *user-mode* systemd unit (issue #84).

Sites that cannot run install.sh as root (no sudo password available — see
`gpu-exchange:docs/ops/appliance-sites.md`) got no supervision at all: the
appliance was launched by a hand-rolled `nohup setsid` one-liner and was
silently gone after every reboot. That cost a ~18 h production outage on
`cameraboy` (2026-07-14 → 2026-07-16).

This unit is the root-unit's counterpart for `systemctl --user`. It is a
*separate file* rather than a variant of `cctv-client.service` because the
two differ in ways systemd enforces (no `User=`, a different `WantedBy=`),
and because the root unit's `Restart=on-failure` decision is deliberate and
tested — see `unit_file_test.py::test_restart_policy_resilient`.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest
from systemd_unit import directive_values, parse_unit

UNIT_FILE = Path(__file__).resolve().parent.parent / "cctv-client-user.service"


@pytest.fixture(scope="module")
def unit() -> configparser.ConfigParser:
    return parse_unit(UNIT_FILE)


def test_user_unit_file_exists() -> None:
    assert UNIT_FILE.is_file(), f"missing user unit file: {UNIT_FILE}"


def test_user_unit_has_no_user_directive(unit: configparser.ConfigParser) -> None:
    """A ``--user`` unit already runs as the owning user, and systemd *rejects*
    ``User=``/``Group=`` there ("Unit ... has User= set, refusing"). Copying
    the root unit's ``User=cctv`` across is the single most likely way to make
    this file fail to start, so we pin the absence explicitly."""
    service = unit["Service"] if unit.has_section("Service") else {}
    assert "User" not in service, "User= is invalid in a --user unit"
    assert "Group" not in service, "Group= is invalid in a --user unit"


def test_user_exec_start_invokes_appliance_module(unit: configparser.ConfigParser) -> None:
    """ExecStart must launch the appliance entrypoint from the user venv that
    install-user.sh provisions.

    ``%h`` is systemd's home-directory specifier — it expands to an absolute
    path before exec, which satisfies systemd's "first token must be absolute"
    rule while keeping this file static (no templating at install time). We
    accept ``%h/`` or a literal ``/`` prefix and reject a bare ``python``,
    which systemd would refuse since it does not resolve $PATH in ExecStart.
    """
    exec_start = unit["Service"]["ExecStart"]
    assert exec_start.startswith(("%h/", "/")), (
        f"ExecStart must be absolute after specifier expansion: {exec_start}"
    )
    assert "-m client_agent.appliance" in exec_start, exec_start
    assert "--env-dir" in exec_start, exec_start


def test_user_unit_restart_policy_matches_root(unit: configparser.ConfigParser) -> None:
    """Parity with the root unit is the point: ``on-failure`` (not ``always``)
    so an explicit ``systemctl --user stop`` stays stopped during operator
    maintenance, and ``RestartSec`` keeps a crash loop from hammering the
    network when RTSP is flapping. Issue #84 is explicit that the root unit's
    policy was never the bug — the unit was simply never installed — so this
    file must not "fix" a decision that was already correct."""
    assert unit["Service"]["Type"] == "simple"
    assert unit["Service"]["Restart"] == "on-failure"
    assert int(unit["Service"]["RestartSec"]) >= 5


def test_user_unit_logs_to_journald(unit: configparser.ConfigParser) -> None:
    """The user-mode site is exactly where log discipline matters most: the
    incident's hand-rolled launcher appended to ``appliance.log`` in the env
    dir, which nothing rotated (that box has ~27 MB of stale ``.pre-*`` logs).
    journald gives the operator ``journalctl --user -u cctv-client`` and
    rotation for free."""
    assert unit["Service"]["StandardOutput"] == "journal"
    assert unit["Service"]["StandardError"] == "journal"


def test_user_install_wantedby_default_target(unit: configparser.ConfigParser) -> None:
    """``default.target`` is the user manager's boot target — it has no
    ``multi-user.target`` (that lives in the system manager only). Copying the
    root unit's ``WantedBy=multi-user.target`` here would make ``systemctl
    --user enable`` a no-op on reboot, which is precisely the failure this
    issue exists to kill."""
    assert unit["Install"]["WantedBy"] == "default.target"


def test_user_environment_files_optional_prefix() -> None:
    """Every ``EnvironmentFile=`` must carry the leading ``-`` (missing file is
    non-fatal) — same rationale as the root unit: install-user.sh enables the
    unit *before* the operator has filled in real creds, so without the dash
    the first start would die on ENOENT. Asserted over *every* entry rather
    than the two we ship, so a future third env file cannot skip the dash."""
    files = directive_values(UNIT_FILE, "Service", "EnvironmentFile")
    assert files, "user unit references no EnvironmentFile at all"
    for entry in files:
        assert entry.startswith("-"), f"EnvironmentFile without '-' prefix: {entry}"
    assert any("cameras.env" in f for f in files), files
    assert any("platform.env" in f for f in files), files


def test_user_environment_files_live_under_home() -> None:
    """The whole point of the user-mode path is that it touches nothing
    root-owned. An env file under ``/etc`` would be unwritable for the
    unprivileged operator and silently fall back to defaults."""
    for entry in directive_values(UNIT_FILE, "Service", "EnvironmentFile"):
        assert entry.startswith("-%h/"), f"env file not under %h: {entry}"


def test_user_unit_does_not_order_on_system_targets() -> None:
    """The root unit orders on ``network-online.target``; this one must not.

    That target exists only in the *system* manager. Probed on cameraboy
    (systemd 255, Ubuntu 24.04)::

        $ systemctl --user show network-online.target -p LoadState
        LoadState=not-found

    A ``Wants=`` on a not-found unit is a weak dependency, so systemd would
    warn and continue rather than refuse to start — meaning this mistake
    survives review, looks correct in the file, and buys exactly nothing. The
    appliance instead tolerates a not-yet-up network by retrying (and, on a
    hard exit, ``Restart=on-failure``). Asserted against the raw text so an
    ``After=`` on any system target is caught too.
    """
    for directive in ("After", "Wants", "Requires", "BindsTo", "PartOf"):
        for value in directive_values(UNIT_FILE, "Unit", directive):
            assert "network-online" not in value, (
                f"{directive}={value} — network-online.target is not-found in "
                "the --user manager; this ordering is a no-op at best"
            )
