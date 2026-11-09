"""Tests for harden-appliance.sh - the root step that closes a new cameraboy
to the client LAN.

Source incident (2026-11-09, cameraboy): ufw inactive and sshd taking
passwords, so the unauthenticated panel on 0.0.0.0:8080 and a password SSH
login were open to the whole client LAN. cameraboy2 had been closed by a
bootstrap script that lived only in a session scratchpad.

The script runs for real against stubbed ufw/ip/sshd/systemctl/loginctl/id/
getent, which log their argv, so the tests assert what would hit the box.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARDEN = ROOT / "harden-appliance.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _box(
    tmp_path: Path,
    *,
    uid: int = 0,
    tailscale: bool = True,
    keys: str = "ssh-ed25519 AAAAtest test@example\n",
    sshd_test_ok: bool = True,
) -> dict[str, str]:
    stub = tmp_path / "bin"
    stub.mkdir()
    log = tmp_path / "calls.log"
    home = tmp_path / "home" / "mvp"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "authorized_keys").write_text(keys)
    dropins = tmp_path / "sshd_config.d"
    dropins.mkdir()

    def logging_stub(name: str, body: str = "") -> None:
        (stub / name).write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> "{log}"\n{body}')

    (stub / "id").write_text(f"#!/usr/bin/env bash\necho {uid}\n")
    (stub / "getent").write_text(f'#!/usr/bin/env bash\necho "mvp:x:1000:1000::{home}:/bin/bash"\n')
    logging_stub("ip", "" if tailscale else "exit 1\n")
    logging_stub("ufw")
    logging_stub(
        "sshd",
        'if [[ "$1" == "-t" ]]; then exit ' + ("0" if sshd_test_ok else "1") + "; fi\n"
        'echo "passwordauthentication no"\n',
    )
    logging_stub("systemctl")
    logging_stub("loginctl")
    for f in stub.iterdir():
        f.chmod(0o755)
    return {
        "PATH": f"{stub}:/usr/bin:/bin",
        "SSHD_DROPIN_DIR": str(dropins),
        "LOG": str(log),
        "DROPIN": str(dropins / "10-key-only.conf"),
    }


def _harden(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(HARDEN), *(args or ("mvp",))],
        capture_output=True,
        text=True,
        env={k: env[k] for k in ("PATH", "SSHD_DROPIN_DIR")},
        check=False,
    )


def _calls(env: dict[str, str]) -> list[str]:
    log = Path(env["LOG"])
    return log.read_text().splitlines() if log.exists() else []


def test_script_is_executable_bash() -> None:
    assert HARDEN.is_file()
    assert HARDEN.stat().st_mode & 0o111
    assert subprocess.run(["bash", "-n", str(HARDEN)], check=False).returncode == 0


def test_missing_user_exits_usage(tmp_path: Path) -> None:
    env = _box(tmp_path)
    result = subprocess.run(
        [str(HARDEN)], capture_output=True, text=True, env={"PATH": env["PATH"]}, check=False
    )
    assert result.returncode == 64
    assert "usage" in result.stderr


def test_refuses_without_root(tmp_path: Path) -> None:
    env = _box(tmp_path, uid=1000)
    result = _harden(env)
    assert result.returncode != 0
    assert not any(c.startswith("ufw") for c in _calls(env))


def test_refuses_without_tailscale_and_touches_nothing(tmp_path: Path) -> None:
    """deny incoming without tailscale0 cuts the only remote way in."""
    env = _box(tmp_path, tailscale=False)
    result = _harden(env)
    assert result.returncode != 0
    assert "tailscale0" in result.stderr
    assert not any(c.startswith(("ufw", "systemctl", "loginctl")) for c in _calls(env))
    assert not Path(env["DROPIN"]).exists()


def test_refuses_without_authorized_keys_and_touches_nothing(tmp_path: Path) -> None:
    """Key-only SSH with no key locks everyone out of the box."""
    env = _box(tmp_path, keys="")
    result = _harden(env)
    assert result.returncode != 0
    assert "authorized_keys" in result.stderr
    assert not any(c.startswith(("ufw", "systemctl", "loginctl")) for c in _calls(env))
    assert not Path(env["DROPIN"]).exists()


def test_firewall_rules_match_cameraboy(tmp_path: Path) -> None:
    env = _box(tmp_path)
    result = _harden(env)
    assert result.returncode == 0, result.stderr
    ufw = [c for c in _calls(env) if c.startswith("ufw ")]
    assert "ufw default deny incoming" in ufw
    assert "ufw default allow outgoing" in ufw
    assert "ufw allow in on tailscale0" in ufw
    assert "ufw allow 41641/udp" in ufw
    for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"):
        assert f"ufw allow proto udp from {net} port 3702" in ufw
        assert f"ufw allow proto udp from {net} to any port 6666:6667" in ufw
    assert ufw.index("ufw allow in on tailscale0") < ufw.index("ufw --force enable")
    assert not any("8080" in c for c in ufw), "the panel must stay closed to the LAN"


def test_ssh_dropin_validated_before_reload(tmp_path: Path) -> None:
    env = _box(tmp_path)
    assert _harden(env).returncode == 0
    assert Path(env["DROPIN"]).read_text() == "PasswordAuthentication no\n"
    calls = _calls(env)
    reload = next(i for i, c in enumerate(calls) if c.startswith("systemctl") and "ssh" in c)
    assert calls.index("sshd -t") < reload


def test_invalid_sshd_config_removes_dropin_and_skips_reload(tmp_path: Path) -> None:
    env = _box(tmp_path, sshd_test_ok=False)
    result = _harden(env)
    assert result.returncode != 0
    assert not Path(env["DROPIN"]).exists()
    assert not any(c.startswith("systemctl") and "ssh" in c for c in _calls(env))


def test_enables_linger_for_the_appliance_user(tmp_path: Path) -> None:
    env = _box(tmp_path)
    assert _harden(env).returncode == 0
    assert "loginctl enable-linger mvp" in _calls(env)


def test_second_run_is_a_no_op_for_the_dropin(tmp_path: Path) -> None:
    env = _box(tmp_path)
    assert _harden(env).returncode == 0
    assert _harden(env).returncode == 0
    assert Path(env["DROPIN"]).read_text() == "PasswordAuthentication no\n"
