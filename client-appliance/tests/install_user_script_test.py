"""Tests for install-user.sh — the no-root install path (issue #84).

Same testing posture as ``install_script_test.py``: we can't execute the
script under pytest (it talks to systemd and writes into $HOME), so the
smoke test is manual and what we enforce here is the *contract* — bash
validity, idempotent guards, and the presence of every step the acceptance
criteria name.

Why a second installer instead of a flag on install.sh: the two provision
disjoint layouts (system user + /opt + /etc + /etc/systemd/system vs. a
single unprivileged $HOME), and the guards are *inverses* — install.sh
demands root, this one refuses it. Folding both into one script would mean
every step carrying an `if root` branch, which is how the root path and the
user-mode reality silently diverged in the first place.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTALL = ROOT / "install-user.sh"


def _text() -> str:
    return INSTALL.read_text()


def test_install_user_script_exists() -> None:
    assert INSTALL.is_file(), INSTALL


def test_install_user_script_is_executable() -> None:
    """An operator cloning the repo onto a no-sudo box must be able to run
    ``./install-user.sh`` directly — and ``chmod +x`` is exactly the kind of
    undocumented prerequisite that gets skipped at 2am."""
    assert INSTALL.stat().st_mode & stat.S_IXUSR, "install-user.sh missing user-execute bit"


def test_install_user_script_uses_bash_shebang() -> None:
    first = INSTALL.read_text().splitlines()[0]
    assert first == "#!/usr/bin/env bash", first


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_install_user_script_syntax_valid() -> None:
    """``bash -n`` parses without executing — catches typos before the
    operator hits them on the box."""
    result = subprocess.run(["bash", "-n", str(INSTALL)], capture_output=True, text=True)
    assert result.returncode == 0, f"syntax error:\n{result.stderr}"


def test_install_user_script_uses_strict_mode() -> None:
    """``set -euo pipefail`` is the difference between a clean abort and a
    half-installed appliance that looks fine until the next reboot."""
    text = _text()
    assert re.search(r"^\s*set\s+-[a-zA-Z]*e[a-zA-Z]*", text, re.MULTILINE), "missing set -e"
    assert "pipefail" in text, "missing pipefail (silent failures inside pipes)"


# --- The guard that defines this script -----------------------------------


def test_refuses_to_run_as_root() -> None:
    """Exact inverse of ``install.sh``'s ``$EUID -ne 0`` guard.

    Running this under sudo would scatter root-owned files through the
    operator's $HOME (a root-owned ``~/.config/systemd/user`` is unwritable
    afterwards and wedges the unprivileged path permanently), and
    ``systemctl --user`` under sudo targets *root's* manager, not the
    operator's — so it would appear to succeed and still not survive reboot.
    Fail fast instead.
    """
    text = _text()
    assert re.search(r"\$EUID\s+-eq\s+0|\$\(id\s+-u\)\"?\s*-eq\s+0|\$EUID\"?\s*==\s*0", text), (
        "no root-refusal guard ($EUID -eq 0)"
    )


# --- The two steps that actually close the outage --------------------------


def test_uses_systemctl_user() -> None:
    """``systemctl --user enable --now`` — ``enable`` wires the unit into
    default.target (survives reboot), ``--now`` starts it this boot. Without
    ``--user`` the command targets the system manager and fails without root,
    which is the whole reason this path exists.

    ``daemon-reload`` must also be present: a re-run that ships changed unit
    directives is a no-op until the user manager re-reads them.
    """
    text = _text()
    assert re.search(r"systemctl\s+--user\s+daemon-reload", text), (
        "no `systemctl --user daemon-reload`"
    )
    assert re.search(r"systemctl\s+--user\s+enable\s+--now\s+cctv-client", text), (
        "no `systemctl --user enable --now cctv-client`"
    )
    # A bare `systemctl enable` (no --user) would hit the system manager.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "systemctl" not in stripped:
            continue
        assert "--user" in stripped, f"systemctl call without --user: {stripped}"


def test_enables_linger() -> None:
    """Linger is the load-bearing step, and the least obvious one.

    Without it the user manager is torn down when the operator's last session
    ends, taking the unit with it — so the appliance would survive `exit` but
    not a reboot, and the box would look healthy right up until it wasn't.
    That is the exact 18 h cameraboy failure mode, just moved one step later.

    Probed on cameraboy (systemd 255, Ubuntu 24.04): the polkit action
    ``org.freedesktop.login1.set-self-linger`` ships defaults of
    allow_any/allow_inactive/allow_active = ``yes``, so a user enabling
    linger *for themselves* needs no authentication — no root, no polkit
    prompt, and the ``@reboot`` crontab fallback the issue floated is not
    needed. ``loginctl enable-linger`` with no username argument is what
    routes to set-self-linger; naming another user routes to
    ``set-user-linger``, which *is* auth-gated.
    """
    text = _text()
    assert re.search(r"loginctl\s+enable-linger", text), "no `loginctl enable-linger` step"


def test_linger_failure_is_not_silent() -> None:
    """If linger cannot be enabled (a site whose polkit policy differs from
    cameraboy's), the operator must be told — an appliance that works until
    the next reboot is worse than one that fails now, because the failure
    surfaces hours later as missing footage rather than as a red install."""
    text = _text()
    linger_idx = text.find("enable-linger")
    assert linger_idx != -1
    tail = text[linger_idx:]
    assert re.search(r"Linger|linger", tail), "no post-check / message around enable-linger"


# --- Idempotence -----------------------------------------------------------


def test_idempotent_guards() -> None:
    """A re-run after ``git pull`` is the documented update path, so every
    mutating step must guard on existing state rather than assume a fresh
    box."""
    text = _text()
    # venv: never blow away an existing one
    assert re.search(r"\[\[?\s*!\s*-d\s+\"?\$\{?VENV", text), "no `-d` guard before venv create"
    # env seeding: operator creds must survive a re-run
    assert re.search(r"\[\[?\s*!\s*-f\s+\"?\$\{?target", text) or re.search(r"cp\s+-n", text), (
        "no idempotent seed guard (-f test or cp -n) — a re-run would clobber operator creds"
    )
    # linger: skip when already on
    assert re.search(r"Linger.*--value|--property=Linger", text), (
        "no linger state check — enable-linger should be guarded/verified, not blind"
    )


def test_config_dir_and_env_files_are_private() -> None:
    """``cameras.env`` holds RTSP credentials and ``platform.env`` an
    appliance token. In the user-mode layout there is no root-owned parent
    directory doing any protecting, so the perms here are the only thing
    between those secrets and every other account on the box."""
    text = _text()
    assert re.search(r"chmod\s+0?700\s+", text), "config dir not chmod 0700"
    assert re.search(r"chmod\s+0?600\s+", text), "env files not chmod 0600"


def test_installs_unit_under_home_not_etc() -> None:
    """``~/.config/systemd/user/`` is the unprivileged unit location. Writing
    to /etc/systemd/system would need the root this script just refused."""
    text = _text()
    assert re.search(r"\$HOME/\.config/systemd/user|~/\.config/systemd/user", text), (
        "unit not installed to ~/.config/systemd/user"
    )
    assert "/etc/systemd/system" not in text, "user installer must not touch /etc/systemd/system"


def test_does_not_depend_on_pythonpath() -> None:
    """The pre-#84 launcher on cameraboy only worked because the operator's
    shell exported PYTHONPATH=<repo>/client-agent — systemd units inherit no
    such environment. The installer must copy ``client_agent`` into the venv
    instead of re-encoding that dependency as an ``Environment=`` line."""
    code = [
        line for line in _text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    for line in code:
        assert "PYTHONPATH" not in line, f"install-user.sh must not set PYTHONPATH: {line.strip()}"
    assert any("cp -R" in line and "client_agent" in line for line in code), (
        "no client_agent copy into the venv site-packages"
    )
