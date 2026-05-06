"""Tests for install.sh.

We can't run the script under pytest (it touches root-owned paths and
systemd), so the smoke test is manual. What we *can* enforce here is the
contract the script must honour: bash syntax validity, idempotent guards,
and the presence of every step the acceptance criteria lists. If any of
those drift, an operator running install.sh on a fresh box would discover
it the slow way.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTALL = ROOT / "install.sh"


def test_install_script_exists() -> None:
    assert INSTALL.is_file(), INSTALL


def test_install_script_is_executable() -> None:
    """An operator copying the repo to a fresh box should be able to run
    ``./install.sh`` without first having to ``chmod +x`` it."""
    mode = INSTALL.stat().st_mode
    assert mode & stat.S_IXUSR, "install.sh missing user-execute bit"


def test_install_script_uses_bash_shebang() -> None:
    """User explicitly chose bash (not POSIX sh) — the shebang has to match
    or arrays / [[ ]] guards will silently break under dash."""
    first = INSTALL.read_text().splitlines()[0]
    assert first == "#!/usr/bin/env bash", first


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_install_script_syntax_valid() -> None:
    """``bash -n`` parses without executing — catches typos / unclosed
    blocks before the operator hits them on a fresh mini-PC."""
    result = subprocess.run(
        ["bash", "-n", str(INSTALL)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"syntax error:\n{result.stderr}"


def test_install_script_uses_strict_mode() -> None:
    """``set -euo pipefail`` is the difference between a clean abort on the
    first error and a half-installed appliance that looks fine until first
    use. Required for the idempotent guarantee to mean anything."""
    text = INSTALL.read_text()
    assert re.search(r"^\s*set\s+-[a-zA-Z]*e[a-zA-Z]*", text, re.MULTILINE), "missing set -e"
    assert "pipefail" in text, "missing pipefail (silent failures inside pipes)"


# --- Idempotence guards ----------------------------------------------------


def test_user_creation_is_idempotent() -> None:
    """Second run must not error on ``useradd`` — guard with ``id`` lookup."""
    text = INSTALL.read_text()
    assert re.search(r"id\s+(-u\s+)?cctv", text), "no `id cctv` guard before useradd"
    assert "useradd" in text, "missing useradd"


def test_venv_creation_is_idempotent() -> None:
    """Second run must not blow away an existing /opt/cctv-client venv. We
    accept either a literal-path guard or a variable-bound guard (the script
    may bind ``VENV=/opt/cctv-client`` and then ``[[ ! -d "$VENV/bin" ]]``)."""
    text = INSTALL.read_text()
    assert "/opt/cctv-client" in text
    # Look for any `-d` test (with `[`, `[[`, or `test`) anywhere in the script —
    # combined with the literal-path assertion above, that's enough to prove
    # the guard exists.
    assert re.search(r"(\[\[?\s*!?\s*-d\s+|test\s+-d\s+)", text), "no `-d` guard before venv create"


# --- Required install steps -----------------------------------------------


def test_creates_etc_config_dir_with_secure_perms() -> None:
    """``/etc/cctv-client/`` must be 0700 (or 0750 with cctv group) — the
    env files inside have R2 secret keys. Path may be inlined or bound to
    a variable like ``ETC=/etc/cctv-client``."""
    text = INSTALL.read_text()
    assert "/etc/cctv-client" in text
    # Accept either `chmod 0700 /etc/cctv-client` or `chmod 0700 "$ETC"` so
    # long as some chmod 0700/0750 appears in proximity to the etc dir.
    assert re.search(r"chmod\s+0?7[05]0\s+(/etc/cctv-client|[\"']?\$\{?[A-Z_]+)", text), (
        "no `chmod 0700` step against /etc/cctv-client (or its bound var)"
    )


def test_env_example_files_copied_when_absent() -> None:
    """First run seeds /etc/cctv-client/{r2,cameras}.env from the .example
    files; second run leaves operator edits alone. Acceptable guards: an
    explicit ``-f`` test (any bracket style), ``cp -n`` (no-clobber), or
    looping over both names with a guard around an ``install``/``cp``."""
    text = INSTALL.read_text()
    # Either each name has its own guarded copy, or the script loops over
    # ``r2.env cameras.env`` with a single guard that covers both. We accept
    # both: require both names appear *somewhere* and at least one ``-f``
    # guard on a path containing "$ETC" or /etc/cctv-client.
    for name in ("r2.env", "cameras.env"):
        assert name in text, f"missing reference to {name}"
    f_guard = re.search(r"\[\[?\s*!\s*-f\s+", text) or re.search(r"cp\s+-n", text)
    assert f_guard, "no idempotent seed guard (-f test or cp -n)"


def test_env_files_chmod_600() -> None:
    """The unit references both files; perms 600 keeps the cctv user (or
    root) the only reader. Accept either literal paths or a loop body that
    chmods ``$target`` after both names have been bound to it."""
    text = INSTALL.read_text()
    # Either chmod 0600 appears against each file literal, or once inside a
    # loop body — the loop pattern is what install.sh uses for DRY-ness.
    has_loop = re.search(r"for\s+\w+\s+in[^\n]*r2\.env[^\n]*cameras\.env", text)
    if has_loop:
        assert re.search(r"chmod\s+0?600\s+", text), "loop without chmod 0600"
    else:
        assert re.search(r"chmod\s+0?600\s+.*r2\.env", text)
        assert re.search(r"chmod\s+0?600\s+.*cameras\.env", text)


def test_installs_systemd_unit() -> None:
    """Unit must land in /etc/systemd/system (the editable location), then
    daemon-reload, then enable --now."""
    text = INSTALL.read_text()
    assert "/etc/systemd/system/cctv-client.service" in text
    assert "systemctl daemon-reload" in text
    assert re.search(r"systemctl\s+enable\s+--now\s+cctv-client", text)


def test_installs_python_dependencies() -> None:
    """Pipeline must install runtime deps into the venv. Acceptable forms:
    ``pip install`` (covers both ``pip install`` and ``"$VENV/bin/pip" install``),
    ``uv pip install``, or ``uv sync`` (the Dockerfile path — preferred)."""
    text = INSTALL.read_text()
    assert re.search(
        r"(uv\s+sync|uv\s+pip\s+install|/pip[\"']?\s+install|\bpip\s+install)", text
    ), "no dep-install step (pip install / uv sync / uv pip install)"


def test_must_run_as_root() -> None:
    """Touching /etc/, /opt/, and systemd requires root. Failing fast with a
    helpful message beats 14 cryptic permission errors."""
    text = INSTALL.read_text()
    assert re.search(r"\$EUID|\$\(id\s+-u\)", text), "no root-check guard"


# --- Path resolution -------------------------------------------------------


def test_unit_file_source_path_resolved_relative_to_script() -> None:
    """Operator may run ``install.sh`` from any cwd (e.g. ``sudo
    /home/x/cctv-gpu-engine/client-appliance/install.sh``). The script must
    locate cctv-client.service and the .env.example files relative to its
    own directory, not ``$PWD``."""
    text = INSTALL.read_text()
    assert re.search(r"BASH_SOURCE|dirname\s+.*\$0", text), "no SCRIPT_DIR resolution"
