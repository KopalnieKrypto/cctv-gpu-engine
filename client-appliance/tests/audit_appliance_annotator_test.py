"""Checks 7-13 of audit-appliance.sh: zone-annotator, its tunnel, its shipping key and host key.

The audit stays read-only. Repair is a re-run of install-annotator.sh, so "FAIL then repair then
PASS" is a probe blob with the drift, then the same blob as the installer leaves it.

Assumptions: property lines arrive in the format ``systemctl show`` prints on systemd 255
(Ubuntu 24.04, checked 2026-11-23); a box without /opt/zone-annotator and without the unit has no
annotation and gets SKIP, which leaves the exit code alone.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "audit-appliance.sh"
PIN = "d800ae42d8ed45ee5346e13c3f3e2561001bcc34"
TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
KNOWN_HOSTS = "# cctv-vps\ncctv-vps,100.64.0.1 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAItest\n"
KNOWN_HOSTS_SHA = hashlib.sha256(KNOWN_HOSTS.encode()).hexdigest()

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

# Checks 1-6 healthy, as in audit_appliance_script_test.py.
BASE = (
    "PROC_COUNT=1\nACTIVE_STATE=active\nSUB_STATE=running\nN_RESTARTS=0\nUNIT_ENABLED=enabled\n"
    "LINGER=yes\nINSTALL_COMMIT=5b43d72\nCHECKOUT_COMMIT=5b43d72\nBUILD_MODIFIED=False\n"
    "UFW_ENABLED=yes\nSSHD_STATE=present\nSSH_PASSWORD_AUTH=no\nSSH_UNREADABLE=\n"
)

HEALTHY = {
    "ZA_COMMIT": PIN,
    "ZA_TREE": TREE,
    "ZA_LOAD": "loaded",
    "ZA_ACTIVE": "active",
    "ZA_LISTEN": "127.0.0.1:8713",
    "ZA_IPAddressDeny": "0.0.0.0/0 ::/0",
    "ZA_IPAddressAllow": "::1/128 127.0.0.0/8",
    "ZA_NoNewPrivileges": "yes",
    "ZA_ProtectSystem": "strict",
    "ZA_InaccessiblePaths": (
        "-/home/cameraboy/.config/cctv-client -/home/cameraboy/.ssh -/etc/cctv-client"
    ),
    "ZT_ACTIVE": "active",
    "ZT_TOKEN_MODE": "600",
    "ZT_TOKEN_OWNER": "root",
    "EGRESS_7844": "ok",
    "ZA_DROP_KEY_MODE": "600",
    "ZA_DROP_HOSTS": KNOWN_HOSTS_SHA,
}


def _audit(
    tmp_path: Path, blob: str, ssh: Path | None = None, tree: str = TREE, **env: str
) -> subprocess.CompletedProcess[str]:
    pin = tmp_path / "zone-annotator.pin"
    pin.write_text(f"{PIN}\n{tree}\n")
    # As a Windows checkout has it: the audit compares without carriage returns.
    known = tmp_path / "annotation-drop.known_hosts"
    known.write_bytes(KNOWN_HOSTS.replace("\n", "\r\n").encode())
    if ssh is None:
        ssh = tmp_path / "fake-ssh"
        ssh.write_text(f"#!/usr/bin/env bash\ncat <<'BLOB'\n{blob}\nBLOB\n")
        ssh.chmod(0o755)
    return subprocess.run(
        [str(AUDIT), "cameraboy"],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "SSH": str(ssh),
            "PIN_FILE": str(pin),
            "KNOWN_HOSTS_FILE": str(known),
            **env,
        },
        check=False,
        # A probe that blocks (a FIFO opened for its tree) fails here instead of hanging.
        timeout=60,
    )


def _blob(**overrides: str) -> str:
    return BASE + "\n".join(f"{k}={v}" for k, v in {**HEALTHY, **overrides}.items())


def _line(stdout: str, check: int) -> str:
    return next((ln for ln in stdout.splitlines() if f"check {check}:" in ln), "")


def test_a_healthy_annotation_box_passes_checks_7_to_13(tmp_path: Path) -> None:
    result = _audit(tmp_path, _blob())

    assert result.returncode == 0, result.stdout + result.stderr
    for check in range(7, 14):
        assert _line(result.stdout, check).startswith("PASS"), result.stdout


def test_a_box_without_annotation_skips_and_keeps_its_exit_code(tmp_path: Path) -> None:
    result = _audit(tmp_path, BASE + "ZA_COMMIT=\nZA_LOAD=not-found\n")

    assert result.returncode == 0, result.stdout
    assert "SKIP" in result.stdout and "zone-annotator not installed" in result.stdout
    assert not any(_line(result.stdout, c) for c in range(7, 14))


@pytest.mark.parametrize(
    ("check", "drift", "named"),
    [
        (7, {"ZA_COMMIT": "1" * 40}, "1111111"),
        (7, {"ZA_COMMIT": ""}, "not installed"),
        # Files changed in the release after the install: the directory name still says PIN.
        (7, {"ZA_TREE": "f" * 40}, "tree ffffffff"),
        (8, {"ZA_LISTEN": "0.0.0.0:8713"}, "0.0.0.0:8713"),
        (8, {"ZA_ACTIVE": "failed", "ZA_LISTEN": ""}, "failed"),
        (9, {"ZA_IPAddressDeny": ""}, "IPAddressDeny"),
        (9, {"ZA_ProtectSystem": "no"}, "ProtectSystem"),
        (9, {"ZA_InaccessiblePaths": "-/etc/cctv-client"}, "cctv-client"),
        (10, {"ZT_TOKEN_MODE": "644"}, "644"),
        # 0600 is no guard when the agent owns the file: it reads the tunnel token.
        (10, {"ZT_TOKEN_OWNER": "cameraboy"}, "owned by cameraboy"),
        (10, {"ZT_ACTIVE": "inactive"}, "inactive"),
        (11, {"EGRESS_7844": "fail"}, "7844"),
        (12, {"ZA_DROP_KEY_MODE": ""}, "missing"),
        (12, {"ZA_DROP_KEY_MODE": "644"}, "644"),
        (13, {"ZA_DROP_HOSTS": ""}, "missing"),
        (13, {"ZA_DROP_HOSTS": "0" * 64}, "not the cctv-vps key pinned"),
    ],
)
def test_drift_fails_with_its_name_and_the_installer_state_passes(
    tmp_path: Path, check: int, drift: dict[str, str], named: str
) -> None:
    broken = _audit(tmp_path, _blob(**drift))
    line = _line(broken.stdout, check)
    assert broken.returncode == 1
    assert line.startswith("FAIL") and named in line, broken.stdout
    assert "install-annotator.sh" in line or check == 11

    repaired = _audit(tmp_path, _blob())
    assert _line(repaired.stdout, check).startswith("PASS")


def _release_like_the_installers(release: Path) -> str:
    """Files whose tree is easy to get wrong, and the tree git itself gives them."""
    (release / "pkg" / "sub").mkdir(parents=True)
    (release / "pkg" / "b.py").write_text("b\n")
    (release / "pkg-x").write_text("ordered before pkg/\n")
    (release / "pkg.txt").write_text("and so is this\n")
    (release / "run.sh").write_text("#!/bin/sh\n")
    (release / "run.sh").chmod(0o755)
    (release / "empty").mkdir()
    (release / "link").symlink_to("pkg-x")
    (release / ".venv" / "bin").mkdir(parents=True)
    (release / ".venv" / "bin" / "python").write_text("the venv is not in the tree")
    copy = release.parent / "for-git"
    shutil.copytree(release, copy, symlinks=True, ignore=shutil.ignore_patterns(".venv"))
    git = ["git", "-C", str(copy)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    tree = subprocess.run([*git, "write-tree"], capture_output=True, text=True, check=True)
    shutil.rmtree(copy)
    return tree.stdout.strip()


def test_the_real_probe_reads_a_stubbed_annotation_box(tmp_path: Path) -> None:
    """Runs the probe itself, so a quoting slip in the new lines fails here and not on a box."""
    stub = tmp_path / "bin"
    stub.mkdir()
    prefix = tmp_path / "opt" / "zone-annotator"
    (prefix / "releases" / PIN).mkdir(parents=True)
    (prefix / "current").symlink_to(f"releases/{PIN}")
    release = prefix / "releases" / PIN
    tree = _release_like_the_installers(release)
    token = tmp_path / "etc" / "zone-annotator" / "tunnel.token"
    token.parent.mkdir(parents=True)
    token.write_text("t")
    token.chmod(0o600)
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "annotation-drop").write_text("k")
    (home / ".ssh" / "annotation-drop").chmod(0o600)
    (home / ".ssh" / "annotation-drop.known_hosts").write_text(KNOWN_HOSTS)
    names = ("IPAddressDeny", "IPAddressAllow", "NoNewPrivileges", "ProtectSystem")
    props = "\n".join(f"{n}={HEALTHY[f'ZA_{n}']}" for n in (*names, "InaccessiblePaths"))
    (stub / "systemctl").write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  *zone-annotator-tunnel*ActiveState*) echo active ;;\n"
        "  *zone-annotator.service*LoadState*) echo loaded ;;\n"
        "  *zone-annotator.service*ActiveState*) echo active ;;\n"
        f"  *zone-annotator.service*IPAddressDeny*) cat <<'P'\n{props}\nP\n ;;\n"
        "  *is-enabled*) echo enabled ;;\n"
        "  *ActiveState*) echo active ;;\n"
        "  *SubState*) echo running ;;\n"
        "  *NRestarts*) echo 0 ;;\n"
        "esac\n"
    )
    (stub / "ss").write_text("#!/usr/bin/env bash\necho 'LISTEN 0 128 127.0.0.1:8713 0.0.0.0:*'\n")
    (stub / "pgrep").write_text("#!/usr/bin/env bash\necho 1\n")
    (stub / "loginctl").write_text("#!/usr/bin/env bash\necho yes\n")
    for f in stub.iterdir():
        f.chmod(0o755)
    with socket.create_server(("127.0.0.1", 0)) as edge:
        ssh = tmp_path / "fake-ssh"
        ssh.write_text(
            f'#!/usr/bin/env bash\ncmd="${{@: -1}}"\nexport PATH="{stub}:$PATH"\n'
            f'export ZA_PREFIX="{prefix}" ZA_TOKEN="{token}" HOME="{home}"\n'
            f"export EGRESS_HOST=127.0.0.1 EGRESS_PORT={edge.getsockname()[1]}\n"
            'eval "$cmd"\n'
        )
        ssh.chmod(0o755)

        # The stub token belongs to whoever runs the test, not to root.
        me = subprocess.run(["id", "-un"], capture_output=True, text=True).stdout.strip()
        result = _audit(tmp_path, "", ssh=ssh, tree=tree, TOKEN_OWNER=me)

    for check in range(7, 14):
        assert _line(result.stdout, check).startswith("PASS"), result.stdout + result.stderr


def test_a_release_holding_something_other_than_files_fails_check_7_without_opening_it(
    tmp_path: Path,
) -> None:
    """A FIFO would block the probe for ever and /dev/zero fill its memory; neither is a file."""
    prefix = tmp_path / "opt" / "zone-annotator"
    release = prefix / "releases" / PIN
    release.mkdir(parents=True)
    (prefix / "current").symlink_to(f"releases/{PIN}")
    os.mkfifo(release / "fifo")
    ssh = tmp_path / "fake-ssh"
    ssh.write_text(
        f'#!/usr/bin/env bash\ncmd="${{@: -1}}"\nexport ZA_PREFIX="{prefix}" EGRESS_PORT=1\n'
        'eval "$cmd"\n'
    )
    ssh.chmod(0o755)

    result = _audit(tmp_path, "", ssh=ssh)

    line = _line(result.stdout, 7)
    assert line.startswith("FAIL") and "fifo is not a regular file" in line, result.stdout
