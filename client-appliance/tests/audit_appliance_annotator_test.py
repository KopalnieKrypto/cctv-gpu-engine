"""Checks 7-13 of audit-appliance.sh: zone-annotator, its tunnel, its shipping key and host key.

The audit stays read-only. Repair is a re-run of install-annotator.sh, so "FAIL then repair then
PASS" is a probe blob with the drift, then the same blob as the installer leaves it.

Assumptions: property lines arrive in the format ``systemctl show`` prints on systemd 255
(Ubuntu 24.04, checked 2026-11-23); a box without /opt/zone-annotator and without the unit has no
annotation and gets SKIP, which leaves the exit code alone.
"""

from __future__ import annotations

import hashlib
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "audit-appliance.sh"
PIN = "d800ae42d8ed45ee5346e13c3f3e2561001bcc34"
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
    "EGRESS_7844": "ok",
    "ZA_DROP_KEY_MODE": "600",
    "ZA_DROP_HOSTS": KNOWN_HOSTS_SHA,
}


def _audit(tmp_path: Path, blob: str, ssh: Path | None = None) -> subprocess.CompletedProcess[str]:
    pin = tmp_path / "zone-annotator.pin"
    pin.write_text(PIN + "\n")
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
        },
        check=False,
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
        (8, {"ZA_LISTEN": "0.0.0.0:8713"}, "0.0.0.0:8713"),
        (8, {"ZA_ACTIVE": "failed", "ZA_LISTEN": ""}, "failed"),
        (9, {"ZA_IPAddressDeny": ""}, "IPAddressDeny"),
        (9, {"ZA_ProtectSystem": "no"}, "ProtectSystem"),
        (9, {"ZA_InaccessiblePaths": "-/etc/cctv-client"}, "cctv-client"),
        (10, {"ZT_TOKEN_MODE": "644"}, "644"),
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


def test_the_real_probe_reads_a_stubbed_annotation_box(tmp_path: Path) -> None:
    """Runs the probe itself, so a quoting slip in the new lines fails here and not on a box."""
    stub = tmp_path / "bin"
    stub.mkdir()
    prefix = tmp_path / "opt" / "zone-annotator"
    (prefix / "releases" / PIN).mkdir(parents=True)
    (prefix / "current").symlink_to(f"releases/{PIN}")
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

        result = _audit(tmp_path, "", ssh=ssh)

    for check in range(7, 14):
        assert _line(result.stdout, check).startswith("PASS"), result.stdout + result.stderr
