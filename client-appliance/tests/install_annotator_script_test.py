"""Tests for install-annotator.sh: zone-annotator and its tunnel on an appliance (gpu-exchange#298).

Unlike ``install_script_test.py`` this installer runs for real under pytest: every root-only
location sits under ``DESTDIR`` and every systemd call goes through ``$SYSTEMCTL``, so the test
installs a throwaway zone-annotator package built from a temporary git repo, with the same
``git archive`` the operator uses for the real one.

Assumptions these tests encode:

- the package is an uncompressed ``git archive`` tar; its commit comes from the archive itself
  (``git get-tar-commit-id``), never from its file name;
- the agent user exists; ``AGENT_HOME`` stands in for its home directory;
- not tested here: the cloudflared download and dpkg (static checks only), and the service
  actually listening, which the audit checks on a live box.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from systemd_unit import directive_values

ROOT = Path(__file__).resolve().parent.parent
INSTALL = ROOT / "install-annotator.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not all(shutil.which(t) for t in ("bash", "git", "uv")),
    reason="runs the installer for real: needs Linux, bash, git and uv",
)


def test_the_pin_is_a_commit_and_its_tree() -> None:
    """The one place the zone-annotator version lives (PRD stories 16, 20, 21, 23, 24)."""
    assert re.fullmatch(r"[0-9a-f]{40}\n[0-9a-f]{40}\n?", (ROOT / "zone-annotator.pin").read_text())


SIBLING = ROOT.parent.parent / "zone-annotator"


@pytest.mark.skipif(not (SIBLING / ".git").exists(), reason="no zone-annotator checkout beside")
def test_the_pinned_tree_is_the_pinned_commits() -> None:
    commit, tree = (ROOT / "zone-annotator.pin").read_text().split()
    actual = subprocess.run(
        ["git", "-C", str(SIBLING), "rev-parse", f"{commit}^{{tree}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert actual == tree


def test_cloudflared_is_pinned_and_checked_before_dpkg() -> None:
    text = INSTALL.read_text()
    assert "releases/latest" not in text
    assert re.search(r"^CLOUDFLARED_VERSION=\d{4}\.\d+\.\d+$", text, re.MULTILINE)
    assert text.index("sha256sum -c") < text.index("dpkg -i")


def test_installer_is_executable_bash() -> None:
    assert INSTALL.stat().st_mode & stat.S_IXUSR
    assert subprocess.run(["bash", "-n", str(INSTALL)], check=False).returncode == 0


def _package(tmp_path: Path, smoke: str = 'print("smoke-phase1 OK")') -> tuple[Path, str]:
    """A minimal zone-annotator as a ``git archive`` tar, and the commit it was cut from."""
    src = tmp_path / "zone-annotator-src"
    (src / "zone_annotator").mkdir(parents=True)
    (src / "pyproject.toml").write_text(
        '[project]\nname = "zone-annotator"\nversion = "0.0.0"\n'
        'requires-python = ">=3.11"\ndependencies = []\n'
    )
    (src / "zone_annotator" / "__init__.py").write_text("")
    (src / "zone_annotator" / "smoke_phase1.py").write_text(smoke + "\n")
    subprocess.run(["uv", "lock", "--offline", "-q"], cwd=src, check=True)
    git = ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    tar = tmp_path / "zone-annotator.tar"
    subprocess.run([*git, "archive", "-o", str(tar), "HEAD"], check=True)
    # The pin's second line; _run reads it from beside the tar.
    tree = subprocess.run([*git, "rev-parse", "HEAD^{tree}"], capture_output=True, text=True)
    tar.with_suffix(".tree").write_text(tree.stdout.strip())
    return tar, commit


@pytest.fixture
def tunnel_port() -> Iterator[int]:
    """A listening local port standing in for the Cloudflare edge on 7844."""
    with socket.create_server(("127.0.0.1", 0)) as server:
        yield server.getsockname()[1]


def _closed_port() -> int:
    with socket.create_server(("127.0.0.1", 0)) as server:
        return server.getsockname()[1]


APPLIANCE = "3f2b8c1e-9d4a-4e6b-8a7f-2c5d1e0b9a64"


def _run(
    tmp_path: Path,
    tar: Path,
    pin: str,
    *,
    egress_port: int,
    appliance: str = APPLIANCE,
    tree: str | None = None,
    **extra_env: str,
) -> subprocess.CompletedProcess[str]:
    destdir = tmp_path / "root"
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    systemctl = tmp_path / "systemctl"
    systemctl.write_text(f'#!/usr/bin/env bash\necho "$*" >>"{tmp_path / "systemctl.log"}"\n')
    systemctl.chmod(0o755)
    # Already at the pinned version, so the installer never downloads in a test.
    cloudflared = tmp_path / "cloudflared"
    cloudflared.write_text(
        '#!/usr/bin/env bash\necho "cloudflared version 2026.7.1 (built 2026-07-01-0000 UTC)"\n'
    )
    cloudflared.chmod(0o755)
    # Drops `-u <agent> --` and runs the rest as the test's own user, keeping a log.
    runuser = tmp_path / "runuser"
    runuser.write_text(
        f'#!/usr/bin/env bash\necho "$*" >>"{tmp_path / "runuser.log"}"\nshift 3\nexec "$@"\n'
    )
    runuser.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "DESTDIR": str(destdir),
        "AGENT_HOME": str(home),
        "ZONE_ANNOTATOR_PIN": pin,
        "ZONE_ANNOTATOR_TREE": tree or tar.with_suffix(".tree").read_text(),
        "SYSTEMCTL": str(systemctl),
        "RUNUSER": str(runuser),
        "CLOUDFLARED": str(cloudflared),
        "EGRESS_HOST": "127.0.0.1",
        "EGRESS_PORT": str(egress_port),
        **extra_env,
    }
    user = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    return subprocess.run(
        ["bash", str(INSTALL), user, str(tar), appliance],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_creates_the_shipping_key_and_prints_the_line_to_register_it(
    tmp_path: Path, tunnel_port: int
) -> None:
    """gpu-exchange#300: packages leave over rsync with this key, into cctv-vps's drop account."""
    tar, commit = _package(tmp_path)

    result = _run(tmp_path, tar, commit, egress_port=tunnel_port)

    assert result.returncode == 0, result.stdout + result.stderr
    key = tmp_path / "home" / ".ssh" / "annotation-drop"
    assert key.stat().st_mode & 0o777 == 0o600
    public = (key.parent / "annotation-drop.pub").read_text().split()
    assert public[0] == "ssh-ed25519"
    assert f'annotation-drop.sh add {APPLIANCE} "ssh-ed25519 {public[1]}' in result.stdout
    assert key.read_text() not in result.stdout + result.stderr


def test_pins_cctv_vps_host_key_for_the_agent_and_replaces_one_learned_on_first_use(
    tmp_path: Path, tunnel_port: int
) -> None:
    """The poller connects with StrictHostKeyChecking=yes, so this file is all ssh trusts."""
    tar, commit = _package(tmp_path)
    pinned = (ROOT / "annotation-drop.known_hosts").read_text()
    assert re.search(r"^cctv-vps,100\.[0-9.]+ ssh-ed25519 \S+$", pinned, re.MULTILINE), pinned
    known = tmp_path / "home" / ".ssh" / "annotation-drop.known_hosts"
    known.parent.mkdir(parents=True)
    known.write_text("cctv-vps ssh-ed25519 AAAAlearnedFromWhoeverAnswered\n")

    result = _run(tmp_path, tar, commit, egress_port=tunnel_port)

    assert result.returncode == 0, result.stdout + result.stderr
    assert known.read_text() == pinned
    assert known.stat().st_mode & 0o777 == 0o644
    assert "annotation-drop.known_hosts" in (tmp_path / "runuser.log").read_text()


def test_installs_the_pinned_package_and_its_smoke_passes(tmp_path: Path, tunnel_port: int) -> None:
    tar, commit = _package(tmp_path)

    result = _run(tmp_path, tar, commit, egress_port=tunnel_port)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "smoke-phase1 OK" in result.stdout
    current = tmp_path / "root" / "opt" / "zone-annotator" / "current"
    assert current.resolve().name == commit
    assert (current / ".venv" / "bin" / "python").exists()


def test_refuses_a_package_cut_from_another_commit_before_changing_anything(
    tmp_path: Path, tunnel_port: int
) -> None:
    tar, commit = _package(tmp_path)

    result = _run(tmp_path, tar, "0" * 40, egress_port=tunnel_port)

    assert result.returncode != 0
    assert commit[:7] in result.stderr and "0000000" in result.stderr
    assert not (tmp_path / "root").exists()


def test_installs_a_confined_unit_with_absolute_paths(tmp_path: Path, tunnel_port: int) -> None:
    """``%h`` in a system unit is /root, not the home of ``User=``, so every path is spelled out."""
    tar, commit = _package(tmp_path)

    result = _run(tmp_path, tar, commit, egress_port=tunnel_port)

    assert result.returncode == 0, result.stdout + result.stderr
    unit = tmp_path / "root" / "etc" / "systemd" / "system" / "zone-annotator.service"
    home = tmp_path / "home"
    projects = home / ".local" / "state" / "zone-annotator" / "projects"
    windows = home / ".local" / "state" / "zone-annotator" / "windows"
    user = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()

    def values(key: str) -> list[str]:
        return directive_values(unit, "Service", key)

    assert "%h" not in unit.read_text()
    assert values("User") == [user]
    assert values("ExecStart") == [
        "/opt/zone-annotator/current/.venv/bin/python -m zone_annotator.serve"
        f" --root {projects} --windows {windows} --max-zones 3 --port 8713"
    ]
    assert values("IPAddressDeny") == ["any"]
    assert values("IPAddressAllow") == ["localhost"]
    assert values("NoNewPrivileges") == ["yes"]
    assert values("ProtectSystem") == ["strict"]
    assert values("ReadWritePaths") == [str(projects)]
    inaccessible = " ".join(values("InaccessiblePaths")).split()
    assert f"-{home}/.config/cctv-client" in inaccessible
    assert f"-{home}/.ssh" in inaccessible
    assert values("CPUWeight") and values("IOWeight") and values("MemoryMax")
    assert projects.is_dir() and windows.is_dir()
    calls = (tmp_path / "systemctl.log").read_text().splitlines()
    assert "daemon-reload" in calls
    assert "enable zone-annotator.service" in calls
    assert "restart zone-annotator.service" in calls


def _tree(*roots: Path) -> dict[str, tuple[int, int, int]]:
    """Every entry under the roots with its mode, size and mtime, symlinks themselves included."""
    entries = {}
    for root in roots:
        for path in [root, *root.rglob("*")]:
            st = path.lstat()
            entries[str(path)] = (st.st_mode, st.st_size, st.st_mtime_ns)
    return entries


def test_second_run_changes_nothing(tmp_path: Path, tunnel_port: int) -> None:
    tar, commit = _package(tmp_path)
    first = _run(tmp_path, tar, commit, egress_port=tunnel_port)
    assert first.returncode == 0, first.stdout + first.stderr
    before = _tree(tmp_path / "root", tmp_path / "home")
    log = tmp_path / "systemctl.log"
    log.unlink()

    second = _run(tmp_path, tar, commit, egress_port=tunnel_port)

    assert second.returncode == 0, second.stdout + second.stderr
    assert _tree(tmp_path / "root", tmp_path / "home") == before
    calls = log.read_text().splitlines() if log.exists() else []
    assert not [c for c in calls if c.split()[0] in {"daemon-reload", "enable", "restart"}], calls


TOKEN = "eyJhIjoiZmFrZS10dW5uZWwtdG9rZW4ifQ"


def test_moves_the_staged_tunnel_token_out_of_reach_and_never_prints_it(
    tmp_path: Path, tunnel_port: int
) -> None:
    """The tunnel script from gpu-exchange stages the token over ssh into the agent's config."""
    tar, commit = _package(tmp_path)
    staged = tmp_path / "home" / ".config" / "cctv-client" / "zone-annotator-tunnel.token"
    staged.parent.mkdir(parents=True)
    staged.write_text(TOKEN)
    staged.chmod(0o600)

    result = _run(tmp_path, tar, commit, egress_port=tunnel_port)

    assert result.returncode == 0, result.stdout + result.stderr
    token = tmp_path / "root" / "etc" / "zone-annotator" / "tunnel.token"
    assert token.read_text() == TOKEN
    assert token.stat().st_mode & 0o777 == 0o600
    assert not staged.exists()
    unit = tmp_path / "root" / "etc" / "systemd" / "system" / "zone-annotator-tunnel.service"
    assert directive_values(unit, "Service", "LoadCredential") == [
        "tunnel-token:/etc/zone-annotator/tunnel.token"
    ]
    assert directive_values(unit, "Service", "ExecStart") == [
        "/usr/bin/cloudflared tunnel --no-autoupdate run --token-file %d/tunnel-token"
    ]
    assert directive_values(unit, "Service", "DynamicUser") == ["yes"]
    calls = (tmp_path / "systemctl.log").read_text()
    assert "restart zone-annotator-tunnel.service" in calls
    for text in (result.stdout, result.stderr, calls, unit.read_text()):
        assert TOKEN not in text


def test_a_rerun_repairs_the_token_mode_and_leaves_it_auditable(
    tmp_path: Path, tunnel_port: int
) -> None:
    """Check 10 of the audit stats the token as the unprivileged user, so its directory is 0755."""
    tar, commit = _package(tmp_path)
    token = tmp_path / "root" / "etc" / "zone-annotator" / "tunnel.token"
    token.parent.mkdir(parents=True)
    token.write_text(TOKEN)
    token.chmod(0o644)

    result = _run(tmp_path, tar, commit, egress_port=tunnel_port)

    assert result.returncode == 0, result.stdout + result.stderr
    assert token.stat().st_mode & 0o777 == 0o600
    assert token.parent.stat().st_mode & 0o777 == 0o755


def test_stops_before_changing_anything_without_egress_to_7844(tmp_path: Path) -> None:
    tar, commit = _package(tmp_path)

    result = _run(tmp_path, tar, commit, egress_port=_closed_port())

    assert result.returncode != 0
    assert "7844" in result.stderr
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        ({"RSYNC": "/nonexistent/rsync"}, "rsync"),
        # sudo's secure_path rarely holds the operator's ~/.local/bin, where uv usually is.
        ({"UV": "/nonexistent/uv"}, "uv not on root's PATH"),
        ({"GIT": "/nonexistent/git"}, "git not installed"),
        ({"appliance": "../3f2b8c1e"}, "UUID"),
    ],
)
def test_stops_before_changing_anything_without_rsync_or_with_a_bad_appliance_id(
    tmp_path: Path, tunnel_port: int, extra: dict[str, str], named: str
) -> None:
    tar, commit = _package(tmp_path)

    result = _run(tmp_path, tar, commit, egress_port=tunnel_port, **extra)

    assert result.returncode != 0
    assert named in result.stderr
    assert not (tmp_path / "root").exists()
    assert not (tmp_path / "home" / ".ssh").exists()


def test_a_rerun_keeps_the_shipping_key_and_repairs_its_mode(
    tmp_path: Path, tunnel_port: int
) -> None:
    tar, commit = _package(tmp_path)
    assert _run(tmp_path, tar, commit, egress_port=tunnel_port).returncode == 0
    key = tmp_path / "home" / ".ssh" / "annotation-drop"
    before = key.read_bytes()
    key.chmod(0o644)

    assert _run(tmp_path, tar, commit, egress_port=tunnel_port).returncode == 0

    assert key.read_bytes() == before
    assert key.stat().st_mode & 0o777 == 0o600


def test_a_unit_that_cannot_be_reloaded_stops_the_installer(
    tmp_path: Path, tunnel_port: int
) -> None:
    """put_unit ran in an `&&` list, where set -e is off: a failed reload read as unchanged."""
    tar, commit = _package(tmp_path)
    failing = tmp_path / "systemctl-failing"
    failing.write_text('#!/usr/bin/env bash\n[[ "$1" == daemon-reload ]] && exit 1\nexit 0\n')
    failing.chmod(0o755)

    result = _run(tmp_path, tar, commit, egress_port=tunnel_port, SYSTEMCTL=str(failing))

    assert result.returncode != 0, result.stdout + result.stderr


def test_root_changes_the_agents_home_only_as_the_agent(tmp_path: Path, tunnel_port: int) -> None:
    """The agent owns its home and may leave links in it; as root, writing there follows them."""
    tar, commit = _package(tmp_path)
    staged = tmp_path / "home" / ".config" / "cctv-client" / "zone-annotator-tunnel.token"
    staged.parent.mkdir(parents=True)
    staged.write_text(TOKEN)

    result = _run(tmp_path, tar, commit, egress_port=tunnel_port)

    assert result.returncode == 0, result.stdout + result.stderr
    user = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    calls = (tmp_path / "runuser.log").read_text().splitlines()
    assert all(c.startswith(f"-u {user} -- ") for c in calls), calls
    ran = "\n".join(calls)
    for step in (
        "install -d -m 0750",
        "ssh-keygen",
        "chmod 0600",
        f"cat {staged}",
        f"rm -f {staged}",
    ):
        assert step in ran, calls


def test_the_package_is_read_once_as_the_agent_and_its_smoke_runs_as_the_agent(
    tmp_path: Path, tunnel_port: int
) -> None:
    """The tar sits in the agent's home: what root checks and unpacks is one copy of it,
    and code from it runs as the agent, not as root."""
    tar, commit = _package(tmp_path)
    other, _ = _package(tmp_path / "other", smoke='print("ANOTHER PACKAGE RAN")')
    swapping = tmp_path / "runuser-swapping"
    swapping.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >>"{tmp_path / "runuser.log"}"\n'
        "shift 3\n"
        # The agent swaps its file the moment root has read it.
        f'if [[ "$*" == "cat {tar}" ]]; then cat "{tar}"; cp "{other}" "{tar}"; exit; fi\n'
        'exec "$@"\n'
    )
    swapping.chmod(0o755)

    result = _run(tmp_path, tar, commit, egress_port=tunnel_port, RUNUSER=str(swapping))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "smoke-phase1 OK" in result.stdout and "ANOTHER" not in result.stdout
    calls = (tmp_path / "runuser.log").read_text()
    assert f"cat {tar}" in calls
    assert ".venv/bin/python -m zone_annotator.smoke_phase1" in calls


def _forged(tar: Path, out: Path, *, add: dict[str, bytes]) -> Path:
    """``tar`` with files added and replaced, under the same commit id in its header.

    A one-byte tar type (``tarfile.FIFOTYPE``, ``CHRTYPE``) adds an entry of that kind.
    """
    with (
        tarfile.open(tar) as original,
        tarfile.open(
            out, "w", format=tarfile.PAX_FORMAT, pax_headers=original.pax_headers
        ) as forged,
    ):
        for member in original.getmembers():
            if member.name not in add:
                forged.addfile(member, original.extractfile(member))
        for name, content in add.items():
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            if content in (tarfile.FIFOTYPE, tarfile.CHRTYPE):
                info.type, info.devmajor, info.devminor = content, 1, 3  # /dev/null
                forged.addfile(info)
                continue
            info.size = len(content)
            forged.addfile(info, io.BytesIO(content))
    return out


@pytest.mark.parametrize(
    "add",
    [
        {"zone_annotator/smoke_phase1.py": b'print("FORGED CODE RAN")\n'},
        # Not in any tree git computes, so the comparison alone would let it through.
        {".git/config": b"[core]\n\tfsmonitor = /bin/false\n"},
        # git add skips both, so they too would pass the comparison, and tar makes them for real.
        {"zone_annotator/fifo": tarfile.FIFOTYPE},
        pytest.param(
            {"zone_annotator/disk": tarfile.CHRTYPE},
            marks=pytest.mark.skipif(
                getattr(os, "geteuid", lambda: -1)() != 0, reason="only root makes a device node"
            ),
        ),
    ],
    ids=["changed-file", "git-metadata", "fifo", "device-node"],
)
def test_refuses_a_package_that_is_not_the_pinned_tree_before_running_any_of_it(
    tmp_path: Path, tunnel_port: int, add: dict[str, bytes]
) -> None:
    """The commit id in a tar header is one line anybody can write; the tree is the content."""
    tar, commit = _package(tmp_path)
    forged = _forged(tar, tmp_path / "forged.tar", add=add)
    header = subprocess.run(
        ["git", "get-tar-commit-id"], stdin=forged.open("rb"), capture_output=True
    )
    assert header.stdout.decode().strip() == commit, "the forgery keeps the pinned commit id"

    result = _run(
        tmp_path, forged, commit, egress_port=tunnel_port, tree=tar.with_suffix(".tree").read_text()
    )

    assert result.returncode != 0
    assert "tree" in result.stderr, result.stderr
    assert "FORGED" not in result.stdout + result.stderr
    assert not (tmp_path / "root").exists()
