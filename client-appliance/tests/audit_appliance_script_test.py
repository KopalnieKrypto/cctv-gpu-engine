"""Tests for audit-appliance.sh — the double-run / restart-loop detector.

Unlike ``install_script_test.py`` (which cannot execute its subject, since the
installer talks to systemd and writes into $HOME), this script is *fully
executable* under pytest: every remote call goes through ``$SSH``, so a fake
ssh emitting a canned probe blob exercises the real parsing and exit-code
paths. Contract-only assertions would be too weak here — the whole value of
this script is the verdict it returns, and a verdict is testable.

Source incident (2026-07-20, cameraboy): a stray hand-launched process held
:8080 while the systemd unit crash-looped 4484 times behind it. The box served
production from three-day-old code. Nothing detected it because a unit in
``Restart=on-failure`` backoff reports ActiveState=activating /
SubState=auto-restart — never ``failed`` — so `systemctl is-active` prints
"activating" and reads as healthy at a glance.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "audit-appliance.sh"

# Exit codes the script promises in its usage block.
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_WARN = 2


def _text() -> str:
    return AUDIT.read_text()


def _fake_ssh(tmp_path: Path, blob: str) -> Path:
    """A stand-in for ssh that ignores its args and prints a canned probe blob."""
    fake = tmp_path / "fake-ssh"
    fake.write_text(f"#!/usr/bin/env bash\ncat <<'BLOB'\n{blob}\nBLOB\n")
    fake.chmod(0o755)
    return fake


def _run(tmp_path: Path, blob: str, *args: str) -> subprocess.CompletedProcess[str]:
    fake = _fake_ssh(tmp_path, blob)
    return subprocess.run(
        [str(AUDIT), *(args or ("cameraboy",))],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "SSH": str(fake)},
        check=False,
    )


def _blob(
    *,
    procs: int = 1,
    active: str = "active",
    sub: str = "running",
    restarts: int = 0,
    enabled: str = "enabled",
    linger: str = "yes",
) -> str:
    return (
        f"PROC_COUNT={procs}\nACTIVE_STATE={active}\nSUB_STATE={sub}\n"
        f"N_RESTARTS={restarts}\nUNIT_ENABLED={enabled}\nLINGER={linger}"
    )


def test_audit_script_exists() -> None:
    assert AUDIT.is_file(), AUDIT


def test_audit_script_is_executable() -> None:
    """Operators run this straight from a clone; a missing +x bit is exactly
    the undocumented prerequisite that gets skipped at 2am."""
    assert AUDIT.stat().st_mode & stat.S_IXUSR, "audit-appliance.sh missing user-execute bit"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_audit_script_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(AUDIT)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_healthy_box_passes(tmp_path: Path) -> None:
    result = _run(tmp_path, _blob())
    assert result.returncode == EXIT_PASS, result.stdout + result.stderr
    assert "PASS" in result.stdout
    assert "FAIL" not in result.stdout


def test_double_run_fails(tmp_path: Path) -> None:
    """Two processes = the cameraboy incident. Must FAIL, not warn: one of the
    two is serving production from code nobody chose."""
    result = _run(tmp_path, _blob(procs=2))
    assert result.returncode == EXIT_FAIL
    assert "FAIL" in result.stdout
    assert "2 client_agent processes" in result.stdout


def test_double_run_warns_against_pkill(tmp_path: Path) -> None:
    """The obvious cleanup — `pkill -f client_agent` over ssh — matches the
    operator's own ssh command line and kills the session mid-write. The
    remediation hint must say so at the moment of failure, not in a doc."""
    result = _run(tmp_path, _blob(procs=3))
    assert "pkill" in result.stdout, "double-run message must warn against pkill -f"
    assert "exact PID" in result.stdout


def test_no_process_fails(tmp_path: Path) -> None:
    """Zero processes is its own failure: the unit may be `inactive` and no
    other check here would notice the appliance is simply gone."""
    result = _run(tmp_path, _blob(procs=0, active="inactive", sub="dead"))
    assert result.returncode == EXIT_FAIL
    assert "no client_agent process running" in result.stdout


def test_restart_loop_detected_via_auto_restart(tmp_path: Path) -> None:
    """The exact shape `is-active` hides. ActiveState=activating +
    SubState=auto-restart must FAIL even when NRestarts is still low, because
    a probe can land on the very first loop iteration."""
    result = _run(tmp_path, _blob(procs=0, active="activating", sub="auto-restart", restarts=1))
    assert result.returncode == EXIT_FAIL
    assert "restart loop" in result.stdout


def test_restart_loop_message_notes_is_active_blind_spot(tmp_path: Path) -> None:
    """Whoever reads this FAIL will have just seen `is-active` say something
    reassuring. Say why the two disagree, or the finding gets dismissed."""
    result = _run(tmp_path, _blob(active="activating", sub="auto-restart", restarts=4484))
    assert "never 'failed'" in result.stdout
    assert "journalctl" in result.stdout, "must tell the operator where to look next"


def test_high_restart_count_fails_even_when_momentarily_active(tmp_path: Path) -> None:
    """A loop probed during an up-swing looks `active`/`running`. The restart
    counter is what gives it away."""
    result = _run(tmp_path, _blob(active="active", sub="running", restarts=4484))
    assert result.returncode == EXIT_FAIL
    assert "4484 times" in result.stdout


def test_single_restart_warns_but_does_not_fail(tmp_path: Path) -> None:
    """One restart after an OOM or a transient platform 502 is normal
    operation. Failing on it would train operators to ignore this script."""
    result = _run(tmp_path, _blob(active="active", sub="running", restarts=1))
    assert result.returncode == EXIT_WARN
    assert "WARN" in result.stdout
    assert "FAIL" not in result.stdout


def test_unreachable_host_warns_rather_than_false_passing(tmp_path: Path) -> None:
    """An empty probe means the ssh failed. That must never read as PASS —
    a silent green on an unreachable box is the failure this script exists
    to prevent, one level up."""
    result = _run(tmp_path, "")
    assert result.returncode == EXIT_WARN
    assert "WARN" in result.stdout
    assert "PASS" not in result.stdout


def test_reboot_survival_passes_when_enabled_and_lingering(tmp_path: Path) -> None:
    result = _run(tmp_path, _blob())
    assert result.returncode == EXIT_PASS
    assert "survives reboot" in result.stdout


def test_enabled_without_linger_fails(tmp_path: Path) -> None:
    """The trap this check exists for. `is-enabled` says "enabled" and the unit
    is running right now, so the box looks entirely healthy — but without
    lingering systemd tears the user manager down at logout and never rebuilds
    it on a headless box. The unit simply never starts after a reboot."""
    result = _run(tmp_path, _blob(linger="no"))
    assert result.returncode == EXIT_FAIL
    assert "will NOT survive reboot" in result.stdout
    assert "Linger=no" in result.stdout
    assert "enable-linger" in result.stdout, "must give the fix, not just the diagnosis"


def test_lingering_but_unit_disabled_fails(tmp_path: Path) -> None:
    """The other half. Linger alone starts the user manager but nothing pulls
    a disabled unit into default.target."""
    result = _run(tmp_path, _blob(enabled="disabled"))
    assert result.returncode == EXIT_FAIL
    assert "not enabled" in result.stdout


def test_both_missing_reports_both_causes(tmp_path: Path) -> None:
    """Fixing one and rebooting only to find it still dead is the worst
    possible feedback loop — report both causes in one pass."""
    result = _run(tmp_path, _blob(enabled="disabled", linger="no"))
    assert result.returncode == EXIT_FAIL
    assert "not enabled" in result.stdout
    assert "Linger=no" in result.stdout


def test_reboot_survival_failure_is_independent_of_running_state(tmp_path: Path) -> None:
    """A box can be perfectly healthy *now* and still be one unattended-upgrades
    reboot away from a silent outage. checks 1 and 2 must pass while 3 fails."""
    result = _run(tmp_path, _blob(procs=1, active="active", sub="running", linger="no"))
    assert result.returncode == EXIT_FAIL
    assert "PASS" in result.stdout, "checks 1 and 2 should still pass"
    assert "check 3" in result.stdout


def test_check_flag_runs_only_that_check(tmp_path: Path) -> None:
    result = _run(tmp_path, _blob(procs=2, restarts=4484), "--check", "1", "cameraboy")
    assert "check 1" in result.stdout
    assert "check 2" not in result.stdout


def test_missing_host_exits_usage(tmp_path: Path) -> None:
    fake = _fake_ssh(tmp_path, _blob())
    result = subprocess.run(
        [str(AUDIT)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "SSH": str(fake), "HOSTS": ""},
        check=False,
    )
    assert result.returncode == 64
    assert "usage" in result.stderr


def test_multiple_hosts_are_all_audited(tmp_path: Path) -> None:
    """One unreachable or broken box must not abort the fan-out — that is why
    `set -e` is deliberately off."""
    result = _run(tmp_path, _blob(procs=2), "boxa", "boxb")
    assert "boxa" in result.stdout
    assert "boxb" in result.stdout


def test_probe_script_actually_executes_against_stubbed_remote(tmp_path: Path) -> None:
    """Regression: the probe used to be a double-quoted ssh argument, and the
    escaping mangled the nested ``$(id -un)`` inside the linger lookup. Every
    box then reported ``Linger=`` empty → a FAIL on correctly-configured hosts.

    A canned-blob fake ssh cannot catch that class of bug — it never runs the
    script. So this fake *executes* the received remote script against stubbed
    pgrep/systemctl/loginctl/id, which is what a real box would do. Any future
    quoting regression in the probe fails here instead of on a live host.
    """
    stub = tmp_path / "bin"
    stub.mkdir()

    (stub / "pgrep").write_text("#!/usr/bin/env bash\necho 1\n")
    (stub / "id").write_text("#!/usr/bin/env bash\necho cameraboy\n")
    # Echoes the user it was given, so a mangled $(id -un) yields an empty or
    # wrong value rather than silently still saying "yes".
    (stub / "loginctl").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$2" == "cameraboy" ]]; then echo yes; else echo "BAD_USER:$2"; fi\n'
    )
    (stub / "systemctl").write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  *is-enabled*) echo enabled ;;\n"
        "  *ActiveState*) echo active ;;\n"
        "  *SubState*) echo running ;;\n"
        "  *NRestarts*) echo 0 ;;\n"
        "esac\n"
    )
    for f in stub.iterdir():
        f.chmod(0o755)

    fake = tmp_path / "fake-ssh"
    fake.write_text(
        f'#!/usr/bin/env bash\ncmd="${{@: -1}}"\nexport PATH="{stub}:$PATH"\neval "$cmd"\n'
    )
    fake.chmod(0o755)

    result = subprocess.run(
        [str(AUDIT), "cameraboy"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "SSH": str(fake)},
        check=False,
    )

    assert result.returncode == EXIT_PASS, result.stdout + result.stderr
    assert "survives reboot" in result.stdout, (
        f"nested $(id -un) must survive transport to the remote shell; got: {result.stdout}"
    )
    assert "BAD_USER" not in result.stdout


def test_probe_uses_bracket_trick_against_self_match() -> None:
    """Without the bracket, the pattern matches the probe's own remote command
    line and every box reports a phantom extra process — turning check 1 into
    a permanent false FAIL."""
    assert "client_agent[.]appliance" in _text()


def test_script_is_read_only(tmp_path: Path) -> None:
    """Remediation is deliberately manual: stopping the wrong process
    mid-recording is worse than the drift it fixes, and choosing correctly
    needs a human to read the two start times.

    Asserted against what the script actually *sends to the box*, not against
    its source text — the source legitimately says "pkill" in a comment and in
    the remediation hint (see the double-run test), so a substring scan would
    fail on correct code while still missing a mutation built up at runtime.
    """
    capture = tmp_path / "sent"
    fake = tmp_path / "fake-ssh"
    # Capture BOTH argv and stdin: the remote script body travels on stdin, so
    # an argv-only capture would inspect almost nothing and pass vacuously.
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" >> {capture}\n'
        f"cat >> {capture}\n"
        "printf 'PROC_COUNT=1\\nACTIVE_STATE=active\\nSUB_STATE=running\\n"
        "N_RESTARTS=0\\nUNIT_ENABLED=enabled\\nLINGER=yes\\n'\n"
    )
    fake.chmod(0o755)

    subprocess.run(
        [str(AUDIT), "cameraboy"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "SSH": str(fake)},
        check=False,
    )

    sent = capture.read_text()
    assert "pgrep" in sent, "probe should have run"
    for mutating in ("pkill", "kill ", "rm ", "systemctl --user restart", "systemctl --user stop"):
        assert mutating not in sent, f"audit must stay read-only; remote command had {mutating!r}"
