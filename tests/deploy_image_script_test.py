"""Tests for scripts/deploy-image.sh.

The script's real work is network-shaped (resolve a digest from GHCR, ssh to
every GPU node, pull, compare) so it can't run under pytest. What we pin here
is the contract an operator depends on: it parses, it refuses to run with a
half-configured environment, and — the part that actually prevents a silent
stale deploy — its digest verdict is right in both directions.

The script is written so ``DEPLOY_IMAGE_LIB=1 source``-ing it defines the
functions without running main, which is what lets the verdict be tested
offline.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "deploy-image.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


def _source_and_run(snippet: str) -> subprocess.CompletedProcess[str]:
    """Run ``snippet`` with the script sourced as a library (main suppressed)."""
    return subprocess.run(
        ["bash", "-c", f'DEPLOY_IMAGE_LIB=1 source "{SCRIPT}"\n{snippet}'],
        capture_output=True,
        text=True,
    )


class TestScriptContract:
    def test_script_exists_and_is_executable(self) -> None:
        assert SCRIPT.is_file(), SCRIPT
        assert SCRIPT.stat().st_mode & stat.S_IXUSR, "deploy-image.sh missing user-execute bit"

    def test_uses_bash_shebang_and_strict_mode(self) -> None:
        # Arrays and [[ ]] break under dash; an unset var or a failed curl must
        # abort a deploy rather than push on with an empty digest.
        text = SCRIPT.read_text(encoding="utf-8")
        assert text.splitlines()[0] == "#!/usr/bin/env bash"
        assert "set -euo pipefail" in text

    def test_syntax_is_valid(self) -> None:
        proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

    def test_sourcing_as_a_library_does_not_deploy(self) -> None:
        # Guards the test harness itself: if main ran on source, every test here
        # would ssh into production.
        proc = _source_and_run('echo "sourced-only"')
        assert proc.returncode == 0, proc.stderr
        assert "sourced-only" in proc.stdout


class TestDigestVerdict:
    """The check that makes the script worth running (issue #96 follow-up).

    The gpu-agent runs ``docker run`` with no pull, so a node holding a stale
    ``:latest`` keeps serving old code indefinitely and nothing says so. The
    verdict has to fail loudly on a mismatch, not just print both digests.
    """

    EXPECTED = "sha256:a8982cbbbfd7f2e8fa15673bc41175a39c26ea34adbf9f61f53d8cf048e4eef2"
    STALE = "sha256:8c38a0331c6c45dd3f8b306dbc1c288eb9dc9661d95ed991f27d82c005a7db84"

    def test_matching_digest_passes(self) -> None:
        proc = _source_and_run(f'verdict "a-node" "{self.EXPECTED}" "{self.EXPECTED}"')

        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_stale_digest_fails_and_names_the_node(self) -> None:
        proc = _source_and_run(f'verdict "cctv-vps" "{self.EXPECTED}" "{self.STALE}"')

        assert proc.returncode != 0
        assert "cctv-vps" in proc.stdout + proc.stderr

    def test_missing_digest_fails(self) -> None:
        # A node that never had the image reports an empty digest; treating that
        # as "nothing to compare, carry on" is exactly the silent pass to avoid.
        proc = _source_and_run(f'verdict "cctv-vps" "{self.EXPECTED}" ""')

        assert proc.returncode != 0
