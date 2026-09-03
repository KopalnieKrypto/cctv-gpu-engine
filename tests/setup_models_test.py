"""Contract tests for `setup-models.sh` (#122).

The script's job is that every checkout lands the *same* weights. Two properties
carry that, and both are tested here rather than assumed:

- **sha256-verified**: a swapped release asset fails the script instead of
  silently substituting weights.
- **idempotent**: re-running verifies what is on disk and downloads nothing, so
  a provisioning step can be repeated without cost and a partially-set-up box
  can be topped up.

The model list is read out of the script rather than written down here, so a
model added tomorrow is held to the same two properties without anyone
remembering to extend this file.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_MODELS_SCRIPT = REPO_ROOT / "setup-models.sh"

# fetch_model "${MODEL_URL}" "${MODEL_FILE}" "${MODEL_SHA256}" "MODEL_SHA256"
FETCH_CALL = re.compile(
    r'fetch_model\s+"\$\{(\w+)_URL\}"\s+"\$\{(\w+)_FILE\}"\s+"\$\{(\w+)_SHA256\}"'
)


def _fetched_prefixes() -> list[str]:
    text = SETUP_MODELS_SCRIPT.read_text()
    prefixes = []
    for url, file, sha in FETCH_CALL.findall(text):
        assert url == file == sha, (
            f"setup-models.sh calls fetch_model with mismatched variable prefixes "
            f"({url}_URL / {file}_FILE / {sha}_SHA256). Keep them consistent so this "
            "test can enumerate the models."
        )
        prefixes.append(url)
    assert prefixes, "setup-models.sh has no parseable fetch_model calls"
    return prefixes


def _fake_release(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Point every model at a local file:// URL with its real sha256.

    The script is exercised end to end - same curl, same verification, same
    idempotence branch - without reaching the network or moving 330 MB.
    """
    assets = tmp_path / "assets"
    assets.mkdir()
    env = {}
    for i, prefix in enumerate(_fetched_prefixes()):
        blob = assets / f"{prefix.lower()}.bin"
        blob.write_bytes(f"fake weights for {prefix} #{i}".encode())
        env[f"{prefix}_URL"] = blob.as_uri()
        env[f"{prefix}_FILE"] = blob.name
        env[f"{prefix}_SHA256"] = hashlib.sha256(blob.read_bytes()).hexdigest()
    return assets, env


def _run(tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    import os

    return subprocess.run(
        ["bash", str(SETUP_MODELS_SCRIPT)],
        cwd=tmp_path,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )


class TestSetupModels:
    def test_every_fetched_model_is_idempotent(self, tmp_path: Path):
        # Run twice: the second run must verify what is on disk and download
        # nothing, so provisioning a box is repeatable and a half-set-up box can
        # be topped up without deleting anything first.
        _, env = _fake_release(tmp_path)

        first = _run(tmp_path, env)
        assert first.returncode == 0, first.stderr
        assert "Downloading" in first.stdout

        models = sorted((tmp_path / "models").iterdir())
        assert models, "first run fetched nothing"
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in models}

        second = _run(tmp_path, env)
        assert second.returncode == 0, second.stderr
        assert "Downloading" not in second.stdout, (
            "the second run re-downloaded a model that was already present and "
            f"verified:\n{second.stdout}"
        )
        assert "already present" in second.stdout

        after = {
            p.name: (p.stat().st_mtime_ns, p.read_bytes())
            for p in sorted((tmp_path / "models").iterdir())
        }
        assert after == before, "the second run modified files on disk"

    def test_a_swapped_asset_fails_instead_of_substituting_weights(self, tmp_path: Path):
        # The reason every pin is here at all: a release asset that changed
        # underneath us must stop the script, not land quietly in models/.
        assets, env = _fake_release(tmp_path)
        prefix = _fetched_prefixes()[0]
        (assets / env[f"{prefix}_FILE"]).write_bytes(b"substituted weights")

        result = _run(tmp_path, env)
        assert result.returncode != 0
        assert "sha256 mismatch" in result.stderr

    def test_the_dinov2_backbone_is_fetched_and_pinned(self):
        # #122: the frozen backbone is half of the station classifier and never
        # changes, so it is pinned like YOLO and OSNet rather than resolved
        # through transformers at build time - where a silently different
        # backbone would change every embedding the station head was fitted on.
        prefixes = _fetched_prefixes()
        assert any("DINOV2" in p for p in prefixes), (
            f"setup-models.sh does not fetch the DINOv2 backbone. Fetched: {prefixes}"
        )
        script = SETUP_MODELS_SCRIPT.read_text()
        pin = re.search(r'DINOV2\w*_SHA256="\$\{DINOV2\w*_SHA256:-([0-9a-fA-F]{64})\}"', script)
        assert pin, "the DINOv2 entry has no pinned 64-hex sha256 default"


@pytest.mark.parametrize("prefix", _fetched_prefixes())
def test_every_model_pins_a_real_sha256(prefix: str):
    """No model may ship with a placeholder pin."""
    script = SETUP_MODELS_SCRIPT.read_text()
    pin = re.search(rf'{prefix}_SHA256="\$\{{{prefix}_SHA256:-([0-9a-fA-F]{{64}})\}}"', script)
    assert pin, f"{prefix}_SHA256 has no parseable 64-hex default in setup-models.sh"


class TestTheStationHead:
    """#123: the head and the card production loads are pinned like every model.

    The card is pinned too, and that is not bookkeeping. It carries the rectangle
    the head was fitted on and the measured time ratio quoted beside every total,
    so a card that changed underneath a deployment would move client-facing
    numbers with nothing in the artefact to say so.
    """

    def test_the_head_and_its_card_are_both_fetched(self):
        prefixes = _fetched_prefixes()
        assert any("STATION_HEAD" in p for p in prefixes), (
            f"setup-models.sh does not fetch the station head. Fetched: {prefixes}"
        )
        assert any("STATION_CARD" in p for p in prefixes), (
            "setup-models.sh fetches the station head without its card. The head "
            "cannot be used without one — the card is where the delivered "
            f"vocabulary and the measured time ratios come from. Fetched: {prefixes}"
        )

    def test_the_head_pin_is_the_released_artefact(self):
        # Hard-coded on purpose: this is the digest the model card names as the
        # weights its figures describe, so the two are checked against each other
        # rather than both read from the same file.
        script = SETUP_MODELS_SCRIPT.read_text()
        assert "ef6d6f1edc1996563e7dc05c89685c2032e2cb1d3d503859a85b7e8d237adfaf" in script, (
            "the station head pin is not the sha256 the shipped model card names"
        )

    def test_the_pinned_card_is_the_one_committed_to_the_repo(self):
        """A deployment must not read a different card than this checkout tests.

        `pipeline/station_card_test.py` asserts the shipped card's rectangle and
        vocabulary; if the pinned release asset were a different file, those tests
        would be describing something production never loads.
        """
        script = SETUP_MODELS_SCRIPT.read_text()
        pin = re.search(
            r'STATION_CARD_SHA256="\$\{STATION_CARD_SHA256:-([0-9a-fA-F]{64})\}"', script
        )
        assert pin, "the station card entry has no pinned 64-hex sha256 default"
        committed = REPO_ROOT / "models" / "station-head-hala-prawe-v1-v2.0.0.card.json"
        assert pin.group(1) == hashlib.sha256(committed.read_bytes()).hexdigest()
