"""Tests for the operator-facing env example files.

The example files are the contract between the appliance package and the
operator: every key the running service reads via env must appear here, or
the operator will hit confusing runtime errors instead of "edit this line".
We parse them like systemd's EnvironmentFile=: ``KEY=VALUE`` per line, plus
``#`` comments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CAMERAS_EXAMPLE = ROOT / "cameras.env.example"
PLATFORM_EXAMPLE = ROOT / "platform.env.example"


def _keys(path: Path) -> set[str]:
    """Return the set of KEY tokens declared in a systemd EnvironmentFile-style
    file. Ignores blanks and ``#`` comments. Unlike the appliance loader we do
    not care about values here — only that the key is *mentioned* (so the
    operator knows it exists)."""
    keys: set[str] = set()
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, _ = stripped.partition("=")
        if sep:
            keys.add(key.strip())
    return keys


# NOTE: ``r2.env.example`` was retired in #29 — the appliance no longer uses
# R2 credentials (uploads go through presigned URLs). Only cameras.env and
# platform.env remain operator-facing.


# --- cameras.env.example ----------------------------------------------------


def test_cameras_example_exists() -> None:
    assert CAMERAS_EXAMPLE.is_file(), CAMERAS_EXAMPLE


def test_cameras_example_has_default_user_pass() -> None:
    """``RTSP_DEFAULT_USER`` + ``RTSP_DEFAULT_PASS`` are the fallback the
    credentials resolver checks last (discovery.py). Without them an operator
    with a single shared cred hits a 400 on every recording."""
    keys = _keys(CAMERAS_EXAMPLE)
    assert "RTSP_DEFAULT_USER" in keys
    assert "RTSP_DEFAULT_PASS" in keys


def test_cameras_example_documents_per_ip_override() -> None:
    """The per-IP override is the only non-obvious part of the contract — the
    sanitization rule (dots → underscores from ``_ip_to_env_suffix``) cannot
    be inferred from the variable name alone, so the example file must show
    one concrete instance. We accept the example commented out (the desired
    UX — opt-in, not auto-applied) by scanning the raw text rather than the
    parsed key set."""
    import re

    text = CAMERAS_EXAMPLE.read_text()
    pattern = re.compile(r"RTSP_CAM_(\d+_\d+_\d+_\d+)_(USER|PASS)\s*=", re.MULTILINE)
    matches = pattern.findall(text)
    assert matches, "expected at least one RTSP_CAM_<sanitized_ip>_USER/_PASS line (commented OK)"
    # The original dotted IP must be in a comment so the operator can map back.
    assert re.search(r"\b\d+\.\d+\.\d+\.\d+\b", text), "include a sample dotted IP in comments"


# --- platform.env.example (issue #30) --------------------------------------


def test_platform_example_exists() -> None:
    assert PLATFORM_EXAMPLE.is_file(), PLATFORM_EXAMPLE


@pytest.mark.parametrize(
    "key",
    [
        "PLATFORM_URL",
        "APPLIANCE_TOKEN",
        "BUFFER_HOURS",
    ],
)
def test_platform_example_has_key(key: str) -> None:
    """The three keys the appliance reads in platform mode:
    ``PLATFORM_URL`` + ``APPLIANCE_TOKEN`` toggle platform mode (DD-09),
    ``BUFFER_HOURS`` controls the rolling-buffer retention window. All
    three must surface in the example so the operator copying the file
    does not need to consult external docs to know what to set."""
    keys: set[str] = set()
    for line in PLATFORM_EXAMPLE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        candidate, sep, _ = stripped.partition("=")
        if sep:
            keys.add(candidate.strip())
    assert key in keys, f"{key} missing from platform.env.example (active KEY=VALUE lines)"


def test_platform_example_buffer_hours_default_is_one() -> None:
    """1-hour default matches the dev/MVP guidance. Production overrides
    to 8+ via the same file, but the example file ships with the safe
    default so a fresh install boots without the operator having to make
    a sizing decision before first start."""
    text = PLATFORM_EXAMPLE.read_text()
    import re

    match = re.search(r"^\s*BUFFER_HOURS\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    assert match, "BUFFER_HOURS must be set to a literal integer in the example"
    assert match.group(1) == "1", "default BUFFER_HOURS in the example should be 1"


def test_platform_example_mentions_production_buffer_hours_recommendation() -> None:
    """The 8h-in-prod recommendation lives in comments, not as an active
    value (we don't want to surprise dev operators with 8h retention on
    first install). Asserting via grep keeps the README and the example
    in sync — drift would mean the operator never learns about the prod
    knob."""
    text = PLATFORM_EXAMPLE.read_text().lower()
    assert "8" in text and "hour" in text, (
        "platform.env.example should mention the 8h+ production guidance in a comment"
    )
