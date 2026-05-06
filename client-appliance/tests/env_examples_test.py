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
R2_EXAMPLE = ROOT / "r2.env.example"
CAMERAS_EXAMPLE = ROOT / "cameras.env.example"


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


# --- r2.env.example ---------------------------------------------------------


def test_r2_example_exists() -> None:
    assert R2_EXAMPLE.is_file(), R2_EXAMPLE


@pytest.mark.parametrize(
    "key",
    [
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
    ],
)
def test_r2_example_has_key(key: str) -> None:
    """All four R2 keys ``client_agent.agent.build_app`` reads must be in the
    example. Three are required (no default → raises KeyError); ``R2_BUCKET``
    has a default of ``surveillance-data`` but we still surface it so the
    operator knows the override exists."""
    assert key in _keys(R2_EXAMPLE)


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
