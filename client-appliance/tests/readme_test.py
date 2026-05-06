"""Tests for client-appliance/README.md.

The README is the operator's only documentation surface for the appliance —
acceptance criteria call out specific sections it must contain. Rather than
test exact wording (brittle), we assert the section headers and the
keywords that anchor each required topic.
"""

from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def _text() -> str:
    return README.read_text()


def test_readme_exists() -> None:
    assert README.is_file(), README


def test_section_wymagania() -> None:
    """Hardware requirements section — links to the Pi 5 / N100 table from
    the plan. Title can be ``Wymagania`` or ``Wymagania sprzętowe``."""
    assert "## Wymagania" in _text()


def test_section_instalacja() -> None:
    """Step-by-step install — must mention ``install.sh`` and the git clone
    path the operator follows."""
    text = _text()
    assert "## Instalacja" in text
    assert "install.sh" in text
    assert "git clone" in text


def test_section_update() -> None:
    """Update procedure — git pull + re-running install.sh covers it, but
    the README must spell that out so the operator does not re-clone."""
    text = _text()
    assert "## Update" in text or "## Aktualizacja" in text
    assert "git pull" in text


def test_section_troubleshooting_multicast() -> None:
    """ONVIF discovery uses multicast — first thing that fails on a router
    that filters or VLAN-isolates 239.255.255.250:3702. Document it."""
    text = _text()
    assert "## Troubleshooting" in text or "## Problemy" in text
    assert "multicast" in text.lower() or "239.255.255.250" in text


def test_section_troubleshooting_firewall() -> None:
    """UI on :8080 — UFW or nftables can block it on a fresh Ubuntu install.
    Acceptance criteria list firewall as a documented troubleshooting case."""
    text = _text().lower()
    assert "firewall" in text or "ufw" in text or "8080" in text


def test_section_smoke_test_runbook() -> None:
    """5-minute smoke test from tar/clone to UI in LAN — operators use this
    to verify the install on a fresh box."""
    text = _text().lower()
    assert "smoke" in text or "test e2e" in text or "weryfikacja" in text


def test_documents_tarball_vs_git_decision() -> None:
    """The plan mandates the architectural decision (tarball vs git clone)
    is captured in the README with a rationale — otherwise a future
    contributor cannot judge whether to flip it."""
    text = _text().lower()
    assert "tarball" in text and "git clone" in text
    # Some marker that this is documented as a decision, not just mentioned
    # in passing — accept any of: "decyzja", "rationale", "wybór", "ADR".
    assert any(marker in text for marker in ("decyzja", "rationale", "wybór", "adr"))


def test_links_to_unit_file_and_envs() -> None:
    """Cross-references inside the package directory — README mentions the
    files the operator will edit."""
    text = _text()
    assert "cctv-client.service" in text
    assert "r2.env" in text
    assert "cameras.env" in text
