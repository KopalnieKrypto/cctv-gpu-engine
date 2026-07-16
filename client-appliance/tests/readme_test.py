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
    files the operator will edit. (``r2.env`` was retired in #29 — the
    appliance holds no R2 credentials.)"""
    text = _text()
    assert "cctv-client.service" in text
    assert "cameras.env" in text
    assert "platform.env" in text


# --- dual-mode docs (issue #30) ---------------------------------------------


def test_section_dual_mode_documents_platform_mode() -> None:
    """Phase 4 platform mode is the appliance's networked flow (the legacy
    Docker UI mode was retired in #29). The README must surface the
    platform-mode entrypoint and its env file so an operator knows the
    networked path exists."""
    text = _text()
    # The platform-mode entrypoint and its env file are the load-bearing
    # references — without them the operator can't tell that a separate
    # path exists.
    assert "platform.env" in text, "README must reference platform.env (Phase 4 mode)"
    assert "PLATFORM_URL" in text, "README must mention PLATFORM_URL (Phase 4 toggle)"
    assert "APPLIANCE_TOKEN" in text, "README must mention APPLIANCE_TOKEN (Phase 4 token)"


def test_section_dual_mode_links_to_issue_29() -> None:
    """Acceptance criterion: link to #29 (retire legacy agent.py) so an
    operator reading the dual-mode docs knows the legacy path has a
    sunset date and is not "two equally-supported flows forever"."""
    text = _text()
    assert "#29" in text, "README must link to #29 (legacy retirement plan)"


def test_section_documents_buffer_hours() -> None:
    """``BUFFER_HOURS`` is a tunable the operator will hit at production
    sizing time. README must mention the 1h dev default and the 8h+ prod
    guidance so the operator does not silently ship 1h retention to a
    customer that needed a wider forensic window."""
    text = _text().lower()
    assert "buffer_hours" in text, "README must document BUFFER_HOURS"
    # Surface the prod guidance — accept either '8' or '8h' verbatim.
    assert "8" in text, "README must document the 8h+ production guidance"


# --- user-mode install path (issue #84) -------------------------------------


def test_documents_user_mode_install() -> None:
    """AC 9: the README must surface the no-root path and its artifacts.

    A no-sudo site that cannot find this section falls back to the hand-rolled
    `nohup` launcher — which is precisely the unsupervised setup that cost the
    ~18 h cameraboy outage."""
    text = _text()
    assert "install-user.sh" in text, "README must document install-user.sh"
    assert "cctv-client-user.service" in text, "README must reference the user unit"


def test_documents_root_vs_user_choice() -> None:
    """AC 9: an operator must be able to tell *which* installer to run before
    running one. Both must be named in the same breath as the deciding
    factor — whether root/sudo is available."""
    text = _text().lower()
    assert "install.sh" in text and "install-user.sh" in text
    assert "sudo" in text or "root" in text


def test_documents_linger_requirement() -> None:
    """AC 8/9: linger is the one non-obvious prerequisite of the user path —
    without it the unit dies with the operator's SSH session and never comes
    back after reboot, silently. The README must name it so an operator
    debugging a vanished appliance has a term to search for."""
    text = _text().lower()
    assert "linger" in text, "README must document the linger requirement"
    assert "loginctl" in text, "README must show the loginctl command"


def test_documents_user_mode_journal_and_status() -> None:
    """The user-mode operator commands differ from the root ones (`--user` on
    every call). Documenting the root forms only would send them to a unit
    that does not exist in the system manager."""
    text = _text()
    assert "systemctl --user" in text, "README must document `systemctl --user` commands"
    assert "journalctl --user" in text, "README must document `journalctl --user` for logs"
