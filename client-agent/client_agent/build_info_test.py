"""Tests for build_info — what version is actually on this box.

Before this, the appliance reported ``agent_version="0.5.0"``: a literal
default argument that tracked nothing. It stayed identical across a real code
change on 2026-07-20, so the admin UI showed an authoritative-looking value
that could not answer "is this box running the newest code".

Three questions have to stay separable, because they fail independently:

    commit    — what was installed
    dirty     — was the checkout clean when it was installed
    modified  — has anything changed on disk SINCE the install

The third exists because a ``tar``/``scp`` straight into site-packages
bypasses the installer, leaving ``commit`` frozen at the last real install
while different code actually runs. That happened on cameraboy the same day.
A commit field alone would have kept reporting the stale SHA.
"""

from __future__ import annotations

from pathlib import Path

from client_agent.build_info import compute_content_hash, resolve_build_state


def _pkg(tmp_path: Path, files: dict[str, str]) -> Path:
    pkg = tmp_path / "client_agent"
    pkg.mkdir(exist_ok=True)
    for name, body in files.items():
        target = pkg / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return pkg


def test_content_hash_is_deterministic(tmp_path: Path) -> None:
    pkg = _pkg(tmp_path, {"a.py": "x = 1", "b.py": "y = 2"})
    assert compute_content_hash(pkg) == compute_content_hash(pkg)


def test_content_hash_changes_when_a_file_changes(tmp_path: Path) -> None:
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    before = compute_content_hash(pkg)
    (pkg / "a.py").write_text("x = 2")
    assert compute_content_hash(pkg) != before


def test_content_hash_changes_when_a_file_is_added(tmp_path: Path) -> None:
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    before = compute_content_hash(pkg)
    (pkg / "c.py").write_text("z = 3")
    assert compute_content_hash(pkg) != before


def test_content_hash_covers_paths_not_just_bytes(tmp_path: Path) -> None:
    """A rename with identical content must still change the hash — otherwise
    swapping two modules' names reads as an untouched build."""
    pkg = _pkg(tmp_path, {"a.py": "same"})
    before = compute_content_hash(pkg)
    (pkg / "a.py").rename(pkg / "renamed.py")
    assert compute_content_hash(pkg) != before


def test_content_hash_ignores_generated_build_info(tmp_path: Path) -> None:
    """``_build_info.py`` is written *after* the hash is computed, so including
    it would make the recorded hash unreproducible at runtime — every box would
    report itself modified the moment it started."""
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    before = compute_content_hash(pkg)
    (pkg / "_build_info.py").write_text('COMMIT = "abc"\n')
    assert compute_content_hash(pkg) == before


def test_content_hash_ignores_pycache(tmp_path: Path) -> None:
    """Bytecode appears on first import. Counting it would flip every box to
    "modified" seconds after starting."""
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    before = compute_content_hash(pkg)
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00\x01")
    assert compute_content_hash(pkg) == before


def test_build_state_without_build_info_reports_unknown(tmp_path: Path) -> None:
    """Running from a checkout (dev, or a legacy hand-rolled launcher) has no
    install record. ``modified`` must be None — "cannot tell" — never False.
    Reporting a confident "unmodified" with no reference is the false green
    this whole feature exists to remove."""
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    state = resolve_build_state(pkg)
    assert state.commit is None
    assert state.modified is None
    assert state.installed_at is None


def test_build_state_matches_when_untouched(tmp_path: Path) -> None:
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    recorded = compute_content_hash(pkg)
    (pkg / "_build_info.py").write_text(
        f'COMMIT = "cd87eb8"\nDIRTY = False\n'
        f'INSTALLED_AT = "2026-07-20T09:47:12Z"\nCONTENT_HASH = "{recorded}"\n'
    )
    state = resolve_build_state(pkg)
    assert state.commit == "cd87eb8"
    assert state.dirty is False
    assert state.installed_at == "2026-07-20T09:47:12Z"
    assert state.modified is False


def test_build_state_flags_modification_after_install(tmp_path: Path) -> None:
    """The cameraboy tar-deploy signature: install record intact, contents
    different. commit still reads cd87eb8 and would look perfectly healthy."""
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    recorded = compute_content_hash(pkg)
    (pkg / "_build_info.py").write_text(
        f'COMMIT = "cd87eb8"\nDIRTY = False\n'
        f'INSTALLED_AT = "2026-07-20T09:47:12Z"\nCONTENT_HASH = "{recorded}"\n'
    )
    (pkg / "a.py").write_text("x = 999")  # hand-patched after install

    state = resolve_build_state(pkg)
    assert state.commit == "cd87eb8", "commit still reports the last real install"
    assert state.modified is True, "but the build must be flagged as modified"


def test_build_state_survives_a_corrupt_build_info(tmp_path: Path) -> None:
    """A malformed install record must not crash the appliance at boot — it
    runs before the platform session, so an exception here means the box never
    registers at all. Degrade to "unknown", which is honest."""
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    (pkg / "_build_info.py").write_text("COMMIT = (((\n")
    state = resolve_build_state(pkg)
    assert state.commit is None
    assert state.modified is None


def test_parses_the_exact_file_shape_the_installers_emit(tmp_path: Path) -> None:
    """Pins the contract between the installers' bash heredoc and this parser.

    They are written in different languages and only meet at this file, so the
    literal shapes matter: bash emits Python's capital-F ``False`` (not shell's
    lowercase), a UTF-8 comment header, and an ISO-8601 Z timestamp from
    ``date -u``. Any drift degrades silently to "unknown" — the box keeps
    running and simply stops reporting its identity, which is precisely the
    blind spot this replaced.
    """
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    recorded = compute_content_hash(pkg)
    # Byte-for-byte the heredoc in install.sh / install-user.sh.
    (pkg / "_build_info.py").write_text(
        "# Generated by install.sh — do not edit.\n"
        "# Hand-editing this file makes the appliance lie about what it runs.\n"
        'COMMIT = "5b43d72f69b2bb607a1d86eb31070b39c3e11c34"\n'
        "DIRTY = False\n"
        'INSTALLED_AT = "2026-07-20T10:02:10Z"\n'
        f'CONTENT_HASH = "{recorded}"\n'
    )

    state = resolve_build_state(pkg)
    assert state.commit == "5b43d72f69b2bb607a1d86eb31070b39c3e11c34"
    assert state.dirty is False
    assert state.installed_at == "2026-07-20T10:02:10Z"
    assert state.modified is False


def test_dirty_true_from_the_installer_heredoc_parses(tmp_path: Path) -> None:
    """The bash branch writes bare ``True``. Python-truthy, but only because
    the installers deliberately emit Python literals rather than shell ones —
    a lowercase ``true`` would raise NameError and lose the whole record."""
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    (pkg / "_build_info.py").write_text(
        'COMMIT = "abc1234"\nDIRTY = True\nINSTALLED_AT = "2026-07-20T10:02:10Z"\n'
        'CONTENT_HASH = "deadbeef"\n'
    )
    state = resolve_build_state(pkg)
    assert state.dirty is True
    assert state.modified is True, "stale hash must read as modified"


def test_dirty_checkout_is_carried_through(tmp_path: Path) -> None:
    """Installing from a dirty worktree means the SHA does not fully describe
    what was installed — the operator has to know that."""
    pkg = _pkg(tmp_path, {"a.py": "x = 1"})
    recorded = compute_content_hash(pkg)
    (pkg / "_build_info.py").write_text(
        f'COMMIT = "cd87eb8"\nDIRTY = True\n'
        f'INSTALLED_AT = "2026-07-20T09:47:12Z"\nCONTENT_HASH = "{recorded}"\n'
    )
    assert resolve_build_state(pkg).dirty is True
