"""What version of the appliance is actually on this box.

The appliance used to report ``agent_version="0.5.0"`` — a literal default
argument tracking nothing, unchanged across real code changes. This module
replaces it with three independently-failing facts:

    commit       git SHA the installer copied from
    dirty        was that worktree clean at install time
    modified     has site-packages changed SINCE the install

``modified`` is the one that catches a deploy which bypassed the installer
(``tar``/``scp`` straight into site-packages). In that case ``commit`` stays
frozen at the last real install while different code runs — a stale SHA that
looks perfectly healthy. Detecting it needs a content hash, not a version
string.

The hash algorithm lives here, in the *package*, so the installer and the
runtime cannot disagree: the installer calls this module through the venv's
own python rather than reimplementing the digest in bash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Written by install.sh / install-user.sh into the installed package.
BUILD_INFO_MODULE = "_build_info.py"

# Excluded from the content hash:
#   _build_info.py  — written after the hash is computed, so including it
#                     would make the recorded value unreproducible and flip
#                     every box to "modified" on first run.
#   __pycache__     — bytecode appears on first import; counting it would do
#                     the same thing a few seconds later.
_HASH_EXCLUDED_NAMES = frozenset({BUILD_INFO_MODULE, "__pycache__"})
_HASH_EXCLUDED_SUFFIXES = (".pyc", ".pyo")


@dataclass(frozen=True)
class BuildState:
    """Install identity of the running package.

    ``modified`` is deliberately tri-state. ``None`` means "no install record,
    cannot tell" — running from a checkout, or from a legacy hand-rolled
    launcher. It must never collapse to ``False``: claiming an unmodified build
    without a reference to compare against is exactly the confident-but-wrong
    signal this module exists to remove.
    """

    commit: str | None = None
    dirty: bool | None = None
    installed_at: str | None = None
    modified: bool | None = None


def _hashable_files(package_dir: Path) -> list[Path]:
    files = []
    for path in package_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in _HASH_EXCLUDED_SUFFIXES:
            continue
        if any(part in _HASH_EXCLUDED_NAMES for part in path.relative_to(package_dir).parts):
            continue
        files.append(path)
    # Sorted by POSIX relative path so the digest does not depend on filesystem
    # iteration order, which differs between machines and filesystems.
    return sorted(files, key=lambda p: p.relative_to(package_dir).as_posix())


def compute_content_hash(package_dir: Path | None = None) -> str:
    """SHA-256 over the package's file *paths and contents*.

    Paths are folded in, not just bytes: renaming a module without editing it
    still changes the build, and a bytes-only digest would call that identical.
    """
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in _hashable_files(root):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_build_info(package_dir: Path) -> dict[str, object] | None:
    """Parse the generated install record.

    Executed rather than imported so the caller controls which directory is
    inspected (tests, and the installer verifying what it just wrote). Any
    malformation degrades to ``None``: this runs at boot, before the platform
    session, so raising here would stop the box registering at all — a
    cosmetic field must never be able to do that.
    """
    source = package_dir / BUILD_INFO_MODULE
    if not source.is_file():
        return None
    namespace: dict[str, object] = {}
    try:
        exec(compile(source.read_text(), str(source), "exec"), {}, namespace)  # noqa: S102
    except Exception:
        return None
    return namespace


def resolve_build_state(package_dir: Path | None = None) -> BuildState:
    """Identity of the installed package, safe to call at boot."""
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    info = _read_build_info(root)
    if info is None:
        return BuildState()

    commit = info.get("COMMIT")
    recorded_hash = info.get("CONTENT_HASH")
    dirty = info.get("DIRTY")
    installed_at = info.get("INSTALLED_AT")

    modified: bool | None = None
    if isinstance(recorded_hash, str) and recorded_hash:
        try:
            modified = compute_content_hash(root) != recorded_hash
        except OSError:
            modified = None

    return BuildState(
        commit=commit if isinstance(commit, str) else None,
        dirty=dirty if isinstance(dirty, bool) else None,
        installed_at=installed_at if isinstance(installed_at, str) else None,
        modified=modified,
    )


def build_payload(package_dir: Path | None = None) -> dict[str, object]:
    """Wire shape for the register ``host_info.build`` block.

    Rides in ``host_info`` (an existing jsonb column the platform refreshes on
    every register) rather than in new columns — the appliance re-registers
    each cycle, so tampering surfaces within a cycle with no schema change.

    ``None`` values are kept, not stripped: the platform must be able to tell
    "unknown" from "false". A box with no install record has to render as
    unknown, never as a confident green.
    """
    state = resolve_build_state(package_dir)
    return {
        "commit": state.commit,
        "dirty": state.dirty,
        "installed_at": state.installed_at,
        "modified": state.modified,
    }
