"""Shared parsing helpers for the systemd unit files we ship.

Two units now exist — ``cctv-client.service`` (root) and
``cctv-client-user.service`` (user-mode, issue #84) — and both are asserted
directive-by-directive. Keeping one parser means a fix here (e.g. handling a
new systemd quirk) lands for both, rather than in whichever copy the next
contributor happens to open.

Not collected by pytest: ``python_files = ["*_test.py"]`` in pyproject.toml.
"""

from __future__ import annotations

import configparser
from pathlib import Path


def parse_unit(path: Path) -> configparser.ConfigParser:
    """Parse a unit file into a ConfigParser.

    ``strict=False`` so duplicate keys (systemd allows repeated
    ``EnvironmentFile=``) do not raise; ``interpolation=None`` so systemd's
    ``%h``/``%i`` specifiers survive rather than being read as configparser
    interpolation syntax.
    """
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # preserve case (systemd directives are CamelCase)
    parser.read_string(path.read_text())
    return parser


def directive_values(path: Path, section: str, key: str) -> list[str]:
    """Return *every* value for a directive in ``section``.

    configparser with ``strict=False`` accepts duplicate keys but silently
    keeps only the last — and for ``EnvironmentFile=``, which systemd
    deliberately allows to repeat, "only the last one" is exactly the bug we
    would fail to catch. So we scan the raw text instead.
    """
    in_section = False
    values: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == f"[{section}]"
            continue
        if not in_section or not stripped or stripped.startswith("#"):
            continue
        k, sep, v = stripped.partition("=")
        if sep and k.strip() == key:
            values.append(v.strip())
    return values
