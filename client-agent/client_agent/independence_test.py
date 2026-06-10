"""Package-independence invariant: client_agent does not import gpu_service.

Issue #40 carves the appliance off the gpu_service module so the installer
no longer needs to copy ``gpu-service/gpu_service`` into the appliance's
site-packages. That promise only holds if no module in ``client_agent/``
reaches into ``gpu_service.*`` at import time — a single ``from
gpu_service.x import Y`` at module level reintroduces the dead-weight dep
and the appliance will fail to boot with ``ModuleNotFoundError``.

We parse each file's AST (not its raw text) so docstring mentions of
``gpu_service`` don't trip the check; only real ``import`` / ``from ... import``
statements count.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


def _python_modules() -> list[Path]:
    """Every .py file under client_agent/ (test files included — they ship
    in the source tree the installer copies, so a stray import there would
    still pull gpu_service onto the appliance)."""
    return sorted(p for p in PKG_DIR.glob("*.py") if p.name != "__pycache__")


def _gpu_service_imports(path: Path) -> list[str]:
    """Return human-readable descriptions of any gpu_service imports in
    ``path``. Empty list means the file is clean."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "gpu_service" or alias.name.startswith("gpu_service."):
                    hits.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "gpu_service" or module.startswith("gpu_service."):
                names = ", ".join(a.name for a in node.names)
                hits.append(f"line {node.lineno}: from {module} import {names}")
    return hits


def test_no_module_in_client_agent_imports_gpu_service() -> None:
    """Walks every .py file under client_agent/ and asserts none of them
    has an ``import gpu_service`` or ``from gpu_service ... import ...``
    statement. The appliance installer drops only ``client_agent`` into
    site-packages; any surviving gpu_service import would crash boot."""
    offenders: dict[str, list[str]] = {}
    for path in _python_modules():
        hits = _gpu_service_imports(path)
        if hits:
            offenders[path.name] = hits

    assert not offenders, (
        "client_agent must not import from gpu_service (issue #40). "
        "Found:\n" + "\n".join(f"  {fname}: {', '.join(hits)}" for fname, hits in offenders.items())
    )
