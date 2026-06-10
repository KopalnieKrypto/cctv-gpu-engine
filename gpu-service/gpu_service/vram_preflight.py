"""Pre-flight VRAM check for the gpu-service container (issue #43).

Operators previously saw mid-load PyTorch OOM tracebacks when another CUDA
process held most of the GPU. This module turns that into a one-line,
structured fail-fast at container startup.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import IO

MemGetInfo = Callable[[], tuple[int, int]]
_MIB = 1024 * 1024


class InsufficientVRAMError(Exception):
    """Free VRAM is below the configured budget for this classifier."""


# Per-classifier VRAM budgets (MiB). Source: warm-load occupancy on a clean
# GPU + ~20% headroom for KV cache / allocator fragmentation. Tune by setting
# VRAM_BUDGET_MB at runtime.
DEFAULT_BUDGETS_MB: dict[str, int] = {
    "heuristic": 1024,
    "vlm": 7168,
}


def resolve_required_mb(classifier: str, env_override: str | None) -> int:
    """Decide the budget for this run: env var beats classifier default."""
    if env_override:
        return int(env_override)
    return DEFAULT_BUDGETS_MB[classifier]


def _default_mem_get_info() -> tuple[int, int]:
    """Lazy-import torch.cuda so unit tests don't pull onnxruntime / cu128."""
    import torch

    return torch.cuda.mem_get_info(0)


def preflight_or_exit(
    classifier: str,
    env_override: str | None,
    mem_get_info: MemGetInfo | None = None,
    stderr: IO[str] = sys.stderr,
    exit_fn: Callable[[int], None] = sys.exit,
) -> None:
    """Run the VRAM check at container entrypoint.

    On insufficient VRAM: write one ``VRAM_PREFLIGHT_FAIL …`` line to stderr
    (flushed, single line so gpu-agent's failure capture stays clean) and call
    ``exit_fn(2)``. On success: no output, no exit — caller proceeds to load
    models. Both ``mem_get_info`` and ``exit_fn`` are injectable for tests.
    """
    required = resolve_required_mb(classifier, env_override)
    if mem_get_info is None:
        mem_get_info = _default_mem_get_info
    try:
        check_vram_budget(required, mem_get_info)
    except InsufficientVRAMError as exc:
        print(f"VRAM_PREFLIGHT_FAIL {exc}", file=stderr, flush=True)
        exit_fn(2)


def check_vram_budget(required_mb: int, mem_get_info: MemGetInfo) -> None:
    free_bytes, total_bytes = mem_get_info()
    free_mb = free_bytes // _MIB
    total_mb = total_bytes // _MIB
    if free_mb < required_mb:
        raise InsufficientVRAMError(
            f"insufficient VRAM on GPU 0: required={required_mb} MiB, "
            f"free={free_mb} MiB, total={total_mb} MiB. "
            f"Another CUDA process is likely consuming VRAM; "
            f"nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
        )
