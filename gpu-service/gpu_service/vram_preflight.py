"""Pre-flight VRAM check + GPU selection for the gpu-service container.

Two jobs, run once at container startup before any model loads:

1. **Select** the visible GPU with the most free VRAM and pin every downstream
   CUDA consumer to it via ``CUDA_VISIBLE_DEVICES=<uuid>``. This lets the box
   host a warm SGLang LLM on one GPU while CCTV inference lands on the free one
   — no static per-GPU assignment, the container adapts to whatever's free.
2. **Fail fast** (issue #43) if *no* GPU has enough free VRAM: one structured
   ``VRAM_PREFLIGHT_FAIL …`` stderr line + ``exit(2)`` instead of a mid-load
   PyTorch OOM traceback that masks the real cause (another CUDA process).

The free-VRAM query goes through ``nvidia-smi`` (NVML), *not* torch — so no
CUDA context is created here. That's what makes the ``CUDA_VISIBLE_DEVICES``
pin effective: it is set before torch/onnxruntime first initialise CUDA, so
their default ``cuda:0`` / ``device_id=0`` transparently maps to the chosen
physical GPU. UUID (not index) is used so the pin is immune to device ordering.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import IO

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024


class InsufficientVRAMError(Exception):
    """No visible GPU has enough free VRAM for the configured budget."""


class NoGpuError(Exception):
    """nvidia-smi reported no usable GPU (or is unavailable)."""


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    free_mb: int
    total_mb: int


# Per-classifier VRAM budgets (MiB). Source: warm-load occupancy on a clean
# GPU + ~20% headroom for KV cache / allocator fragmentation. Tune by setting
# VRAM_BUDGET_MB at runtime.
DEFAULT_BUDGETS_MB: dict[str, int] = {
    "heuristic": 1024,
    # Issue #34 same-image Film 1 peak: 540 MiB. Rounded to 768 MiB for
    # allocator/driver variation; raw 0.5-second samples are committed.
    "mlp": 768,
    "vlm": 7168,
}

QueryGpus = Callable[[], list[GpuInfo]]


def resolve_required_mb(classifier: str, env_override: str | None) -> int:
    """Decide the budget for this run: env var beats classifier default."""
    if env_override:
        return int(env_override)
    return DEFAULT_BUDGETS_MB[classifier]


def parse_nvidia_smi(csv_text: str) -> list[GpuInfo]:
    """Parse ``index,uuid,memory.free,memory.total`` CSV rows (nounits → MiB).

    Blank lines and short/garbled rows are skipped rather than raising — a
    partial parse still lets ``select_best_gpu`` pick from the usable GPUs.
    """
    gpus: list[GpuInfo] = []
    for raw in csv_text.strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    uuid=parts[1],
                    free_mb=int(parts[2]),
                    total_mb=int(parts[3]),
                )
            )
        except ValueError:
            continue
    return gpus


def _default_query_gpus() -> list[GpuInfo]:
    """Query every visible GPU's free VRAM via nvidia-smi (no CUDA context)."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NoGpuError(f"nvidia-smi unavailable: {exc}") from exc
    return parse_nvidia_smi(completed.stdout)


def select_best_gpu(gpus: list[GpuInfo]) -> GpuInfo:
    """The GPU with the most free VRAM; ties break to the lowest index."""
    if not gpus:
        raise NoGpuError("nvidia-smi reported no GPU")
    return max(gpus, key=lambda g: (g.free_mb, -g.index))


def preflight_or_exit(
    classifier: str,
    env_override: str | None,
    query_gpus: QueryGpus | None = None,
    environ: MutableMapping[str, str] | None = None,
    stderr: IO[str] = sys.stderr,
    exit_fn: Callable[[int], None] = sys.exit,
) -> None:
    """Select the freest GPU and pin CUDA to it, or fail fast.

    On success: sets ``CUDA_VISIBLE_DEVICES`` to the chosen GPU's UUID and
    returns (no stderr output). On no-GPU or insufficient-VRAM: writes one
    ``VRAM_PREFLIGHT_FAIL …`` line to ``stderr`` and calls ``exit_fn(2)``.
    ``query_gpus``, ``environ``, and ``exit_fn`` are injectable for tests.
    """
    required = resolve_required_mb(classifier, env_override)
    if query_gpus is None:
        query_gpus = _default_query_gpus
    if environ is None:
        environ = os.environ

    try:
        gpus = query_gpus()
        best = select_best_gpu(gpus)
        if best.free_mb < required:
            raise InsufficientVRAMError(
                f"insufficient VRAM: required={required} MiB; freest of {len(gpus)} "
                f"GPU(s) is GPU {best.index} with free={best.free_mb} MiB, "
                f"total={best.total_mb} MiB. "
                f"Another CUDA process is likely consuming VRAM; "
                f"nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
            )
    except (InsufficientVRAMError, NoGpuError) as exc:
        print(f"VRAM_PREFLIGHT_FAIL {exc}", file=stderr, flush=True)
        exit_fn(2)
        return

    # Pin every downstream CUDA consumer (torch cuda:0, onnxruntime device_id=0)
    # to the chosen GPU. Set BEFORE the first torch import initialises CUDA —
    # rest_server/worker call this before _warm_up_pipeline, so the pin holds.
    environ["CUDA_VISIBLE_DEVICES"] = best.uuid
    logger.info(
        "VRAM preflight: selected GPU %d (%s) free=%d MiB of %d MiB "
        "for classifier=%s (required=%d MiB)",
        best.index,
        best.uuid,
        best.free_mb,
        best.total_mb,
        classifier,
        required,
    )
