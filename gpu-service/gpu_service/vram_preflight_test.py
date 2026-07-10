"""Tests for the VRAM pre-flight check + GPU selection.

Two behaviours under test:

* **Selection** — of the visible GPUs, the one with the most free VRAM wins,
  and its UUID is pinned into ``CUDA_VISIBLE_DEVICES`` so the pipeline's default
  ``cuda:0`` lands there. This is what lets a warm SGLang LLM on GPU 0 coexist
  with CCTV inference on GPU 1.
* **Fail-fast** (issue #43) — when *no* GPU clears the budget (or none exists),
  one structured ``VRAM_PREFLIGHT_FAIL`` line + ``exit(2)`` instead of a
  multi-line PyTorch OOM trace.

The system boundary (``nvidia-smi``) is injected as ``query_gpus`` so tests run
on the macOS CPU dev box with no torch / driver.
"""

from __future__ import annotations

import io

import pytest

from gpu_service.vram_preflight import (
    GpuInfo,
    NoGpuError,
    parse_nvidia_smi,
    preflight_or_exit,
    resolve_required_mb,
    select_best_gpu,
)


class TestResolveRequiredMb:
    def test_env_override_wins_over_classifier_default(self) -> None:
        assert resolve_required_mb(classifier="vlm", env_override="2048") == 2048
        assert resolve_required_mb(classifier="heuristic", env_override="9999") == 9999

    def test_heuristic_classifier_uses_1024_mib_default(self) -> None:
        assert resolve_required_mb(classifier="heuristic", env_override=None) == 1024

    def test_vlm_classifier_uses_7168_mib_default(self) -> None:
        assert resolve_required_mb(classifier="vlm", env_override=None) == 7168

    def test_empty_string_env_override_falls_back_to_classifier_default(self) -> None:
        # Docker's missing-env vs empty-env distinction: `-e VRAM_BUDGET_MB=`
        # gives "". Treat as "not set" — otherwise int("") crashes boot.
        assert resolve_required_mb(classifier="vlm", env_override="") == 7168


class TestParseNvidiaSmi:
    def test_parses_multiple_gpus(self) -> None:
        # Real shape of `--query-gpu=index,uuid,memory.free,memory.total
        # --format=csv,noheader,nounits` on a 2-GPU box.
        csv = "0, GPU-aaaa, 2447, 11774\n1, GPU-bbbb, 11772, 11774\n"
        gpus = parse_nvidia_smi(csv)
        assert gpus == [
            GpuInfo(index=0, uuid="GPU-aaaa", free_mb=2447, total_mb=11774),
            GpuInfo(index=1, uuid="GPU-bbbb", free_mb=11772, total_mb=11774),
        ]

    def test_skips_blank_and_garbled_rows(self) -> None:
        csv = "\n0, GPU-aaaa, 8000, 12000\ngarbage-line\n1, GPU-bbbb, notanint, 12000\n"
        gpus = parse_nvidia_smi(csv)
        assert gpus == [GpuInfo(index=0, uuid="GPU-aaaa", free_mb=8000, total_mb=12000)]

    def test_empty_output_yields_no_gpus(self) -> None:
        assert parse_nvidia_smi("") == []


class TestSelectBestGpu:
    def test_picks_gpu_with_most_free_vram(self) -> None:
        gpus = [
            GpuInfo(index=0, uuid="GPU-aaaa", free_mb=2447, total_mb=11774),
            GpuInfo(index=1, uuid="GPU-bbbb", free_mb=11772, total_mb=11774),
        ]
        assert select_best_gpu(gpus).index == 1

    def test_ties_break_to_lowest_index(self) -> None:
        gpus = [
            GpuInfo(index=1, uuid="GPU-bbbb", free_mb=8000, total_mb=12000),
            GpuInfo(index=0, uuid="GPU-aaaa", free_mb=8000, total_mb=12000),
        ]
        assert select_best_gpu(gpus).index == 0

    def test_no_gpu_raises(self) -> None:
        with pytest.raises(NoGpuError):
            select_best_gpu([])


def _fixed(gpus: list[GpuInfo]):
    return lambda: gpus


class TestPreflightOrExit:
    def test_selects_free_gpu_and_pins_cuda_visible_devices(self) -> None:
        # Source incident layout: SGLang on GPU 0 (2447 MiB free), GPU 1 empty.
        # vlm needs 7168 — must pick GPU 1 and pin its UUID.
        gpus = [
            GpuInfo(index=0, uuid="GPU-sglang", free_mb=2447, total_mb=11774),
            GpuInfo(index=1, uuid="GPU-free", free_mb=11772, total_mb=11774),
        ]
        environ: dict[str, str] = {}
        stderr = io.StringIO()
        exit_calls: list[int] = []

        preflight_or_exit(
            classifier="vlm",
            env_override=None,
            query_gpus=_fixed(gpus),
            environ=environ,
            stderr=stderr,
            exit_fn=exit_calls.append,
        )

        assert exit_calls == []
        assert stderr.getvalue() == ""
        assert environ["CUDA_VISIBLE_DEVICES"] == "GPU-free"

    def test_single_free_gpu_is_pinned(self) -> None:
        gpus = [GpuInfo(index=0, uuid="GPU-only", free_mb=11772, total_mb=11774)]
        environ: dict[str, str] = {}
        preflight_or_exit(
            classifier="vlm",
            env_override=None,
            query_gpus=_fixed(gpus),
            environ=environ,
            stderr=io.StringIO(),
            exit_fn=lambda _c: None,
        )
        assert environ["CUDA_VISIBLE_DEVICES"] == "GPU-only"

    def test_all_gpus_busy_exits_2_with_structured_line(self) -> None:
        # Both GPUs occupied below the vlm budget → fail fast, no CUDA pin.
        gpus = [
            GpuInfo(index=0, uuid="GPU-aaaa", free_mb=2447, total_mb=11774),
            GpuInfo(index=1, uuid="GPU-bbbb", free_mb=139, total_mb=11774),
        ]
        environ: dict[str, str] = {}
        stderr = io.StringIO()
        exit_calls: list[int] = []

        preflight_or_exit(
            classifier="vlm",
            env_override=None,
            query_gpus=_fixed(gpus),
            environ=environ,
            stderr=stderr,
            exit_fn=exit_calls.append,
        )

        assert exit_calls == [2]
        line = stderr.getvalue()
        assert line.startswith("VRAM_PREFLIGHT_FAIL ")
        assert "required=7168" in line
        # Reports the *freest* GPU (the best candidate that still failed).
        assert "free=2447" in line
        assert "nvidia-smi" in line
        assert line.count("\n") == 1  # one line only — no torch traceback
        assert "CUDA_VISIBLE_DEVICES" not in environ

    def test_no_gpu_visible_exits_2(self) -> None:
        environ: dict[str, str] = {}
        stderr = io.StringIO()
        exit_calls: list[int] = []

        preflight_or_exit(
            classifier="vlm",
            env_override=None,
            query_gpus=_fixed([]),
            environ=environ,
            stderr=stderr,
            exit_fn=exit_calls.append,
        )

        assert exit_calls == [2]
        assert stderr.getvalue().startswith("VRAM_PREFLIGHT_FAIL ")
        assert "CUDA_VISIBLE_DEVICES" not in environ

    def test_query_raising_no_gpu_error_exits_2(self) -> None:
        # nvidia-smi unavailable (raises NoGpuError) must fail fast, not crash.
        def boom() -> list[GpuInfo]:
            raise NoGpuError("nvidia-smi unavailable: [Errno 2] No such file")

        stderr = io.StringIO()
        exit_calls: list[int] = []
        preflight_or_exit(
            classifier="vlm",
            env_override=None,
            query_gpus=boom,
            environ={},
            stderr=stderr,
            exit_fn=exit_calls.append,
        )
        assert exit_calls == [2]
        assert stderr.getvalue().startswith("VRAM_PREFLIGHT_FAIL ")

    def test_env_override_relaxes_budget(self) -> None:
        # Operator forces a tight box: VRAM_BUDGET_MB=200 → a 256-MiB-free GPU
        # now passes and gets pinned.
        gpus = [GpuInfo(index=0, uuid="GPU-tight", free_mb=256, total_mb=12000)]
        environ: dict[str, str] = {}
        exit_calls: list[int] = []
        preflight_or_exit(
            classifier="vlm",
            env_override="200",
            query_gpus=_fixed(gpus),
            environ=environ,
            stderr=io.StringIO(),
            exit_fn=exit_calls.append,
        )
        assert exit_calls == []
        assert environ["CUDA_VISIBLE_DEVICES"] == "GPU-tight"
