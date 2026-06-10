"""Tests for the VRAM pre-flight check (issue #43).

Goal: replace the multi-line PyTorch OOM trace operators currently see when
another CUDA process holds VRAM with a single, structured ``VRAM_PREFLIGHT_FAIL``
line and a hard ``exit(2)`` before any model load is attempted.

Tests inject ``mem_get_info`` (the only system boundary) so they run on the
macOS CPU dev box without onnxruntime / torch.cuda.
"""

from __future__ import annotations

import io

import pytest

from gpu_service.vram_preflight import (
    InsufficientVRAMError,
    check_vram_budget,
    preflight_or_exit,
    resolve_required_mb,
)

MIB = 1024 * 1024


class TestCheckVramBudget:
    def test_raises_when_free_below_required(self) -> None:
        # The exact numbers from the source incident: 139 MiB free of 12 GiB,
        # required budget for VLM = 7168 MiB.
        free_bytes = 139 * MIB
        total_bytes = 12 * 1024 * MIB

        with pytest.raises(InsufficientVRAMError) as exc_info:
            check_vram_budget(
                required_mb=7168,
                mem_get_info=lambda: (free_bytes, total_bytes),
            )

        msg = str(exc_info.value)
        assert "required=7168" in msg
        assert "free=139" in msg
        assert "total=12288" in msg

    def test_passes_silently_when_free_above_required(self) -> None:
        # 8 GiB free, 12 GiB total, required=7168 — typical clean-GPU case.
        result = check_vram_budget(
            required_mb=7168,
            mem_get_info=lambda: (8 * 1024 * MIB, 12 * 1024 * MIB),
        )
        assert result is None

    def test_passes_when_free_equals_required(self) -> None:
        # Exact-equal is acceptable — the budget already includes headroom.
        check_vram_budget(
            required_mb=1024,
            mem_get_info=lambda: (1024 * MIB, 12 * 1024 * MIB),
        )


class TestResolveRequiredMb:
    def test_env_override_wins_over_classifier_default(self) -> None:
        # Operator override path: VRAM_BUDGET_MB beats the per-classifier default.
        assert resolve_required_mb(classifier="vlm", env_override="2048") == 2048
        assert resolve_required_mb(classifier="heuristic", env_override="9999") == 9999

    def test_heuristic_classifier_uses_1024_mib_default(self) -> None:
        assert resolve_required_mb(classifier="heuristic", env_override=None) == 1024

    def test_vlm_classifier_uses_7168_mib_default(self) -> None:
        assert resolve_required_mb(classifier="vlm", env_override=None) == 7168

    def test_empty_string_env_override_falls_back_to_classifier_default(self) -> None:
        # Docker's missing-env vs empty-env distinction: os.environ.get returns
        # "" for `-e VRAM_BUDGET_MB=`. Treat empty as "not set" — otherwise
        # int("") raises and a forgotten empty flag would crash boot.
        assert resolve_required_mb(classifier="vlm", env_override="") == 7168


class TestPreflightOrExit:
    def test_insufficient_vram_exits_2_with_structured_stderr_line(self) -> None:
        # Faithful reproduction of the source incident: 139 MiB free of 12 GiB,
        # VLM budget 7168 MiB. Operator should see one line, then exit(2).
        stderr = io.StringIO()
        exit_calls: list[int] = []

        def fake_exit(code: int) -> None:
            exit_calls.append(code)

        preflight_or_exit(
            classifier="vlm",
            env_override=None,
            mem_get_info=lambda: (139 * MIB, 12 * 1024 * MIB),
            stderr=stderr,
            exit_fn=fake_exit,
        )

        assert exit_calls == [2]
        line = stderr.getvalue()
        assert line.startswith("VRAM_PREFLIGHT_FAIL ")
        assert "required=7168" in line
        assert "free=139" in line
        assert "total=12288" in line
        # nvidia-smi hint must be in the line so operator has a one-glance
        # next step rather than searching docs for what to run.
        assert "nvidia-smi" in line
        # One line only — no PyTorch traceback bleeding through.
        assert line.count("\n") == 1

    def test_sufficient_vram_proceeds_silently(self) -> None:
        # Healthy GPU: 8 GiB free of 12 GiB, vlm needs 7168. Caller must reach
        # the next instruction with no stderr noise and no exit.
        stderr = io.StringIO()
        exit_calls: list[int] = []

        preflight_or_exit(
            classifier="vlm",
            env_override=None,
            mem_get_info=lambda: (8 * 1024 * MIB, 12 * 1024 * MIB),
            stderr=stderr,
            exit_fn=exit_calls.append,
        )

        assert exit_calls == []
        assert stderr.getvalue() == ""

    def test_env_override_drives_the_check(self) -> None:
        # Operator sets VRAM_BUDGET_MB=200 to force-allow a tight box; 256 MiB
        # free should now PASS even though it's far below the vlm default.
        stderr = io.StringIO()
        exit_calls: list[int] = []

        preflight_or_exit(
            classifier="vlm",
            env_override="200",
            mem_get_info=lambda: (256 * MIB, 12 * 1024 * MIB),
            stderr=stderr,
            exit_fn=exit_calls.append,
        )

        assert exit_calls == []
        assert stderr.getvalue() == ""
