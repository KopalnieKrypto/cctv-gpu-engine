"""Tests for appliance self-telemetry (issue #92).

The appliance is the only thing that can see its own disk. A production box
(`cameraboy`) burned 1.2 GB/day for seven days while every heartbeat reported
healthy; it surfaced only via a manual ``df -h`` during an unrelated audit.

``shutil.disk_usage`` is stubbed here rather than the caller — it is the
syscall boundary, and a real reading is whatever the CI runner's disk happens
to hold, which cannot be asserted on.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# ----- 1. disk sampling reports raw bytes from the volume -----


def test_sample_disk_bytes_reports_free_and_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw bytes, not a band enum: the platform derives OK/LOW/CRITICAL in
    ``applianceDiskHealth`` so thresholds can be retuned without redeploying
    every appliance (gpu-exchange#157). Its Zod schema is
    ``z.number().int().nonnegative()`` — floats would be rejected."""
    from client_agent.telemetry import sample_disk_bytes

    sampled: list[Path] = []

    def fake_disk_usage(path: str | Path) -> shutil._ntuple_diskusage:
        sampled.append(Path(path))
        return shutil._ntuple_diskusage(
            total=123_480_309_760, used=84_825_604_096, free=38_654_705_664
        )

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)

    free, total = sample_disk_bytes(tmp_path)

    assert (free, total) == (38_654_705_664, 123_480_309_760)
    assert all(isinstance(v, int) for v in (free, total))
    # Sampled at the buffer volume, which is what fills up — not at "/",
    # which on the appliance can be a different filesystem entirely.
    assert sampled == [tmp_path]


# ----- 2. an unreadable volume degrades, it does not kill the beat -----


def test_sample_disk_bytes_returns_none_when_volume_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Telemetry is cosmetic; the heartbeat is not. A vanished or
    permission-denied buffer dir must degrade to "unknown" so the box still
    registers, reconciles recorders, and pulls config. Both values drop
    together — a free byte count without a total cannot be banded, and
    ``applianceDiskHealth`` treats that pair as unknown anyway."""
    from client_agent.telemetry import sample_disk_bytes

    def boom(path: str | Path) -> shutil._ntuple_diskusage:
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr(shutil, "disk_usage", boom)

    assert sample_disk_bytes(tmp_path) == (None, None)
