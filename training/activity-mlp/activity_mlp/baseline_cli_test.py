"""Public-command behavior for freezing pre-training baseline evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image

from activity_mlp.baseline_cli import HeartbeatPredictor, main


def test_command_freezes_heuristic_and_vlm_baselines_together(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(dataset_root / "frame.jpg")
    rows = [
        {
            "sample_id": f"test-{index:03d}",
            "split": "test",
            "activity": "standing",
            "camera_geometry_id": "controlled-garden",
            "frame_path": "frame.jpg",
            "bbox": [10.0, 20.0, 20.0, 80.0],
            "keypoints": [{"x": 0.0, "y": 0.0, "vis": 0.0} for _ in range(17)],
        }
        for index in range(150)
    ]
    (dataset_root / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    output_path = tmp_path / "baselines.json"

    class FakeVlm:
        def classify_frame(self, _frame_bgr) -> str:
            return "standing"

    exit_code = main(
        ["--dataset", str(dataset_root), "--output", str(output_path)],
        vlm_factory=FakeVlm,
    )

    artifact = json.loads(output_path.read_text())
    assert exit_code == 0
    assert list(artifact["baselines"]) == ["heuristic", "vlm"]
    assert artifact["baselines"]["heuristic"]["accuracy"] == 1.0
    assert artifact["baselines"]["vlm"]["accuracy"] == 1.0


def test_predictor_emits_flushed_progress_heartbeats() -> None:
    class TrackingStream(StringIO):
        flushed = False

        def flush(self) -> None:
            self.flushed = True
            super().flush()

    stream = TrackingStream()
    predictor = HeartbeatPredictor("vlm", lambda row: row["activity"], every=2, stream=stream)

    predictor({"activity": "sitting"})
    predictor({"activity": "standing"})

    assert "baseline classifier=vlm processed=2" in stream.getvalue()
    assert stream.flushed is True


def test_module_exposes_the_reproducible_public_command() -> None:
    repo_root = Path(__file__).parents[3]
    completed = subprocess.run(
        [sys.executable, "-m", "activity_mlp.baseline_cli", "--help"],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "training/activity-mlp"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--dataset" in completed.stdout
    assert "--output" in completed.stdout
