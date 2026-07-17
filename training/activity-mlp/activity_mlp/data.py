"""Deterministic issue #33 row preprocessing for MLP training."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pipeline.activity_features import ACTIVITY_CLASSES, extract_activity_features
from pipeline.postprocessing import Detection, Keypoint


def _row_detection(row: dict) -> Detection:
    x, y, width, height = row["bbox"]
    return Detection(
        bbox=[x, y, x + width, y + height],
        confidence=1.0,
        keypoints=[Keypoint(**keypoint) for keypoint in row["keypoints"]],
    )


def build_feature_matrix(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert ordered release rows into features and class indices."""
    class_indices = {activity: index for index, activity in enumerate(ACTIVITY_CLASSES)}
    features = np.stack([extract_activity_features(_row_detection(row)) for row in rows])
    labels = np.asarray([class_indices[row["activity"]] for row in rows], dtype=np.int64)
    return features.astype(np.float32, copy=False), labels
