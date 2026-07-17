"""Training-data behavior through the shared runtime feature contract."""

from __future__ import annotations

import numpy as np

from activity_mlp.data import build_feature_matrix
from pipeline.activity_features import ACTIVITY_CLASSES, FEATURE_DIM


def test_feature_matrix_uses_frozen_class_order_and_float32_features() -> None:
    rows = [
        {
            "sample_id": activity,
            "activity": activity,
            "bbox": [10.0, 20.0, 100.0, 200.0],
            "keypoints": [
                {"x": 20.0 + index, "y": 30.0 + 2 * index, "vis": index / 16} for index in range(17)
            ],
        }
        for activity in ACTIVITY_CLASSES
    ]

    features, labels = build_feature_matrix(rows)

    assert features.shape == (4, FEATURE_DIM)
    assert features.dtype == np.float32
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels, [0, 1, 2, 3])
    np.testing.assert_array_equal(features[0], features[1])
