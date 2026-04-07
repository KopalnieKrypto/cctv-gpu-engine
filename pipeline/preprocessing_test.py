"""Tests for image preprocessing — produces YOLO-pose ready tensor."""

import numpy as np

from pipeline.preprocessing import IMG_SIZE, preprocess


class TestPreprocess:
    def test_returns_tensor_with_yolo_pose_shape_and_original_dimensions(self):
        # given a non-square BGR image like a real surveillance frame
        img_bgr = np.full((720, 1280, 3), 128, dtype=np.uint8)

        tensor, orig_w, orig_h = preprocess(img_bgr)

        # tensor matches the YOLO-pose ONNX input contract
        assert tensor.shape == (1, 3, IMG_SIZE, IMG_SIZE)
        assert tensor.dtype == np.float32
        # values are normalized into [0, 1]
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        # uniform gray 128/255 ≈ 0.502
        assert np.isclose(tensor.mean(), 128.0 / 255.0, atol=1e-4)
        # original dimensions are preserved for later coordinate scaling
        assert orig_w == 1280
        assert orig_h == 720
