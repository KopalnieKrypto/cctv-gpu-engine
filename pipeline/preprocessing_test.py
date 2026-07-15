"""Tests for image preprocessing — produces YOLO-pose ready tensor."""

import numpy as np

from pipeline.preprocessing import IMG_SIZE, PAD_VALUE, letterbox_params, preprocess


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
        # original dimensions are preserved for later coordinate scaling
        assert orig_w == 1280
        assert orig_h == 720


class TestLetterbox:
    """Issue #83 — the frame must keep its aspect ratio inside the square input.

    Squashing 16:9 into 640×640 stretches people out of proportion, and YOLO
    was trained on aspect-preserved (letterboxed) input. Measured on real
    4K pilot footage: on a tile with 4 workers visible, squashing found 1 and
    letterboxing found all 4.
    """

    def test_wide_frame_is_padded_rather_than_stretched(self):
        # 1280×720 is 16:9 — it fits the 640 box across, leaving 140 rows of
        # padding above and below a 640×360 image.
        img_bgr = np.full((720, 1280, 3), 255, dtype=np.uint8)

        tensor, _, _ = preprocess(img_bgr)

        rows = tensor[0, 0]  # one channel, (IMG_SIZE, IMG_SIZE)
        assert rows[0, 0] == np.float32(PAD_VALUE / 255.0)  # padded band on top
        assert rows[IMG_SIZE - 1, 0] == np.float32(PAD_VALUE / 255.0)  # and below
        assert rows[IMG_SIZE // 2, IMG_SIZE // 2] == np.float32(1.0)  # image in the middle
        # Exactly 140 padded rows top and bottom, 360 rows of picture between.
        padded_rows = int((rows == np.float32(PAD_VALUE / 255.0)).all(axis=1).sum())
        assert padded_rows == 280

    def test_tall_frame_is_padded_on_the_sides(self):
        img_bgr = np.full((1280, 720, 3), 255, dtype=np.uint8)

        tensor, _, _ = preprocess(img_bgr)

        rows = tensor[0, 0]
        padded_cols = int((rows == np.float32(PAD_VALUE / 255.0)).all(axis=0).sum())
        assert padded_cols == 280

    def test_square_frame_needs_no_padding(self):
        img_bgr = np.full((640, 640, 3), 255, dtype=np.uint8)

        tensor, _, _ = preprocess(img_bgr)

        assert (tensor == np.float32(1.0)).all()

    def test_letterbox_params_describe_the_fit(self):
        # Shared with postprocessing so the forward and inverse transforms can
        # never drift apart — a drift would put every bbox in the wrong place.
        scale, pad_x, pad_y = letterbox_params(1920, 1080)

        assert scale == 640 / 1920  # limited by the wider side
        assert pad_x == 0
        assert pad_y == (640 - 360) // 2
