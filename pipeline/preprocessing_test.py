"""Tests for image preprocessing — produces YOLO-pose ready tensor."""

import numpy as np
import pytest

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


class TestNonSquareInput:
    """Issue #100 — 44% of every 640×640 tensor is grey padding.

    Pixels-on-target for a 16:9 frame is set by the *width* ratio alone (width
    is the binding side into a square), so a 640×384 export sees the frame at
    exactly the same scale as 640×640 for 0.60× the compute. This widens the
    contract — square stays first-class, it just stops being the only shape.
    """

    def test_letterbox_scale_is_unchanged_when_only_the_padding_is_removed(self):
        # 16:9 into 640×640 and into 640×384 must place the image identically:
        # same scale, same x offset. Only the wasted grey band differs.
        square = letterbox_params(1920, 1080, 640)
        wide = letterbox_params(1920, 1080, (640, 384))

        assert square[0] == wide[0] == 640 / 1920
        assert square[1] == wide[1] == 0
        assert square[2] == (640 - 360) // 2
        assert wide[2] == (384 - 360) // 2

    def test_padding_is_computed_per_axis(self):
        # A frame taller than the input's aspect is bound by height instead.
        scale, pad_x, pad_y = letterbox_params(1000, 1000, (1280, 736))

        assert scale == 736 / 1000
        assert pad_x == (1280 - 736) // 2
        assert pad_y == 0

    def test_preprocess_emits_the_models_own_tensor_shape(self):
        img_bgr = np.full((720, 1280, 3), 255, dtype=np.uint8)

        tensor, orig_w, orig_h = preprocess(img_bgr, input_size=(1280, 736))

        assert tensor.shape == (1, 3, 736, 1280)
        assert (orig_w, orig_h) == (1280, 720)

    def test_non_square_input_wastes_far_less_of_the_tensor_on_padding(self):
        img_bgr = np.full((2160, 3840, 3), 255, dtype=np.uint8)

        square, _, _ = preprocess(img_bgr, input_size=640)
        wide, _, _ = preprocess(img_bgr, input_size=(640, 384))

        pad = np.float32(PAD_VALUE / 255.0)
        square_waste = float((square == pad).mean())
        wide_waste = float((wide == pad).mean())
        # 3840×2160 into 640×640 leaves 640×360 of picture: 43.75% grey.
        assert square_waste == pytest.approx(0.4375, abs=0.005)
        assert wide_waste < 0.07
        assert wide.size / square.size == pytest.approx(0.6, abs=0.005)

    def test_square_input_size_still_accepted_as_a_bare_int(self):
        # Widening, not replacing: every existing caller passes an int.
        img_bgr = np.full((720, 1280, 3), 255, dtype=np.uint8)

        from_int, _, _ = preprocess(img_bgr, input_size=640)
        from_pair, _, _ = preprocess(img_bgr, input_size=(640, 640))

        assert np.array_equal(from_int, from_pair)
