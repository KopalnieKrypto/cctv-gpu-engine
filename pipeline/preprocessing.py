"""Image preprocessing for YOLO-pose ONNX inference.

Converts BGR frames into the float32 NCHW tensor declared by the fixed
YOLO-pose ONNX model, while returning original dimensions so postprocessing
can scale predictions back to image space.

The frame is **letterboxed**, not stretched: it is scaled by a single factor
that fits it inside the model input, and the leftover strip is filled with
YOLO's neutral grey. Squashing a 16:9 frame into 640×640 instead — which this
module used to do — distorts people out of the proportions the model was
trained on, and costs detections badly on wide frames (issue #83): on a 4K
tile with four workers visible, squashing found one and letterboxing found all
four.

The input need not be square (issue #100). For a 16:9 frame the *width* ratio
is the binding one, so a 640×384 export sees the frame at exactly the same
scale as 640×640 while skipping the 43.75% of the tensor that was constant
grey. Square exports stay first-class — this widened the contract rather than
replacing it, and a bare int still means a square input.
"""

from __future__ import annotations

import cv2
import numpy as np

IMG_SIZE = 640

# A model input size: either a square side or an explicit ``(width, height)``.
InputSize = int | tuple[int, int]

# Neutral grey YOLO pads letterboxed images with. Matching it matters: the
# model has seen this exact value around training images, so it reads as
# "nothing here" rather than as content.
PAD_VALUE = 114


def input_wh(input_size: InputSize) -> tuple[int, int]:
    """Normalize a model input size to ``(width, height)``.

    A bare int is the square case (``640`` → ``(640, 640)``): square exports
    remain first-class, and every caller predating issue #100 passes one.
    """
    if isinstance(input_size, int):
        return input_size, input_size
    width, height = input_size
    return int(width), int(height)


def letterbox_params(
    orig_w: int, orig_h: int, input_size: InputSize = IMG_SIZE
) -> tuple[float, int, int]:
    """Scale and padding used to fit ``orig_w × orig_h`` into the model input.

    Returns ``(scale, pad_x, pad_y)``: multiply original coordinates by
    ``scale`` then add the padding to reach model space, and invert to come
    back. :func:`preprocess` and :func:`pipeline.postprocessing.postprocess`
    both derive the transform from here rather than each computing their own,
    so the forward and inverse can never drift apart — a drift would silently
    put every bbox in the wrong place.

    The scale is the single factor that fits both axes, so aspect ratio is
    preserved whatever the input shape; only how much grey is left over
    changes.
    """
    in_w, in_h = input_wh(input_size)
    scale = min(in_w / orig_w, in_h / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    return scale, (in_w - new_w) // 2, (in_h - new_h) // 2


def preprocess(
    img_bgr: np.ndarray, input_size: InputSize = IMG_SIZE
) -> tuple[np.ndarray, int, int]:
    """Letterbox, normalize, and reshape a BGR image for YOLO-pose inference.

    Returns:
        tensor: float32 array of shape ``(1, 3, input_h, input_w)`` with
            values in ``[0, 1]``.
        orig_w: original image width in pixels.
        orig_h: original image height in pixels.
    """
    in_w, in_h = input_wh(input_size)
    orig_h, orig_w = img_bgr.shape[:2]
    scale, pad_x, pad_y = letterbox_params(orig_w, orig_h, input_size)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((in_h, in_w, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    arr = canvas.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, 0)  # add batch dim
    return arr, orig_w, orig_h
