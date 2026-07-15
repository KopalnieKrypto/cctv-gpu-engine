"""Image preprocessing for YOLO-pose ONNX inference.

Converts BGR frames into the [1, 3, 640, 640] float32 NCHW tensor that the
YOLO-pose ONNX model expects, while returning original dimensions so the
postprocessing stage can scale predictions back to image space.

The frame is **letterboxed**, not stretched: it is scaled by a single factor
that fits it inside the square input, and the leftover strip is filled with
YOLO's neutral grey. Squashing a 16:9 frame into 640×640 instead — which this
module used to do — distorts people out of the proportions the model was
trained on, and costs detections badly on wide frames (issue #83): on a 4K
tile with four workers visible, squashing found one and letterboxing found all
four.
"""

from __future__ import annotations

import cv2
import numpy as np

IMG_SIZE = 640

# Neutral grey YOLO pads letterboxed images with. Matching it matters: the
# model has seen this exact value around training images, so it reads as
# "nothing here" rather than as content.
PAD_VALUE = 114


def letterbox_params(orig_w: int, orig_h: int) -> tuple[float, int, int]:
    """Scale and padding used to fit ``orig_w × orig_h`` into ``IMG_SIZE²``.

    Returns ``(scale, pad_x, pad_y)``: multiply original coordinates by
    ``scale`` then add the padding to reach model space, and invert to come
    back. :func:`preprocess` and :func:`pipeline.postprocessing.postprocess`
    both derive the transform from here rather than each computing their own,
    so the forward and inverse can never drift apart — a drift would silently
    put every bbox in the wrong place.
    """
    scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    return scale, (IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2


def preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Letterbox, normalize, and reshape a BGR image for YOLO-pose inference.

    Returns:
        tensor: float32 array of shape ``(1, 3, IMG_SIZE, IMG_SIZE)`` with
            values in ``[0, 1]``.
        orig_w: original image width in pixels.
        orig_h: original image height in pixels.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    scale, pad_x, pad_y = letterbox_params(orig_w, orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    arr = canvas.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, 0)  # add batch dim
    return arr, orig_w, orig_h
