"""Image preprocessing for YOLO-pose ONNX inference.

Converts BGR frames into the [1, 3, 640, 640] float32 NCHW tensor that the
YOLO-pose ONNX model expects, while returning original dimensions so the
postprocessing stage can scale predictions back to image space.
"""

from __future__ import annotations

import cv2
import numpy as np

IMG_SIZE = 640


def preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Resize, normalize, and reshape a BGR image for YOLO-pose inference.

    Returns:
        tensor: float32 array of shape ``(1, 3, IMG_SIZE, IMG_SIZE)`` with
            values in ``[0, 1]``.
        orig_w: original image width in pixels.
        orig_h: original image height in pixels.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = img_resized.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, 0)  # add batch dim
    return arr, orig_w, orig_h
