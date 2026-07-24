"""Detection-scale / recall-risk signal for ``result.json`` diagnostics (#113).

A survivorship-bias-free flag for how much a scene's geometry limits detection
recall. #101 measured on ``magazyn-hall-v1`` that recall is ~0 below ~60 px of
person height at model input, so "can this scene be detected" is, to first
order, a geometry question answerable at analysis time without ground truth:

    input_scale             = min(in_w/src_w, in_h/src_h)   # how far the frame shrinks
    resolvable_height_native = floor_input_px / input_scale  # smallest resolvable native px

Because it derives from the resolved input size and the frame size alone, it is
free of the survivorship bias that makes "median detected height" useless here
(the detector only ever sees the resolvable people). The detected-height median
is carried too, as supporting context only — never the trigger.
"""

from __future__ import annotations

import statistics

# Model-input person height (px) below which #101 measured recall ~= 0.
DETECTION_FLOOR_INPUT_PX = 60

# resolvable_height_frac at/above which the scene is flagged: the detector can
# only resolve people taller than this fraction of the frame. Policy default
# tied to #101; magazyn @640 (0.167) flags, @1280x736 (0.083) does not.
RECALL_RISK_FRAC_THRESHOLD = 0.10


def detection_scale(source_frame, input_size_wh, detected_heights):
    if source_frame is None:
        return None
    src_w, src_h = source_frame
    in_w, in_h = input_size_wh
    input_scale = min(in_w / src_w, in_h / src_h)
    resolvable_native = DETECTION_FLOOR_INPUT_PX / input_scale
    frac = resolvable_native / src_h
    heights = list(detected_heights)
    return {
        "input_scale": input_scale,
        "floor_input_px": DETECTION_FLOOR_INPUT_PX,
        "resolvable_height_native_px": resolvable_native,
        "resolvable_height_frac": frac,
        "median_detected_height_native_px": statistics.median(heights) if heights else None,
        "detections_measured": len(heights),
        "recall_risk": "high" if frac >= RECALL_RISK_FRAC_THRESHOLD else "normal",
    }
