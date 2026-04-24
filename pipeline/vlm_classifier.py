"""VLM-based activity classification using Qwen2.5-VL-3B.

Replaces geometric heuristics with a vision-language model that classifies
each frame by understanding the visual scene (posture, furniture, context).
The model is loaded lazily on first call and reused for all subsequent frames.

Requires: torch (cu128 for Blackwell GPUs), transformers, qwen-vl-utils.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

Activity = Literal["sitting", "standing", "walking", "running", "unknown"]

ACTIVITIES = ("sitting", "standing", "walking", "running")

PROMPT = (
    "Look at this surveillance camera frame. "
    "Classify the person's activity as exactly one of: "
    "sitting, standing, walking, running. "
    "Reply with only one word."
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class VLMClassifier:
    """Lazy-loaded Qwen2.5-VL activity classifier."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._model = None
        self._processor = None

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        log.info("Loading VLM %s ...", self._model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        log.info("VLM loaded.")

    def classify_frame(self, frame_bgr: np.ndarray) -> Activity:
        """Classify a single BGR frame into an activity label."""
        import torch
        from PIL import Image

        if self._model is None:
            self._load()

        rgb = frame_bgr[:, :, ::-1]
        img = Image.fromarray(rgb)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[img], padding=True, return_tensors="pt").to(
            self._model.device
        )

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=10, do_sample=False)

        generated = output_ids[0][inputs.input_ids.shape[1] :]
        answer = self._processor.decode(generated, skip_special_tokens=True)
        answer = answer.strip().lower()

        for act in ACTIVITIES:
            if act in answer:
                return act
        return "standing"
