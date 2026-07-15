"""Tests for OSNet Re-ID appearance embedding (issue #32).

The embedder is the association metric behind :mod:`pipeline.tracker`: it turns
each person's bbox crop into an appearance vector whose cosine similarity says
"same person" or "someone else". The advisory (2026-05-27) locked this to a
dedicated OSNet-class network rather than reusing YOLO's intermediate features,
because "specialized Re-ID models are much better at handling the identity
consistency you need at 1 fps".

The ONNX session is a genuine system boundary, so these tests inject a fake one
and assert on the contract around it: what the model is fed, and what callers
get back. The tracker's correctness rests on the vectors being L2-normalized
(it takes dot products and calls them cosines), so that is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.postprocessing import Detection, Keypoint
from pipeline.reid import REID_INPUT_H, REID_INPUT_W, OSNetEmbedder


class FakeSession:
    """Stands in for the OSNet ONNX session; records every call it receives."""

    def __init__(self, features) -> None:
        self._features = [np.asarray(f, dtype=np.float32) for f in features]
        self.inputs: list[np.ndarray] = []

    def run(self, output_names, feed: dict) -> list[np.ndarray]:
        self.inputs.append(next(iter(feed.values())))
        # Hand back the scripted vector for this call; the last one repeats if
        # a test scripted fewer vectors than it makes calls.
        index = min(len(self.inputs) - 1, len(self._features) - 1)
        return [self._features[index].reshape(1, -1)]


def _frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(bbox) -> Detection:
    return Detection(
        bbox=list(bbox),
        confidence=0.9,
        keypoints=[Keypoint(0.0, 0.0, 0.0)] * 17,
    )


class TestEmbedding:
    def test_returns_one_unit_length_vector_per_detection(self):
        # The tracker compares identities with a bare dot product, which only
        # equals cosine similarity when both vectors are unit length. If raw
        # model output leaked through, every similarity — and so every match
        # threshold — would be meaningless.
        session = FakeSession(features=[[3.0, 4.0], [0.0, 5.0]])
        embedder = OSNetEmbedder(session=session, input_name="images")

        vectors = embedder.embed(_frame(), [_det((0, 0, 10, 20)), _det((20, 20, 30, 40))])

        assert len(vectors) == 2
        assert vectors[0] == pytest.approx([0.6, 0.8])
        assert np.linalg.norm(vectors[1]) == pytest.approx(1.0)

    def test_every_crop_is_sent_at_a_constant_input_shape(self):
        """Crops go one per call, never as a variable-size batch.

        Measured on an RTX 5070: onnxruntime's CUDA provider re-optimises the
        graph on *every* input-shape change — ~0.5 s, and 4.3 s even when
        flipping between two shapes it has already warmed. A batch sized by the
        person count changes shape almost every frame, which made the embedder
        6.6x more expensive than the entire rest of the pipeline. A constant
        shape is worth far more here than batching is.
        """
        session = FakeSession(features=[[1.0, 1.0]])
        embedder = OSNetEmbedder(session=session, input_name="images")

        embedder.embed(_frame(), [_det((0, 0, 10, 20))] * 3)

        assert len(session.inputs) == 3
        assert all(t.shape == (1, 3, REID_INPUT_H, REID_INPUT_W) for t in session.inputs)

    def test_no_detections_means_no_model_call(self):
        session = FakeSession(features=[[1.0, 1.0]])
        embedder = OSNetEmbedder(session=session, input_name="images")

        assert embedder.embed(_frame(), []) == []
        assert session.inputs == []


class TestAwkwardBoxes:
    """Boxes YOLO really emits, which naive cropping turns into crashes."""

    def test_bbox_overhanging_the_frame_edge_is_clamped_to_the_image(self):
        # Someone walking out of shot gets a bbox with negative coordinates.
        # Numpy slicing reads those as "from the end", so a naive crop silently
        # grabs the *opposite* corner of the image and embeds the wrong pixels —
        # the person's identity would flip for no visible reason.
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[0:32, 0:32] = 255  # the region the box actually overlaps
        session = FakeSession(features=[[1.0, 1.0]])
        embedder = OSNetEmbedder(session=session, input_name="images")

        embedder.embed(frame, [_det((-20.0, -20.0, 32.0, 32.0))])

        # White pixels normalize positive, black ones negative — so a positive
        # mean proves we embedded the top-left region and not the black corner.
        assert session.inputs[0].mean() > 0.0

    def test_zero_area_bbox_does_not_crash_the_run(self):
        # A degenerate box slices to an empty array and blows up cv2.resize.
        # One bad box at minute 19 must not kill a 20-minute GPU job.
        session = FakeSession(features=[[1.0, 1.0]])
        embedder = OSNetEmbedder(session=session, input_name="images")

        vectors = embedder.embed(_frame(), [_det((10.0, 10.0, 10.0, 10.0))])

        assert len(vectors) == 1
