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
    """Stands in for the OSNet ONNX session; records what it was fed."""

    def __init__(self, features: np.ndarray) -> None:
        self._features = np.asarray(features, dtype=np.float32)
        self.last_input: np.ndarray | None = None

    def run(self, output_names, feed: dict) -> list[np.ndarray]:
        self.last_input = next(iter(feed.values()))
        return [self._features[: self.last_input.shape[0]]]


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
        session = FakeSession(features=np.array([[3.0, 4.0], [0.0, 5.0]]))
        embedder = OSNetEmbedder(session=session, input_name="images")

        vectors = embedder.embed(_frame(), [_det((0, 0, 10, 20)), _det((20, 20, 30, 40))])

        assert len(vectors) == 2
        assert vectors[0] == pytest.approx([0.6, 0.8])
        assert np.linalg.norm(vectors[1]) == pytest.approx(1.0)

    def test_every_person_is_embedded_in_one_batched_pass(self):
        # Three people must cost one forward pass, not three: at 1 fps on an
        # hour of footage the per-call overhead is what eats the 10% throughput
        # budget the issue allows.
        session = FakeSession(features=np.ones((3, 512)))
        embedder = OSNetEmbedder(session=session, input_name="images")

        embedder.embed(_frame(), [_det((0, 0, 10, 20))] * 3)

        assert session.last_input.shape == (3, 3, REID_INPUT_H, REID_INPUT_W)

    def test_no_detections_means_no_model_call(self):
        session = FakeSession(features=np.ones((1, 512)))
        embedder = OSNetEmbedder(session=session, input_name="images")

        assert embedder.embed(_frame(), []) == []
        assert session.last_input is None


class TestAwkwardBoxes:
    """Boxes YOLO really emits, which naive cropping turns into crashes."""

    def test_bbox_overhanging_the_frame_edge_is_clamped_to_the_image(self):
        # Someone walking out of shot gets a bbox with negative coordinates.
        # Numpy slicing reads those as "from the end", so a naive crop silently
        # grabs the *opposite* corner of the image and embeds the wrong pixels —
        # the person's identity would flip for no visible reason.
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[0:32, 0:32] = 255  # the region the box actually overlaps
        session = FakeSession(features=np.array([[1.0, 1.0]]))
        embedder = OSNetEmbedder(session=session, input_name="images")

        embedder.embed(frame, [_det((-20.0, -20.0, 32.0, 32.0))])

        # White pixels normalize positive, black ones negative — so a positive
        # mean proves we embedded the top-left region and not the black corner.
        assert session.last_input.mean() > 0.0

    def test_zero_area_bbox_does_not_crash_the_run(self):
        # A degenerate box slices to an empty array and blows up cv2.resize.
        # One bad box at minute 19 must not kill a 20-minute GPU job.
        session = FakeSession(features=np.array([[1.0, 1.0]]))
        embedder = OSNetEmbedder(session=session, input_name="images")

        vectors = embedder.embed(_frame(), [_det((10.0, 10.0, 10.0, 10.0))])

        assert len(vectors) == 1
