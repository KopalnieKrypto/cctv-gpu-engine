"""Tests for the station classifier path (issue #123).

The zone is the unit of measurement here, not a person (assumption A5), so this
path takes a rectangle from `zones.json`, crops it out of the **native** frame,
and reads it with a frozen backbone plus a ~1 MB temporal head. No pose model, no
OSNet, no VLM — the arm is cheap precisely because it removes three heavy
components rather than adding a fourth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.station_card import StationCard
from pipeline.station_classifier import (
    DEFAULT_STATION_CARD_PATH,
    StationClassifier,
    StationCropError,
    StationRectMismatchError,
    StationZoneError,
    load_station_classifier,
    preprocess_crop,
    resolve_station_zone,
    station_crop,
    station_rect,
    zone_rect,
)
from pipeline.zones import ZoneConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CARD_NAME = DEFAULT_STATION_CARD_PATH

# The welding bench: 900x800 at (1700, 1360) in the 3840x2160 native frame,
# written as a rectangle polygon because that is what a station zone is.
#
# y is 1360 and not the 1400 the fixture manifest records. `crop=900:800:1700:1400`
# does not fit a 2160-tall frame, and ffmpeg's crop filter silently clamps y to
# `in_h - h`, so every training crop was cut at 1360. Verified against the fixture
# rather than reasoned about: mean |diff| between the shipped crop and the native
# frame is 2.02 (JPEG noise) at y=1360 and 24.74 at the next candidate offset.
STATION_POLYGON = [[1700, 1360], [2600, 1360], [2600, 2160], [1700, 2160]]


def _station_config(**zone_overrides) -> ZoneConfig:
    zone = {
        "id": "spawanie",
        "name": "Stanowisko spawalnicze",
        "polygon": STATION_POLYGON,
        "rules": {"type": "station"},
        **zone_overrides,
    }
    return ZoneConfig.from_dict({"zones": [zone]})


def _card(**overrides) -> StationCard:
    """A card carrying the fields under test, defaulted to the shipped station."""
    fields = {
        "version": "1.0.0",
        "station_id": "hala-prawe-v1",
        "zone_rect": (1700, 1360, 900, 800),
        "stride_s": 2,
        "model_outputs": ("spawanie", "ukladanie_pretow", "postoj", "nierozpoznane"),
        "delivered_classes": ("spawanie", "ukladanie_pretow", "pozostale", "nierozpoznane"),
        "bucket": "pozostale",
        "bucket_members": frozenset({"postoj"}),
        "abstention": "nierozpoznane",
        "time_ratios": {
            "spawanie": 1.06,
            "ukladanie_pretow": 1.10,
            "pozostale": 0.84,
            "nierozpoznane": 0.5,
        },
        "window": 64,
        "model_input": (224, 224),
        "training_windows": (),
        **overrides,
    }
    return StationCard(**fields)


class TestFindingTheStationZone:
    def test_a_zone_declaring_the_station_ruleset_is_the_station(self) -> None:
        zone = resolve_station_zone(_station_config())

        assert zone.id == "spawanie"
        assert zone.name == "Stanowisko spawalnicze"

    def test_a_config_with_no_station_zone_is_refused(self) -> None:
        bending = ZoneConfig.from_dict(
            {"zones": [{"id": "gietarka", "name": "Giętarka", "polygon": STATION_POLYGON}]}
        )

        with pytest.raises(StationZoneError) as exc:
            resolve_station_zone(bending)

        assert "station" in str(exc.value)

    def test_two_station_zones_are_refused_rather_than_silently_picking_one(self) -> None:
        # #123 scopes this to one station. Picking the first of two would report
        # one bench's totals under a config that asked for two.
        both = ZoneConfig.from_dict(
            {
                "zones": [
                    {
                        "id": "spawanie",
                        "name": "Spawanie",
                        "polygon": STATION_POLYGON,
                        "rules": {"type": "station"},
                    },
                    {
                        "id": "spawanie-2",
                        "name": "Spawanie 2",
                        "polygon": STATION_POLYGON,
                        "rules": {"type": "station"},
                    },
                ]
            }
        )

        with pytest.raises(StationZoneError) as exc:
            resolve_station_zone(both)

        assert "spawanie" in str(exc.value)
        assert "spawanie-2" in str(exc.value)


class TestTheCropIsTakenFromTheNativeFrame:
    """Never from a downscaled copy — that discards the 3x station framing gives.

    The client declined to reframe the camera (assumption A2), so the whole arm
    rests on cropping pixels we already receive at full resolution. A crop taken
    from the 1280x736 pose downscale would throw away exactly what makes it work.
    """

    def test_the_crop_matches_the_zone_rectangle_in_original_resolution_pixels(self) -> None:
        # A 3840x2160 native frame with markers planted at the rectangle's two
        # corners, so a crop taken from a resized copy cannot pass.
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        frame[1360, 1700] = (11, 22, 33)
        frame[2159, 2599] = (44, 55, 66)

        crop = station_crop(frame, (1700, 1360, 900, 800))

        assert crop.shape == (800, 900, 3)
        assert tuple(crop[0, 0]) == (11, 22, 33)
        assert tuple(crop[799, 899]) == (44, 55, 66)

    def test_the_rectangle_comes_from_the_zone_polygon_bounding_box(self) -> None:
        assert zone_rect(resolve_station_zone(_station_config())) == (1700, 1360, 900, 800)

    def test_the_configured_rectangle_must_be_the_one_the_head_was_fitted_on(self) -> None:
        """A head fed a different rectangle still returns confident logits.

        Nothing downstream would say so — the totals would simply be wrong, which
        is the same class of silent substitution the sha256 pins in
        `setup-models.sh` exist to catch.
        """
        card = _card(zone_rect=(1700, 1360, 900, 800))
        shifted = _station_config(polygon=[[1700, 1340], [2600, 1340], [2600, 2140], [1700, 2140]])

        with pytest.raises(StationRectMismatchError) as exc:
            station_rect(resolve_station_zone(shifted), card)

        message = str(exc.value)
        assert "900x800 at (1700, 1340)" in message
        assert "900x800 at (1700, 1360)" in message

    def test_a_matching_rectangle_is_accepted(self) -> None:
        card = _card(zone_rect=(1700, 1360, 900, 800))

        assert station_rect(resolve_station_zone(_station_config()), card) == (1700, 1360, 900, 800)

    def test_a_rectangle_that_runs_off_the_frame_is_refused(self) -> None:
        """The failure this exists to catch is silent, and it already happened.

        numpy slicing clips: ``frame[1400:2200]`` on a 2160-row frame returns 760
        rows without complaint, and ffmpeg's crop filter does the same thing by
        moving the rectangle instead — which is how the fixture's own crops came
        to be cut 40 px above where every record of them says they were.
        """
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        with pytest.raises(StationCropError) as exc:
            station_crop(frame, (1700, 1400, 900, 800))

        assert "3840x2160" in str(exc.value)


class TestPreprocessing:
    """The tensor the backbone receives, reproduced without `transformers`.

    Resolving `facebook/dinov2-base`'s processor through the hub at run time would
    make the preprocessing depend on what HuggingFace serves that day, which is
    the same failure `setup-models.sh` pins the weights against — and a changed
    resize or normalisation moves every embedding the head was fitted on.

    Note `resize` and `model_input` are different numbers: 518 is the resize
    target and the processor then centre-crops to 224, so 518 never reaches the
    model. Reading only the first is how this fixture's "-518" arms came to be
    described as seeing 518 pixels.
    """

    def test_the_tensor_has_the_shape_the_exported_backbone_declares(self) -> None:
        crop = np.zeros((800, 900, 3), dtype=np.uint8)

        tensor = preprocess_crop(crop, model_input=(224, 224))

        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == np.float32

    def test_bgr_pixels_arrive_as_rgb_rescaled_and_imagenet_normalised(self) -> None:
        # Solid red, written the way ffmpeg hands frames over: BGR.
        crop = np.zeros((800, 900, 3), dtype=np.uint8)
        crop[:, :, 2] = 255

        tensor = preprocess_crop(crop, model_input=(224, 224))[0]

        assert tensor[0].mean() == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)
        assert tensor[1].mean() == pytest.approx((0.0 - 0.456) / 0.224, abs=1e-4)
        assert tensor[2].mean() == pytest.approx((0.0 - 0.406) / 0.225, abs=1e-4)

    def test_the_crop_is_centred_and_the_image_is_not_flipped(self) -> None:
        # Left half white, right half black. A centred crop keeps that split down
        # the middle; an off-centre or flipped one does not.
        crop = np.zeros((800, 900, 3), dtype=np.uint8)
        crop[:, :450] = 255

        tensor = preprocess_crop(crop, model_input=(224, 224))[0]

        white = (1.0 - 0.485) / 0.229
        black = (0.0 - 0.485) / 0.229
        assert tensor[0, :, :100].mean() == pytest.approx(white, abs=1e-3)
        assert tensor[0, :, 124:].mean() == pytest.approx(black, abs=1e-3)


@pytest.mark.gpu
class TestPreprocessingMatchesTheRealProcessor:
    """The claim above, checked against the library it reproduces.

    This cannot run on the dev box — `transformers` is not in the CPU-stub
    environment — so it runs on the GPU box, where the embeddings the head was
    fitted on were actually produced. A reimplementation that is only ever
    compared against its own expectations is not verified at all.
    """

    def test_it_matches_auto_image_processor_for_dinov2_base(self) -> None:
        from PIL import Image
        from transformers import AutoImageProcessor

        rng = np.random.default_rng(117)
        crop = rng.integers(0, 256, size=(800, 900, 3), dtype=np.uint8)

        reference = AutoImageProcessor.from_pretrained("facebook/dinov2-base")(
            images=[Image.fromarray(crop[:, :, ::-1])],
            return_tensors="np",
            size={"height": 518, "width": 518},
        )["pixel_values"]

        ours = preprocess_crop(crop, model_input=(224, 224))

        assert ours.shape == reference.shape
        assert np.abs(ours - reference).max() < 1e-5


class _FakeSession:
    """An onnxruntime session stand-in: one named input, one array out.

    A boundary fake, not an internal one. The real thing is a CUDA session over
    weights this test has no business loading; what it stands in for is the
    contract — a named input, a batched array back — and the test drives the
    module through the same public method production does.
    """

    def __init__(self, name: str, fn) -> None:
        self._name = name
        self._fn = fn
        self.calls: list[tuple[int, ...]] = []

    def get_inputs(self):
        return [SimpleNamespace(name=self._name, shape=[1, 768, "time"])]

    def get_providers(self):
        return ["CUDAExecutionProvider"]

    def run(self, _outputs, feed):
        array = feed[self._name]
        self.calls.append(array.shape)
        return [self._fn(array)]


def _echo_head(n_class: int):
    """A head that reads the class index straight off feature 0 of each sample.

    Keeps the averaging a no-op so a failure points at the plumbing — the time
    axis, the per-sample alignment, the collapse — rather than at arithmetic.
    """

    def run(embeddings: np.ndarray) -> np.ndarray:
        _, _, time = embeddings.shape
        logits = np.zeros((1, n_class, time), dtype=np.float32)
        for t in range(time):
            logits[0, int(embeddings[0, 0, t]), t] = 1.0
        return logits

    return run


def _embeddings(class_indices: list[int], feature: int = 768) -> np.ndarray:
    x = np.zeros((len(class_indices), feature), dtype=np.float32)
    x[:, 0] = class_indices
    return x


class TestReadingTheHead:
    def test_each_sample_gets_the_category_its_logits_argmax_to(self) -> None:
        card = _card()
        head = _FakeSession("embeddings", _echo_head(len(card.model_outputs)))
        classifier = StationClassifier(backbone=None, head=head, card=card)

        # spawanie, spawanie, ukladanie_pretow, nierozpoznane
        categories = classifier.categories(_embeddings([0, 0, 1, 3]))

        assert categories == ["spawanie", "spawanie", "ukladanie_pretow", "nierozpoznane"]

    def test_the_collapse_happens_after_argmax(self) -> None:
        """The head is seven-class and the delivered vocabulary is four.

        `postoj` is a member of the `pozostale` bucket, so it must reach the
        consumer as the bucket. Collapsing before the argmax would sum logits
        across members and change which class wins.
        """
        card = _card()
        head = _FakeSession("embeddings", _echo_head(len(card.model_outputs)))
        classifier = StationClassifier(backbone=None, head=head, card=card)

        assert classifier.categories(_embeddings([2])) == ["pozostale"]

    def test_the_abstention_is_never_folded_into_the_bucket(self) -> None:
        """`nierozpoznane` keeps its own row: it is neither work nor downtime.

        Folding the honest cannot-tell into a collective work bucket would
        convert unknown time into measured time.
        """
        card = _card()
        head = _FakeSession("embeddings", _echo_head(len(card.model_outputs)))
        classifier = StationClassifier(backbone=None, head=head, card=card)

        assert classifier.categories(_embeddings([3])) == ["nierozpoznane"]

    def test_a_clip_longer_than_the_window_is_covered_end_to_end(self) -> None:
        card = _card(window=8)
        head = _FakeSession("embeddings", _echo_head(len(card.model_outputs)))
        classifier = StationClassifier(backbone=None, head=head, card=card)

        indices = [i % 2 for i in range(20)]
        categories = classifier.categories(_embeddings(indices))

        assert len(categories) == 20
        assert categories == [card.model_outputs[i] for i in indices]
        # Every window is the width the head was trained on — a head fed a
        # different span sees a different amount of the production cycle.
        assert {shape[2] for shape in head.calls} == {8}

    def test_a_clip_shorter_than_the_window_still_predicts_every_sample(self) -> None:
        card = _card(window=64)
        head = _FakeSession("embeddings", _echo_head(len(card.model_outputs)))
        classifier = StationClassifier(backbone=None, head=head, card=card)

        categories = classifier.categories(_embeddings([0, 1, 2]))

        assert categories == ["spawanie", "ukladanie_pretow", "pozostale"]
        assert head.calls == [(1, 768, 3)]

    def test_no_samples_produce_no_categories_and_no_head_call(self) -> None:
        card = _card()
        head = _FakeSession("embeddings", _echo_head(len(card.model_outputs)))
        classifier = StationClassifier(backbone=None, head=head, card=card)

        assert classifier.categories(np.zeros((0, 768), dtype=np.float32)) == []
        assert head.calls == []


class _FakeOrt:
    """The onnxruntime module surface this loader touches, and nothing else."""

    def __init__(self, available=("CUDAExecutionProvider",), active=None) -> None:
        self._available = list(available)
        self._active = list(active if active is not None else available)
        self.preloaded = False
        self.sessions: list[tuple[str, list[str]]] = []

    def get_available_providers(self):
        return list(self._available)

    def preload_dlls(self, cuda: bool = False, cudnn: bool = False):
        self.preloaded = True

    def InferenceSession(self, path, providers):  # noqa: N802 — mirrors onnxruntime
        self.sessions.append((path, list(providers)))
        active = self._active
        return SimpleNamespace(
            get_providers=lambda: list(active),
            get_inputs=lambda: [SimpleNamespace(name="x", shape=[1, 768, "time"])],
        )


def _artefacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    head = tmp_path / "station-head-hala-prawe-v1-v2.0.0.onnx"
    head.write_bytes(b"head-weights")
    backbone = tmp_path / "dinov2-base.onnx"
    backbone.write_bytes(b"backbone-weights")
    card = tmp_path / "station-head-hala-prawe-v1-v2.0.0.card.json"
    # The real card with the one field a re-exported head carries. The shipped
    # v1.0.0 document predates the removal of the centre-crop and is refused by
    # `StationCard` on purpose, so these session tests - which are about CUDA and
    # sha256, not about preprocessing - state the current contract rather than
    # inheriting a document the loader is designed to reject.
    document = json.loads(REPO_ROOT.joinpath(SHIPPED_CARD_NAME).read_text(encoding="utf-8"))
    document["backbone"]["preprocessing"] = {"model_input": [420, 882], "center_crop": False}
    card.write_text(json.dumps(document))
    return head, card, backbone


class TestLoadingTheSessions:
    def test_it_records_the_weights_that_produced_the_numbers(self, tmp_path: Path) -> None:
        head, card, backbone = _artefacts(tmp_path)

        classifier = load_station_classifier(head, card, backbone, ort_module=_FakeOrt())

        assert classifier.head_sha256 == hashlib.sha256(b"head-weights").hexdigest()
        assert classifier.card.station_id == "hala-prawe-v1"

    def test_it_refuses_to_run_without_cuda(self, tmp_path: Path) -> None:
        # The repo rule, unchanged here: no CPU fallback, because it breaks the
        # 1:1 processing guarantee the whole schedule is costed against.
        head, card, backbone = _artefacts(tmp_path)

        with pytest.raises(RuntimeError) as exc:
            load_station_classifier(
                head, card, backbone, ort_module=_FakeOrt(available=("CPUExecutionProvider",))
            )

        assert "CUDAExecutionProvider" in str(exc.value)

    def test_it_catches_a_silent_fallback_after_the_session_opens(self, tmp_path: Path) -> None:
        """`get_available_providers` is not enough — microsoft/onnxruntime#25145.

        CUDA can register and then be inactive on the session, which looks like a
        working run at a third of the speed and on different arithmetic.
        """
        head, card, backbone = _artefacts(tmp_path)

        with pytest.raises(RuntimeError) as exc:
            load_station_classifier(
                head,
                card,
                backbone,
                ort_module=_FakeOrt(
                    available=("CUDAExecutionProvider",), active=("CPUExecutionProvider",)
                ),
            )

        assert "refusing CPU fallback" in str(exc.value)

    def test_it_preloads_the_cuda_libraries_before_opening_a_session(self, tmp_path: Path) -> None:
        # The onnxruntime-gpu wheel has no RPATH to site-packages/nvidia, so a
        # session opened before this silently lands on CPU.
        head, card, backbone = _artefacts(tmp_path)
        ort = _FakeOrt()

        load_station_classifier(head, card, backbone, ort_module=ort)

        assert ort.preloaded
        assert [providers for _, providers in ort.sessions] == [
            ["CUDAExecutionProvider"],
            ["CUDAExecutionProvider"],
        ]
