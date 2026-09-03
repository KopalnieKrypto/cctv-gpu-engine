"""Station activity inference: zone crop → frozen backbone → temporal head (#123).

The chronometraż at one station answers a duration question — how much of the
shift went on welding, how much on laying rods, how much on everything else — and
the client agreed the **zone** is the unit of measurement, not a person
(assumption A5). So this path is deliberately not the person pipeline:

    zone crop from the native frame at a fixed stride
      → frozen image backbone
      → small temporal head
      → per-sample category

No pose model, no OSNet, no VLM. That removes three heavy components rather than
adding a fourth, and it is why this arm costs ~49 GPU-seconds per video-hour
where the pose-based one costs 555.

The crop is taken from the **native** frame and never from a downscaled copy. The
client declined to reframe the camera (assumption A2), so the whole approach
rests on cropping full-resolution pixels we already receive; taking the crop from
the 1280x736 pose downscale would discard the 3x magnification that makes station
framing work at all.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.station_card import Rect, StationCard, StationError
from pipeline.zones import StationRuleset, Zone, ZoneConfig

DEFAULT_STATION_HEAD_PATH = "models/station-head-hala-prawe-v1-v2.0.0.onnx"
DEFAULT_STATION_CARD_PATH = "models/station-head-hala-prawe-v1-v2.0.0.card.json"
DEFAULT_BACKBONE_PATH = "models/dinov2-base.onnx"

CUDA_PROVIDER = "CUDAExecutionProvider"


class StationZoneError(StationError):
    """Raised when a zones config does not name exactly one station zone."""


class StationCropError(StationError):
    """Raised when the station rectangle does not fit inside the native frame."""


class StationRectMismatchError(StationError):
    """Raised when the configured zone is not the rectangle the head was fitted on."""


def resolve_station_zone(zones: ZoneConfig) -> Zone:
    """The single zone this run measures, from ``rules.type: "station"``.

    Raises:
        StationZoneError: when no zone declares the station ruleset, or more than
            one does. #123 is scoped to one station, and picking the first of two
            would report one bench's totals under a config that asked for both.
    """
    stations = [z for z in zones.zones if z.rules.get("type") == StationRuleset.type]
    if not stations:
        raise StationZoneError(
            'no zone in the zones config declares `"rules": {"type": "station"}`, '
            "so there is no station to measure. The station classifier reads its "
            "crop from that zone's polygon."
        )
    if len(stations) > 1:
        named = ", ".join(repr(z.id) for z in stations)
        raise StationZoneError(
            f"{len(stations)} zones declare the station ruleset ({named}), and this "
            "build measures one station per run. Split them across runs rather than "
            "reporting one bench's totals under a config that asked for both."
        )
    return stations[0]


def zone_rect(zone: Zone) -> Rect:
    """The zone polygon's bounding box as ``(x, y, w, h)`` in native pixels."""
    xs = [p[0] for p in zone.polygon]
    ys = [p[1] for p in zone.polygon]
    x, y = int(round(min(xs))), int(round(min(ys)))
    return (x, y, int(round(max(xs))) - x, int(round(max(ys))) - y)


# `facebook/dinov2-base`'s preprocessor, transcribed rather than resolved through
# the hub at run time. Fetching `preprocessor_config.json` on a live system would
# make every embedding depend on what HuggingFace serves that day — the same
# failure mode `setup-models.sh` pins the weights against — and a changed resize
# or normalisation silently moves the feature space the head was fitted on.
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

#: DINOv2 splits its input into 14x14 patches, so both input dimensions have to
#: be multiples of 14 or the position embeddings are interpolated to a grid the
#: weights were not fitted on.
PATCH = 14


def model_input_for_rect(rect: Rect, target_height: int) -> tuple[int, int]:
    """The ``(height, width)`` tensor size for a station rectangle.

    Derived from the rectangle rather than typed, so the tensor keeps the bench's
    own aspect ratio to within one patch. Both dimensions are rounded to a
    multiple of :data:`PATCH`, and the width is rounded to nearest rather than
    down — down would shave a column of patches off the wider axis, which is the
    axis the rectangle was widened along in the first place.

    One function, used by the exporter, the training embedder and the card
    generator, because the previous arrangement had two independent
    implementations of one convention and they were both wrong in the same way.
    """
    _, _, width, height = rect
    if width < 1 or height < 1:
        raise ValueError(f"station rectangle {width}x{height} is not an image")
    h = max(PATCH, round(target_height / PATCH) * PATCH)
    w = max(PATCH, round(h * width / height / PATCH) * PATCH)
    return (h, w)


def preprocess_crop(crop_bgr: np.ndarray, model_input: tuple[int, int]) -> np.ndarray:
    """One station crop → the ``(1, 3, h, w)`` tensor the exported backbone takes.

    **The whole rectangle, resized once.** Convert to RGB, **bicubic** resize
    straight to ``model_input``, rescale to ``[0, 1]``, normalise by the ImageNet
    statistics. No centre-crop, and the card must say so (`center_crop: false`)
    or it does not load.

    That second step used to be here, and it was never intended. DINOv2's
    processor takes ``size`` and ``crop_size`` independently; the training code
    set ``size=518`` to give the model a high-resolution view — the record says as
    much, "224 px leaves a rod a couple of pixels wide" — and left ``crop_size``
    at its 224 default. The result was the opposite of the intent: the head read
    the middle 224/518 of each axis, 18.7% of the rectangle's area, magnified.
    Every document, the panel and ``zone_native_px`` reported the full rectangle.
    Nothing failed. A production run simply reported ``spawanie: 0.0 s`` across an
    hour in which the welder is on camera striking arcs, because the bench had
    been rearranged and the work now sat outside that hidden centre.

    The aspect ratio is not preserved, deliberately: the rectangle is squeezed
    into ``model_input`` exactly as it is during training, and ``model_input`` is
    chosen close to the rectangle's own ratio so the squeeze is a few percent
    rather than a halving. Letterboxing instead would spend patches on grey bars
    and change the feature space for no gain.

    PIL does the resize because PIL did it during training. cv2's ``INTER_CUBIC``
    is a different kernel and would shift every embedding by more than the head's
    decision boundaries can absorb.

    One difference from training is not emulated and is recorded instead: the
    training crops were JPEG-encoded by ffmpeg at ``-q:v 2`` before the processor
    saw them, while here the pixels come straight from the decoder. That is a real
    domain shift, and a smaller one than re-encoding every frame to imitate it.
    """
    from PIL import Image

    height, width = model_input
    image = Image.fromarray(crop_bgr[:, :, ::-1]).resize(
        (width, height), resample=Image.Resampling.BICUBIC
    )

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)


def _describe(rect: Rect) -> str:
    x, y, w, h = rect
    return f"{w}x{h} at ({x}, {y})"


def station_rect(zone: Zone, card: StationCard) -> Rect:
    """The rectangle to crop, once the config and the card are shown to agree.

    A head fed a rectangle it was not fitted on does not fail — it returns
    confident logits over embeddings from pixels it has never seen, and nothing
    downstream would report that. This is the same silent substitution the sha256
    pins in ``setup-models.sh`` exist to catch, so the disagreement stops the run.

    Raises:
        StationRectMismatchError: when the zone's bounding box is not the card's
            ``zone_native_px``.
    """
    configured = zone_rect(zone)
    if configured != card.zone_rect:
        raise StationRectMismatchError(
            f"zone {zone.id!r} is {_describe(configured)}, but the station head "
            f"{card.station_id} v{card.version} was fitted on "
            f"{_describe(card.zone_rect)} (native pixels). The head would score "
            "embeddings from a rectangle it never saw. Author the zone polygon as "
            "the card's rectangle, or ship a head trained on this one."
        )
    return configured


@dataclass
class StationClassifier:
    """The frozen backbone and the station's temporal head, as one reader.

    Two ONNX sessions with one job between them: turn native station crops into a
    category per sample. The split is the economics of the offer — the backbone
    is identical at every station and ships once inside the container image, and
    what is station-specific is the ~1 MB head. Onboarding another bench is
    "annotate twenty minutes, train a head", with no new large model and no engine
    redeploy.
    """

    backbone: Any
    head: Any
    card: StationCard
    head_sha256: str = ""

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """One native station crop → its frozen-backbone CLS vector."""
        tensor = preprocess_crop(crop_bgr, self.card.model_input)
        name = self.backbone.get_inputs()[0].name
        return np.asarray(self.backbone.run(None, {name: tensor})[0][0], dtype=np.float32)

    def categories(self, embeddings: np.ndarray) -> list[str]:
        """Per-sample delivered categories for a whole clip's embeddings.

        ``embeddings`` is ``(samples, feature)`` in time order. Scored with
        overlapping windows of the width the head was trained on, logits averaged,
        so every sample is predicted with as much surrounding context as the clip
        allows — the same inference the cross-validated folds used, which is what
        makes the card's figures describe this path.

        A clip shorter than the window is scored in one pass over what there is.
        The head is fully convolutional in time, so that is a shorter sequence and
        not a truncated one; it simply sees less context, which is the honest
        consequence of a short clip.
        """
        count = len(embeddings)
        if count == 0:
            return []

        name = self.head.get_inputs()[0].name
        span = min(count, self.card.window)
        n_class = len(self.card.model_outputs)
        totals = np.zeros((count, n_class), dtype=np.float32)
        counts = np.zeros((count, 1), dtype=np.float32)
        for start in range(0, count - span + 1):
            segment = embeddings[start : start + span].T[None].astype(np.float32)
            logits = np.asarray(self.head.run(None, {name: segment})[0][0], dtype=np.float32)
            totals[start : start + span] += logits.T
            counts[start : start + span] += 1

        winners = (totals / np.maximum(counts, 1)).argmax(axis=1)
        return [self.card.deliver(self.card.model_outputs[int(i)]) for i in winners]


def station_crop(frame: np.ndarray, rect: Rect) -> np.ndarray:
    """Cut ``rect`` out of a native-resolution ``frame``.

    ``frame`` is the full-resolution BGR frame as decoded, so the returned array
    is exactly ``rect``'s pixels at their original scale.

    A rectangle that runs off the frame is refused rather than clipped. Both
    obvious ways of handling it are silent: numpy slicing returns a shorter array
    and ffmpeg's ``crop`` filter slides the rectangle back inside. The second one
    is not hypothetical — it is why this fixture's own crops were cut 40 px above
    the offset every record of them states.

    Raises:
        StationCropError: when ``rect`` is not wholly inside ``frame``.
    """
    x, y, w, h = rect
    height, width = frame.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise StationCropError(
            f"the station rectangle {w}x{h} at ({x}, {y}) is not inside the "
            f"{width}x{height} frame. Cropping it anyway would silently measure a "
            "different rectangle than the one configured — numpy returns a short "
            "array and ffmpeg slides the rectangle back inside, and neither says so."
        )
    return frame[y : y + h, x : x + w]


def _cuda_session(ort_module: Any, path: Path, what: str) -> Any:
    """Open a CUDA-only ONNX session, or refuse.

    Two checks, not one. ``get_available_providers`` reports what the build
    supports; a provider can register and still be inactive on the session
    (microsoft/onnxruntime#25145), which looks like a working run at a fraction of
    the speed and on different arithmetic. There is no CPU fallback here for the
    same reason the rest of this engine has none — it breaks the 1:1 processing
    guarantee the schedule is costed against.
    """
    available = ort_module.get_available_providers()
    if CUDA_PROVIDER not in available:
        raise RuntimeError(
            f"{CUDA_PROVIDER} not available for the {what} — GPU required, no CPU "
            f"fallback. Available providers: {available}"
        )
    session = ort_module.InferenceSession(str(path), providers=[CUDA_PROVIDER])
    active = session.get_providers()
    if CUDA_PROVIDER not in active:
        raise RuntimeError(
            f"{CUDA_PROVIDER} registered but inactive after the {what} session init "
            f"(active providers: {active}) — refusing CPU fallback"
        )
    return session


def load_station_classifier(
    head_path: str | Path,
    card_path: str | Path,
    backbone_path: str | Path,
    *,
    ort_module: Any | None = None,
) -> StationClassifier:
    """Open the frozen backbone and the station head, with the card that describes them.

    The head's sha256 is read here and travels into ``station_activity``. Mounting
    other weights over ``./models`` is the documented way to run a different model,
    and without the digest in the artefact it would change every total in silence.
    """
    if ort_module is None:
        import onnxruntime as ort_module

    card = StationCard.load(card_path)
    head = Path(head_path)
    # Before any session: the onnxruntime-gpu wheel has no RPATH to
    # site-packages/nvidia, so a session opened first lands silently on CPU.
    ort_module.preload_dlls(cuda=True, cudnn=True)
    return StationClassifier(
        backbone=_cuda_session(ort_module, Path(backbone_path), "station backbone"),
        head=_cuda_session(ort_module, head, "station head"),
        card=card,
        head_sha256=hashlib.sha256(head.read_bytes()).hexdigest(),
    )
