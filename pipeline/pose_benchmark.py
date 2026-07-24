"""Measured pose-mode evaluation for the bending-station pilot (issue #86)."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import IO, Any

from pipeline.postprocessing import CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, Detection, _iou
from pipeline.preprocessing import input_wh
from pipeline.zones import Zone, ZoneConfig, foot_point

MIN_RECALL = 0.90
MIN_PRECISION = 0.95
MAX_THROUGHPUT_REGRESSION = 0.10
MAX_ONE_HOUR_WALLCLOCK_S = 22 * 60
MAX_PROCESS_GPU_VRAM_MB = 8 * 1024
HEARTBEAT_INTERVAL_S = 60.0


def query_process_gpu_vram_mb(
    pid: int,
    run: Callable[..., Any] = subprocess.run,
) -> float:
    """Return this process's GPU allocation from ``nvidia-smi`` in MiB."""
    completed = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    total = 0.0
    for line in completed.stdout.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 2:
            continue
        try:
            row_pid = int(columns[0])
            used_mb = float(columns[1])
        except ValueError:
            continue
        if row_pid == pid:
            total += used_mb
    return total


class PeakProcessVramMonitor:
    """Sample one PID during a run and retain its peak GPU allocation."""

    def __init__(
        self,
        *,
        pid: int | None = None,
        sample: Callable[[int], float] = query_process_gpu_vram_mb,
        interval_s: float = 0.5,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("VRAM sample interval must be positive")
        self.pid = os.getpid() if pid is None else pid
        self.sample = sample
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _take_sample(self) -> None:
        self.peak_mb = max(self.peak_mb, self.sample(self.pid))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._take_sample()
            except Exception as exc:
                self._error = exc
                self._stop.set()

    def __enter__(self) -> PeakProcessVramMonitor:
        self._take_sample()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self._thread.join()
        if exc_type is None and self._error is not None:
            raise RuntimeError(f"failed to sample process GPU VRAM: {self._error}") from self._error


class BenchmarkHeartbeat:
    """Flush progress and a partial result artifact throughout a long arm run."""

    def __init__(
        self,
        *,
        label: str,
        partial_path: Path,
        snapshot: Callable[[], dict[str, Any]],
        stream: IO[str] = sys.stderr,
        interval_s: float = HEARTBEAT_INTERVAL_S,
    ) -> None:
        if interval_s <= 0 or interval_s > HEARTBEAT_INTERVAL_S:
            raise ValueError(
                f"heartbeat interval must be > 0 and <= {HEARTBEAT_INTERVAL_S:g} seconds"
            )
        self.label = label
        self.partial_path = partial_path
        self.snapshot = snapshot
        self.stream = stream
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._write_lock = threading.Lock()

    def _write_partial(self) -> None:
        payload = self.snapshot()
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.partial_path.with_suffix(self.partial_path.suffix + ".tmp")
        with self._write_lock:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.partial_path)

    def checkpoint(self) -> None:
        """Persist progress immediately, in addition to timed heartbeats."""
        self._write_partial()

    def _emit(self) -> None:
        self._write_partial()
        print(f"BENCHMARK_HEARTBEAT arm={self.label}", file=self.stream, flush=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._emit()

    def __enter__(self) -> BenchmarkHeartbeat:
        self._emit()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self._thread.join()
        self._write_partial()


@dataclass(frozen=True)
class DetectionScore:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class BenchmarkFrame:
    id: str
    window_id: str
    image: Any
    ground_truth: list[list[float]]


@dataclass(frozen=True)
class DetectionArmRun:
    name: str
    tp: int
    fp: int
    fn: int
    pose_wallclock_s: list[float]
    frames: list[dict[str, Any]]


@dataclass(frozen=True)
class EndToEndMeasurement:
    wallclock_s: float
    measured_video_duration_s: float
    measured_frame_count: int


@dataclass(frozen=True)
class ArmMetrics:
    name: str
    tp: int
    fp: int
    fn: int
    pose_wallclock_s: list[float]
    end_to_end_wallclock_s: float
    measured_video_duration_s: float
    peak_process_gpu_vram_mb: float
    film_recall: dict[str, float]
    frame_count: int = 0

    @property
    def precision(self) -> float:
        return _metrics(self.tp, self.fp, self.fn).precision

    @property
    def recall(self) -> float:
        return _metrics(self.tp, self.fp, self.fn).recall

    @property
    def f1(self) -> float:
        return _metrics(self.tp, self.fp, self.fn).f1

    @property
    def one_hour_extrapolated_s(self) -> float:
        return self.end_to_end_wallclock_s / self.measured_video_duration_s * 3600


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def arm_metrics_to_dict(metrics: ArmMetrics) -> dict[str, Any]:
    """Serialize one arm with reproducible timing/extrapolation evidence."""
    return {
        "name": metrics.name,
        "quality": {
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        },
        "pose_inference_wallclock_s": {
            "samples": metrics.pose_wallclock_s,
            "p50": _percentile(metrics.pose_wallclock_s, 0.50),
            "p95": _percentile(metrics.pose_wallclock_s, 0.95),
        },
        "end_to_end": {
            "measured_frame_count": metrics.frame_count,
            "measured_video_duration_s": metrics.measured_video_duration_s,
            "measured_wallclock_s": metrics.end_to_end_wallclock_s,
            "one_hour_extrapolated_s": metrics.one_hour_extrapolated_s,
            "formula": (
                "one_hour_extrapolated_s = measured_wallclock_s / measured_video_duration_s * 3600"
            ),
            "linearity_assumption": (
                "Wallclock scales linearly with decoded video duration at the "
                "locked sampling rate, model, classifier, and hardware."
            ),
        },
        "peak_process_gpu_vram_mb": metrics.peak_process_gpu_vram_mb,
        "film_recall": metrics.film_recall,
    }


@dataclass(frozen=True)
class BoundCheck:
    name: str
    measured: float
    operator: str
    limit: float
    passed: bool


@dataclass(frozen=True)
class EligibilityResult:
    arm: str
    eligible: bool
    checks: list[BoundCheck]


@dataclass(frozen=True)
class WinnerSelection:
    winner: str | None
    eligibility: list[EligibilityResult]


class BenchmarkConfigError(ValueError):
    """Fixture or run configuration cannot produce comparable evidence."""


# The exact ``(width, height)`` each arm is measured at. An arm's identity is
# its input shape, so this is declared rather than inferred: scoring a 640×384
# export while the harness believes it ran 640×640 would put two different
# experiments in the same row of the gate table. ``baseline_640x384`` (issue
# #100) has the same *width* as ``baseline_640`` — for a 16:9 frame the width
# ratio alone sets detection scale — so it tests compute, not resolution.
ARM_INPUT_SIZES: dict[str, tuple[int, int]] = {
    "baseline_640": (640, 640),
    "baseline_640x384": (640, 384),
    "full_frame_1280": (1280, 1280),
    # Issue #101: double baseline_640's width — so double the detection scale —
    # at 2.24x its measured cost rather than full_frame_1280's 3.87x.
    "full_frame_1280x736": (1280, 736),
    "focused_roi_640": (640, 640),
    # Issue #110: native-resolution tiling. Both arms run the same 1280x736 model
    # per tile — a 1280x736 native tile letterboxes at scale 1.0, so an 80-120 px
    # person keeps their pixels instead of shrinking below the ~60 px input floor
    # a full-frame downscale forced them under. The arms differ only in reach:
    # the whole frame (grid) versus only the authored zones (focused compute).
    "tiled_1280x736": (1280, 736),
    "tiled_zones_1280x736": (1280, 736),
}

# The full-frame arms measured by ``run-arm`` (#86/#101). Kept distinct from the
# tiling arms so ``run-arm`` cannot silently measure a tiling arm as a plain
# full-frame detector — the input size is identical, only the mode differs.
FULL_FRAME_ARMS: tuple[str, ...] = (
    "baseline_640",
    "baseline_640x384",
    "full_frame_1280",
    "full_frame_1280x736",
    "focused_roi_640",
)

# The tiling arms measured by ``run-tiling-arm`` (#110).
TILING_ARMS: tuple[str, ...] = ("tiled_1280x736", "tiled_zones_1280x736")

# Fraction of a tile shared with each neighbour (#110). 20% keeps a person on a
# tile seam whole inside at least one tile at the cost of the extra tiles.
DEFAULT_TILE_OVERLAP = 0.2


def expected_input_size_for_arm(arm: str) -> tuple[int, int]:
    """Return the ``(width, height)`` an arm's model must declare."""
    try:
        return ARM_INPUT_SIZES[arm]
    except KeyError:
        raise BenchmarkConfigError(
            f"unknown benchmark arm {arm!r}; known arms: {sorted(ARM_INPUT_SIZES)}"
        ) from None


@dataclass(frozen=True)
class FixtureSummary:
    fixture_id: str
    frame_count: int
    window_ids: tuple[str, ...]
    annotation_methodology: str


@dataclass(frozen=True)
class BenchmarkFixture:
    summary: FixtureSummary
    frames: list[BenchmarkFrame]


def evaluate_eligibility(candidate: ArmMetrics, baseline: ArmMetrics) -> EligibilityResult:
    """Evaluate every locked numeric selection bound from issue #86."""
    candidate_rate = candidate.end_to_end_wallclock_s / candidate.measured_video_duration_s
    baseline_rate = baseline.end_to_end_wallclock_s / baseline.measured_video_duration_s
    regression = candidate_rate / baseline_rate - 1
    checks = [
        BoundCheck(
            "pilot_recall", candidate.recall, ">=", MIN_RECALL, candidate.recall >= MIN_RECALL
        ),
        BoundCheck(
            "pilot_precision",
            candidate.precision,
            ">=",
            MIN_PRECISION,
            candidate.precision >= MIN_PRECISION,
        ),
        BoundCheck(
            "throughput_regression",
            regression,
            "<=",
            MAX_THROUGHPUT_REGRESSION,
            regression <= MAX_THROUGHPUT_REGRESSION,
        ),
        BoundCheck(
            "one_hour_extrapolated_s",
            candidate.one_hour_extrapolated_s,
            "<=",
            MAX_ONE_HOUR_WALLCLOCK_S,
            candidate.one_hour_extrapolated_s <= MAX_ONE_HOUR_WALLCLOCK_S,
        ),
        BoundCheck(
            "peak_process_gpu_vram_mb",
            candidate.peak_process_gpu_vram_mb,
            "<=",
            MAX_PROCESS_GPU_VRAM_MB,
            candidate.peak_process_gpu_vram_mb <= MAX_PROCESS_GPU_VRAM_MB,
        ),
    ]
    for film_name, baseline_recall in sorted(baseline.film_recall.items()):
        candidate_recall = candidate.film_recall.get(film_name, float("-inf"))
        checks.append(
            BoundCheck(
                f"{film_name}_recall_no_regression",
                candidate_recall,
                ">=",
                baseline_recall,
                candidate_recall >= baseline_recall,
            )
        )
    return EligibilityResult(
        arm=candidate.name,
        eligible=all(check.passed for check in checks),
        checks=checks,
    )


def select_winner(arms: list[ArmMetrics], baseline_name: str) -> WinnerSelection:
    """Choose the eligible arm with the lowest normalized end-to-end wallclock."""
    baseline = next(arm for arm in arms if arm.name == baseline_name)
    eligibility = [evaluate_eligibility(arm, baseline) for arm in arms]
    by_name = {arm.name: arm for arm in arms}
    eligible = [result.arm for result in eligibility if result.eligible]
    winner = min(
        eligible,
        key=lambda name: (
            by_name[name].end_to_end_wallclock_s / by_name[name].measured_video_duration_s,
            name,
        ),
        default=None,
    )
    return WinnerSelection(winner=winner, eligibility=eligibility)


def validate_fixture_manifest(
    manifest: object,
    *,
    minimum_frame_count: int = 60,
    minimum_window_count: int = 3,
) -> FixtureSummary:
    """Validate the locked minimum fixture size and return its summary."""
    if not isinstance(manifest, dict):
        raise BenchmarkConfigError("fixture manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise BenchmarkConfigError("fixture manifest schema_version must be 1")
    fixture_id = manifest.get("fixture_id")
    methodology = manifest.get("annotation_methodology")
    frames = manifest.get("frames")
    if not isinstance(fixture_id, str) or not fixture_id:
        raise BenchmarkConfigError("fixture manifest needs a non-empty fixture_id")
    if not isinstance(frames, list):
        raise BenchmarkConfigError("fixture manifest needs a frames list")
    if not isinstance(methodology, str) or not methodology:
        raise BenchmarkConfigError("fixture manifest needs an annotation_methodology reference")
    windows = tuple(
        sorted(
            {
                frame.get("window_id")
                for frame in frames
                if isinstance(frame, dict) and isinstance(frame.get("window_id"), str)
            }
        )
    )
    if len(frames) < minimum_frame_count:
        raise BenchmarkConfigError(
            f"fixture must contain at least {minimum_frame_count} annotated frames, "
            f"found {len(frames)}"
        )
    if len(windows) < minimum_window_count:
        raise BenchmarkConfigError(
            f"fixture must contain at least {minimum_window_count} distinct recording "
            f"windows, found {len(windows)}"
        )
    return FixtureSummary(
        fixture_id=fixture_id,
        frame_count=len(frames),
        window_ids=windows,
        annotation_methodology=methodology,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path) -> Any:
    import cv2

    return cv2.imread(str(path))


def load_benchmark_fixture(
    manifest_path: Path,
    *,
    zone: Zone | None,
    image_reader: Callable[[Path], Any] = _read_image,
    sha256_file: Callable[[Path], str] = _sha256_file,
    minimum_frame_count: int = 60,
    minimum_window_count: int = 3,
) -> BenchmarkFixture:
    """Load a checksummed, manually annotated fixture into memory one frame at a time."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = validate_fixture_manifest(
        manifest,
        minimum_frame_count=minimum_frame_count,
        minimum_window_count=minimum_window_count,
    )
    base = manifest_path.parent.resolve()
    loaded: list[BenchmarkFrame] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(manifest["frames"]):
        if not isinstance(raw, dict):
            raise BenchmarkConfigError(f"frame {index} must be a JSON object")
        frame_id = raw.get("id")
        window_id = raw.get("window_id")
        relative_path = raw.get("path")
        expected_sha = raw.get("sha256")
        persons = raw.get("persons")
        if not isinstance(frame_id, str) or not frame_id or frame_id in seen_ids:
            raise BenchmarkConfigError(f"frame {index} needs a unique non-empty id")
        seen_ids.add(frame_id)
        if not isinstance(window_id, str) or not window_id:
            raise BenchmarkConfigError(f"frame {frame_id!r} needs a non-empty window_id")
        if not isinstance(relative_path, str) or not relative_path:
            raise BenchmarkConfigError(f"frame {frame_id!r} needs a relative path")
        path = (base / relative_path).resolve()
        if not path.is_relative_to(base):
            raise BenchmarkConfigError(f"frame {frame_id!r} path must stay inside the fixture")
        if not path.is_file():
            raise BenchmarkConfigError(f"frame {frame_id!r} file does not exist: {path}")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha.lower())
        ):
            raise BenchmarkConfigError(f"frame {frame_id!r} needs a 64-character sha256")
        actual_sha = sha256_file(path)
        if actual_sha.lower() != expected_sha.lower():
            raise BenchmarkConfigError(
                f"frame {frame_id!r} sha256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        image = image_reader(path)
        if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
            raise BenchmarkConfigError(f"frame {frame_id!r} is not a readable image")
        frame_height, frame_width = image.shape[:2]
        if not isinstance(persons, list):
            raise BenchmarkConfigError(f"frame {frame_id!r} needs a persons list")
        boxes: list[list[float]] = []
        for person_index, person in enumerate(persons):
            bbox = person.get("bbox") if isinstance(person, dict) else None
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    for value in bbox
                )
            ):
                raise BenchmarkConfigError(
                    f"frame {frame_id!r} person {person_index} needs a finite xyxy bbox"
                )
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if not (0 <= x1 < x2 <= frame_width and 0 <= y1 < y2 <= frame_height):
                raise BenchmarkConfigError(
                    f"frame {frame_id!r} person {person_index} bbox must be inside the frame"
                )
            if zone is not None and not zone.contains((x1 + x2) / 2, y2):
                raise BenchmarkConfigError(
                    f"frame {frame_id!r} person {person_index} foot point is outside "
                    f"zone {zone.id!r}"
                )
            boxes.append([x1, y1, x2, y2])
        loaded.append(
            BenchmarkFrame(
                id=frame_id,
                window_id=window_id,
                image=lambda path=path: image_reader(path),
                ground_truth=boxes,
            )
        )
    return BenchmarkFixture(summary=summary, frames=loaded)


def build_results_artifact(
    *,
    fixture: BenchmarkFixture,
    metrics: list[ArmMetrics],
    raw_frames: dict[str, list[dict[str, Any]]],
    model_evidence: dict[str, dict[str, Any]],
    reference_tiling: dict[str, Any],
    follow_up_issue: str | None = None,
    production_default_changed: bool = False,
) -> dict[str, Any]:
    """Combine raw and aggregate evidence into the versioned result contract."""
    expected = {"baseline_640", "full_frame_1280", "focused_roi_640"}
    metric_names = {item.name for item in metrics}
    if metric_names != expected or set(raw_frames) != expected or set(model_evidence) != expected:
        raise BenchmarkConfigError(
            "results need exactly baseline_640, full_frame_1280, and focused_roi_640 arms"
        )
    selection = select_winner(metrics, baseline_name="baseline_640")
    eligibility_by_arm = {item.arm: item for item in selection.eligibility}
    arms: dict[str, Any] = {}
    for item in metrics:
        arm = arm_metrics_to_dict(item)
        arm["model"] = model_evidence[item.name]
        arm["frames"] = raw_frames[item.name]
        arm["eligibility_checks"] = [
            asdict(check) for check in eligibility_by_arm[item.name].checks
        ]
        arms[item.name] = arm
    winner = selection.winner
    return {
        "schema_version": 1,
        "fixture": {
            "id": fixture.summary.fixture_id,
            "frame_count": fixture.summary.frame_count,
            "frame_ids": [frame.id for frame in fixture.frames],
            "window_ids": list(fixture.summary.window_ids),
            "annotation_methodology": fixture.summary.annotation_methodology,
        },
        "locked_settings": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "true_positive_iou_threshold": 0.5,
        },
        "arms": arms,
        "reference_tiling": reference_tiling,
        "decision": {
            "winner": winner,
            "production_default_changed": production_default_changed,
            "follow_up_issue": follow_up_issue if winner is None else None,
        },
    }


def _metrics(tp: int, fp: int, fn: int) -> DetectionScore:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionScore(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


# The recall-by-native-height bands #101 made the headline metric. Recall is
# flatly 0% below a ~60 px input-height floor and steps to ~70% above it, so the
# story is which band an arm reaches — the 80-120 px bucket (90 of 296 people)
# is the one tiling exists to recover. Half-open [lo, hi); the top band is open.
DEFAULT_HEIGHT_BANDS: list[tuple[float, float]] = [
    (0.0, 80.0),
    (80.0, 120.0),
    (120.0, 180.0),
    (180.0, 260.0),
    (260.0, float("inf")),
]


def match_ground_truth(
    detections: list[Detection],
    ground_truth: list[list[float]],
    iou_threshold: float = 0.5,
) -> set[int]:
    """Return the indices of ground-truth boxes matched one-to-one to detections.

    Detections are consumed in descending confidence, each claiming its
    highest-IoU still-unclaimed ground-truth box when the overlap clears
    ``iou_threshold``. The single matching rule shared by :func:`score_frame`
    (which reads the count as true positives) and :func:`recall_by_height`
    (which reads *which* people were found), so the two can never disagree.
    """
    unmatched = set(range(len(ground_truth)))
    matched: set[int] = set()
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        best = max(
            unmatched,
            key=lambda index: _iou(detection.bbox, ground_truth[index]),
            default=None,
        )
        if best is not None and _iou(detection.bbox, ground_truth[best]) >= iou_threshold:
            unmatched.remove(best)
            matched.add(best)
    return matched


def recall_by_height(
    frames: Iterable[tuple[list[Detection], list[list[float]]]],
    bands: list[tuple[float, float]] = DEFAULT_HEIGHT_BANDS,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Aggregate detection recall split by native ground-truth person height.

    ``frames`` yields ``(detections, ground_truth)`` per frame — the detections
    already in full-frame coordinates. Each ground-truth box is bucketed by its
    pixel height (``y2 - y1``) into the half-open band containing it and counted
    as found when :func:`match_ground_truth` paired it. Returns one record per
    band in ``bands`` order, so a zero-recall band is still reported rather than
    silently dropped.
    """
    people = [0 for _ in bands]
    found = [0 for _ in bands]
    for detections, ground_truth in frames:
        matched = match_ground_truth(detections, ground_truth, iou_threshold)
        for index, box in enumerate(ground_truth):
            height = box[3] - box[1]
            for band_index, (low, high) in enumerate(bands):
                if low <= height < high:
                    people[band_index] += 1
                    if index in matched:
                        found[band_index] += 1
                    break
    return [
        {
            "min_height": low,
            "max_height": None if high == float("inf") else high,
            "people": people[band_index],
            "matched": found[band_index],
            "recall": (found[band_index] / people[band_index]) if people[band_index] else 0.0,
        }
        for band_index, (low, high) in enumerate(bands)
    ]


def recall_by_height_from_evidence(
    evidence_frames: list[dict[str, Any]],
    benchmark_frames: list[BenchmarkFrame],
    bands: list[tuple[float, float]] = DEFAULT_HEIGHT_BANDS,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Recompute recall-by-height from a run's stored per-frame detection evidence.

    :func:`run_detection_arm` already persisted every detection's bbox and
    confidence, so the size breakdown reads back from that rather than paying a
    second GPU pass. Only bbox and confidence drive matching, so keypoints are
    left empty in the reconstructed detections.
    """
    frames = [
        (
            [
                Detection(
                    bbox=list(evidence["bbox"]),
                    confidence=float(evidence["confidence"]),
                    keypoints=[],
                )
                for evidence in evidence_frame["detections"]
            ],
            benchmark_frame.ground_truth,
        )
        for evidence_frame, benchmark_frame in zip(evidence_frames, benchmark_frames, strict=True)
    ]
    return recall_by_height(frames, bands=bands, iou_threshold=iou_threshold)


def pose_min_per_hour(pose_wallclock_s: list[float], fps: int) -> float:
    """Extrapolate the mean per-frame pose cost to minutes of pose per video hour.

    This is the detector's *isolated* cost (#110 methodology): a tiling arm runs
    N pose calls per frame, and ``pose_wallclock_s`` already sums them, so this
    is the per-hour floor that tiling's pose stage alone imposes — decode and the
    VLM (which scales with detections found) are read separately.
    """
    if not pose_wallclock_s:
        return 0.0
    mean_s = sum(pose_wallclock_s) / len(pose_wallclock_s)
    return mean_s * fps * 3600 / 60


def _polygon_bbox(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Axis-aligned ``xyxy`` bounding box of a polygon's vertices."""
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def zone_bounding_boxes(zones: ZoneConfig) -> list[tuple[float, float, float, float]]:
    """The ``xyxy`` bounding box of every authored zone — the with-zones tiling reach."""
    return [_polygon_bbox(zone.polygon) for zone in zones.zones]


def score_frame(
    detections: list[Detection],
    ground_truth: list[list[float]],
    zone: Zone | None,
    iou_threshold: float = 0.5,
) -> DetectionScore:
    """Score station detections using confidence-ordered one-to-one matching."""
    in_zone = (
        detections
        if zone is None
        else [detection for detection in detections if zone.contains(*foot_point(detection))]
    )
    matched = match_ground_truth(in_zone, ground_truth, iou_threshold)
    tp = len(matched)
    return _metrics(tp=tp, fp=len(in_zone) - tp, fn=len(ground_truth) - tp)


def _detection_to_evidence(detection: Detection) -> dict[str, Any]:
    return {
        "bbox": detection.bbox,
        "confidence": detection.confidence,
        "keypoints": [
            {"x": point.x, "y": point.y, "visibility": point.vis} for point in detection.keypoints
        ],
    }


def run_detection_arm(
    *,
    name: str,
    detector: Any,
    frames: list[BenchmarkFrame],
    zone: Zone,
    partial_path: Path,
    clock: Callable[[], float] = time.perf_counter,
    heartbeat_stream: IO[str] = sys.stderr,
    heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
) -> DetectionArmRun:
    """Run one pose arm frame-by-frame and persist raw partial evidence."""
    progress: dict[str, Any] = {
        "schema_version": 1,
        "arm": name,
        "frames_completed": 0,
        "frames_total": len(frames),
        "frames": [],
    }
    pose_times: list[float] = []
    tp = fp = fn = 0
    with BenchmarkHeartbeat(
        label=name,
        partial_path=partial_path,
        snapshot=lambda: deepcopy(progress),
        stream=heartbeat_stream,
        interval_s=heartbeat_interval_s,
    ) as heartbeat:
        for frame in frames:
            started = clock()
            image = frame.image() if callable(frame.image) else frame.image
            detections = detector.detect(image)
            elapsed = clock() - started
            score = score_frame(detections, frame.ground_truth, zone)
            pose_times.append(elapsed)
            tp += score.tp
            fp += score.fp
            fn += score.fn
            progress["frames"].append(
                {
                    "frame_id": frame.id,
                    "window_id": frame.window_id,
                    "pose_wallclock_s": elapsed,
                    "tp": score.tp,
                    "fp": score.fp,
                    "fn": score.fn,
                    "detections": [_detection_to_evidence(item) for item in detections],
                }
            )
            progress["frames_completed"] += 1
            heartbeat.checkpoint()
    return DetectionArmRun(
        name=name,
        tp=tp,
        fp=fp,
        fn=fn,
        pose_wallclock_s=pose_times,
        frames=progress["frames"],
    )


def measure_end_to_end(
    *,
    label: str,
    run: Callable[[Callable[[int], None]], Any],
    measured_video_duration_s: float,
    measured_frame_count: int,
    partial_path: Path,
    clock: Callable[[], float] = time.perf_counter,
    heartbeat_stream: IO[str] = sys.stderr,
    heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
) -> EndToEndMeasurement:
    """Measure one real pipeline run with flushed progress and calibration inputs."""
    progress: dict[str, Any] = {
        "arm": label,
        "stage": "end_to_end_vlm",
        "state": "running",
        "progress_pct": 0,
        "measured_video_duration_s": measured_video_duration_s,
        "measured_frame_count": measured_frame_count,
    }
    with BenchmarkHeartbeat(
        label=f"{label}:end_to_end_vlm",
        partial_path=partial_path,
        snapshot=lambda: deepcopy(progress),
        stream=heartbeat_stream,
        interval_s=heartbeat_interval_s,
    ) as heartbeat:

        def report(percent: int) -> None:
            progress["progress_pct"] = percent
            heartbeat.checkpoint()

        started = clock()
        reported_frame_count = run(report)
        wallclock = clock() - started
        if isinstance(reported_frame_count, int) and not isinstance(reported_frame_count, bool):
            measured_frame_count = reported_frame_count
        progress["state"] = "completed"
        progress["progress_pct"] = 100
        progress["measured_frame_count"] = measured_frame_count
        progress["measured_wallclock_s"] = wallclock
        heartbeat.checkpoint()
    return EndToEndMeasurement(
        wallclock_s=wallclock,
        measured_video_duration_s=measured_video_duration_s,
        measured_frame_count=measured_frame_count,
    )


def measure_fixture_recall(detector: Any, fixture: BenchmarkFixture) -> float:
    """Measure aggregate whole-frame recall for a regression fixture."""
    tp = fn = 0
    for frame in fixture.frames:
        image = frame.image() if callable(frame.image) else frame.image
        score = score_frame(detector.detect(image), frame.ground_truth, zone=None)
        tp += score.tp
        fn += score.fn
    return tp / (tp + fn) if tp + fn else 0.0


def _video_duration_s(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return float(completed.stdout.strip())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_arm_command(args: argparse.Namespace) -> int:
    from pipeline.analyze import run_full_video_to_json
    from pipeline.pose_detector import load_pose_model

    zones = ZoneConfig.load(args.zones)
    if zones.inference_roi is None:
        raise BenchmarkConfigError("zones config needs inference_roi for the focused arm")
    scoring_zone = next(zone for zone in zones.zones if zone.id == zones.inference_roi.zone_id)
    inference_zones = zones if args.arm == "focused_roi_640" else replace(zones, inference_roi=None)
    expected_input_size = expected_input_size_for_arm(args.arm)
    fixture = load_benchmark_fixture(Path(args.fixture), zone=scoring_zone)
    film_fixtures = {
        "film_1": load_benchmark_fixture(
            Path(args.film_1_fixture),
            zone=None,
            minimum_frame_count=1,
            minimum_window_count=1,
        ),
        "film_2": load_benchmark_fixture(
            Path(args.film_2_fixture),
            zone=None,
            minimum_frame_count=1,
            minimum_window_count=1,
        ),
    }
    clips = [Path(path) for path in args.throughput_clip]
    if not clips or any(not path.is_file() for path in clips):
        raise BenchmarkConfigError("every --throughput-clip must name an existing video")
    measured_duration_s = sum(_video_duration_s(path) for path in clips)
    output = Path(args.output)
    partial_dir = output.parent / f"{output.stem}.partial"

    with PeakProcessVramMonitor() as vram:
        detector = load_pose_model(args.model, zones=inference_zones)
        if input_wh(detector.input_size) != expected_input_size:
            raise BenchmarkConfigError(
                f"arm {args.arm} requires a fixed {expected_input_size} model, "
                f"but {args.model} declares {input_wh(detector.input_size)}"
            )
        detection_run = run_detection_arm(
            name=args.arm,
            detector=detector,
            frames=fixture.frames,
            zone=scoring_zone,
            partial_path=partial_dir / "detections.json",
        )
        regression_detector = replace(detector, zones=None)
        film_recall = {
            name: measure_fixture_recall(regression_detector, regression_fixture)
            for name, regression_fixture in film_fixtures.items()
        }
        del regression_detector
        del detector
        gc.collect()

        def run_pipeline(progress: Callable[[int], None]) -> int:
            result = run_full_video_to_json(
                chunks=clips,
                progress=progress,
                model_path=args.model,
                classifier="vlm",
                zones=inference_zones,
            )
            return int(json.loads(result)["total_frames"])

        end_to_end = measure_end_to_end(
            label=args.arm,
            run=run_pipeline,
            measured_video_duration_s=measured_duration_s,
            measured_frame_count=0,
            partial_path=partial_dir / "end_to_end.json",
        )

    metrics = ArmMetrics(
        name=args.arm,
        tp=detection_run.tp,
        fp=detection_run.fp,
        fn=detection_run.fn,
        pose_wallclock_s=detection_run.pose_wallclock_s,
        end_to_end_wallclock_s=end_to_end.wallclock_s,
        measured_video_duration_s=end_to_end.measured_video_duration_s,
        peak_process_gpu_vram_mb=vram.peak_mb,
        film_recall=film_recall,
        frame_count=end_to_end.measured_frame_count,
    )
    payload = arm_metrics_to_dict(metrics)
    payload.update(
        {
            "schema_version": 1,
            "fixture": fixture.summary.fixture_id,
            "frames": detection_run.frames,
            "model": {
                "path": args.model,
                "sha256": _sha256_file(Path(args.model)),
                # [w, h], matching result.json's diagnostics (issue #98/#100).
                "input_size": list(expected_input_size),
            },
        }
    )
    _write_json(output, payload)
    print(
        f"BENCHMARK_ARM_COMPLETE arm={args.arm} frames={fixture.summary.frame_count} "
        f"wallclock_s={end_to_end.wallclock_s:.6f} vram_mb={vram.peak_mb:.1f}",
        flush=True,
    )
    return 0


def _run_tiling_arm_command(args: argparse.Namespace) -> int:
    """Measure one native-resolution tiling arm (#110), detector-isolated.

    Scores whole-frame recall (the scoring zone stays the fixture's inference_roi
    zone, unchanged from #101) plus recall-by-native-height, and reports the
    pose-only per-hour cost and peak VRAM. No end-to-end VLM run: the tiling
    pipeline is not wired end-to-end yet (a follow-up), and #110 isolates the
    detector on purpose — the VLM scales with detections found and is read apart.
    """
    from pipeline.pose_detector import load_pose_model
    from pipeline.tiled_detector import TiledPoseDetector

    zones = ZoneConfig.load(args.zones)
    if zones.inference_roi is None:
        raise BenchmarkConfigError("zones config needs inference_roi for the tiling scoring zone")
    scoring_zone = next(zone for zone in zones.zones if zone.id == zones.inference_roi.zone_id)
    expected_input_size = expected_input_size_for_arm(args.arm)
    tile_w, tile_h = expected_input_size

    zone_bounds: list[tuple[float, float, float, float]] | None = None
    if args.arm == "tiled_zones_1280x736":
        if not args.roi_zones:
            raise BenchmarkConfigError(
                "the with-zones arm needs --roi-zones naming the authored zones to tile within"
            )
        zone_bounds = zone_bounding_boxes(ZoneConfig.load(args.roi_zones))
        if not zone_bounds:
            raise BenchmarkConfigError(
                f"--roi-zones {args.roi_zones} defines no zones to tile within"
            )

    fixture = load_benchmark_fixture(Path(args.fixture), zone=scoring_zone)
    output = Path(args.output)
    partial_dir = output.parent / f"{output.stem}.partial"

    with PeakProcessVramMonitor() as vram:
        base_detector = load_pose_model(args.model, zones=None)
        if input_wh(base_detector.input_size) != expected_input_size:
            raise BenchmarkConfigError(
                f"arm {args.arm} requires a fixed {expected_input_size} model, "
                f"but {args.model} declares {input_wh(base_detector.input_size)}"
            )
        detector = TiledPoseDetector(
            detector=base_detector,
            tile_w=tile_w,
            tile_h=tile_h,
            overlap=args.overlap,
            zone_bounds=zone_bounds,
        )
        detection_run = run_detection_arm(
            name=args.arm,
            detector=detector,
            frames=fixture.frames,
            zone=scoring_zone,
            partial_path=partial_dir / "detections.json",
        )

    height_recall = recall_by_height_from_evidence(detection_run.frames, fixture.frames)

    metrics = _metrics(detection_run.tp, detection_run.fp, detection_run.fn)
    total_detections = sum(len(frame["detections"]) for frame in detection_run.frames)
    payload = {
        "schema_version": 1,
        "name": args.arm,
        "fixture": fixture.summary.fixture_id,
        "mode": "tiling",
        "tiling": {
            "tile_size": list(expected_input_size),
            "overlap": args.overlap,
            "ios_threshold": detector.ios_threshold,
            "scope": "zones" if zone_bounds is not None else "whole_frame",
            "roi_zone_bounds": [list(box) for box in zone_bounds] if zone_bounds else None,
        },
        "quality": {
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        },
        "recall_by_height": height_recall,
        "pose_inference_wallclock_s": {
            "samples": detection_run.pose_wallclock_s,
            "p50": _percentile(detection_run.pose_wallclock_s, 0.50),
            "p95": _percentile(detection_run.pose_wallclock_s, 0.95),
        },
        "detector_cost": {
            "total_detections": total_detections,
            "pose_min_per_hour": pose_min_per_hour(detection_run.pose_wallclock_s, fps=1),
            "fps": 1,
            "formula": "mean(pose_wallclock_s) * fps * 3600 / 60",
            "note": (
                "Pose-only, detector-isolated (#110). The N tiles per frame are "
                "summed into each pose_wallclock_s sample; decode and VLM are "
                "excluded and the VLM scales with total_detections."
            ),
        },
        "peak_process_gpu_vram_mb": vram.peak_mb,
        "frames": detection_run.frames,
        "model": {
            "path": args.model,
            "sha256": _sha256_file(Path(args.model)),
            "input_size": list(expected_input_size),
        },
    }
    _write_json(output, payload)
    print(
        f"BENCHMARK_TILING_ARM_COMPLETE arm={args.arm} frames={fixture.summary.frame_count} "
        f"detections={total_detections} vram_mb={vram.peak_mb:.1f} "
        f"pose_min_per_h={payload['detector_cost']['pose_min_per_hour']:.2f}",
        flush=True,
    )
    return 0


def _metrics_from_arm_payload(payload: dict[str, Any]) -> ArmMetrics:
    quality = payload["quality"]
    end_to_end = payload["end_to_end"]
    return ArmMetrics(
        name=payload["name"],
        tp=int(quality["tp"]),
        fp=int(quality["fp"]),
        fn=int(quality["fn"]),
        pose_wallclock_s=[
            float(value) for value in payload["pose_inference_wallclock_s"]["samples"]
        ],
        end_to_end_wallclock_s=float(end_to_end["measured_wallclock_s"]),
        measured_video_duration_s=float(end_to_end["measured_video_duration_s"]),
        peak_process_gpu_vram_mb=float(payload["peak_process_gpu_vram_mb"]),
        film_recall={name: float(value) for name, value in payload["film_recall"].items()},
        frame_count=int(end_to_end["measured_frame_count"]),
    )


def _select_command(args: argparse.Namespace) -> int:
    manifest_path = Path(args.fixture)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = validate_fixture_manifest(manifest)
    fixture = BenchmarkFixture(
        summary=summary,
        frames=[
            BenchmarkFrame(
                id=frame["id"],
                window_id=frame["window_id"],
                image=None,
                ground_truth=[],
            )
            for frame in manifest["frames"]
        ],
    )
    arm_payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.arm_result]
    metrics = [_metrics_from_arm_payload(payload) for payload in arm_payloads]
    raw_frames = {payload["name"]: payload["frames"] for payload in arm_payloads}
    model_evidence = {payload["name"]: payload["model"] for payload in arm_payloads}
    selection = select_winner(metrics, baseline_name="baseline_640")
    if selection.winner is None and not args.follow_up_issue:
        raise BenchmarkConfigError(
            "no arm is eligible; pass --follow-up-issue after filing the required follow-up"
        )
    artifact = build_results_artifact(
        fixture=fixture,
        metrics=metrics,
        raw_frames=raw_frames,
        model_evidence=model_evidence,
        reference_tiling={
            "full_frame_640_detections": 1,
            "tiled_3x3_640_detections": 97,
            "single_1280x720_tile_visible_people": 4,
            "source": "issues #32/#83",
            "productized": False,
        },
        follow_up_issue=args.follow_up_issue,
        production_default_changed=args.production_default_changed,
    )
    _write_json(Path(args.output), artifact)
    print(
        f"BENCHMARK_SELECTION winner={selection.winner or 'none'} output={args.output}",
        flush=True,
    )
    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Issue #86 measured pose-mode benchmark")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_arm = subcommands.add_parser("run-arm", help="measure one full-frame arm in isolation")
    run_arm.add_argument(
        "--arm",
        required=True,
        choices=sorted(FULL_FRAME_ARMS),
    )
    run_arm.add_argument("--fixture", required=True)
    run_arm.add_argument("--zones", required=True)
    run_arm.add_argument("--model", required=True)
    run_arm.add_argument("--throughput-clip", action="append", required=True)
    run_arm.add_argument("--film-1-fixture", required=True)
    run_arm.add_argument("--film-2-fixture", required=True)
    run_arm.add_argument("--output", required=True)
    run_arm.set_defaults(handler=_run_arm_command)

    run_tiling = subcommands.add_parser(
        "run-tiling-arm", help="measure one native-resolution tiling arm (#110)"
    )
    run_tiling.add_argument("--arm", required=True, choices=sorted(TILING_ARMS))
    run_tiling.add_argument("--fixture", required=True)
    run_tiling.add_argument("--zones", required=True)
    run_tiling.add_argument(
        "--roi-zones",
        help="zones config whose polygons bound the with-zones arm's tiling (required for it)",
    )
    run_tiling.add_argument("--model", required=True)
    run_tiling.add_argument("--overlap", type=float, default=DEFAULT_TILE_OVERLAP)
    run_tiling.add_argument("--output", required=True)
    run_tiling.set_defaults(handler=_run_tiling_arm_command)

    select = subcommands.add_parser("select", help="select winner from three measured arm files")
    select.add_argument("--fixture", required=True)
    select.add_argument("--arm-result", action="append", required=True)
    select.add_argument("--follow-up-issue")
    select.add_argument("--production-default-changed", action="store_true")
    select.add_argument("--output", required=True)
    select.set_defaults(handler=_select_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (BenchmarkConfigError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"BENCHMARK_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
