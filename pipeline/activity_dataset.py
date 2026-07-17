"""Validation entry point for the activity-classifier dataset (issue #33)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

EXPECTED_SAMPLE_COUNT = 1_000
ACTIVITIES = frozenset({"sitting", "standing", "walking", "running"})
SPLITS = frozenset({"train", "validation", "test"})
REQUIRED_KEYPOINT_FIELDS = frozenset({"vis", "x", "y"})
REQUIRED_SAMPLE_FIELDS = frozenset(
    {
        "activity",
        "bbox",
        "camera_geometry_id",
        "frame_height",
        "frame_path",
        "frame_sha256",
        "frame_width",
        "keypoints",
        "review_status",
        "sample_id",
        "source_id",
        "split",
        "synthetic",
    }
)
REQUIRED_GEOMETRY_FIELDS = frozenset(
    {
        "horizontal_fov_deg",
        "id",
        "license",
        "mounting_height_m",
        "parameter_basis",
        "source",
        "source_id",
        "tilt_deg",
    }
)
REQUIRED_README_SECTIONS = (
    "Camera geometries",
    "Sources and licenses",
    "Split strategy",
    "Labeling tool decision",
)


class DatasetValidationError(ValueError):
    """Raised when an activity dataset violates its published contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as asset_file:
        for chunk in iter(lambda: asset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(dataset_dir: str | Path, *, verify_assets: bool = True) -> None:
    """Validate the activity dataset rooted at ``dataset_dir``."""
    dataset_path = Path(dataset_dir)
    labels_path = dataset_path / "labels.jsonl"
    with labels_path.open(encoding="utf-8") as labels_file:
        lines = [line for line in labels_file if line.strip()]

    if len(lines) != EXPECTED_SAMPLE_COUNT:
        raise DatasetValidationError(
            f"expected exactly {EXPECTED_SAMPLE_COUNT} person-frame samples; found {len(lines)}"
        )

    activity_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    geometry_ids: set[str] = set()
    split_geometries = {split: set() for split in SPLITS}
    split_activities = {split: set() for split in SPLITS}
    frame_splits: dict[str, set[str]] = {}
    validated_samples: list[dict] = []
    sample_ids: set[str] = set()
    for sample_number, line in enumerate(lines, start=1):
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"sample {sample_number}: invalid JSON") from exc
        missing_fields = sorted(REQUIRED_SAMPLE_FIELDS - sample.keys())
        if missing_fields:
            raise DatasetValidationError(
                f"sample {sample_number}: missing required fields: {', '.join(missing_fields)}"
            )
        validated_samples.append(sample)
        if sample["sample_id"] in sample_ids:
            raise DatasetValidationError(f"duplicate sample_id: {sample['sample_id']}")
        sample_ids.add(sample["sample_id"])
        if sample["review_status"] != "reviewed":
            raise DatasetValidationError(
                f"sample {sample_number}: review_status must be 'reviewed'; "
                f"found {sample['review_status']!r}"
            )

        x, y, width, height = sample["bbox"]
        frame_width = sample["frame_width"]
        frame_height = sample["frame_height"]
        if width <= 0 or height <= 0:
            raise DatasetValidationError(
                f"sample {sample_number}: bbox width and height must be positive; "
                f"found {width}x{height}"
            )
        if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
            raise DatasetValidationError(
                f"sample {sample_number}: bbox extends outside the "
                f"{frame_width}x{frame_height} source frame"
            )

        if sample["activity"] not in ACTIVITIES:
            raise DatasetValidationError(
                f"sample {sample_number}: unsupported activity {sample['activity']!r}"
            )
        activity_counts[sample["activity"]] += 1
        if sample["split"] not in SPLITS:
            raise DatasetValidationError(
                f"sample {sample_number}: unsupported split {sample['split']!r}"
            )
        split_counts[sample["split"]] += 1
        split_activities[sample["split"]].add(sample["activity"])
        geometry_ids.add(sample["camera_geometry_id"])
        split_geometries[sample["split"]].add(sample["camera_geometry_id"])
        frame_splits.setdefault(sample["frame_sha256"], set()).add(sample["split"])

        keypoint_count = len(sample["keypoints"])
        if keypoint_count != 17:
            raise DatasetValidationError(
                f"sample {sample_number}: expected exactly 17 keypoints; found {keypoint_count}"
            )

        for keypoint_number, keypoint in enumerate(sample["keypoints"], start=1):
            missing_keypoint_fields = sorted(REQUIRED_KEYPOINT_FIELDS - keypoint.keys())
            if missing_keypoint_fields:
                raise DatasetValidationError(
                    f"sample {sample_number} keypoint {keypoint_number}: "
                    "missing required fields: " + ", ".join(missing_keypoint_fields)
                )
            visibility = keypoint["vis"]
            if visibility < 0 or visibility > 1:
                raise DatasetValidationError(
                    f"sample {sample_number} keypoint {keypoint_number}: vis must be between "
                    f"0 and 1; found {visibility}"
                )
            keypoint_x = keypoint["x"]
            keypoint_y = keypoint["y"]
            if visibility > 0 and not (
                0 <= keypoint_x < frame_width and 0 <= keypoint_y < frame_height
            ):
                raise DatasetValidationError(
                    f"sample {sample_number} keypoint {keypoint_number}: coordinates "
                    f"({keypoint_x}, {keypoint_y}) are outside {frame_width}x{frame_height}"
                )

    if any(activity_counts[activity] != 250 for activity in ACTIVITIES):
        counts = ", ".join(
            f"{activity}={activity_counts[activity]}" for activity in sorted(ACTIVITIES)
        )
        raise DatasetValidationError(f"expected exactly 250 samples per activity; found {counts}")

    for frame_sha256, assigned_splits in sorted(frame_splits.items()):
        if len(assigned_splits) > 1:
            splits = " and ".join(sorted(assigned_splits))
            raise DatasetValidationError(f"source frame {frame_sha256} appears in both {splits}")

    expected_split_counts = {"train": 700, "validation": 150, "test": 150}
    if split_counts != expected_split_counts:
        found = ", ".join(f"{split}={split_counts[split]}" for split in sorted(SPLITS))
        raise DatasetValidationError(
            f"expected split counts train=700, validation=150, test=150; found {found}"
        )

    for split in ("train", "validation", "test"):
        missing_activities = sorted(ACTIVITIES - split_activities[split])
        if missing_activities:
            raise DatasetValidationError(
                f"{split} split is missing activities: {', '.join(missing_activities)}"
            )

    if len(geometry_ids) != 6:
        raise DatasetValidationError(
            f"expected exactly 6 camera geometries; found {len(geometry_ids)}"
        )

    train_test_overlap = split_geometries["train"] & split_geometries["test"]
    if train_test_overlap:
        raise DatasetValidationError(
            "camera geometries cross train/test boundary: " + ", ".join(sorted(train_test_overlap))
        )

    validation_only = split_geometries["validation"] - split_geometries["train"]
    if validation_only:
        raise DatasetValidationError(
            "validation geometries absent from training: " + ", ".join(sorted(validation_only))
        )

    train_geometry_count = len(split_geometries["train"])
    test_geometry_count = len(split_geometries["test"])
    if train_geometry_count != 4 or test_geometry_count != 2:
        raise DatasetValidationError(
            "expected 4 train geometries and 2 test-only geometries; found "
            f"train={train_geometry_count}, test={test_geometry_count}"
        )

    geometries_path = dataset_path / "geometries.json"
    if not geometries_path.is_file():
        raise DatasetValidationError("missing required camera metadata: geometries.json")

    with geometries_path.open(encoding="utf-8") as geometries_file:
        geometry_metadata = json.load(geometries_file)
    for geometry in geometry_metadata["geometries"]:
        missing_fields = sorted(REQUIRED_GEOMETRY_FIELDS - geometry.keys())
        if missing_fields:
            geometry_id = geometry.get("id", "<unknown>")
            raise DatasetValidationError(
                f"geometry {geometry_id}: missing required fields: {', '.join(missing_fields)}"
            )

    metadata_geometry_ids = {geometry["id"] for geometry in geometry_metadata["geometries"]}
    if metadata_geometry_ids != geometry_ids:
        undocumented = ", ".join(sorted(geometry_ids - metadata_geometry_ids)) or "none"
        unused = ", ".join(sorted(metadata_geometry_ids - geometry_ids)) or "none"
        raise DatasetValidationError(
            f"geometry metadata mismatch: undocumented labels={undocumented}; "
            f"unused metadata={unused}"
        )

    geometries_by_id = {geometry["id"]: geometry for geometry in geometry_metadata["geometries"]}
    for sample_number, sample in enumerate(validated_samples, start=1):
        geometry = geometries_by_id[sample["camera_geometry_id"]]
        if sample["source_id"] != geometry["source_id"]:
            raise DatasetValidationError(
                f"sample {sample_number}: source_id {sample['source_id']} does not match "
                f"{sample['camera_geometry_id']} source_id {geometry['source_id']}"
            )

    camera_heights = [geometry["mounting_height_m"] for geometry in geometry_metadata["geometries"]]
    min_height = min(camera_heights)
    max_height = max(camera_heights)
    if min_height > 1.8 or max_height < 3.5:
        raise DatasetValidationError(
            f"camera heights must span 1.8..3.5 m; found {min_height}..{max_height} m"
        )

    camera_tilts = [geometry["tilt_deg"] for geometry in geometry_metadata["geometries"]]
    min_tilt = min(camera_tilts)
    max_tilt = max(camera_tilts)
    if min_tilt > 15 or max_tilt < 75:
        raise DatasetValidationError(
            f"camera tilts must span 15..75 degrees; found {min_tilt}..{max_tilt} degrees"
        )

    fov_values = {geometry["horizontal_fov_deg"] for geometry in geometry_metadata["geometries"]}
    if len(fov_values) < 3:
        raise DatasetValidationError(
            f"expected at least 3 distinct horizontal FOV values; found {len(fov_values)}"
        )

    readme_path = dataset_path / "README.md"
    if not readme_path.is_file():
        raise DatasetValidationError("missing required dataset documentation: README.md")
    readme = readme_path.read_text(encoding="utf-8")
    missing_sections = [
        section for section in REQUIRED_README_SECTIONS if f"## {section}" not in readme
    ]
    if missing_sections:
        raise DatasetValidationError(
            f"README.md missing required sections: {', '.join(missing_sections)}"
        )

    for sample_number, sample in enumerate(validated_samples, start=1):
        frame_path = Path(sample["frame_path"])
        geometry_prefix = sample["camera_geometry_id"]
        if not frame_path.parts or frame_path.parts[0] != geometry_prefix:
            raise DatasetValidationError(
                f"sample {sample_number}: frame_path must be under {geometry_prefix}/; "
                f"found {sample['frame_path']}"
            )
        if verify_assets:
            asset_path = dataset_path / frame_path
            if not asset_path.is_file():
                raise DatasetValidationError(
                    f"sample {sample_number}: missing frame asset {sample['frame_path']}"
                )
            if _sha256_file(asset_path) != sample["frame_sha256"]:
                raise DatasetValidationError(
                    f"sample {sample_number}: frame checksum mismatch for {sample['frame_path']}"
                )
            with Image.open(asset_path) as frame:
                asset_width, asset_height = frame.size
            if (asset_width, asset_height) != (
                sample["frame_width"],
                sample["frame_height"],
            ):
                raise DatasetValidationError(
                    f"sample {sample_number}: frame dimensions mismatch for "
                    f"{sample['frame_path']}; label={sample['frame_width']}x"
                    f"{sample['frame_height']}, asset={asset_width}x{asset_height}"
                )
