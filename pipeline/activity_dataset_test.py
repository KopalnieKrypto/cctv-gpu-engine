"""Behavior tests for the issue #33 activity-classifier dataset contract.

Assumptions fixed before the first RED:

* one JSONL row represents one person in one source frame;
* a complete dataset has exactly 1,000 rows and exactly 250 rows for each of
  sitting, standing, walking, and running;
* bbox and visible-keypoint coordinates are pixels in the full source frame,
  while keypoint visibility is a float in the inclusive range 0..1;
* the split is exactly 700/150/150, with four train/validation geometries and
  two test-only geometries; every split contains every activity;
* source frames never cross split boundaries;
* large frame assets may live in a checksummed release payload, but labels,
  provenance, licenses, and camera-geometry metadata are tracked in git;
* empty frames, identity tracking, MLP training, and synthetic camera
  re-projection are intentionally outside issue #33.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image

from pipeline.activity_dataset import DatasetValidationError, validate_dataset


def _valid_sample(index: int) -> dict:
    frame_bytes = f"frame-{index}".encode()
    return {
        "sample_id": f"sample-{index:04d}",
        "frame_path": f"geometry-1/frame-{index:04d}.jpg",
        "frame_sha256": hashlib.sha256(frame_bytes).hexdigest(),
        "frame_width": 640,
        "frame_height": 360,
        "bbox": [100.0, 50.0, 120.0, 240.0],
        "activity": "standing",
        "keypoints": [{"x": 150.0, "y": 100.0, "vis": 0.9} for _ in range(17)],
        "camera_geometry_id": "geometry-1",
        "split": "train",
        "source_id": "fixture-source",
        "synthetic": False,
        "review_status": "reviewed",
    }


def _write_samples(root, samples: list[dict]) -> None:  # noqa: ANN001
    payload = "".join(json.dumps(sample) + "\n" for sample in samples)
    (root / "labels.jsonl").write_text(payload, encoding="utf-8")


def _balanced_split_samples() -> list[dict]:
    samples = [_valid_sample(index) for index in range(1_000)]
    activities = ("sitting", "standing", "walking", "running")
    for index, sample in enumerate(samples):
        sample["activity"] = activities[index % len(activities)]
        if index < 700:
            sample["split"] = "train"
        elif index < 850:
            sample["split"] = "validation"
        else:
            sample["split"] = "test"
    return samples


def _six_geometry_samples() -> list[dict]:
    samples = _balanced_split_samples()
    for index, sample in enumerate(samples):
        if sample["split"] == "test":
            sample["camera_geometry_id"] = f"geometry-{index % 2 + 5}"
        else:
            sample["camera_geometry_id"] = f"geometry-{index % 4 + 1}"
        sample["frame_path"] = f"{sample['camera_geometry_id']}/frame-{index:04d}.jpg"
    return samples


def _geometry_metadata() -> dict:
    heights = (1.8, 2.1, 2.4, 2.8, 3.2, 3.5)
    tilts = (15.0, 30.0, 45.0, 55.0, 65.0, 75.0)
    fovs = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0)
    return {
        "schema_version": 1,
        "geometries": [
            {
                "id": f"geometry-{index + 1}",
                "source_id": "fixture-source",
                "source": "consented fixture footage",
                "license": "test fixture",
                "mounting_height_m": heights[index],
                "tilt_deg": tilts[index],
                "horizontal_fov_deg": fovs[index],
                "parameter_basis": "measured",
            }
            for index in range(6)
        ],
    }


def _write_geometry_metadata(root, metadata: dict | None = None) -> None:  # noqa: ANN001
    payload = metadata if metadata is not None else _geometry_metadata()
    (root / "geometries.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_valid_readme(root) -> None:  # noqa: ANN001
    (root / "README.md").write_text(
        """# Activity classifier dataset

## Camera geometries

Documented in geometries.json.

## Sources and licenses

Fixture source and license.

## Split strategy

Four development geometries and two test-only geometries.

## Labeling tool decision

Custom review workflow for this fixture.
""",
        encoding="utf-8",
    )


def test_dataset_requires_exactly_1000_person_frame_samples(tmp_path) -> None:  # noqa: ANN001
    """A partial label file cannot masquerade as the completed dataset."""
    (tmp_path / "labels.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(
        DatasetValidationError,
        match=r"expected exactly 1000 person-frame samples; found 0",
    ):
        validate_dataset(tmp_path)


def test_each_sample_requires_the_published_label_fields(tmp_path) -> None:  # noqa: ANN001
    """A row is useful to training only when label and provenance fields exist."""
    records = "\n".join("{}" for _ in range(1_000)) + "\n"
    (tmp_path / "labels.jsonl").write_text(records, encoding="utf-8")

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"sample 1: missing required fields: activity, bbox, camera_geometry_id, "
            r"frame_height, frame_path, frame_sha256, frame_width, keypoints, review_status, "
            r"sample_id, source_id, split, synthetic"
        ),
    ):
        validate_dataset(tmp_path)


def test_every_final_sample_must_be_visually_reviewed(tmp_path) -> None:  # noqa: ANN001
    """Pending prelabels cannot pass as the completed ground-truth corpus."""
    samples = _six_geometry_samples()
    samples[0]["review_status"] = "pending"
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1: review_status must be 'reviewed'; found 'pending'",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_bbox_must_be_xywh_pixels_inside_the_source_frame(tmp_path) -> None:  # noqa: ANN001
    """The stored bbox is positive xywh and cannot extend past image bounds."""
    samples = [_valid_sample(index) for index in range(1_000)]
    samples[0]["bbox"] = [600.0, 100.0, 50.0, 100.0]
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1: bbox extends outside the 640x360 source frame",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_bbox_width_and_height_must_be_positive(tmp_path) -> None:  # noqa: ANN001
    """Degenerate boxes cannot be normalized into pose features."""
    samples = _six_geometry_samples()
    samples[0]["bbox"] = [100.0, 50.0, -1.0, 240.0]
    _write_samples(tmp_path, samples)
    _write_geometry_metadata(tmp_path)
    _write_valid_readme(tmp_path)

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1: bbox width and height must be positive; found -1.0x240.0",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_activity_is_one_of_the_four_product_classes(tmp_path) -> None:  # noqa: ANN001
    """Issue #33 must not grow a fifth label that runtime cannot consume."""
    samples = [_valid_sample(index) for index in range(1_000)]
    samples[0]["activity"] = "lying"
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1: unsupported activity 'lying'",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_each_sample_has_exactly_17_coco_keypoints(tmp_path) -> None:  # noqa: ANN001
    """The downstream MLP consumes one fixed COCO-17 pose per sample."""
    samples = [_valid_sample(index) for index in range(1_000)]
    samples[0]["keypoints"] = samples[0]["keypoints"][:-1]
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1: expected exactly 17 keypoints; found 16",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_each_keypoint_requires_x_y_and_visibility(tmp_path) -> None:  # noqa: ANN001
    """Malformed pose points fail with their sample and COCO joint position."""
    samples = _six_geometry_samples()
    del samples[0]["keypoints"][0]["x"]
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1 keypoint 1: missing required fields: x",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_keypoint_visibility_is_a_unit_interval_confidence(tmp_path) -> None:  # noqa: ANN001
    """Visibility stores YOLO confidence, not an unbounded or COCO 0/1/2 flag."""
    samples = [_valid_sample(index) for index in range(1_000)]
    samples[0]["keypoints"][0]["vis"] = 2
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1 keypoint 1: vis must be between 0 and 1; found 2",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_visible_keypoints_stay_inside_the_source_frame(tmp_path) -> None:  # noqa: ANN001
    """A visible joint cannot name a pixel beyond the image used for review."""
    samples = [_valid_sample(index) for index in range(1_000)]
    samples[0]["keypoints"][0]["x"] = 641
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1 keypoint 1: coordinates \(641, 100.0\) are outside 640x360",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_split_is_train_validation_or_test(tmp_path) -> None:  # noqa: ANN001
    """Every sample must have one stable split assignment."""
    samples = [_valid_sample(index) for index in range(1_000)]
    samples[0]["split"] = "holdout"
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1: unsupported split 'holdout'",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_dataset_has_exactly_250_samples_per_activity(tmp_path) -> None:  # noqa: ANN001
    """Global balance is exact so the long-pole running class cannot disappear."""
    samples = [_valid_sample(index) for index in range(1_000)]
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"expected exactly 250 samples per activity; found "
            r"running=0, sitting=0, standing=1000, walking=0"
        ),
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_dataset_split_is_exactly_70_15_15(tmp_path) -> None:  # noqa: ANN001
    """The completed corpus has exact, reproducible train/validation/test counts."""
    samples = [_valid_sample(index) for index in range(1_000)]
    activities = ("sitting", "standing", "walking", "running")
    for index, sample in enumerate(samples):
        sample["activity"] = activities[index // 250]
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"expected split counts train=700, validation=150, test=150; found .*train=1000",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_every_split_contains_every_activity(tmp_path) -> None:  # noqa: ANN001
    """Each reported split must support a four-class confusion matrix."""
    samples = _six_geometry_samples()
    validation_running = [
        sample
        for sample in samples
        if sample["split"] == "validation" and sample["activity"] == "running"
    ]
    train_sitting = [
        sample
        for sample in samples
        if sample["split"] == "train" and sample["activity"] == "sitting"
    ][: len(validation_running)]
    for sample in validation_running:
        sample["activity"] = "sitting"
    for sample in train_sitting:
        sample["activity"] = "running"
    _write_samples(tmp_path, samples)
    _write_geometry_metadata(tmp_path)
    _write_valid_readme(tmp_path)

    with pytest.raises(
        DatasetValidationError,
        match=r"validation split is missing activities: running",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_dataset_uses_the_approved_six_camera_geometries(tmp_path) -> None:  # noqa: ANN001
    """Fewer views cannot demonstrate the requested tilt robustness."""
    samples = _balanced_split_samples()
    for index, sample in enumerate(samples):
        sample["camera_geometry_id"] = f"geometry-{index % 4 + 1}"
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"expected exactly 6 camera geometries; found 4",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_test_geometries_are_absent_from_training(tmp_path) -> None:  # noqa: ANN001
    """Camera leakage would make the tilt-robustness result look better than it is."""
    samples = _six_geometry_samples()
    samples[850]["camera_geometry_id"] = "geometry-1"
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"camera geometries cross train/test boundary: geometry-1",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_validation_is_stratified_from_training_geometries(tmp_path) -> None:  # noqa: ANN001
    """Validation measures familiar views; test alone measures unseen views."""
    samples = _six_geometry_samples()
    samples[700]["camera_geometry_id"] = "geometry-5"
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"validation geometries absent from training: geometry-5",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_exactly_two_geometries_are_held_out_for_test(tmp_path) -> None:  # noqa: ANN001
    """The approved split keeps four development views and two unseen test views."""
    samples = _balanced_split_samples()
    for index, sample in enumerate(samples):
        if sample["split"] == "test":
            sample["camera_geometry_id"] = f"geometry-{index % 3 + 4}"
        else:
            sample["camera_geometry_id"] = f"geometry-{index % 3 + 1}"
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"expected 4 train geometries and 2 test-only geometries; found train=3, test=3",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_source_frame_never_crosses_split_boundaries(tmp_path) -> None:  # noqa: ANN001
    """Near-duplicate people from one frame cannot leak across evaluation splits."""
    samples = _six_geometry_samples()
    samples[850]["frame_sha256"] = samples[0]["frame_sha256"]
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"source frame .* appears in both test and train",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_camera_geometry_metadata_is_required(tmp_path) -> None:  # noqa: ANN001
    """Camera parameters and source rights cannot live only in tribal knowledge."""
    _write_samples(tmp_path, _six_geometry_samples())

    with pytest.raises(
        DatasetValidationError,
        match=r"missing required camera metadata: geometries.json",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_each_geometry_documents_source_rights_and_camera_parameters(tmp_path) -> None:  # noqa: ANN001
    """Every geometry carries provenance, rights, height, tilt, and FOV."""
    _write_samples(tmp_path, _six_geometry_samples())
    metadata = _geometry_metadata()
    del metadata["geometries"][0]["license"]
    _write_geometry_metadata(tmp_path, metadata)

    with pytest.raises(
        DatasetValidationError,
        match=r"geometry geometry-1: missing required fields: license",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_geometry_metadata_matches_every_label_geometry(tmp_path) -> None:  # noqa: ANN001
    """Labels cannot refer to an undocumented camera setup."""
    _write_samples(tmp_path, _six_geometry_samples())
    metadata = _geometry_metadata()
    metadata["geometries"][-1]["id"] = "geometry-7"
    _write_geometry_metadata(tmp_path, metadata)

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"geometry metadata mismatch: undocumented labels=geometry-6; "
            r"unused metadata=geometry-7"
        ),
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_sample_source_matches_its_camera_geometry(tmp_path) -> None:  # noqa: ANN001
    """Per-frame provenance cannot disagree with the geometry data card."""
    samples = _six_geometry_samples()
    samples[0]["source_id"] = "wrong-source"
    _write_samples(tmp_path, samples)
    _write_geometry_metadata(tmp_path)
    _write_valid_readme(tmp_path)

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"sample 1: source_id wrong-source does not match geometry-1 source_id "
            r"fixture-source"
        ),
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_geometries_span_the_required_1_8_to_3_5_m_height_range(tmp_path) -> None:  # noqa: ANN001
    """The numeric mounting-height AC is enforced, not merely mentioned."""
    _write_samples(tmp_path, _six_geometry_samples())
    metadata = _geometry_metadata()
    for geometry in metadata["geometries"]:
        geometry["mounting_height_m"] = 2.5
    _write_geometry_metadata(tmp_path, metadata)

    with pytest.raises(
        DatasetValidationError,
        match=r"camera heights must span 1.8..3.5 m; found 2.5..2.5 m",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_geometries_span_the_required_15_to_75_degree_tilt_range(tmp_path) -> None:  # noqa: ANN001
    """Tilt diversity is a numeric gate because it is the dataset's purpose."""
    _write_samples(tmp_path, _six_geometry_samples())
    metadata = _geometry_metadata()
    for geometry in metadata["geometries"]:
        geometry["tilt_deg"] = 45.0
    _write_geometry_metadata(tmp_path, metadata)

    with pytest.raises(
        DatasetValidationError,
        match=r"camera tilts must span 15..75 degrees; found 45.0..45.0 degrees",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_geometries_include_at_least_three_fov_values(tmp_path) -> None:  # noqa: ANN001
    """FOV 'ranges' means more than one token variation in the metadata."""
    _write_samples(tmp_path, _six_geometry_samples())
    metadata = _geometry_metadata()
    for geometry in metadata["geometries"]:
        geometry["horizontal_fov_deg"] = 80.0
    _write_geometry_metadata(tmp_path, metadata)

    with pytest.raises(
        DatasetValidationError,
        match=r"expected at least 3 distinct horizontal FOV values; found 1",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_dataset_readme_is_required(tmp_path) -> None:  # noqa: ANN001
    """The dataset is not complete without its human-readable data card."""
    _write_samples(tmp_path, _six_geometry_samples())
    _write_geometry_metadata(tmp_path)

    with pytest.raises(
        DatasetValidationError,
        match=r"missing required dataset documentation: README.md",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_readme_documents_sources_licenses_split_and_labeling_decision(tmp_path) -> None:  # noqa: ANN001
    """All narrative ACs have stable, reviewable sections in the data card."""
    _write_samples(tmp_path, _six_geometry_samples())
    _write_geometry_metadata(tmp_path)
    (tmp_path / "README.md").write_text(
        """# Activity classifier dataset

## Camera geometries

Documented in geometries.json.

## Sources and licenses

Fixture source and license.

## Split strategy

Four development geometries and two test-only geometries.
""",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetValidationError,
        match=r"README.md missing required sections: Labeling tool decision",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_frame_is_stored_under_its_geometry_subdirectory(tmp_path) -> None:  # noqa: ANN001
    """The on-disk layout follows datasets/activity-classifier/<geometry>/... ."""
    samples = _six_geometry_samples()
    samples[0]["frame_path"] = "wrong-geometry/frame-0000.jpg"
    _write_samples(tmp_path, samples)
    _write_geometry_metadata(tmp_path)
    _write_valid_readme(tmp_path)

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"sample 1: frame_path must be under geometry-1/; "
            r"found wrong-geometry/frame-0000.jpg"
        ),
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_materialized_dataset_requires_every_frame_asset(tmp_path) -> None:  # noqa: ANN001
    """Full validation cannot pass a labels-only or partially downloaded corpus."""
    _write_samples(tmp_path, _six_geometry_samples())
    _write_geometry_metadata(tmp_path)
    _write_valid_readme(tmp_path)

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1: missing frame asset geometry-1/frame-0000.jpg",
    ):
        validate_dataset(tmp_path)


def test_materialized_frame_hash_must_match_its_label(tmp_path) -> None:  # noqa: ANN001
    """Checksums make the reviewed image and the training label inseparable."""
    samples = _six_geometry_samples()
    _write_samples(tmp_path, samples)
    _write_geometry_metadata(tmp_path)
    _write_valid_readme(tmp_path)
    frame_path = tmp_path / samples[0]["frame_path"]
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"tampered-frame")

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1: frame checksum mismatch for geometry-1/frame-0000.jpg",
    ):
        validate_dataset(tmp_path)


def test_materialized_frame_dimensions_match_the_label(tmp_path) -> None:  # noqa: ANN001
    """Pixel-space boxes/keypoints are meaningful only for the reviewed dimensions."""
    samples = _six_geometry_samples()
    frame_path = tmp_path / samples[0]["frame_path"]
    frame_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(frame_path, format="JPEG")
    samples[0]["frame_sha256"] = hashlib.sha256(frame_path.read_bytes()).hexdigest()
    _write_samples(tmp_path, samples)
    _write_geometry_metadata(tmp_path)
    _write_valid_readme(tmp_path)

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"sample 1: frame dimensions mismatch for geometry-1/frame-0000.jpg; "
            r"label=640x360, asset=32x32"
        ),
    ):
        validate_dataset(tmp_path)


def test_sample_ids_are_unique(tmp_path) -> None:  # noqa: ANN001
    """Stable unique IDs are required for review corrections and reproducibility."""
    samples = _six_geometry_samples()
    samples[1]["sample_id"] = samples[0]["sample_id"]
    _write_samples(tmp_path, samples)

    with pytest.raises(
        DatasetValidationError,
        match=r"duplicate sample_id: sample-0000",
    ):
        validate_dataset(tmp_path, verify_assets=False)


def test_malformed_jsonl_reports_the_sample_number(tmp_path) -> None:  # noqa: ANN001
    """A bad annotation gives the reviewer an actionable line number."""
    valid_lines = [json.dumps(_valid_sample(index)) for index in range(999)]
    (tmp_path / "labels.jsonl").write_text(
        "not-json\n" + "\n".join(valid_lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetValidationError,
        match=r"sample 1: invalid JSON",
    ):
        validate_dataset(tmp_path, verify_assets=False)
