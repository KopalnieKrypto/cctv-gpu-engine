"""The station head trained from a zone-annotator package (gpu-exchange#301).

What these tests encode:

* **Input**: `testdata/zone-annotator-package`, exported by zone-annotator from
  two windows of its own test clip (one zone, every sample answered, folds
  `fold-1` = W1 and `fold-2` = W2), and stand-ins for the three GPU steps.
* **Output**: a card the pipeline's own loader accepts, with the package's
  geometry and a measured figure for every delivered class.
* **Boundaries**: `check_export` failing, one fold, a fold without answers, an
  unanswered sample inside a window, geometry the engine's rule disagrees with.
* **Not tested here**: DINOv2, the TCN and the ONNX export, which are the
  functions the manifest path trains with, and `check_export` itself, which is
  zone-annotator's (the stand-in exits as it would).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import station_package as sp  # noqa: E402

from pipeline.station_card import StationCard  # noqa: E402
from pipeline.station_classifier import model_input_for_rect  # noqa: E402

PACKAGE = Path(__file__).parent / "testdata" / "zone-annotator-package"
SOUND = [sys.executable, "-c", "raise SystemExit(0)"]
TAMPERED = [
    sys.executable,
    "-c",
    "print('images/W1-d20fccbd/stanowisko-1/t00001.jpg: sha256 differs'); raise SystemExit(1)",
]


def mean_colour(files: list[Path], _model_input: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        [
            np.asarray(Image.open(f).convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
            for f in files
        ]
    )


def nearest_centroid(train, tests, _n_class):
    x = np.concatenate([x for x, _ in train])
    y = np.concatenate([y for _, y in train])
    centroids = {int(c): x[y == c].mean(0) for c in np.unique(y)}
    return [
        np.asarray([min(centroids, key=lambda c: np.linalg.norm(r - centroids[c])) for r in t])
        for t in tests
    ]


def a_file(train, n_class: int, path: Path) -> dict:
    path.write_bytes(b"not onnx")
    return {
        "name": path.name,
        "version": "0.1.0",
        "sha256": "0" * 64,
        "bytes": 8,
        "opset": 18,
        "input": "embeddings (batch, 3, time)",
        "output": f"logits (batch, {n_class}, time)",
    }


def train(tmp_path: Path, package: Path = PACKAGE, *, check=SOUND, embed=mean_colour) -> Path:
    return sp.train_package(
        package,
        station="hala-test",
        version="0.1.0",
        collapse="pozostale=postoj",
        zone=None,
        out_dir=tmp_path / "models",
        check_cmd=check,
        embed=embed,
        fit_predict=nearest_centroid,
        fit_export=a_file,
    )


def a_package(tmp_path: Path, change) -> Path:
    copy = tmp_path / "package"
    shutil.copytree(PACKAGE, copy)
    doc = json.loads((copy / "export.json").read_text(encoding="utf-8"))
    change(doc)
    (copy / "export.json").write_text(json.dumps(doc), encoding="utf-8")
    return copy


def test_a_two_fold_package_trains_into_a_card_the_pipeline_loads(tmp_path):
    card_path = train(tmp_path)

    card = json.loads(card_path.read_text(encoding="utf-8"))
    loaded = StationCard.load(card_path)
    assert loaded.delivered_classes == ("spawanie", "pozostale", "nierozpoznane")
    for c in loaded.delivered_classes:
        figures = card["accuracy"]["union"][c]
        assert figures["support"] > 0 and None not in figures.values(), (c, figures)
    assert sorted(card["accuracy"]["per_window"]) == ["W1.mp4", "W2.mp4"]
    assert card["accuracy"]["source"] == sp.SOURCE
    assert card["artifact"]["name"] == "station-head-hala-test-v0.1.0.onnx"
    assert [w["samples"] for w in card["training_material"]["windows"]] == [4, 4]
    assert card["training_material"]["package"]["folds"] == {
        "fold-1": ["W1.mp4"],
        "fold-2": ["W2.mp4"],
    }
    assert card["training_material"]["package"]["bound_to"] == "station-head-test/v1"
    # Where it was read from: publish_station_head.py checks it is the export's drop.
    assert card["training_material"]["package"]["dir"] == str(PACKAGE.resolve())


def test_the_cards_geometry_is_the_packages_and_the_golden_rule(tmp_path):
    doc = json.loads((PACKAGE / "export.json").read_text(encoding="utf-8"))
    geometry = doc["clips"][0]["zones"][0]["crop_geometry"]

    card = json.loads(train(tmp_path).read_text(encoding="utf-8"))

    assert card["station"]["zone_native_px"] == geometry["zone_native_px"]
    model_input = card["backbone"]["preprocessing"]["model_input"]
    assert model_input == [geometry["model_input"]["height"], geometry["model_input"]["width"]]
    assert model_input == list(model_input_for_rect((40, 30, 120, 60), 518))
    assert card["backbone"]["preprocessing"]["center_crop"] is False


def test_a_package_check_export_refuses_is_not_trained_on(tmp_path):
    def must_not_embed(*_args):
        raise AssertionError("embedded a package check_export refused")

    with pytest.raises(SystemExit, match="check_export exited 1") as refused:
        train(tmp_path, check=TAMPERED, embed=must_not_embed)

    assert "sha256 differs" in str(refused.value)
    assert not (tmp_path / "models").exists()


def test_a_package_with_one_fold_is_refused(tmp_path):
    def one_fold(doc):
        folds = doc["split"]["folds"]
        doc["split"]["folds"] = {"fold-1": folds["fold-1"] + folds["fold-2"]}

    with pytest.raises(SystemExit, match="at least two folds.*one fold"):
        train(tmp_path, a_package(tmp_path, one_fold))


def test_a_fold_without_answers_is_refused(tmp_path):
    def unanswer_w2(doc):
        doc["samples"] = [s for s in doc["samples"] if s["clip_id"] != 2]

    with pytest.raises(SystemExit, match="at least two folds.*fold-2 have no answers"):
        train(tmp_path, a_package(tmp_path, unanswer_w2))


def test_an_unanswered_sample_inside_a_window_is_refused(tmp_path):
    def drop_second_of_w1(doc):
        w1 = [s["id"] for s in doc["samples"] if s["clip_id"] == 1]
        doc["samples"] = [s for s in doc["samples"] if s["id"] != w1[1]]

    with pytest.raises(SystemExit, match="W1.mp4: unanswered samples sit inside"):
        train(tmp_path, a_package(tmp_path, drop_second_of_w1))


def test_geometry_the_engines_rule_disagrees_with_is_refused(tmp_path):
    def wider(doc):
        for clip in doc["clips"]:
            clip["zones"][0]["crop_geometry"]["model_input"]["width"] = 1050

    with pytest.raises(SystemExit, match="golden geometry tables have drifted"):
        train(tmp_path, a_package(tmp_path, wider))


@pytest.mark.parametrize(
    "where", ["absolute", "../outside.jpg", "images/../../outside.jpg", "link"]
)
def test_a_sample_image_outside_the_package_is_refused(tmp_path, where):
    """`image` comes from the box; the sha check in check_export only limits what a path can do."""
    outside = tmp_path / "outside.jpg"
    shutil.copy(PACKAGE / "images" / "W1-d20fccbd" / "stanowisko-1" / "t00001.jpg", outside)
    package = a_package(tmp_path, lambda doc: None)
    if where == "link":
        link = package / "images" / "link.jpg"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("this account cannot create symlinks")
        image = "images/link.jpg"
    else:
        image = str(outside) if where == "absolute" else where

    doc = json.loads((package / "export.json").read_text(encoding="utf-8"))
    doc["samples"][0]["image"] = image
    (package / "export.json").write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(SystemExit, match="outside the package"):
        train(tmp_path, package)
