"""The station head from a zone-annotator package, not a benchmark manifest (gpu-exchange#301).

## What the package decides

The package is what left the client's box: `export.json` and `images/`. It says
which picture carries which answer (`samples`), which windows are measured
against which (`split.folds`), and where the station is and how its crop
becomes a tensor (`crop_geometry`). Nothing here derives any of these a second
time. The geometry is read and only checked against the engine's own rule: on
2026-09-03 a second derivation fed the head the middle 43% of the bench, and a
run reported zero seconds of welding over an hour of it.

Two things a package cannot say come from the command line: the delivered
vocabulary (`--collapse`, which the manifest declares and a package does not)
and the station id the head ships under.

## The order

`check_export` runs first, before the package is read, so a head is trained only
on the bytes that left the box. Then the folds: fewer than two, or one without
a single answer, leaves the card nothing held out to measure on. The card is the
document `station_head.build_card` writes for the manifest path, with its figures
from this run's folds instead of `C0-report.json`, and the package it came from
named under `training_material.package`.

## Where the GPU is

Embedding, fitting and exporting are passed in. `train_station_head.py` passes
DINOv2, `run_tcn_arm.fit_model` and the ONNX export, the code the manifest path
trains with; the tests pass stand-ins and check the rest, which is this module.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_arms import (  # noqa: E402
    NON_ACTIVITY,
    collapse_classes,
    collapse_pairs,
    per_activity_scores,
    resolve_collapse,
)
from run_pixel_probe_arm import BACKBONE  # noqa: E402
from station_head import (  # noqa: E402
    build_card,
    choose_vocabulary,
    head_hyperparameters,
    remap_labels,
)

from pipeline.station_classifier import model_input_for_rect  # noqa: E402

EXPORT_NAME = "export.json"
SOURCE = "export.json folds (cross-validated in this run)"
_STATION_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")

Clips = list[tuple[np.ndarray, np.ndarray]]
#: Crops of one window, and the (height, width) the package says they are resized to.
Embed = Callable[[list[Path], tuple[int, int]], np.ndarray]
#: Fit once on the training windows, return one prediction per held-out window.
FitPredict = Callable[[Clips, list[np.ndarray], int], list[np.ndarray]]
#: Fit on every window, write the head to the path, return its `artifact` block.
FitExport = Callable[[Clips, int, Path], dict]


@dataclass(frozen=True)
class Window:
    """One recording's answered samples for the station's zone, in time order."""

    name: str
    fold: str
    images: list[Path]
    labels: list[str]
    stride_s: float
    recorded: dict | None
    annotated_at: str | None


@dataclass(frozen=True)
class Package:
    classes: list[str]
    windows: list[Window]
    zone_native_px: dict
    model_input: tuple[int, int]
    camera_id: str
    provenance: dict


def check_package(package_dir: Path, check_cmd: Sequence[str]) -> None:
    """Refuse a package zone-annotator's own `check_export` does not pass."""
    result = subprocess.run([*check_cmd, str(package_dir)], capture_output=True, text=True)
    if result.returncode != 0:
        said = (result.stdout + result.stderr).strip()
        sys.exit(
            f"{package_dir}: check_export exited {result.returncode}, so nothing is "
            f"trained on it.\n{said}"
        )


def read_package(package_dir: Path, zone: str | None = None) -> Package:
    """The windows, folds and geometry of one zone, or a refusal naming what is missing."""
    doc = json.loads((package_dir / EXPORT_NAME).read_text(encoding="utf-8"))
    labels = doc["vocabulary"]["labels"]
    classes = [label["name"] for label in labels]
    cannot_tell = [label["name"] for label in labels if label["means_cannot_tell"]]
    if cannot_tell != [NON_ACTIVITY]:
        sys.exit(
            f"the package's cannot-tell answer is {cannot_tell or 'not marked'}, and the "
            f"engine scores and delivers `{NON_ACTIVITY}` as the abstention: the card "
            "would keep the wrong label out of the bucket."
        )
    zone = _station_zone(doc, zone)
    clips = [c for c in doc["clips"] if any(z["name"] == zone for z in c["zones"])]
    zones = {c["id"]: next(z for z in c["zones"] if z["name"] == zone) for c in clips}
    geometry, model_input = _one_geometry(clips, zones)
    cameras = {c["conditions"]["camera"]["id"] for c in clips}
    if len(cameras) != 1:
        sys.exit(f"the windows of `{zone}` come from cameras {sorted(cameras)}; a head is one view")

    split = doc.get("split") or {}
    if not split.get("declared"):
        sys.exit(
            "the package declares no split, so nothing is held out to measure the head "
            "on. Declare the folds in zone-annotator (`split declare`) and export again."
        )
    fold_of = {e["clip_id"]: fold for fold, entries in split["folds"].items() for e in entries}
    unassigned = [c["name"] for c in clips if c["id"] not in fold_of]
    if unassigned:
        sys.exit(
            f"{', '.join(unassigned)} is in no fold: the head would train on a window "
            "no fold measures. Assign every window, or leave it out of the export."
        )
    windows = [_window(package_dir, doc, c, zones[c["id"]], fold_of[c["id"]]) for c in clips]
    folds = sorted({fold_of[c["id"]] for c in clips})
    answered = {f: sum(len(w.labels) for w in windows if w.fold == f) for f in folds}
    unanswered = [f for f, n in answered.items() if not n]
    if len(folds) < 2 or unanswered:
        why = (
            f"fold(s) {', '.join(unanswered)} have no answers"
            if unanswered
            else f"the package declares one fold ({', '.join(folds)})"
        )
        sys.exit(
            f"training needs at least two folds with answers, so every figure on the "
            f"card is measured on material its model did not see; {why}."
        )
    return Package(
        classes=classes,
        windows=[w for w in windows if w.labels],
        zone_native_px=geometry["zone_native_px"],
        model_input=model_input,
        camera_id=cameras.pop(),
        provenance={
            "export": EXPORT_NAME,
            "schema": doc.get("schema"),
            "exported_utc": doc.get("exported_utc"),
            "bound_to": (doc.get("retention") or {}).get("bound_to"),
            # Where the bytes were read: publishing checks it is the export's drop on cctv-vps.
            "dir": str(package_dir.resolve()),
            "zone": zone,
            "split_decided_utc": split.get("decided_utc"),
            "folds": {f: sorted(w.name for w in windows if w.fold == f) for f in folds},
        },
    )


def train_package(
    package_dir: Path,
    *,
    station: str,
    version: str,
    collapse: str,
    zone: str | None,
    out_dir: Path,
    check_cmd: Sequence[str],
    embed: Embed,
    fit_predict: FitPredict,
    fit_export: FitExport,
) -> Path:
    """Check, read, cross-validate, train on everything, and write the card. Returns its path."""
    package_dir = Path(package_dir)
    if not _STATION_ID.fullmatch(station):
        sys.exit(f"--station {station!r}: lowercase letters, digits, dot, dash, underscore")
    check_package(package_dir, check_cmd)
    pkg = read_package(package_dir, zone)
    manifest = _as_manifest(pkg, station)
    vocabulary = resolve_collapse(manifest, collapse)
    delivered = collapse_classes(pkg.classes, vocabulary)

    xs = {w.name: embed(w.images, pkg.model_input) for w in pkg.windows}
    ys = {
        w.name: np.asarray([pkg.classes.index(label) for label in w.labels], dtype=np.int64)
        for w in pkg.windows
    }
    ys_direct = {name: remap_labels(y, pkg.classes, vocabulary)[0] for name, y in ys.items()}

    # #122's comparison on this package's folds: the full vocabulary scored after
    # the merge is the floor, and the delivered one trained directly must beat it.
    merged = {
        name: collapse_pairs(pairs, vocabulary)
        for name, pairs in _cross_validate(pkg.windows, xs, ys, pkg.classes, fit_predict).items()
    }
    direct = _cross_validate(pkg.windows, xs, ys_direct, delivered, fit_predict)
    floor = per_activity_scores(_union(merged), delivered)
    unmeasured = [c for c, s in floor.items() if not s["support"]]
    if unmeasured:
        sys.exit(
            f"{', '.join(unmeasured)} has no answer in any held-out fold, so the card "
            "would have no time ratio for it and the pipeline refuses such a card."
        )
    comparison = choose_vocabulary(per_activity_scores(_union(direct), delivered), floor)
    ships_direct = comparison["ships"] == "direct"
    output_classes = delivered if ships_direct else pkg.classes
    targets = ys_direct if ships_direct else ys

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = fit_export(
        [(xs[w.name], targets[w.name]) for w in pkg.windows],
        len(output_classes),
        out_dir / f"station-head-{station}-v{version}.onnx",
    )
    arm = f"tcn-pixel-{pkg.model_input[0]}"
    training = {
        "backbone": BACKBONE,
        "preprocessing": {"model_input": list(pkg.model_input), "center_crop": False},
        "output_classes": output_classes,
        "trained_as": comparison["ships"],
        "comparison": comparison,
        "hyperparameters": head_hyperparameters(next(iter(xs.values())).shape[1]),
        "artifact": artifact,
    }
    report = {
        "arms": [
            {
                "name": arm,
                "collapsed": {
                    "scores": floor,
                    "per_window_scores": {
                        name: per_activity_scores(pairs, delivered)
                        for name, pairs in sorted(merged.items())
                    },
                },
            }
        ],
        "collapse": {**vocabulary, "classes": delivered},
    }
    card = build_card(manifest, report, arm, training, source=SOURCE)
    card["training_material"]["package"] = pkg.provenance
    card_path = out_dir / f"station-head-{station}-v{version}.card.json"
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return card_path


def _station_zone(doc: dict, zone: str | None) -> str:
    names = sorted({z["name"] for c in doc["clips"] for z in c["zones"]})
    if zone is not None:
        if zone not in names:
            sys.exit(f"no zone {zone!r} in the package; it has {', '.join(names)}")
        return zone
    if len(names) != 1:
        sys.exit(f"the package holds zones {', '.join(names)}; a head is one station: --zone")
    return names[0]


def _one_geometry(clips: list[dict], zones: dict[int, dict]) -> tuple[dict, tuple[int, int]]:
    """The zone's one `crop_geometry`, checked against the engine's rule, never replaced."""
    found = {json.dumps(zones[c["id"]]["crop_geometry"], sort_keys=True) for c in clips}
    if len(found) != 1:
        sys.exit(
            f"the windows were cut under {len(found)} different geometries; one head "
            "reads one rectangle at one tensor size"
        )
    geometry = json.loads(found.pop())
    model_input = geometry["model_input"]
    if model_input.get("center_crop") is not False:
        sys.exit("the package's crop_geometry does not declare `center_crop: false`")
    native = geometry["zone_native_px"]
    rect = (int(native["x"]), int(native["y"]), int(native["w"]), int(native["h"]))
    size = (int(model_input["height"]), int(model_input["width"]))
    engine = tuple(model_input_for_rect(rect, int(model_input["target_height"])))
    if engine != size:
        sys.exit(
            f"the package resizes {rect} to {size} and this engine's rule gives {engine}: "
            "the two golden geometry tables have drifted, so the head would be fed "
            "another tensor than its card states"
        )
    return geometry, size


def _window(package_dir: Path, doc: dict, clip: dict, zone: dict, fold: str) -> Window:
    sampling = clip["sampling"]
    stride, first = float(sampling["stride_s"]), float(sampling["first_pts_s"])
    mine = sorted(
        (
            s
            for s in doc["samples"]
            if s["clip_id"] == clip["id"] and s["zone_id"] == zone["id"] and s.get("label")
        ),
        key=lambda s: s["pts_s"],
    )
    steps = [round((s["pts_s"] - first) / stride) for s in mine]
    # The export leaves unanswered samples out. At the ends that only shortens the
    # window; inside it, a temporal head would read the join as continuous time.
    if steps and steps != list(range(steps[0], steps[0] + len(steps))):
        sys.exit(
            f"{clip['name']}: unanswered samples sit inside the window, not at its ends. "
            "Answer them, or cut the window at the gap."
        )
    recorded = clip.get("recorded_window_utc")
    return Window(
        name=clip["name"],
        fold=fold,
        images=[_inside(package_dir, s["image"]) for s in mine],
        labels=[s["label"] for s in mine],
        stride_s=sampling["stride_s"],
        recorded=recorded,
        annotated_at=max((s.get("labelled_at") or "" for s in mine), default="")[:10] or None,
    )


def _inside(package_dir: Path, image: str) -> Path:
    """A sample's picture, read only from inside the package: no absolute path, `..` or link."""
    root = package_dir.resolve()
    path = root / image
    if path.resolve() != path or not path.is_relative_to(root):
        sys.exit(
            f"sample image {image!r} is outside the package {package_dir}, so nothing is trained"
        )
    return path


def _as_manifest(pkg: Package, station: str) -> dict:
    """The fields of a benchmark manifest `resolve_collapse` and `build_card` read."""
    return {
        "fixture": station,
        "camera": {"id": pkg.camera_id},
        "station_roi": {"crop": pkg.zone_native_px},
        "activities": [{"id": c} for c in pkg.classes],
        "clips": [
            {
                "slot": w.name,
                "annotated": True,
                "window_local": f"{w.recorded['start']}/{w.recorded['end']}"
                if w.recorded
                else None,
                "annotated_at": w.annotated_at,
                "annotation_coverage": {"stride_s": w.stride_s, "labelled": len(w.labels)},
            }
            for w in pkg.windows
        ],
    }


def _cross_validate(
    windows: list[Window],
    xs: dict[str, np.ndarray],
    ys: dict[str, np.ndarray],
    classes: list[str],
    fit_predict: FitPredict,
) -> dict[str, list[tuple[str, str]]]:
    """(truth, predicted) per held-out window, each fold fitted once on the others."""
    pairs: dict[str, list[tuple[str, str]]] = {}
    for fold in sorted({w.fold for w in windows}):
        held = [w for w in windows if w.fold == fold]
        train = [(xs[w.name], ys[w.name]) for w in windows if w.fold != fold]
        predictions = fit_predict(train, [xs[w.name] for w in held], len(classes))
        for w, predicted in zip(held, predictions, strict=True):
            pairs[w.name] = [
                (classes[int(t)], classes[int(p)])
                for t, p in zip(ys[w.name], predicted, strict=True)
            ]
    return pairs


def _union(per_window: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    return [pair for pairs in per_window.values() for pair in pairs]
