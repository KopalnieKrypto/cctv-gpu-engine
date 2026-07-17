"""Per-second Films 1+2 activity agreement and strict promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ACTIVITIES = ("sitting", "standing", "walking", "running")
CLASSIFIERS = ("heuristic", "vlm", "mlp")
AGREEMENT_FLOOR = 0.89


def expand_truth(film: dict) -> dict[int, str]:
    """Expand start-inclusive/end-exclusive annotation intervals."""
    truth: dict[int, str] = {}
    for interval in film["intervals"]:
        start = interval["start_s"]
        end = interval["end_s"]
        activity = interval["activity"]
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            raise ValueError(f"invalid interval: {interval}")
        if activity not in ACTIVITIES:
            raise ValueError(f"invalid activity: {activity}")
        for second in range(start, end):
            if second in truth:
                raise ValueError(f"overlapping truth at second {second}")
            truth[second] = activity
    return dict(sorted(truth.items()))


def load_detection_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        timestamp = float(row["timestamp_s"])
        if not timestamp.is_integer():
            raise ValueError(f"non-integer timestamp on line {line_number}: {timestamp}")
        rows.append(row)
    return rows


def _confirmed_track_counts(rows: list[dict]) -> Counter[int]:
    """Mirror the runtime filter: 3 sightings in any 5-frame window."""
    sightings: dict[int, list[int]] = {}
    for row_index, row in enumerate(rows):
        frame_index = int(row.get("frame_idx", row_index))
        for person in row["persons"]:
            track_id = person.get("track_id")
            if isinstance(track_id, int):
                sightings.setdefault(track_id, []).append(frame_index)

    confirmed = {
        track_id
        for track_id, frames in sightings.items()
        if any(frames[index] - frames[index - 2] <= 4 for index in range(2, len(frames)))
    }
    return Counter(
        person["track_id"]
        for row in rows
        for person in row["persons"]
        if person.get("track_id") in confirmed
    )


def evaluate_film(film: dict, rows: list[dict]) -> dict:
    """Score only annotated seconds using the recording's longest-lived track."""
    truth = expand_truth(film)
    confirmed_counts = _confirmed_track_counts(rows)
    predictions: dict[int, tuple[str, int]] = {}
    for row in rows:
        second = int(float(row["timestamp_s"]))
        candidates = [
            person for person in row["persons"] if person.get("track_id") in confirmed_counts
        ]
        if candidates:
            selected = max(
                candidates,
                key=lambda person: (
                    confirmed_counts[person["track_id"]],
                    -person["track_id"],
                ),
            )
            predictions[second] = (selected["activity"], selected["track_id"])

    per_second = []
    confusion: dict[str, dict[str, int]] = {
        activity: {prediction: 0 for prediction in (*ACTIVITIES, "missing")}
        for activity in ACTIVITIES
    }
    correct = 0
    for second, expected in truth.items():
        selected = predictions.get(second)
        prediction = selected[0] if selected else None
        track_id = selected[1] if selected else None
        is_correct = prediction == expected
        correct += int(is_correct)
        confusion[expected][prediction or "missing"] += 1
        per_second.append(
            {
                "second": second,
                "expected": expected,
                "prediction": prediction,
                "track_id": track_id,
                "correct": is_correct,
            }
        )

    return {
        "confirmed_track_ids": sorted(confirmed_counts),
        "annotated_seconds": len(truth),
        "correct_seconds": correct,
        "agreement": correct / len(truth),
        "confusion": confusion,
        "per_second": per_second,
    }


def film_promotion_gate(results: dict[str, dict[str, dict]]) -> dict:
    films = {}
    passed = True
    for film_id in ("film-1", "film-2"):
        mlp = results[film_id]["mlp"]["agreement"]
        vlm = results[film_id]["vlm"]["agreement"]
        checks = {
            "mlp_at_least_89_percent": mlp >= AGREEMENT_FLOOR,
            "at_least_vlm": mlp >= vlm,
        }
        checks["passed"] = all(checks.values())
        films[film_id] = checks
        passed = passed and checks["passed"]
    return {**films, "passed": passed}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("training/activity-mlp/films-ground-truth.json"),
    )
    parser.add_argument("--film-1-root", type=Path, required=True)
    parser.add_argument("--film-2-root", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable evaluation: {args.output}")

    truth_bytes = args.truth.read_bytes()
    truth_manifest = json.loads(truth_bytes)
    films = {film["film_id"]: film for film in truth_manifest["films"]}
    roots = {"film-1": args.film_1_root, "film-2": args.film_2_root}
    results: dict[str, dict[str, dict]] = {}
    for film_id, root in roots.items():
        results[film_id] = {}
        for classifier in CLASSIFIERS:
            detections_path = root / classifier / "detections.jsonl"
            detection_bytes = detections_path.read_bytes()
            evaluation = evaluate_film(films[film_id], load_detection_rows(detections_path))
            results[film_id][classifier] = {
                "detections_path": str(detections_path),
                "detections_sha256": hashlib.sha256(detection_bytes).hexdigest(),
                **evaluation,
            }
            print(
                f"film agreement film={film_id} classifier={classifier} "
                f"correct={evaluation['correct_seconds']}/{evaluation['annotated_seconds']} "
                f"agreement={evaluation['agreement']:.6f}",
                flush=True,
            )

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "image_id": args.image_id,
        "truth": {
            "path": str(args.truth),
            "sha256": hashlib.sha256(truth_bytes).hexdigest(),
            "interval_semantics": truth_manifest["interval_semantics"],
        },
        "methodology": {
            "sample_rate_fps": 1,
            "scored_scope": "annotated seconds only; visible-presence gaps excluded",
            "person_selection": (
                "runtime-confirmed track present per second; ties by total sightings then track_id"
            ),
            "agreement_floor": AGREEMENT_FLOOR,
        },
        "results": results,
        "promotion_film_gate": film_promotion_gate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"film promotion gate passed={artifact['promotion_film_gate']['passed']} "
        f"artifact={args.output}",
        flush=True,
    )
    return 0 if artifact["promotion_film_gate"]["passed"] else 2
