"""Per-second Films 1+2 agreement methodology and promotion gate."""

from __future__ import annotations

import json
from pathlib import Path

from activity_mlp.film_agreement import (
    evaluate_film,
    expand_truth,
    film_promotion_gate,
    load_detection_rows,
)

GROUND_TRUTH = Path(__file__).parents[1] / "films-ground-truth.json"


def test_ground_truth_expands_to_documented_activity_totals() -> None:
    films = {film["film_id"]: film for film in json.loads(GROUND_TRUTH.read_text())["films"]}

    film_1 = expand_truth(films["film-1"])
    film_2 = expand_truth(films["film-2"])

    assert len(film_1) == 130
    assert list(film_1.values()).count("walking") == 80
    assert list(film_1.values()).count("sitting") == 31
    assert list(film_1.values()).count("standing") == 19
    assert len(film_2) == 90
    assert list(film_2.values()).count("walking") == 47
    assert list(film_2.values()).count("sitting") == 43


def test_evaluator_uses_each_confirmed_track_and_retains_every_annotated_second(
    tmp_path,
) -> None:
    path = tmp_path / "detections.jsonl"
    rows = [
        {"timestamp_s": 0.0, "persons": [{"track_id": 1, "activity": "walking"}]},
        {
            "timestamp_s": 1.0,
            "persons": [
                {"track_id": 1, "activity": "walking"},
                {"track_id": 99, "activity": "sitting"},
            ],
        },
        {"timestamp_s": 2.0, "persons": [{"track_id": 1, "activity": "standing"}]},
        {"timestamp_s": 3.0, "persons": [{"track_id": 2, "activity": "walking"}]},
        {"timestamp_s": 4.0, "persons": [{"track_id": 2, "activity": "walking"}]},
        {"timestamp_s": 5.0, "persons": [{"track_id": 2, "activity": "walking"}]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    truth = {
        "film_id": "fixture",
        "intervals": [{"start_s": 0, "end_s": 7, "activity": "walking"}],
    }

    result = evaluate_film(truth, load_detection_rows(path))

    assert result["confirmed_track_ids"] == [1, 2]
    assert result["annotated_seconds"] == 7
    assert result["correct_seconds"] == 5
    assert result["agreement"] == 5 / 7
    assert [row["prediction"] for row in result["per_second"]] == [
        "walking",
        "walking",
        "standing",
        "walking",
        "walking",
        "walking",
        None,
    ]


def test_promotion_gate_requires_floor_and_no_vlm_regression_on_each_film() -> None:
    passing = film_promotion_gate(
        {
            "film-1": {"vlm": {"agreement": 0.89}, "mlp": {"agreement": 0.90}},
            "film-2": {"vlm": {"agreement": 0.89}, "mlp": {"agreement": 0.89}},
        }
    )
    regressing = film_promotion_gate(
        {
            "film-1": {"vlm": {"agreement": 0.90}, "mlp": {"agreement": 0.89}},
            "film-2": {"vlm": {"agreement": 0.89}, "mlp": {"agreement": 0.89}},
        }
    )

    assert passing["passed"] is True
    assert regressing["film-1"]["at_least_vlm"] is False
    assert regressing["passed"] is False
