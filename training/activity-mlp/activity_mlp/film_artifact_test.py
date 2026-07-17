"""Automated numeric gate over immutable Films 1+2 per-second evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).parents[1] / "results" / "films-agreement.json"
ARTIFACT_SHA256 = "c38e120fa412359ca0185e0f77fa5fbc22fe442b4ba8aefdd4a0a93312a9b044"


@pytest.mark.perf
def test_frozen_film_artifact_enforces_each_promotion_row_without_rounding() -> None:
    raw = ARTIFACT.read_bytes()
    artifact = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA256
    assert artifact["methodology"] == {
        "agreement_floor": 0.89,
        "person_selection": (
            "runtime-confirmed track present per second; ties by total sightings then track_id"
        ),
        "sample_rate_fps": 1,
        "scored_scope": "annotated seconds only; visible-presence gaps excluded",
    }
    results = artifact["results"]
    expected = {
        "film-1": {
            "heuristic": (89, 130),
            "vlm": (115, 130),
            "mlp": (55, 130),
        },
        "film-2": {
            "heuristic": (72, 90),
            "vlm": (80, 90),
            "mlp": (83, 90),
        },
    }
    for film_id, classifiers in expected.items():
        for classifier, (correct, total) in classifiers.items():
            result = results[film_id][classifier]
            assert result["correct_seconds"] == correct
            assert result["annotated_seconds"] == total
            assert result["agreement"] == correct / total
            assert len(result["per_second"]) == total

    assert results["film-1"]["mlp"]["agreement"] < 0.89
    assert results["film-1"]["mlp"]["agreement"] < results["film-1"]["vlm"]["agreement"]
    assert results["film-2"]["mlp"]["agreement"] >= 0.89
    assert artifact["promotion_film_gate"]["passed"] is False
