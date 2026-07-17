"""CLI implementation for freezing heuristic and VLM held-out baselines."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from activity_mlp.evaluation import (
    VlmRowPredictor,
    build_baseline_artifact,
    predict_heuristic,
    write_json_once,
)


class HeartbeatPredictor:
    """Add visible, flushed progress to a row predictor."""

    def __init__(
        self,
        name: str,
        predictor: Callable[[dict], str],
        *,
        every: int = 25,
        stream: TextIO = sys.stderr,
    ) -> None:
        self._name = name
        self._predictor = predictor
        self._every = every
        self._stream = stream
        self._processed = 0

    def __call__(self, row: dict) -> str:
        prediction = self._predictor(row)
        self._processed += 1
        if self._processed % self._every == 0:
            print(
                f"baseline classifier={self._name} processed={self._processed}",
                file=self._stream,
                flush=True,
            )
        return prediction


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    vlm_factory: Callable[[], object] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if vlm_factory is None:
        from pipeline.vlm_classifier import VLMClassifier

        vlm_factory = VLMClassifier

    artifact = build_baseline_artifact(
        args.dataset / "labels.jsonl",
        {
            "heuristic": HeartbeatPredictor("heuristic", predict_heuristic),
            "vlm": HeartbeatPredictor("vlm", VlmRowPredictor(args.dataset, vlm_factory())),
        },
    )
    write_json_once(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
