# Immutable evaluation evidence

`baselines.json` is generated exactly once, before MLP training, by:

```bash
PYTHONPATH=training/activity-mlp uv run python -m activity_mlp.baseline_cli \
  --dataset datasets/activity-classifier \
  --output training/activity-mlp/results/baselines.json
```

The command evaluates the deployed heuristic and VLM classifiers on the fixed issue #33 test rows,
records every prediction, pins the label-file and ordered-test-sample checksums, emits progress every
25 rows, and refuses to overwrite an existing result.

The model was then trained exactly once from the frozen 700/150 development split:

```bash
uv run python training/activity-mlp/train.py \
  --dataset datasets/activity-classifier \
  --output-dir /tmp/issue34-training-v1
```

`model-metadata.json` binds the 95,555-byte ONNX weights (`sha256:
4835d97e368567838d2c6ba2ccaf329ee541de283cfa377e72188783ac89cd67`) to feature schema v1,
the dataset, dependency versions, seed, architecture, and validation-selected epoch.
`training-history.json` contains every flushed epoch measurement.

The frozen weights were evaluated once on the untouched held-out rows:

```bash
uv run python training/activity-mlp/eval.py \
  --dataset datasets/activity-classifier \
  --model /tmp/issue34-training-v1/activity-mlp-v1.0.0.onnx \
  --metadata /tmp/issue34-training-v1/model-metadata.json \
  --baselines training/activity-mlp/results/baselines.json \
  --output training/activity-mlp/results/frozen-test-evaluation.json
```

The MLP scored 62.67% overall (controlled-garden 62%, pexels-marathon 64%) against VLM's 93.33%
(90% / 100%). All four frozen quality checks failed. The result is a negative promotion outcome:
the MLP must not become the default and the held-out labels must not be used for retraining or tuning.
