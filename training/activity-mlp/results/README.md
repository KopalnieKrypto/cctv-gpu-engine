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
