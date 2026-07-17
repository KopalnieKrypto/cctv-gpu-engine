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

Warm batch-1 latency was measured on the idle second RTX 5070 with all 1,000 raw timings retained:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra gpu python training/activity-mlp/benchmark_latency.py \
  --output /tmp/activity-mlp-latency-rtx5070.json \
  --warmup 100 --samples 1000 --heartbeat-every 100
```

`latency-rtx5070.json` has SHA-256
`516bcf4bd92ea612ca655cd51e3a6a7a28e14087f4a0ffce304b112e81b1232f`. It measured feature
extraction plus synchronous ONNX inference at 0.168660 ms p95 (`method=higher`), passing the 5 ms
bound. The artifact pins the 95,555-byte model checksum, CUDA provider, ORT 1.24.4, driver
595.45.04, warm-up/sample counts, GPU inventory, and raw timings.

End-to-end resource use was measured in fresh containers from one locally built image, on the same
tracked 299.883-second Film 1 input and idle RTX 5070:

```bash
uv run python training/activity-mlp/benchmark_resources.py \
  --image cctv-gpu-engine:issue34 \
  --video test-data/issue86-legacy/film-1.mp4 --gpu 1 \
  --hf-cache-volume cctv-hf-cache \
  --work-dir /tmp/issue34-resource-work \
  --output /tmp/activity-mlp-resources-rtx5070.json
```

`resources-rtx5070-film1.json` has SHA-256
`0673d038f4e8d11ddeb7f3b5320c33c1fef6de2cc560f4b52fef5b12d96abbed`. MLP measured
71.325 seconds / 540 MiB peak, VLM 129.660 seconds / 7,808 MiB, and heuristic 73.898 seconds /
508 MiB. The resource gate passed; all 0.5-second VRAM samples and exact result diagnostics remain
in the artifact.

Films 1+2 were scored only on the original annotated seconds from `jobs/notes.md`. Tracks are
confirmed using the runtime's 3-sightings-in-5-frames rule; when a person re-enters after a long gap,
the evaluator selects the strongest confirmed track present on that second rather than assuming one
ID spans the whole recording (#89 remains out of scope).

```bash
uv run python training/activity-mlp/eval_films.py \
  --film-1-root /tmp/issue34-resource-work \
  --film-2-root /tmp/issue34-resource-work-film2 \
  --image-id sha256:e3d905606d7ecd0f1812b4fa8004d6c4bc1dcc164e12944087e9b34b9e88629d \
  --output /tmp/activity-mlp-films-agreement.json
```

`films-agreement.json` has SHA-256
`c38e120fa412359ca0185e0f77fa5fbc22fe442b4ba8aefdd4a0a93312a9b044` and retains every scored
second. MLP scored 42.31% on Film 1 and 92.22% on Film 2; the film promotion gate failed.
