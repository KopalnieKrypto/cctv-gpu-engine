# Activity MLP evaluation and rollout decision (issue #34)

## Decision

**Do not promote or deploy `activity-mlp-v1.0.0`. VLM remains the GPU-service default.**

The frozen MLP is substantially cheaper than VLM, but it fails the locked quality gate:

- held-out #33 accuracy: **62.67%**, below both the **85%** floor and VLM's **93.33%**;
- held-out geometries: **62% / 64%**, below VLM's **90% / 100%**;
- Film 1 annotated-second agreement: **42.31%**, below the **89%** floor and VLM's **88.46%**;
- Film 2 agreement: **92.22%**, which passes that row, but every row must pass.

No held-out label was used to tune or retrain the model after this result. The failed rows are
reported as measured. The service image still declares `CLASSIFIER=vlm`; heuristic and VLM code
remain available, and no production service on `cctv-vps` or `cctv-vps-2` was changed.

## Data and reproducibility

The restricted issue #33 release was materialized and fully validated before training:

| Evidence | Value |
|---|---|
| Full release archive SHA-256 | `c475e7de8b0f3ab9591da1a91d57763af593451f809a24c93e859b8fcb44d147` |
| Review archive SHA-256 | `63a305a9ef8144fbc70bbd2fe587b25c60adc06c521f0837d6b84f3b42b271ca` |
| Validated assets | 1,000 |
| Frozen split | 700 train / 150 validation / 150 test |
| Labels SHA-256 | `4a316809ddfea6443321d7d4f0f0171ac2a05c668b8c45a3741b1486a29a6c1c` |
| Ordered test IDs SHA-256 | `475d834e28531c43fa45af74c3826d0339cf5d75463398bd2e30add2a93c26ff` |

The exact 115-feature order is frozen in `training/activity-mlp/feature-schema.json` as
`activity-mlp-features-v1`. It contains bbox-normalized raw COCO keypoints and unthresholded
visibility, angle, segment, posture-vector, and global features. The output class order is
`sitting`, `standing`, `walking`, `running`.

The fixed network has hidden sizes 128 and 64, ReLU, dropout 0.15, and a four-class softmax. The
training seed is 3407. Adam used learning rate 0.001, weight decay 0.0001, batch size 64, at most 500
epochs, early-stopping patience 50, and minimum delta 0.0001. Training selected epoch 18 by
validation loss and stopped after epoch 68; validation accuracy was 98.67%. Only train/validation
were visible until the frozen evaluation command.

Reproduction commands and immutable artifact hashes are recorded in
`training/activity-mlp/results/README.md`.

## Frozen held-out quality

Baselines were evaluated and committed before MLP training. The immutable baseline artifact has
SHA-256 `3df1e16a619367a809018815826bdb90eae2e389b63df59dcdbe0d67f2dacbee`.

| Classifier | Complete test (150) | controlled-garden (100) | pexels-marathon (50) |
|---|---:|---:|---:|
| Heuristic | 33.33% | 50.00% | 0.00% |
| VLM | **93.33%** | **90.00%** | **100.00%** |
| MLP | 62.67% | 62.00% | 64.00% |

### MLP per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| sitting | 100.00% | 86.00% | 92.47% | 50 |
| standing | 21.62% | 44.44% | 29.09% | 18 |
| walking | 33.33% | 34.38% | 33.85% | 32 |
| running | 86.49% | 64.00% | 73.56% | 50 |

Confusion matrix, rows = truth and columns = prediction, both in the frozen class order:

| Truth \ prediction | sitting | standing | walking | running |
|---|---:|---:|---:|---:|
| sitting | 43 | 7 | 0 | 0 |
| standing | 0 | 8 | 10 | 0 |
| walking | 0 | 16 | 11 | 5 |
| running | 0 | 6 | 12 | 32 |

The complete per-sample predictions, regression IDs, baselines, and gate booleans are in
`training/activity-mlp/results/frozen-test-evaluation.json` (SHA-256
`f024a83e279bc5abee0a338821cd7985ca365ddd4acc154f61bb8ef335a26505`).

## Films 1+2 annotated-second agreement

The evaluator expands the original `jobs/notes.md` intervals as integer seconds with an inclusive
start and exclusive end. It scores only annotated seconds, excluding documented visible-presence
gaps. It mirrors the runtime's minimum-track filter (three sightings in any five-frame window) and
selects a confirmed track present on each second; this permits legitimate new IDs after long gaps
without implementing #89.

| Film | Annotated seconds | Heuristic | VLM | MLP | MLP row |
|---|---:|---:|---:|---:|---|
| Film 1 | 130 | 89/130 = 68.46% | 115/130 = 88.46% | 55/130 = **42.31%** | fail |
| Film 2 | 90 | 72/90 = 80.00% | 80/90 = 88.89% | 83/90 = **92.22%** | pass |

Historical reporting rounded these VLM recordings to 89%. This evaluation retains exact fractions
and does not round a failing comparison upward. The MLP Film 1 regression is large enough that the
promotion decision does not depend on that rounding convention.

`training/activity-mlp/results/films-agreement.json` retains every scored second, confusion counts,
confirmed IDs, and input detection hashes. Its SHA-256 is
`c38e120fa412359ca0185e0f77fa5fbc22fe442b4ba8aefdd4a0a93312a9b044`.

## Model artifact and runtime resources

The model is published as the experimental prerelease
[`activity-mlp-v1.0.0`](https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/tag/activity-mlp-v1.0.0).
It is explicitly not a production promotion.

| Artifact/runtime check | Measured result | Gate |
|---|---:|---|
| ONNX weights | 95,555 bytes (0.091 MiB) | pass, ≤10 MiB |
| ONNX SHA-256 | `4835d97e368567838d2c6ba2ccaf329ee541de283cfa377e72188783ac89cd67` | verified |
| Warm batch-1, 100 warm-ups + 1,000 samples | p95 0.168660 ms | pass, ≤5 ms |
| MLP end-to-end Film 1 | 71.325 s / 540 MiB peak | measured |
| VLM end-to-end Film 1 | 129.660 s / 7,808 MiB peak | measured |
| Heuristic end-to-end Film 1 | 73.898 s / 508 MiB peak | measured |
| MLP vs VLM resource gate | faster and lower peak VRAM | pass |

The resource comparison used fresh containers from local benchmark image
`sha256:e3d905606d7ecd0f1812b4fa8004d6c4bc1dcc164e12944087e9b34b9e88629d`, the same
299.883-second Film 1
(`sha256:2f6ef8a0eaa1b1c96f3171ea48f5e25e6008ca7af4c04d634945220717dbceb8`), tracking enabled, the same idle RTX 5070,
and no source-code bind mount. GPU 0 was excluded because an unrelated process occupied 9,222 MiB;
GPU 1 began at 2 MiB. Peak VRAM was sampled every 0.5 seconds through `nvidia-smi`.

Raw evidence:

- latency: `training/activity-mlp/results/latency-rtx5070.json`, SHA-256
  `516bcf4bd92ea612ca655cd51e3a6a7a28e14087f4a0ffce304b112e81b1232f`;
- resources: `training/activity-mlp/results/resources-rtx5070-film1.json`, SHA-256
  `0673d038f4e8d11ddeb7f3b5320c33c1fef6de2cc560f4b52fef5b12d96abbed`.

The GPU-service MLP preflight budget is 768 MiB, derived from the measured 540 MiB peak and rounded
up for runtime/allocator variation. This budget is exercised only when `CLASSIFIER=mlp`.

## Optional runtime and rollback

The non-default runtime is available for reproducibility:

```bash
./setup-models.sh
uv run python -m pipeline.analyze input.mp4 \
  --output result.json --classifier mlp \
  --activity-model models/activity-mlp-v1.0.0.onnx \
  --activity-model-metadata models/activity-mlp-v1.0.0.json
```

`setup-models.sh` idempotently fetches and verifies the ONNX weights and metadata sidecar. The
loader calls `ort.preload_dlls`, verifies the model checksum, feature schema and class order, creates
an ONNX Runtime CUDA session, and rejects a silent CPU fallback. Each detection is classified
independently before smoothing by `track_id`. `detections.jsonl`, aggregation, and schema-v6
`result.json` preserve per-person predictions. Result diagnostics include classifier, model version,
model SHA-256, and feature-schema version.

The REST server and R2 worker accept `ACTIVITY_MODEL_PATH` and `ACTIVITY_MODEL_METADATA_PATH` while
keeping `CLASSIFIER=vlm` as the image default. The rollback path remains:

```bash
sed -i 's/^CLASSIFIER=.*/CLASSIFIER=vlm/' .env.gpu && \
  docker compose up -d --force-recreate gpu-service
```

No rollout occurred, so this rollback command was documented but not invoked.

## Acceptance and deployment matrix

| Acceptance criterion | Status | Evidence / blocker |
|---|---|---|
| Restricted release SHA + full validation | verified | 1,000 assets; release checksums above |
| Exact 700/150/150 split; test untouched until frozen eval | verified | data/split tests and frozen command |
| Training/eval/export pipeline and frozen versions | verified | `training/activity-mlp/` |
| Feature/order/visibility/leakage/determinism/parity tests | verified | automated CPU/GPU suites |
| Checksummed release + idempotent setup | verified | prerelease and double-run setup evidence |
| Pre-training heuristic/VLM baselines | verified | immutable baseline artifact |
| MLP held-out accuracy ≥85% | **failing** | 62.67% |
| MLP ≥VLM overall | **failing** | 62.67% vs 93.33% |
| No geometry regression | **failing** | 62/64% vs 90/100% |
| Films 1+2 each ≥89% and no VLM regression | **failing** | Film 1 42.31%; Film 2 92.22% |
| Independent multi-person predictions through outputs | verified | automated integration test |
| ONNX ≤10 MiB | verified | 95,555 bytes; automated artifact gate |
| RTX 5070 p95 ≤5 ms | verified | 0.168660 ms; raw 1,000-sample gate |
| Faster and lower peak VRAM than VLM | verified | same-image Film 1 resource artifact |
| Optional CLI/service runtime + diagnostics | verified | automated CLI/REST/worker tests + GPU runs |
| Promote default to MLP | **blocked by failed quality gate** | VLM remains default |
| Deploy both workers + real REST-path smoke | **not run by design** | deployment is conditional on promotion gate |

The issue's completion rule therefore does not permit a “completed/deployed” claim. The valid
outcome is a published negative result, an optional reproducible runtime, and unchanged user-visible
VLM behavior.
