# CCTV GPU Engine

[![Tests](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/tests.yml)
[![Build Docker images](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/docker.yml/badge.svg)](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/docker.yml)

Batch surveillance-video analysis for NVIDIA GPU infrastructure. The system records RTSP cameras on a bare-metal client appliance, detects and tracks people, classifies posture/activity, and produces a structured `result.json` report for the platform to render.

The canonical production artifact is JSON. The older standalone HTML report remains available only for local debugging with `--format html`.

## Current architecture

The repository supports two serving paths:

```text
Platform path
client appliance ──HTTPS──> GPU Exchange ──presigned URLs──> R2
                               │
                               └──gpu-agent──> gpu-service REST :5003
                                                └── result.json

Standalone compatibility path
client/upload source ──> R2 status.json jobs <── gpu-service polling worker :5000
```

- The client appliance is bare-metal systemd only. The retired client Docker image and direct client-side R2 credentials must not return.
- In platform mode, the appliance registers and heartbeats over outbound HTTPS, maintains a rolling camera buffer, claims tasks, and uploads chunks through platform-issued presigned URLs.
- The platform/gpu-agent path starts `gpu_service.rest_server`, submits presigned input/result URLs, and renders the returned `result.json` natively.
- `docker compose up` starts the standalone R2-polling worker and investor dashboard. This remains useful for compatibility and diagnostics but is not the platform dispatch contract.

## What the pipeline does

```text
MP4 chunks
  → ffmpeg frames at 1 fps
  → YOLO11s-pose on CUDA
  → OSNet appearance tracking + minimum-track filter
  → activity classifier
  → optional zone / shift / workstation modes
  → result.json schema 6
```

Activities remain exactly `sitting`, `standing`, `walking`, and `running`.

Classifier modes:

| Mode | Status | Behavior |
|---|---|---|
| `vlm` | Deployed Docker default | Qwen2.5-VL-3B for stationary posture plus displacement-based walking |
| `heuristic` | Supported rollback/baseline | Geometric rules plus displacement smoothing |
| `mlp` | Experimental only | Per-detection ONNX MLP with per-track smoothing; failed the frozen quality gate and must not be promoted |

See [the frozen MLP evaluation](docs/mlp-classifier-eval.md) for the measured negative result.

## Local development

```bash
# macOS or a non-GPU development machine
make sync-dev

# Linux with NVIDIA CUDA
make sync-gpu

# Unit suite
make test

# Single-frame CUDA smoke
make test-gpu TEST_VIDEO=test-data/sample.mp4
```

Install the pre-commit hook once after syncing:

```bash
uv run pre-commit install
```

Run a full video and write the canonical artifact:

```bash
uv run python -m pipeline.analyze input.mp4 \
  --output result.json \
  --classifier vlm \
  --dump-detections detections.jsonl
```

The CLI defaults to `heuristic` for lightweight local runs; the GPU-service image declares `CLASSIFIER=vlm`.

For a legacy local HTML report:

```bash
uv run python -m pipeline.analyze input.mp4 \
  --output report.html \
  --format html \
  --classifier vlm
```

## Models

```bash
./setup-models.sh
```

The script idempotently downloads and checksum-verifies:

- `yolo11s-pose.onnx` — deployed 640×640 pose model;
- `osnet_x0_25.onnx` — required by default person tracking;
- `activity-mlp-v1.0.0.onnx` plus metadata — experimental reproducibility artifact, not a production promotion.

The GPU-service image also bakes these pinned files so the platform REST mode can boot without a host model mount. The standalone compose file overlays `/app/models` with local `./models`, so run `./setup-models.sh` before `docker compose up`.

### Alternative YOLO sizes

The production default is the fixed-square 640×640 YOLO11s export. To reproduce the nano baseline:

```bash
MODEL_TAG=yolo11n-pose-v1.0 \
MODEL_FILE=yolo11n-pose.onnx \
MODEL_SHA256=70bd721f9cb797eb44cbc70bc65213397a0a26da38fe6fd5ccdf699016d33d3c \
./setup-models.sh
```

For `m`, `l`, or `x`, export a fixed 640×640 ONNX model and pass it through `--model` or `MODEL_PATH`. Dynamic or non-square inputs fail fast. A fixed 1280 export is supported for the measured #86 comparison but was not promoted.

## Zone and shift analysis

Pass a JSON configuration with `--zones zones.json`. Detections are assigned by the midpoint of the bounding-box bottom edge (the foot point).

```json
{
  "recording_start": "2026-07-16T06:00:00+02:00",
  "shift": {
    "timezone": "Europe/Warsaw",
    "windows": [["07:00", "15:00"]],
    "breaks": [["11:00", "11:20"]]
  },
  "zones": [
    {
      "id": "bending-1",
      "name": "Giętarka 1",
      "polygon": [[1200, 500], [2600, 500], [2600, 1900], [1200, 1900]],
      "rules": {
        "type": "bending",
        "work": {"min_move_px": 40},
        "conversation": {"proximity_px": 150},
        "absence": {"flag_after_s": 180}
      }
    }
  ]
}
```

The `bending` ruleset emits zone posture totals, anchored-worker presence/absence/work intervals, and conversation intervals. `inference_roi` is optional and exists for measured camera experiments; issue #86 did not justify promoting it as the production default.

In platform REST mode, the service loads `ZONES_CONFIG_PATH`, defaulting to `/config/zones.json`, when that file is mounted. Absence of the file means an ungated whole-frame run.

## GPU service

### Standalone R2 worker

```bash
./setup-models.sh
cp .env.gpu.example .env.gpu
$EDITOR .env.gpu
docker compose pull
docker compose up -d

docker compose ps
curl -fsS http://localhost:5000/dashboard
docker compose logs -f gpu-service
```

Only this standalone worker needs the R2 credentials in `.env.gpu`. The client appliance never stores them.

### Platform REST service

The same image exposes the gpu-agent contract on `:5003` when launched with:

```text
python -m gpu_service.rest_server
```

Routes:

- `GET /healthz` — ready only after CUDA/model warm-up;
- `POST /analyze` — accepts `task_id`, `input_presigned_urls`, and `result_presigned_url`;
- `GET /status/<task_id>` — queued/running/completed/failed state.

The gpu-agent owns the production container lifecycle and presigned URLs. See [GPU setup](docs/SETUP_GPU.md) for the operator contract.

## Client appliance

The canonical runbook is [client-appliance/README.md](client-appliance/README.md).

Root installation:

```bash
sudo ./client-appliance/install.sh
sudo nano /etc/cctv-client/cameras.env
sudo nano /etc/cctv-client/platform.env
sudo systemctl enable --now cctv-client
```

User-mode installation when sudo is unavailable:

```bash
./client-appliance/install-user.sh
nano ~/.config/cctv-client/cameras.env
nano ~/.config/cctv-client/platform.env
systemctl --user status cctv-client
```

User-mode boot persistence requires systemd linger; the installer enables and verifies it.

Platform runtime settings (`buffer_hours`, polling interval, heartbeat interval, and upload chunk bytes) arrive in register/heartbeat responses. Environment values are cold-start fallbacks only; valid platform values win live without restarting the appliance.

## Measured evidence

Performance and quality are fixture-specific; these numbers are not general ETAs.

| Fixture and hardware | Result | Evidence |
|---|---|---|
| #86 bending pilot, fixed-640 VLM, RTX 5070 | 76.267 s measured for 181.995 s input; linear extrapolation 25.14 min/h; 7,866 MiB peak | [#86 result](benchmark-results/issue-86/README.md) |
| Film 1, same-image resource comparison, RTX 5070 | VLM 129.660 s / 7,808 MiB; heuristic 73.898 s / 508 MiB; MLP 71.325 s / 540 MiB | [MLP evaluation](docs/mlp-classifier-eval.md) |
| #33 frozen 150-row test | VLM 93.33%; MLP 62.67%; heuristic 33.33% | [MLP evaluation](docs/mlp-classifier-eval.md) |

Issue #86 found no eligible software resolution/ROI arm for the bending-station camera. Production remains fixed-640; issue #88 requires a station-framed camera stream before pilot validation can resume.

## Project structure

```text
pipeline/                 pose, tracking, classifiers, zones, aggregation, result schema
gpu-service/              R2 polling worker, dashboard, REST gpu-agent contract
client-agent/             shared bare-metal appliance package
client-appliance/         root/user systemd packaging and operator runbook
datasets/                 reviewed activity-classifier release metadata
training/activity-mlp/    reproducible experimental MLP pipeline and evidence
benchmarks/               versioned pose-resolution fixtures and methodology
benchmark-results/        immutable measured selections
docs/                     operator guides and evaluation reports
```

## Documentation

- [SPEC.md](SPEC.md) — current as-built technical specification
- [pipeline/README.md](pipeline/README.md) — pipeline, tracking, zones, and outputs
- [docs/SETUP_GPU.md](docs/SETUP_GPU.md) — GPU host setup for both serving paths
- [client-appliance/README.md](client-appliance/README.md) — canonical appliance runbook
- [docs/POSE_RESOLUTION_BENCHMARK.md](docs/POSE_RESOLUTION_BENCHMARK.md) — reproducible #86 benchmark
- [docs/mlp-classifier-eval.md](docs/mlp-classifier-eval.md) — frozen #34 result and decision
- [DECISION_LOG.md](DECISION_LOG.md) — decisions plus dated superseding updates
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — historical prototype plan, not the current architecture

## License

Proprietary — [KopalnieKrypto](https://github.com/KopalnieKrypto)
