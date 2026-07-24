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

For `m`, `l`, or `x`, export a 640×640 ONNX model and pass it through `--model` or `MODEL_PATH`. Dynamic inputs fail fast; non-square and higher-resolution exports are supported and shipped per-camera — see [Pose resolution and detection modes](#pose-resolution-and-detection-modes). `setup-models.sh` also fetches the pinned `yolo11s-pose-1280x736.onnx` used by those modes.

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
  "restrict_to_zones": false,
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

`restrict_to_zones` (default `false`) decides whether zones only *annotate* or actually *mask* the analysis. Left off, the headline totals (`peak_persons`, `avg_persons`, `person_minutes`, `timeline`, `dominant_activity`) count everyone in the frame and zones just add a per-zone breakdown. Turned on with at least one polygon, those totals count only people standing inside a zone. With an empty `zones` list the flag is ignored — masking to nothing would zero the report.

In platform REST mode, the service loads `ZONES_CONFIG_PATH`, defaulting to `/config/zones.json`, when that file is mounted. Absence of the file means an ungated whole-frame run.

## Pose resolution and detection modes

Two per-camera detector knobs live in the `pose` block of the mounted `zones.json` (resolved at container startup; `--pose-mode` and `--model` on the CLI). Both default to the shipped 640×640 full-frame behaviour, so a camera that sets neither is unchanged.

- **`pose.input_size`** — `640x640` (default) or `1280x736` (issues #100/#109). The 1280×736 export is non-square: it preserves the 16:9 aspect ratio and skips the ~44% grey padding a square 1280 would carry, so it costs ~2.2× the 640 baseline rather than ~3.9×. On the deep-hall `magazyn` fixture it measured 5× the detection recall (7% → 35%) and lifted precision from 64% to 89% (#101).
- **`pose.mode`** — `full_frame` (default, one downscaled pose call) or `hybrid` (native-resolution tiling plus one whole-frame pass; issues #110/#111, cross-tile dedup guard in #112). Hybrid reaches the 80–120 px-native person band that no full-frame resolution can on 12 GB hardware (0% → 23% on that band), lifting whole-frame recall to ~49%, at ~15× the pose cost — strictly opt-in, and it requires a 1280×736 model. With `restrict_to_zones`, hybrid bounds the tiling to the authored zone bboxes.

Neither is a new default: the effective detection floor on a 3840-wide frame is ~60 px of person height at model input, so a whole-hall camera's far field stays out of reach and the highest-leverage fix remains a station-framed camera (deferred issue #88, blocked on client input). The resolved `input_size` and `pose_mode` are recorded in `result.json` `diagnostics`, alongside a `detection_scale` recall-risk signal (#113) — `input_scale`, the smallest resolvable person height, and a `recall_risk` verdict — which the platform renders as a client-facing caveat when a scene is too deep for the detector, so a low-recall run is never presented as a work-time measurement.

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
| `magazyn-hall-v1` detection recall (296 people), RTX 5070 | 640: 7.1% recall / 64% precision · 1280×736: 35.5% / 89% · hybrid tiling: 45.3% / 71% | [#101](benchmark-results/issue-101/README.md) · [#110](benchmark-results/issue-110/README.md) |
| Film 1, same-image resource comparison, RTX 5070 | VLM 129.660 s / 7,808 MiB; heuristic 73.898 s / 508 MiB; MLP 71.325 s / 540 MiB | [MLP evaluation](docs/mlp-classifier-eval.md) |
| #33 frozen 150-row test | VLM 93.33%; MLP 62.67%; heuristic 33.33% | [MLP evaluation](docs/mlp-classifier-eval.md) |

Issue #86 found no eligible software resolution/ROI arm for the bending-station **pilot** camera; #88 (deferred, blocked on the client providing a station-framed stream) is the remaining lever there. For deep-hall production cameras like `magazyn`, per-camera 1280×736 and hybrid tiling are the shipped detection levers ([Pose resolution and detection modes](#pose-resolution-and-detection-modes)) — 640×640 full-frame stays the default. No arm clears the recall gate on its own, so reports carry a `detection_scale` recall-risk caveat rather than presenting a low-recall run as a measurement.

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
- [docs/NONSQUARE_POSE_INPUT.md](docs/NONSQUARE_POSE_INPUT.md) — non-square 1280×736 pose input (#100)
- [docs/mlp-classifier-eval.md](docs/mlp-classifier-eval.md) — frozen #34 result and decision
- [DECISION_LOG.md](DECISION_LOG.md) — decisions plus dated superseding updates
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — historical prototype plan, not the current architecture

## License

Proprietary — [KopalnieKrypto](https://github.com/KopalnieKrypto)
