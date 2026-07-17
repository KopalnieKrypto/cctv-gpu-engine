# GPU Service Setup Guide

This repository ships one GPU-service image with two entrypoints:

| Mode | Entrypoint | Port | Coordination |
|---|---|---:|---|
| Standalone worker | `python -m gpu_service.worker` (image default) | 5000 | R2 `status.json` polling plus dashboard |
| Platform REST service | `python -m gpu_service.rest_server` | 5003 | gpu-agent task dispatch with presigned URLs |

Both modes run the same CUDA pipeline and produce canonical `result.json`. Standalone HTML is not a worker output.

## 1. Host requirements

- Linux x86_64 with Docker and the Compose plugin.
- NVIDIA GPU reachable from containers.
- NVIDIA driver compatible with the image's CUDA 12.8 runtime. The measured RTX 5070 runs used driver `595.45.04`; verify your host rather than relying on an unmeasured minimum-version claim.
- NVIDIA Container Toolkit configured for the system Docker daemon.
- Outbound access required by the chosen mode:
  - GHCR for the image;
  - Hugging Face for the first VLM model-cache fill unless the cache is pre-populated;
  - R2 for standalone worker traffic or presigned object URLs;
  - GPU Exchange/gpu-agent infrastructure for platform mode.

The default VLM path measured a 7,808 MiB process peak on the pinned Film 1 comparison and 7,866 MiB on the #86 fixed-640 run. Those are fixture-specific measurements, not a universal requirement. The service selects the visible GPU with the most free VRAM and fails fast if it misses the configured preflight budget.

## 2. Install Docker and NVIDIA Container Toolkit

Install Docker from the official repository for your distribution, then install `nvidia-container-toolkit`.

Ubuntu example:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Log out and back in after changing Docker group membership.

Verify the actual runtime used by this project:

```bash
nvidia-smi
docker run --rm --gpus all \
  nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 \
  nvidia-smi
```

Both commands must list the same GPU. Fix this boundary before running repository tests or services.

On `cctv-vps`, the working daemon is the system socket:

```bash
export DOCKER_HOST=unix:///var/run/docker.sock
```

## 3. Clone and materialize models

```bash
git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git
cd cctv-gpu-engine
./setup-models.sh
```

The script checksum-verifies:

- `models/yolo11s-pose.onnx` — deployed pose model;
- `models/osnet_x0_25.onnx` — default tracking model;
- `models/activity-mlp-v1.0.0.onnx` and metadata — experimental #34 artifact.

The image bakes these models for platform REST use. The standalone compose file mounts local `./models` over `/app/models`, so the local files are still required before `docker compose up`.

The MLP artifact being present does not make it production-approved. Its frozen quality gate failed; `CLASSIFIER=vlm` remains the image default.

## 4. Standalone R2 worker and dashboard

Use this mode when jobs are coordinated through `surveillance-jobs/{job_id}/status.json`.

### 4.1 Configure R2

Create a bucket named `surveillance-data` and an R2 API token scoped to object read/write for that bucket.

```bash
cp .env.gpu.example .env.gpu
$EDITOR .env.gpu
```

Required values:

```dotenv
R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<access-key>
R2_SECRET_ACCESS_KEY=<secret-key>
R2_BUCKET=surveillance-data
```

Optional worker overrides:

```dotenv
CLASSIFIER=vlm
MODEL_PATH=/app/models/yolo11s-pose.onnx
POLL_INTERVAL_S=10
VRAM_BUDGET_MB=<explicit-preflight-budget-if-needed>
```

The client appliance does not use these R2 credentials. In platform mode it uploads only through presigned URLs.

### 4.2 Start

```bash
docker compose pull
docker compose up -d
```

Verify:

```bash
docker compose ps
curl -fsS http://localhost:5000/dashboard
docker compose logs -f gpu-service
```

The standalone key layout is:

```text
surveillance-jobs/{job_id}/
  status.json
  input/chunk_001.mp4
  output/result.json
  output/detections.jsonl
```

The dashboard is read-only. Job metrics are sampled during pipeline progress and stored in `status.json.metrics`.

## 5. Platform REST service

The gpu-agent uses the same image but overrides the entrypoint:

```text
python -m gpu_service.rest_server
```

The production gpu-agent owns task-container creation, network wiring, presigned URLs, polling, and teardown. Do not add standalone R2 credentials to this path.

REST environment:

| Variable | Default | Meaning |
|---|---|---|
| `REST_PORT` | `5003` | REST listener |
| `MODEL_PATH` | `/app/models/yolo11s-pose.onnx` | Pose model |
| `CLASSIFIER` | `vlm` | `vlm`, `heuristic`, or experimental `mlp` |
| `ACTIVITY_MODEL_PATH` | `/app/models/activity-mlp-v1.0.0.onnx` | MLP weights |
| `ACTIVITY_MODEL_METADATA_PATH` | `/app/models/activity-mlp-v1.0.0.json` | MLP sidecar |
| `WORKDIR` | `/tmp/cctv-jobs` | Per-task workspace |
| `VRAM_BUDGET_MB` | classifier-specific | Optional preflight override |
| `ZONES_CONFIG_PATH` | `/config/zones.json` | Optional server-owned zone/shift config |

If `ZONES_CONFIG_PATH` does not exist, the task runs without zone or shift gating. If it exists but is malformed, the task fails visibly.

Manual contract smoke, without submitting a task:

```bash
docker run --rm --gpus all \
  -p 5003:5003 \
  --entrypoint python \
  -e CLASSIFIER=vlm \
  -v cctv-hf-cache:/root/.cache/huggingface \
  ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service:latest \
  -m gpu_service.rest_server
```

In another shell:

```bash
curl -fsS http://localhost:5003/healthz
```

The service marks readiness only after CUDA and the pose model warm successfully. A full `/analyze` smoke requires valid tenant-scoped presigned input and result URLs and should normally be driven by gpu-agent.

To test a mounted zones file, add:

```text
-v "$PWD/zones.json:/config/zones.json:ro" -e ZONES_CONFIG_PATH=/config/zones.json
```

## 6. Local CUDA checks

Install the GPU dependency profile:

```bash
make sync-gpu
```

Single-frame pose smoke:

```bash
make test-gpu TEST_VIDEO=test-data/sample.mp4
```

Full-video canonical JSON:

```bash
uv run python -m pipeline.analyze test-data/sample.mp4 \
  --output result.json \
  --classifier vlm \
  --dump-detections detections.jsonl
```

Do not infer completion time from another fixture. For any performance claim, run a measured representative calibration or cite one of the committed benchmark artifacts.

## 7. Measured reference runs

| Run | Measured result |
|---|---|
| #86 fixed-640 VLM, 181.995 s bending input, RTX 5070 | 76.267 s wallclock; 25.14 min/h linear extrapolation; 7,866 MiB peak |
| Film 1 VLM, same-image comparison, RTX 5070 | 129.660 s wallclock; 7,808 MiB peak |
| Film 1 heuristic, same-image comparison | 73.898 s; 508 MiB peak |
| Film 1 experimental MLP, same-image comparison | 71.325 s; 540 MiB peak; failed quality gate |

Evidence:

- [issue #86 result](../benchmark-results/issue-86/README.md)
- [activity MLP evaluation](mlp-classifier-eval.md)

These measurements describe their pinned image, hardware, and fixture. They are not general operator ETAs.

## 8. Troubleshooting

| Symptom | Check |
|---|---|
| `CUDA driver version is insufficient` | Host driver cannot run the CUDA 12.8 image; verify the container `nvidia-smi` boundary in §2. |
| `VRAM_PREFLIGHT_FAIL` | Inspect `nvidia-smi`; another process may occupy the freest GPU, or the explicit budget is too high. |
| `CUDAExecutionProvider` missing after session creation | Confirm GPU extras, `nvidia-cublas-cu12`, toolkit, and container GPU access. CPU fallback is intentionally rejected. |
| `libcublasLt.so.12` missing | Pull/rebuild the current image or use `make sync-gpu`; the direct cublas dependency is required. |
| Compose starts but a model is missing | The `./models:/app/models` mount hides baked models. Run `./setup-models.sh`. |
| First VLM start cannot reach ready state | Check Hugging Face connectivity/cache and container logs. The pose readiness boundary may pass before a later lazy VLM load. |
| Standalone dashboard has no jobs | Verify `.env.gpu`, bucket, endpoint, and `surveillance-jobs/` keys. Do not look for client-side R2 credentials. |
| REST task fails on zones | Validate the mounted JSON and confirm `ZONES_CONFIG_PATH` points to the container path. |

## 9. Updating and stopping standalone mode

```bash
git pull --ff-only
./setup-models.sh
docker compose pull
docker compose up -d
```

Stop while retaining volumes:

```bash
docker compose down
```

Remove local compose volumes only when intentionally discarding the Hugging Face cache and workdir:

```bash
docker compose down -v
```

R2 objects are unaffected by either command.

## See also

- [project overview](../README.md)
- [current specification](../SPEC.md)
- [client appliance runbook](../client-appliance/README.md)
- [pose benchmark methodology](POSE_RESOLUTION_BENCHMARK.md)
