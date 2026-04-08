# CCTV GPU Engine

[![Tests](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/tests.yml)
[![Build Docker images](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/docker.yml/badge.svg)](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/docker.yml)

Batch surveillance video analysis powered by idle GPU infrastructure. Upload MP4 footage, get a standalone HTML activity report — no manual review needed.

**What it does:** Detects people in surveillance footage using YOLO-pose, classifies their activity (sitting, standing, walking, running), and generates a self-contained HTML report with charts, timeline, and annotated keyframes.

---

## Installing this image? Pick your role

This repository ships **two separate Docker images** that run on **two different machines**. Skip straight to the section that matches the image you pulled:

| You pulled… | You are… | Jump to |
|---|---|---|
| `ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service` | the **GPU host operator** (investor / data center side, NVIDIA hardware) | [GPU Service Setup](#gpu-service-setup-gpu-host-operator) |
| `ghcr.io/kopalniekrypto/cctv-gpu-engine/client-agent` | the **on-premise operator** (customer office, regular CPU box with cameras on the LAN) | [Client Agent Setup](#client-agent-setup-on-premise-operator) |

The two images talk to each other **only** through a shared Cloudflare R2 bucket — no direct network connection, no VPN, no port forwarding. Operators on each side only need their own image and their own `.env` file.

---

## Architecture

```
┌─ Client LAN ──────────┐       ┌─ Cloudflare R2 ──────┐       ┌─ GPU Server ───────────┐
│                        │       │ surveillance-jobs/    │       │                        │
│  client-agent          │──────>│   {job_id}/           │<──────│  gpu-service            │
│  Flask UI :8080        │upload │     status.json       │ poll  │  YOLO-pose inference   │
│  ffmpeg RTSP → MP4     │       │     input/*.mp4       │       │  activity heuristics   │
│  boto3 → R2            │<──────│     output/report.html│──────>│  HTML report gen       │
│                        │ poll  │                       │upload │                        │
└────────────────────────┘       └───────────────────────┘       └────────────────────────┘
```

Two Docker images connected by an R2 bucket — no database, no direct communication.

## GPU Service Setup (GPU host operator)

You're running the worker that does the actual YOLO-pose inference. Pull jobs from R2, process them, push reports back.

**Host requirements (one-time):**

- Linux (Ubuntu 22.04+ recommended), x86_64
- NVIDIA GPU with compute capability ≥ 7.5 (verified on RTX 4090 and RTX 5070)
- **NVIDIA driver ≥ 560** — older drivers will fail at container start with `CUDA driver version is insufficient`
- **`nvidia-container-toolkit`** installed and configured for Docker:
  ```bash
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- Docker 24.0+ with the `compose` plugin
- ~10 GB free disk for the image + workdir

**Run it:**

```bash
# 1. Clone the repo (only for compose file + setup-models.sh — image is pulled from GHCR)
git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git
cd cctv-gpu-engine

# 2. Download the YOLO-pose model (NOT baked into the image to keep size down)
./setup-models.sh   # produces models/yolo11n-pose.onnx (~12 MB)

# 3. Configure R2 credentials
cp .env.gpu.example .env.gpu
$EDITOR .env.gpu    # fill R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET

# 4. Pull and start (image comes from GHCR — no local build)
docker compose pull
docker compose up -d
```

**Where to get R2 credentials:** Cloudflare dashboard → R2 → "Manage R2 API Tokens" → create a token scoped to the `surveillance-data` bucket with read+write. The endpoint URL is shown on the bucket overview page (`https://<account>.r2.cloudflarestorage.com`). Both sides (GPU and client) need credentials for the **same** bucket.

**Verify it's working:**

```bash
docker compose ps                 # cctv-gpu-service should be "Up (healthy)"
curl http://localhost:5000/dashboard   # investor dashboard — shows job history pulled from R2
docker compose logs -f gpu-service     # should log "polling for pending jobs" every 30s
```

The dashboard on `:5000` shows every job that's ever been processed (job id, status, duration, errors). It's read-only — pure observability. Refreshes itself every 10 seconds.

**Healthcheck:** the container is "healthy" only when the dashboard responds with HTTP 200. If the worker crashes the daemon thread dies with it, so a 200 here means **both** processes are alive.

**If you see "CPU EP fallback"** in the logs: the GPU isn't actually being used. Check `nvidia-smi` runs on the host, that the toolkit is installed, and that `docker info | grep -i nvidia` shows the runtime. Per CLAUDE.md "Don't" we never silently fall back to CPU — the worker exits with an error.

---

## Client Agent Setup (on-premise operator)

You're running the box that records video from your IP cameras and ships it to R2 for processing. No GPU needed — this image is pure CPU.

**Host requirements (one-time):**

- Any Linux box (or Windows/macOS with Docker Desktop) reachable from your camera network
- Docker 24.0+ with the `compose` plugin
- ~1 GB free disk for the image + recording workdir
- Network access to your RTSP cameras AND outbound HTTPS to `r2.cloudflarestorage.com`

**Current state — placeholder!**

> ⚠️ As of this image's first GHCR release, the client-agent ships a **placeholder ENTRYPOINT** that just logs a banner and blocks until SIGTERM. The Flask UI on `:8080`, ffmpeg RTSP recording, and R2 upload are landing in issues #7 and #8 — they will reuse this same image without bumping the layer cache. So setting it up now is fine; it'll just sit idle until the next image release.

**Run it:**

```bash
# 1. Clone the repo (only for the compose file — image is pulled from GHCR)
git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git
cd cctv-gpu-engine

# 2. Configure credentials
cp .env.client.example .env.client
$EDITOR .env.client    # fill R2_*, RTSP_DEFAULT_URL, MAX_RECORDING_HOURS

# 3. Pull and start
docker compose -f docker-compose.client.yml pull
docker compose -f docker-compose.client.yml up -d
```

**Where to get R2 credentials:** see the GPU Service section above — both sides share the same bucket and use the same kind of token. The investor / GPU host operator typically creates the bucket and shares credentials with you out-of-band.

**Verify it's working:**

```bash
docker compose -f docker-compose.client.yml ps      # cctv-client-agent should be "Up"
docker compose -f docker-compose.client.yml logs    # should show the placeholder banner today
```

Once #7 / #8 land, opening `http://localhost:8080` in your browser will give you the Flask UI for triggering recordings and watching upload progress.

---

## Local Pipeline (development)

Two install profiles — pick the one that matches your machine:

```bash
# macOS / dev box (no NVIDIA GPU): CPU-only onnxruntime stub, ~50MB
make sync-dev

# Linux + NVIDIA GPU: real onnxruntime-gpu + cublas, ~1.5GB
make sync-gpu
```

Install the ruff pre-commit hook once after sync: `uv run pre-commit install`.

Then run unit tests or the GPU smoke test:

```bash
make test                                          # unit tests (no GPU needed)
make test-gpu TEST_VIDEO=test-data/your.mp4        # end-to-end CUDA inference
```

Or invoke the CLI directly:

```bash
uv run python -m pipeline.analyze input.mp4 --timestamp 12.5 --model models/yolo11n-pose.onnx
```

> A bare `uv sync` (no `--extra`) installs only numpy/pillow/opencv — handy for
> reading the code or running lint without pulling the ONNX runtime.

## Report Output

Each report is a standalone HTML file (zero external dependencies) containing:

- **Summary table** — video duration, frames analyzed, peak/avg person count, dominant activity
- **Pie chart** — person-minutes per activity class
- **Timeline** — stacked bar chart with 1-minute bins showing activity over time
- **Annotated keyframes** — 5 selected frames with bounding boxes, skeleton overlays, and activity labels

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Pose detection | YOLOv11n-pose (ONNX) |
| Inference | onnxruntime-gpu, CUDAExecutionProvider |
| Frame extraction | ffmpeg at 1 fps |
| Activity classification | Geometric heuristics on COCO 17 keypoints |
| Report | Jinja2 + vendored Chart.js |
| Client UI | Flask |
| Job coordination | Cloudflare R2 (S3-compat), no database |
| GPU Docker base | nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 |

## Performance

On RTX 5070, processing a 1-hour video:

| Stage | Time |
|-------|------|
| Frame extraction | ~36s |
| YOLO-pose inference | ~6 min |
| Classification + report | ~9s |
| **Total** | **~7 min** (~8:1 ratio) |

VRAM usage: ~600MB. Works on RTX 5070 and RTX 4090.

## Project Structure

```
├── pipeline/              # Core AI pipeline
│   ├── analyze.py         # CLI entry point
│   ├── pose_detector.py   # YOLO-pose ONNX inference
│   ├── activity_classifier.py
│   └── report_generator.py
├── gpu-service/           # R2 polling worker + investor dashboard
├── client-agent/          # Flask UI + ffmpeg recorder + R2 uploader
├── models/                # yolo11n-pose.onnx (gitignored)
├── test/                  # Validation scripts
├── plans/                 # Implementation plan
└── SPEC.md                # Full technical specification
```

## Documentation

- [SPEC.md](SPEC.md) — Full technical specification
- [DECISION_LOG.md](DECISION_LOG.md) — Design decisions and rationale
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 4-phase roadmap
- [plans/surveillance-prototype.md](plans/surveillance-prototype.md) — Vertical-slice implementation plan

## License

Proprietary — [KopalnieKrypto](https://github.com/KopalnieKrypto)
