# CCTV GPU Engine

[![Tests](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/tests.yml)
[![Build Docker images](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/docker.yml/badge.svg)](https://github.com/KopalnieKrypto/cctv-gpu-engine/actions/workflows/docker.yml)

Batch surveillance video analysis powered by idle GPU infrastructure. Upload MP4 footage, get a standalone HTML activity report — no manual review needed.

**What it does:** Detects people in surveillance footage using YOLO-pose, classifies their activity (sitting, standing, walking, running) with a hybrid VLM + displacement pipeline, and generates a self-contained HTML report with charts, timeline, and annotated keyframes.

---

## Setting this up? Pick your role

This system runs across **two different machines**. Skip straight to the section that matches your role:

| Your role | You are… | Jump to |
|---|---|---|
| **GPU host operator** — pull the `ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service` Docker image | the investor / data center side, NVIDIA hardware | [GPU Service Setup](#gpu-service-setup-gpu-host-operator) |
| **On-premise operator** — install the bare-metal client appliance (no Docker) | customer office, a regular CPU box with cameras on the LAN | [Client Appliance Setup](#client-appliance-setup-on-premise-operator) |

The GPU side runs a single Docker image; the client side is a bare-metal systemd appliance (Docker for the client was retired in #29). The two sides talk to each other **only** through a shared Cloudflare R2 bucket — no direct network connection, no VPN, no port forwarding. The GPU host uses its `.env.gpu` file with R2 credentials; the client holds no R2 credentials and (in platform mode) uploads via presigned URLs.

---

## Architecture

```
┌─ Client LAN ──────────┐       ┌─ Cloudflare R2 ──────┐       ┌─ GPU Server ───────────┐
│                        │       │ surveillance-jobs/    │       │                        │
│  client appliance      │──────>│   {job_id}/           │<──────│  gpu-service            │
│  (bare-metal systemd)  │upload │     status.json       │ poll  │  YOLO-pose + VLM       │
│  Flask UI :8080        │(pre-  │     input/*.mp4       │       │  activity classif.     │
│  ffmpeg RTSP → buffer  │signed │     output/report.html│──────>│  HTML report gen       │
│  presigned-URL upload  │ URLs) │                       │upload │                        │
└────────────────────────┘       └───────────────────────┘       └────────────────────────┘
```

A bare-metal client appliance and a gpu-service Docker image connected by an R2 bucket — no database, no direct communication.

## GPU Service Setup (GPU host operator)

You're running the worker that does the actual YOLO-pose inference. Pull jobs from R2, process them, push reports back.

> **Looking for the full step-by-step?** See **[docs/SETUP_GPU.md](docs/SETUP_GPU.md)** — exhaustive setup guide with hardware/software requirements, R2 bucket creation, smoke tests, and a troubleshooting matrix. The section below is the speed-run version for operators who already know the stack.

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
- ~25 GB free disk for the image (~16 GB) + VLM model cache (~6 GB) + workdir

**Run it:**

```bash
# 1. Clone the repo (only for compose file + setup-models.sh — image is pulled from GHCR)
git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git
cd cctv-gpu-engine

# 2. Download the YOLO-pose model (NOT baked into the image to keep size down)
./setup-models.sh   # curls models/yolo11s-pose.onnx (~38 MB) from GitHub release, sha256-verified

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

The dashboard on `:5000` shows every job that's ever been processed (job id, status, duration, **per-job telemetry: peak GPU%, GPU temperature, peak CPU%, peak RAM%, disk usage**, and error). It's read-only — pure observability. Refreshes itself every 10 seconds. Telemetry is sampled by the worker on every progress callback (powered by `psutil` + NVML) and persisted into each job's `status.json` under `metrics`, so the same numbers are available programmatically via R2.

**Healthcheck:** the container is "healthy" only when the dashboard responds with HTTP 200. If the worker crashes the daemon thread dies with it, so a 200 here means **both** processes are alive.

**If you see "CPU EP fallback"** in the logs: the GPU isn't actually being used. Check `nvidia-smi` runs on the host, that the toolkit is installed, and that `docker info | grep -i nvidia` shows the runtime. Per CLAUDE.md "Don't" we never silently fall back to CPU — the worker exits with an error.

---

## Client Appliance Setup (on-premise operator)

You're running the box that records video from your IP cameras. No GPU needed, and **no Docker** — the client is a bare-metal systemd appliance (the Docker client image was retired in #29). It holds no R2 credentials; in platform mode it uploads via presigned URLs handed to it by the platform.

> **Looking for the full step-by-step?** See **[client-appliance/README.md](client-appliance/README.md)** — the canonical install / update / troubleshooting runbook (systemd unit, idempotent installer, env templates, 5-minute smoke test). **[docs/SETUP_CLIENT.md](docs/SETUP_CLIENT.md)** is a short redirect to it. The section below is the speed-run version.

**Host requirements (one-time):**

- A mini-PC / small Linux box (Ubuntu 24.04 LTS or Raspberry Pi OS Bookworm — Python 3.12 + systemd + ffmpeg) reachable from your camera network
- ~5 GB free disk for the local rolling buffer (more if you buffer long sessions on high-bitrate cameras)
- Network access to your RTSP cameras
- **No GPU, no Docker.** The appliance is pure CPU.

**Install it:**

```bash
sudo apt-get update && sudo apt-get install -y git python3.12-venv ffmpeg

git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git /opt/src/cctv-gpu-engine
cd /opt/src/cctv-gpu-engine
sudo ./client-appliance/install.sh          # creates the cctv user + venv, installs the systemd unit

# Fill in RTSP camera creds (and, for platform mode, PLATFORM_URL + APPLIANCE_TOKEN)
sudo nano /etc/cctv-client/cameras.env
sudo nano /etc/cctv-client/platform.env

sudo systemctl enable --now cctv-client
```

The installer is idempotent — re-run it after `git pull` and it won't clobber your `/etc/cctv-client/*.env` files.

**Verify it's working:**

```bash
systemctl status cctv-client                # should be active (running)
journalctl -u cctv-client -f                # Flask + appliance boot logs
```

Open `http://<appliance-ip>:8080` in your browser for the Flask UI: camera discovery ("Wykryj kamery"), per-camera snapshots, the managed-cameras panel, and the test-connection / stop controls. The legacy manual "upload an MP4 / record → R2" flow has been retired — those routes now return 503.

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
uv run python -m pipeline.analyze input.mp4 --timestamp 12.5 --model models/yolo11s-pose.onnx
```

> A bare `uv sync` (no `--extra`) installs only numpy/pillow/opencv — handy for
> reading the code or running lint without pulling the ONNX runtime.

### Using a different model size

`./setup-models.sh` ships the **small** variant (`yolo11s-pose.onnx`, 38 MB) —
sweet spot between detection accuracy and inference latency on RTX 5070.
For low-latency / VRAM-constrained runs you can swap to **nano** (12 MB);
for higher accuracy on tougher footage swap up to **m / l / x**.

The pipeline only assumes the standard YOLO11*-pose ONNX shape
(input `[1,3,640,640]`, output `[1,56,N]` — 4 bbox + 1 conf + 17×3 keypoints),
so any `yolo11{n,s,m,l,x}-pose` exported at imgsz=640 is a drop-in.

**Switch back to nano (pinned release on GH):**

```bash
MODEL_TAG=yolo11n-pose-v1.0 \
MODEL_FILE=yolo11n-pose.onnx \
MODEL_SHA256=70bd721f9cb797eb44cbc70bc65213397a0a26da38fe6fd5ccdf699016d33d3c \
./setup-models.sh
```

**Switch up to a bigger variant** (one-off — m/l/x aren't on GH releases,
you export them yourself; ultralytics needs a few onnx deps in the same
throwaway venv as itself):

```bash
# Pick one: yolo11m-pose, yolo11l-pose, yolo11x-pose
uvx --with onnx --with onnxslim --with onnxruntime --from ultralytics \
  yolo export model=yolo11m-pose.pt format=onnx imgsz=640
mv yolo11m-pose.onnx models/
```

**Point the pipeline at it** — every entry point takes `--model` (CLI) or
`MODEL_PATH` (gpu-service container):

```bash
# Direct CLI
uv run python -m pipeline.analyze input.mp4 --model models/yolo11m-pose.onnx

# make targets — override MODEL on the command line
make test-gpu MODEL=models/yolo11m-pose.onnx TEST_VIDEO=test-data/your.mp4

# gpu-service container — set MODEL_PATH in your .env.gpu or compose file
# (defaults to /app/models/yolo11s-pose.onnx)
```

**Trade-offs to expect:**

| Variant | Size  | Speed (RTX 5070, ~ms/frame) | Accuracy |
|---------|-------|------------------------------|----------|
| nano    | 12 MB | ~70–100 ms                   | baseline |
| **small** (default) | 38 MB | ~150 ms          | +        |
| medium  | 76 MB | ~250 ms                      | ++       |
| large   | 96 MB | ~350 ms                      | +++      |
| xlarge  | 222 MB| ~500 ms                      | ++++     |

Numbers are rough — measure on your hardware. The activity classifier and NMS
thresholds (0.25 conf, 0.45 IoU) are size-independent so you don't need to retune.
`setup-models.sh` is pinned to the **nano** and **small** GH releases only;
m/l/x variants are export-it-yourself by design (we don't want to host every
size on GH releases).

## Report Output

Each report is a standalone HTML file (zero external dependencies) containing:

- **Summary table** — video duration, frames analyzed, peak/avg person count, dominant activity
- **Pie chart** — person-minutes per activity class
- **Timeline** — stacked bar chart with 1-minute bins showing activity over time
- **Annotated keyframes** — 5 selected frames with bounding boxes, skeleton overlays, and activity labels

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Pose detection | YOLOv11n-pose (ONNX) via onnxruntime-gpu |
| Activity classification | Hybrid: Qwen2.5-VL-3B (sitting/standing) + bbox displacement (walking) |
| Frame extraction | ffmpeg at 1 fps |
| Report | Jinja2 + vendored Chart.js |
| Client UI | Flask |
| Job coordination | Cloudflare R2 (S3-compat), no database |
| GPU Docker base | nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 |

## Performance

On RTX 5070, processing a 1-hour video at 1 fps (3600 frames):

| Mode | Total time | Ratio | VRAM |
|------|-----------|-------|------|
| **VLM hybrid** (default) | ~20 min | 3:1 | ~6 GB |
| Heuristic only | ~7 min | 8:1 | ~600 MB |

VLM hybrid: YOLO ~100ms + Qwen2.5-VL ~270ms per frame. First frame includes ~40s model load (cached for subsequent jobs). Accuracy: ~85% on ground-truth test (sitting 89%, walking 96%).

Works on RTX 5070 and RTX 4090.

## Project Structure

```
├── pipeline/              # Core AI pipeline
│   ├── analyze.py         # CLI entry point (--classifier vlm|heuristic)
│   ├── pose_detector.py   # YOLO-pose ONNX inference
│   ├── vlm_classifier.py  # Qwen2.5-VL-3B activity classifier
│   ├── activity_classifier.py  # Heuristic fallback + displacement smoother
│   └── report_renderer.py
├── gpu-service/           # R2 polling worker + investor dashboard
├── client-agent/          # Shared client package (Flask UI + ffmpeg recorder), served by the bare-metal appliance
├── client-appliance/      # Bare-metal appliance packaging (systemd unit + installer)
├── scripts/               # Standalone test/benchmark scripts
├── setup-models.sh        # curl + sha256-verify yolo11s-pose.onnx (GH release pin)
├── models/                # yolo11s-pose.onnx (gitignored, fetched by setup-models.sh)
├── test/                  # Validation scripts
└── SPEC.md                # Full technical specification
```

## Documentation

- [docs/SETUP_GPU.md](docs/SETUP_GPU.md) — Step-by-step setup guide for the GPU host operator (hardware, driver, container toolkit, R2, smoke tests, troubleshooting)
- [docs/SETUP_CLIENT.md](docs/SETUP_CLIENT.md) — On-premise operator setup (short redirect to the canonical bare-metal appliance runbook)
- [client-appliance/README.md](client-appliance/README.md) — Canonical client appliance install / update / troubleshooting runbook (bare-metal, systemd)
- [SPEC.md](SPEC.md) — Full technical specification
- [DECISION_LOG.md](DECISION_LOG.md) — Design decisions and rationale
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 4-phase roadmap
- [plans/surveillance-prototype.md](plans/surveillance-prototype.md) — Vertical-slice implementation plan

## License

Proprietary — [KopalnieKrypto](https://github.com/KopalnieKrypto)
