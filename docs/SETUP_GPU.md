# GPU Service — Setup Guide (host operator)

Step-by-step instructions to bring up `cctv-gpu-service` on a Linux box with an
NVIDIA GPU. This is the **investor / data-center side** of the system: it polls
Cloudflare R2 for pending jobs, runs YOLO-pose inference, and uploads the
generated HTML report back to R2. It never talks to the customer LAN directly.

> **TL;DR for the experienced operator:**
> driver ≥ 560 → `nvidia-container-toolkit` → clone repo → `./setup-models.sh`
> → `cp .env.gpu.example .env.gpu` and fill R2 keys → `docker compose up -d`.
> Health check: `curl http://localhost:5000/dashboard` returns 200.

---

## 1. Hardware requirements

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| GPU | NVIDIA, compute capability ≥ 7.5 (Turing or newer) | RTX 4090 / RTX 5070 | Verified on RTX 5070 and RTX 4090. Older cards (GTX 10xx, Tesla K-series) **will not work** — onnxruntime-gpu 1.20 drops sm < 75. |
| VRAM | 2 GB free | 8 GB free | Inference itself uses ~600 MB; the rest is headroom for the cuDNN/cuBLAS workspace. |
| CPU | 2 cores x86_64 | 4+ cores | Frame extraction (ffmpeg) and report rendering are CPU-bound. |
| RAM | 4 GB free | 8 GB | Pipeline streams frame-by-frame; never loads a full video into RAM. |
| Disk | 15 GB free | 50 GB free | ~10 GB image, the rest is workdir for in-flight job chunks (deleted after each job). |
| Network | Outbound HTTPS to `*.r2.cloudflarestorage.com` | — | No inbound ports required. R2 endpoint, no listener. |

The GPU host has **no need** to be reachable from the public internet or from
the customer LAN — communication is one-way outbound to R2.

## 2. Software requirements (one-time host setup)

### 2.1 Operating system

- Linux x86_64 — Ubuntu 22.04 LTS or 24.04 LTS recommended.
- Other distros (Debian 12, RHEL 9, Rocky 9) work as long as they support the
  NVIDIA driver and `nvidia-container-toolkit` packages below.
- Windows Server / WSL2 / macOS are **not supported** for the GPU host.

### 2.2 NVIDIA driver

Install **driver version 560 or newer**. Older drivers will fail at container
start with `CUDA driver version is insufficient for CUDA runtime version`
because the image bundles CUDA 12.6.

```bash
# Ubuntu — easiest path is the graphics-drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install -y nvidia-driver-560
sudo reboot
```

Verify:

```bash
nvidia-smi    # must show driver 560+ and your GPU
```

### 2.3 Docker Engine ≥ 24.0 with the `compose` plugin

```bash
# Official Docker repo (covers compose plugin)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER   # log out / log back in afterwards
```

Verify:

```bash
docker --version           # 24.0+
docker compose version     # v2.x
```

### 2.4 NVIDIA Container Toolkit

This is what lets Docker pass the GPU into the container. **Without it the
container starts but inference falls back to errors immediately** (the worker
refuses to start without `CUDAExecutionProvider`, by design).

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:

```bash
docker info | grep -i nvidia        # should list "nvidia" runtime
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

The second command must print the same output as `nvidia-smi` on the host. If
it doesn't, fix this **before** continuing — every subsequent step assumes
the GPU is reachable from inside containers.

### 2.5 Git + curl + bash

Already on every modern Linux. Used by `setup-models.sh` to fetch the ONNX
model from a pinned GitHub Release.

## 3. Cloudflare R2 credentials

You need an R2 bucket and a token with read+write access to it. The same
bucket is shared with every client agent that ships footage to this GPU host.

1. Cloudflare dashboard → **R2** → **Create bucket** named `surveillance-data`
   (the name is hard-coded in `.env.gpu.example`; if you change it, change the
   bucket name on **every** client agent that uploads to it).
2. **Manage R2 API Tokens** → **Create API token**.
3. Permissions: **Object Read & Write**, scoped to the `surveillance-data`
   bucket only.
4. Copy the values shown:
   - `Access Key ID` → `R2_ACCESS_KEY_ID`
   - `Secret Access Key` → `R2_SECRET_ACCESS_KEY`
   - **S3 endpoint** (the `https://<account>.r2.cloudflarestorage.com` URL on
     the bucket overview page) → `R2_ENDPOINT`. Do **not** use the `r2.dev`
     public URL; the worker speaks the S3 protocol.

Save them somewhere temporarily — they're shown only once.

## 4. Bring the service up

### 4.1 Clone the repository

```bash
git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git
cd cctv-gpu-engine
```

You only need the repository for the `docker-compose.yml`, the `setup-models.sh`
script, and the `.env.gpu.example` template. The container image itself is
pulled from GHCR — you do **not** build locally.

### 4.2 Download the YOLO-pose model

```bash
./setup-models.sh
```

This curls `models/yolo11n-pose.onnx` (~12 MB) from a pinned GitHub release
(`yolo11n-pose-v1.0`) and verifies the sha256 checksum. The model is **not**
baked into the Docker image (keeps the image small) — it's bind-mounted at
runtime via the `./models:/app/models:ro` volume.

If you want a larger / more accurate variant (`s` / `m` / `l` / `x`), see the
"Using a different model size" section in the main [README.md](../README.md).

### 4.3 Configure environment

```bash
cp .env.gpu.example .env.gpu
$EDITOR .env.gpu
```

Fill in:

```dotenv
R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<from step 3>
R2_SECRET_ACCESS_KEY=<from step 3>
R2_BUCKET=surveillance-data

# Defaults below are fine for most deployments
SURVEILLANCE_GPU=0          # CUDA device index (first GPU)
MODELS_DIR=./models
POLL_INTERVAL_S=10          # how often to poll R2 for pending jobs

# Pipeline tuning — leave defaults unless you know what you're doing
CONFIDENCE_THRESHOLD=0.25
SAMPLING_FPS=1
KEYFRAME_COUNT=5
KEYFRAME_MIN_SPACING_S=120
```

### 4.4 Pull and start

```bash
docker compose pull
docker compose up -d
```

`pull` grabs `ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service:latest` from
the public GHCR registry. No login required.

## 5. Verify the deployment

```bash
docker compose ps                              # cctv-gpu-service "Up (healthy)"
curl -fsS http://localhost:5000/dashboard      # HTTP 200 + investor dashboard HTML
docker compose logs -f gpu-service             # "polling for pending jobs" every POLL_INTERVAL_S
```

The container reports `healthy` **only** when the dashboard responds 200. The
dashboard worker thread shares its lifecycle with the inference worker, so a
green health check means **both** processes are alive.

Open `http://<gpu-host-ip>:5000/dashboard` in a browser to see the investor
view: a list of every job ever processed (id, status, duration, errors). It
auto-refreshes every 10 seconds, is read-only, and pulls everything from R2 —
no database is involved.

## 6. Smoke test (end-to-end)

To prove the full pipeline works without waiting for a real client to upload:

```bash
# Place any short MP4 with people in it at test-data/sample.mp4
mkdir -p test-data && cp /path/to/your/sample.mp4 test-data/sample.mp4

# Run the GPU smoke test (only on a host that has uv installed; otherwise
# upload through a client agent — see docs/SETUP_CLIENT.md).
make sync-gpu      # one-time, ~1.5 GB of CUDA wheels
make test-gpu TEST_VIDEO=test-data/sample.mp4
```

The script invokes `python -m pipeline.analyze` directly with
`CUDAExecutionProvider`, asserts the report is generated, and prints the
per-stage timing. If this passes, your GPU + driver + toolkit stack is
correctly wired.

## 7. Troubleshooting

| Symptom | Diagnosis | Fix |
|---|---|---|
| `CUDA driver version is insufficient` at container start | Driver < 560 | Upgrade driver (see §2.2). |
| `Cannot connect to the Docker daemon` | Docker not running, or you're not in the `docker` group | `sudo systemctl start docker` and `usermod -aG docker $USER` + relog. |
| `nvidia-container-cli: initialization error` | NVIDIA Container Toolkit not installed or not registered with Docker | Re-run §2.4. |
| Container is `Up` but `/dashboard` returns 502 / connection refused | Worker thread crashed (e.g. CPU-only fallback was attempted and rejected per CLAUDE.md "Don't") | `docker compose logs gpu-service` — look for `CUDAExecutionProvider` initialization errors. Verify §2.4 + §6 smoke test. |
| Dashboard shows zero jobs after a client uploaded | R2 bucket / credentials mismatch between client and GPU side | Confirm `R2_BUCKET` and `R2_ENDPOINT` match exactly on both `.env.gpu` and the client's `.env.client`. |
| `libcublasLt.so.12: cannot open shared object file` | Old image without `nvidia-cublas-cu12` | `docker compose pull` to grab the latest GHCR image. |

For anything else, attach `docker compose logs --no-color gpu-service` to a
GitHub issue.

## 8. Updating

```bash
cd cctv-gpu-engine
git pull
docker compose pull
docker compose up -d        # recreates the container with the new image
```

The `:latest` tag is rebuilt on every push to `main` by `.github/workflows/docker.yml`.
If you want a frozen version, pin a specific tag in `docker-compose.yml`
(e.g. `gpu-service:sha-450dccb`) and re-deploy on your own schedule.

## 9. Stopping / removing

```bash
docker compose down                  # stop, keep workdir + dashboard data
docker compose down -v               # also wipe the cctv-workdir volume
```

R2 contents are **not** touched by either command — job history persists in
the bucket and survives any container lifecycle.

---

See also:

- [docs/SETUP_CLIENT.md](SETUP_CLIENT.md) — the on-prem side that uploads footage
- [README.md](../README.md) — project overview, architecture diagram, model variants
- [SPEC.md](../SPEC.md) — full technical specification
