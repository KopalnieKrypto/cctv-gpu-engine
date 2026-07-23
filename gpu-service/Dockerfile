# GPU service worker + investor dashboard (issue #6).
#
# Base: nvidia/cuda 12.8 runtime — required by onnxruntime-gpu's
# CUDAExecutionProvider AND PyTorch cu128 for Blackwell (sm_120) GPUs.
# python:slim is NOT acceptable per CLAUDE.md.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# OCI source label — links the GHCR package to the repo so collaborators
# inherit access from repo permissions and the package page renders the
# README. Without this label, GHCR shows the package as orphaned and
# visibility / access must be managed per-package.
LABEL org.opencontainers.image.source="https://github.com/KopalnieKrypto/cctv-gpu-engine"
LABEL org.opencontainers.image.description="GPU service worker — runs YOLO-pose inference on Cloudflare R2 jobs and serves an investor dashboard on :5000. Requires NVIDIA driver >=560 + nvidia-container-toolkit on the host."

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

# System deps:
# - python3.12 + venv: ubuntu24.04 ships 3.12 (matches pyproject requires-python)
# - ffmpeg: frame extraction (1 fps)
# - libgl1 + libglib2.0-0: opencv-python-headless still needs these for some codecs
# - ca-certificates + curl: TLS to R2, fetching uv installer
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        ffmpeg libgl1 libglib2.0-0 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# Bundle the default YOLO pose ONNX model into the image (issue #31). Before
# this step the REST mode (gpu_service.rest_server, port 5003) crashed on
# boot with NoSuchFile because docker-compose.yml's `./models:/app/models:ro`
# bind-mount only applies to the worker entrypoint — gpu-agent and any
# `docker run` without a volume override never had the model available.
#
# Pin URL + sha256 as ARGs so a malicious / accidental swap of the
# released asset fails the build instead of silently substituting weights.
# Keep this in sync with `setup-models.sh`; the build_config_test enforces
# equality between the two pins.
ARG YOLO_MODEL_URL=https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/download/yolo11s-pose-v1.0/yolo11s-pose.onnx
ARG YOLO_MODEL_SHA256=469beac503fdc788ea3980331bc4bfbd2bd00de3772eb0984f4c53032740583f
RUN mkdir -p /app/models \
    && curl -fL "${YOLO_MODEL_URL}" -o /app/models/yolo11s-pose.onnx \
    && echo "${YOLO_MODEL_SHA256}  /app/models/yolo11s-pose.onnx" | sha256sum -c -

# Non-square 1280x736 pose export (#100/#101). NOT the default — MODEL_PATH
# below still points at the square 640 weights. This is baked so a camera can
# opt into it per-task: the gpu-agent sets MODEL_PATH to this path for a camera
# whose config requests 1280x736 (2x detection scale on a 16:9 4K frame, 2.24x
# cost, measured #101). Same weights as yolo11s-pose, different export shape.
ARG YOLO_MODEL_1280X736_URL=https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/download/yolo11s-pose-1280x736-v1.0/yolo11s-pose-1280x736.onnx
ARG YOLO_MODEL_1280X736_SHA256=7ee0fcd86efdf953678a92649a682b4369ee5372ebf2623876602286c37731e9
RUN curl -fL "${YOLO_MODEL_1280X736_URL}" -o /app/models/yolo11s-pose-1280x736.onnx \
    && echo "${YOLO_MODEL_1280X736_SHA256}  /app/models/yolo11s-pose-1280x736.onnx" | sha256sum -c -

# OSNet Re-ID weights (issue #32). Person tracking is on by default, so this
# is as load-bearing as the pose model: without it every job dies on boot in
# load_reid_model(). 0.7 MB, so it costs the image nothing.
ARG OSNET_MODEL_URL=https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/download/osnet-x0_25-v1.0/osnet_x0_25.onnx
ARG OSNET_MODEL_SHA256=86f6314fc903d6b3c7e90c8f4dc5f438fb640faf2840574f3170b749c4765ce6
RUN curl -fL "${OSNET_MODEL_URL}" -o /app/models/osnet_x0_25.onnx \
    && echo "${OSNET_MODEL_SHA256}  /app/models/osnet_x0_25.onnx" | sha256sum -c -

# Experimental per-person activity MLP (issue #34). Bundle both the ONNX
# weights and the metadata sidecar: the runtime refuses to load weights unless
# the checksum, feature schema, and class order match this sidecar. The failed
# frozen quality gate means this is opt-in; CLASSIFIER stays ``vlm`` below.
ARG ACTIVITY_MLP_URL=https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/download/activity-mlp-v1.0.0/activity-mlp-v1.0.0.onnx
ARG ACTIVITY_MLP_SHA256=4835d97e368567838d2c6ba2ccaf329ee541de283cfa377e72188783ac89cd67
ARG ACTIVITY_MLP_METADATA_URL=https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/download/activity-mlp-v1.0.0/model-metadata.json
ARG ACTIVITY_MLP_METADATA_SHA256=d387b156934d8498e3afc0554324959164327848dd9fc57e1d507da9f789d8f4
RUN curl -fL "${ACTIVITY_MLP_URL}" -o /app/models/activity-mlp-v1.0.0.onnx \
    && echo "${ACTIVITY_MLP_SHA256}  /app/models/activity-mlp-v1.0.0.onnx" | sha256sum -c - \
    && curl -fL "${ACTIVITY_MLP_METADATA_URL}" -o /app/models/activity-mlp-v1.0.0.json \
    && echo "${ACTIVITY_MLP_METADATA_SHA256}  /app/models/activity-mlp-v1.0.0.json" | sha256sum -c -

# uv (preferred per CLAUDE.md). Pinned version keeps the build reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Lockstep install: copy only the manifest first so changes to source code
# don't bust the dependency layer. `--extra gpu` is mandatory — without it
# the lightweight default skips onnxruntime-gpu and the worker silently
# falls back to CPU at import time (CLAUDE.md TODO).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra gpu --no-dev \
    && uv cache clean

# Deduplicate cuDNN: the apt-shipped libcudnn9 (~870MB) is unused at runtime
# because onnxruntime-gpu's `ort.preload_dlls(cuda=True, cudnn=True)` loads
# from `site-packages/nvidia/` (the pythonwheels via `nvidia-cudnn-cu12`).
# Keeping both is the canonical-ish layering per CLAUDE.md issue #12 gotcha,
# but we can safely drop the apt copy since the wheels are authoritative.
# nvidia-cublas-cu12 (mandatory direct dep) is unaffected.
RUN apt-mark unhold libcudnn9-cuda-12 \
    && apt-get update \
    && apt-get purge -y libcudnn9-cuda-12 \
    && apt-get autoremove -y --purge \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# VLM classifier deps (Qwen2.5-VL-3B). PyTorch cu128 required for Blackwell
# sm_120 support. Installed via uv pip into the managed venv — uv handles
# onnxruntime + core deps, then adds the VLM layer on top.
#
# PINNED for reproducibility (issue #62). The rest of the image goes to great
# lengths to pin (uv.lock --frozen, uv 0.5.11, YOLO weights by URL + sha256);
# an unpinned VLM stack undoes that — `transformers` has shipped breaking
# Qwen2.5-VL changes across minor versions (the exact symbols
# pipeline/vlm_classifier.py imports), so an unrelated rebuild could silently
# pull a new wheel and break — or change the outputs of — the classifier, with
# nothing in CI to catch it (CI builds the image but never runs VLM inference).
#
# The ARGs below are the single source of truth for the pins and match the
# versions verified working in the deployed gpu-service:latest image on
# cctv-vps. tests/build_config_test.py::TestGpuServiceVlmPins asserts every
# spec carries `==` and that these ARG values equal the known-good versions.
#
# To bump a pin:
#   1. Edit the relevant ARG below (one place).
#   2. Update EXPECTED_VLM_PINS in tests/build_config_test.py to match.
#   3. Rebuild the image on a GPU host and run `make test-gpu` (cctv-vps).
#   4. Run the VLM smoke test: `uv run python scripts/test_vlm_classifier.py`.
#   5. Confirm the classifier output is unchanged, then commit the new pins.
ARG TORCH_VERSION=2.11.0+cu128
ARG TORCHVISION_VERSION=0.26.0+cu128
ARG TRANSFORMERS_VERSION=5.13.0
ARG ACCELERATE_VERSION=1.14.0
ARG QWEN_VL_UTILS_VERSION=0.0.14
RUN uv pip install --python /app/.venv/bin/python --no-cache \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install --python /app/.venv/bin/python --no-cache \
        transformers==${TRANSFORMERS_VERSION} \
        accelerate==${ACCELERATE_VERSION} \
        qwen-vl-utils==${QWEN_VL_UTILS_VERSION}

# Application code. The package is hyphen-named on disk but Python imports it
# as `gpu_service` — the `gpu-service/gpu_service/` layout makes that work
# without a setup.py. We also need pipeline/ for the lazy import inside
# worker.main()'s pipeline closure.
COPY gpu-service/gpu_service /app/gpu_service
COPY pipeline /app/pipeline

# Default model (yolo11s-pose.onnx, ~40 MB) is baked in above so the REST
# entrypoint boots without a host bind-mount (issue #31). Operators that
# want a different model can still mount their own ./models/ over /app/models/
# via docker-compose.yml — the bind-mount overlays the baked file.
ENV MODEL_PATH=/app/models/yolo11s-pose.onnx \
    ACTIVITY_MODEL_PATH=/app/models/activity-mlp-v1.0.0.onnx \
    ACTIVITY_MODEL_METADATA_PATH=/app/models/activity-mlp-v1.0.0.json \
    CLASSIFIER=vlm \
    DASHBOARD_PORT=5000 \
    REST_PORT=5003 \
    WORKDIR=/tmp/cctv-jobs \
    PATH="/app/.venv/bin:${PATH}"

# :5000 — investor dashboard (default ENTRYPOINT = gpu_service.worker, R2 poll).
# :5003 — gpu-agent REST contract (issue #25). Override entrypoint at run time:
#   docker run --entrypoint python <image> -m gpu_service.rest_server
EXPOSE 5000 5003

# Healthcheck: dashboard responds on /dashboard. If the worker crashes the
# daemon thread dies with it, so a 200 here = both processes alive.
# Port is read from $DASHBOARD_PORT (defaults to 5000) so a second worker
# instance launched with `-e DASHBOARD_PORT=5001` is healthchecked correctly.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${DASHBOARD_PORT:-5000}/dashboard" >/dev/null || exit 1

ENTRYPOINT ["python", "-m", "gpu_service.worker"]
