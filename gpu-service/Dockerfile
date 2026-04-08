# GPU service worker + investor dashboard (issue #6).
#
# Base: nvidia/cuda runtime + cudnn — required by onnxruntime-gpu's
# CUDAExecutionProvider. python:slim is NOT acceptable per CLAUDE.md.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

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

# uv (preferred per CLAUDE.md). Pinned version keeps the build reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Lockstep install: copy only the manifest first so changes to source code
# don't bust the dependency layer. `--extra gpu` is mandatory — without it
# the lightweight default skips onnxruntime-gpu and the worker silently
# falls back to CPU at import time (CLAUDE.md TODO).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra gpu --no-dev

# Application code. The package is hyphen-named on disk but Python imports it
# as `gpu_service` — the `gpu-service/gpu_service/` layout makes that work
# without a setup.py. We also need pipeline/ for the lazy import inside
# worker.main()'s pipeline closure.
COPY gpu-service/gpu_service /app/gpu_service
COPY pipeline /app/pipeline

# Models are NOT baked into the image (would push it over 5GB). Mount the
# host's models/ dir at runtime via docker compose.
ENV MODEL_PATH=/app/models/yolo11n-pose.onnx \
    DASHBOARD_PORT=5000 \
    WORKDIR=/tmp/cctv-jobs \
    PATH="/app/.venv/bin:${PATH}"

EXPOSE 5000

# Healthcheck: dashboard responds on /dashboard. If the worker crashes the
# daemon thread dies with it, so a 200 here = both processes alive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5000/dashboard >/dev/null || exit 1

ENTRYPOINT ["python", "-m", "gpu_service.worker"]
