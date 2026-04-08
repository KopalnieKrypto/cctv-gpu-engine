# Client agent: Flask UI + ffmpeg RTSP recorder + R2 uploader (issue #16).
#
# CPU-only — no CUDA. CLAUDE.md "Don't" forbids python:slim ONLY for GPU
# workloads, so slim is the right base here (smallest possible image).
#
# Today this image ships a placeholder ENTRYPOINT that blocks on SIGTERM.
# Real Flask app + RTSP + upload land in #7 / #8 and will reuse this image
# without bumping the layer cache (source files are the LAST COPY).
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

# - ffmpeg: future RTSP capture (#8) + connection probe
# - ca-certificates + curl: TLS to R2, fetching uv installer
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Match the gpu-service Dockerfile pin so both images use the same uv build.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Lockstep install: copy only the manifest first so source edits don't bust
# the dependency layer. NO `--extra gpu` here — client-agent never imports
# onnxruntime, so the lightweight default keeps this image ~200MB instead of
# pulling in 1.5GB of CUDA wheels.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev \
    && uv cache clean

# Source last so a code change only invalidates the final layer.
COPY client-agent/client_agent /app/client_agent

ENV PATH="/app/.venv/bin:${PATH}"

# Flask UI will listen here when #7 lands. Exposing now so docker-compose can
# already wire the port mapping.
EXPOSE 8080

ENTRYPOINT ["python", "-m", "client_agent.agent"]
