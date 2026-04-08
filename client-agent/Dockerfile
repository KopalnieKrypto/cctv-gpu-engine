# Client agent: Flask UI + R2 uploader (issue #7), ffmpeg RTSP recorder (#8).
#
# CPU-only — no CUDA. CLAUDE.md "Don't" forbids python:slim ONLY for GPU
# workloads, so slim is the right base here (smallest possible image).
#
# As of issue #7 the ENTRYPOINT runs the real Flask app on :8080 (MP4 upload
# form, job list with status badges, report viewer). The ffmpeg binary in
# the apt layer below is staged for #8 (RTSP capture); it's harmless until
# the recorder routes land.
FROM python:3.12-slim

# OCI source label — links the GHCR package to the repo so collaborators
# inherit access from repo permissions and the package page renders the
# README. Without this label, GHCR shows the package as orphaned and
# visibility / access must be managed per-package.
LABEL org.opencontainers.image.source="https://github.com/KopalnieKrypto/cctv-gpu-engine"
LABEL org.opencontainers.image.description="On-premise client agent — Flask UI on :8080 for MP4 upload + job status (issue #7) and Cloudflare R2 multipart uploader. ffmpeg bundled for the RTSP recorder landing in issue #8. CPU-only, runs on any Docker host (no NVIDIA GPU required)."

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

# Flask UI listens here — wired to docker-compose.client.yml port mapping.
EXPOSE 8080

ENTRYPOINT ["python", "-m", "client_agent.agent"]
