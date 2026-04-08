# CCTV GPU Engine

Batch surveillance video analysis: MP4 → YOLO-pose → activity classification → standalone HTML report. Full spec: `SPEC.md`.

## Quick reference

- CLI: `uv run python -m pipeline.analyze input.mp4 --output report.html`
- Model: `./setup-models.sh` (curls pinned `yolo11n-pose.onnx` from GitHub release `yolo11n-pose-v1.0`, sha256-verified, idempotent). For non-nano sizes (s/m/l/x) see README "Using a different model size".
- Sync deps (dev/macOS): `make sync-dev` (CPU stub onnxruntime, ~50MB)
- Sync deps (Linux+GPU): `make sync-gpu` (onnxruntime-gpu + cublas, ~1.5GB)
- Install pre-commit hook (one-time, after sync): `uv run pre-commit install` (ruff format + lint on every commit)
- Run unit tests: `make test`
- Run end-to-end GPU smoke test: `make test-gpu`
- GPU service: `docker compose up` (polls R2 for pending jobs)
- Client agent: `docker compose -f docker-compose.client.yml up` (Flask UI :8080)

## Remote infrastructure

- **VPS with NVIDIA GPU**: reachable via `ssh cctv-vps`. Use this for any test that requires real CUDA/onnxruntime-gpu (the `make sync-gpu`, `make test-gpu`, container builds, R2 worker tests). Local macOS dev box only runs CPU-stub unit tests.
- **Project path on VPS**: `/home/mvp/cctv-gpu-engine` — git clone of this repo, tracking `origin/main`.
- **VPS test loop**: the VPS clone is updated only via `git pull`, never rsync/scp. So before running anything on the VPS that depends on local edits: commit + `git push` from local first, then `ssh cctv-vps 'cd /home/mvp/cctv-gpu-engine && git pull'`, then run the test (e.g. `make test-gpu`). Skipping the push/pull means the VPS still runs the previous version.
- **VPS uv path**: `uv` is at `~/.local/bin/uv` on `cctv-vps` and **not** on the default ssh `PATH`. Prefix any non-interactive command that calls `make test` / `uv run` with `export PATH=$HOME/.local/bin:$PATH &&` or it will fail with `make: uv: No such file or directory`.
- **VPS docker socket**: rootless docker on `cctv-vps` is broken (the user systemd unit fails on start); the system-wide dockerd at `/var/run/docker.sock` is what works, and user `mvp` is in the `docker` group. Every non-interactive `docker`/`docker compose` call must `export DOCKER_HOST=unix:///var/run/docker.sock` first or it errors with `Cannot connect to the Docker daemon at unix:///run/user/1000/docker.sock`.
- **VPS env files**: two separate env files live at the repo root — `.env.gpu` (real R2 credentials, used by `docker-compose.yml`) and `.env.client` (historically placeholder-only, used by `docker-compose.client.yml`). Both stacks share the same R2 bucket so the easy fix when bringing up client-agent is to merge `R2_*` keys from `.env.gpu` into `.env.client`. Long-term: consolidate into one file.
- **Local RTSP fake for testing recorder flows**: spin up `bluenviron/mediamtx` on the compose network as an RTSP server, then push a sample MP4 with `ffmpeg -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://mediamtx:8554/stream`. The recorder reaches it as `rtsp://mediamtx:8554/stream` from inside `cctv-client-agent`. HTTP `POST /start` only accepts `duration_h ∈ {1,2,4,8}`, so for short smoke tests bypass the route and call `Recorder.start(url=..., duration_s=30)` directly via `docker exec cctv-client-agent python -c "..."`.

## TODO (deferred)

- **Upload retry (issue #5 follow-up)**: every `gpu_service.r2_client.R2Client` network method (`upload_report`, `download_chunks`, `upload_input_chunk`, `get_report`) is currently single-shot. SPEC §8.2 calls for "retry 3× with exponential backoff, then fail". The worker translates any failure into `status: failed` and the client-agent surfaces R2 errors as a 500, but a flaky network will cause unnecessary job/upload failures. Add a retry decorator (e.g. `botocore.config.Config(retries={"max_attempts": 4, "mode": "adaptive"})` or a small custom backoff) before the first production deploy.

## Architecture

```
client-agent (Flask :8080) → R2 bucket (surveillance-data) → gpu-service (Docker+NVIDIA)
```

- **Pipeline** (`pipeline/`): frame extraction → YOLO-pose → activity heuristics → HTML report
- **Client Agent** (`client-agent/`): Flask UI on :8080 for MP4 upload + job status (#7 ✅) and RTSP recorder with ffmpeg stream-copy + segmented chunks (#8 ✅, e2e-validated on cctv-vps).
- **GPU Service** (`gpu-service/`): R2 polling worker + investor dashboard, downloads video, runs pipeline, uploads report

## Stack

| Component | Technology |
|-----------|-----------|
| Pose model | YOLOv11n-pose ONNX, input `[1,3,640,640]`, output `[1,56,N]` |
| Inference | onnxruntime-gpu, CUDAExecutionProvider only |
| Frame extraction | ffmpeg at 1 fps |
| Image processing | OpenCV + Pillow |
| Classification | Geometric heuristics on COCO keypoints |
| Report | Jinja2 + vendored Chart.js (standalone HTML) |
| Client UI | Flask |
| R2 client | boto3 (S3-compat) |
| Docker base | nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 |
| Python deps | uv preferred, pip fallback |

## Must follow

- Python + onnxruntime-gpu for all inference. No Node.js, no CPU fallback
- `ort.preload_dlls(cuda=True, cudnn=True)` must be called before `InferenceSession()` — onnxruntime-gpu wheel has no RPATH to site-packages/nvidia/. See `pipeline/pose_detector.py`.
- `nvidia-cublas-cu12` must be a direct dep — `onnxruntime-gpu[cuda,cudnn]` extras don't pull it but `libcublasLt.so.12` is required at runtime.
- Docker base: `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` (not python:slim)
- Frame extraction: ffmpeg 1 fps, frame-by-frame processing, never full video in RAM
- YOLO-pose output `[1,56,N]` transposed: rows 0-3 bbox, row 4 conf, rows 5-55 keypoints (17×3)
- Preprocessing: PIL RGB → resize 640×640 → float32 /255 → CHW → batch dim
- Activity classes: sitting, standing, walking, running — no others
- Reports: standalone HTML, zero external deps (vendored Chart.js, base64 images)
- R2 bucket: `surveillance-data`. Key: `surveillance-jobs/{job_id}/`
- Job coordination: `status.json` in R2, no database
- Confidence threshold: 0.25, NMS IoU: 0.45
- uv over pip for Python dependency management

## Don't

- Don't add CPU inference fallback — breaks 1:1 processing guarantee
- Don't trust `ort.get_available_providers()` alone — also check `session.get_providers()` after init to catch silent CPU fallback (microsoft/onnxruntime#25145).
- Don't use `python:slim` Docker images for GPU workloads
- Don't add per-person tracking (ByteTrack/DeepSORT) — deferred
- Don't add RTSP live monitoring — batch only
- Don't add face recognition or person identification
- Don't worry about RODO/GDPR — deferred to production
- Don't add platform integration (billing, tenant isolation, job dispatch) — Phase 4

## File structure

```
├── SPEC.md                    # Full specification
├── DECISION_LOG.md            # Design decisions + rationale
├── IMPLEMENTATION_PLAN.md     # 4-phase roadmap
├── RTX5070_CONSTRAINTS.md     # Hardware compatibility
├── Makefile                   # sync-dev / sync-gpu / test / test-gpu shortcuts
├── pyproject.toml             # uv-managed deps; cpu-stub & gpu extras (issue #9)
├── docker-compose.yml         # gpu-service stack (R2 worker + dashboard)
├── docker-compose.client.yml  # client-agent stack (Flask UI :8080)
├── pipeline/                  # Core AI pipeline (CLI: python -m pipeline.analyze)
│   ├── analyze.py             # full-video CLI entry point
│   ├── pose_detector.py       # ONNX session + CUDAExecutionProvider guard
│   ├── activity_classifier.py # COCO-keypoint heuristics → sit/stand/walk/run
│   ├── report_renderer.py     # Jinja2 → standalone HTML (vendored Chart.js)
│   ├── report_template.html
│   ├── frame_extractor.py / video_frames.py        # ffmpeg @ 1 fps streaming
│   ├── preprocessing.py / postprocessing.py        # PIL→CHW; transpose+NMS
│   ├── annotator.py / aggregator.py                # boxes+keypoints; person-minutes
│   └── vendor/                # vendored Chart.js for offline reports
├── gpu-service/               # R2 polling worker (#5) + investor dashboard (#6)
│   ├── Dockerfile             # nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
│   └── gpu_service/           # worker.py, dashboard.py, r2_client.py (+tests)
├── client-agent/              # Flask UI :8080 (#7) + RTSP recorder (#8)
│   ├── Dockerfile             # python:3.12-slim + ffmpeg, ENTRYPOINT client_agent.agent
│   └── client_agent/          # web.py (Flask), recorder.py (ffmpeg+R2), agent.py (entrypoint) (+tests)
├── tests/                     # Repo-level meta tests (build_config_test.py)
├── test/                      # Legacy single-frame validation scripts (pre-#4)
├── docs/                      # Operator setup guides (SETUP_GPU.md, SETUP_CLIENT.md)
├── setup-models.sh            # curl + sha256 verify yolo11n-pose.onnx from GH release
├── models/                    # yolo11n-pose.onnx (gitignored, fetched by setup-models.sh)
└── test-data/                 # sample MP4s (gitignored)
```

## Performance (RTX 5070, 1h video)

- ~7 min total (8:1 ratio), ~600MB VRAM
- YOLO: ~100ms/frame, 3600 frames/hour at 1fps

## Platform integration (future, not now)

This is a standalone prototype. Integration into ML Compute Exchange (KopalnieKrypto/ml-compute-engine) planned for Phase 4: docker image reference in gpu-agent compose, `problem_type: 'surveillance_analysis'`, billing, tenant isolation.
