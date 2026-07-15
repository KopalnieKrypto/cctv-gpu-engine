# CCTV GPU Engine

Batch surveillance video analysis: MP4 → YOLO-pose + VLM → activity classification → standalone HTML report. Full spec: `SPEC.md`.

## Quick reference

- CLI: `uv run python -m pipeline.analyze input.mp4 --output report.html --classifier vlm`
- Classifier modes: `--classifier heuristic` (fast, geometric rules) or `--classifier vlm` (Qwen2.5-VL-3B hybrid, higher accuracy, default in Docker)
- Model: `./setup-models.sh` (curls pinned `yolo11n-pose.onnx` from GitHub release `yolo11n-pose-v1.0`, sha256-verified, idempotent). For non-nano sizes (s/m/l/x) see README "Using a different model size".
- Sync deps (dev/macOS): `make sync-dev` (CPU stub onnxruntime, ~50MB)
- Sync deps (Linux+GPU): `make sync-gpu` (onnxruntime-gpu + cublas, ~1.5GB)
- Install pre-commit hook (one-time, after sync): `uv run pre-commit install` (ruff format + lint on every commit)
- Run unit tests: `make test`
- Run end-to-end GPU smoke test: `make test-gpu`
- GPU service: `docker compose up` (polls R2 for pending jobs)
- GPU service REST mode (gpu-exchange integration, #25): `docker run --rm --entrypoint python <image> -m gpu_service.rest_server` — Flask on :5003. Routes: `POST /analyze` (presigned URL passthrough), `GET /healthz`, `GET /status/:id`. (`--rm` is required — without it each manual run leaves an exit-0 container behind; see gpu-exchange#64.) For interactive debugging where you want the container removed on Ctrl-C as well, add `-it`: `docker run --rm -it --entrypoint python <image> -m gpu_service.rest_server`.
- Client appliance (bare-metal mini-PC, no Docker — the only client deployment): `sudo ./client-appliance/install.sh` then `systemctl enable --now cctv-client` — see `client-appliance/README.md`
- Client appliance platform mode (#26, optional): set `PLATFORM_URL` + `APPLIANCE_TOKEN` in `/etc/cctv-client/platform.env` (chmod 600, seeded by install.sh from `platform.env.example`) → boot calls platform `register`/`push_cameras`/`heartbeat` and recorders spawn from heartbeat config. Either key missing → auto-fallback to legacy standalone flow.

## Remote infrastructure

- **VPS with NVIDIA GPU**: reachable via `ssh cctv-vps`. Use this for any test that requires real CUDA/onnxruntime-gpu (the `make sync-gpu`, `make test-gpu`, container builds, R2 worker tests). Local macOS dev box only runs CPU-stub unit tests.
- **Project path on VPS**: `/home/mvp/cctv-gpu-engine` — git clone of this repo, tracking `origin/main`.
- **VPS test loop**: the VPS clone is updated only via `git pull`, never rsync/scp. So before running anything on the VPS that depends on local edits: commit + `git push` from local first, then `ssh cctv-vps 'cd /home/mvp/cctv-gpu-engine && git pull'`, then run the test (e.g. `make test-gpu`). Skipping the push/pull means the VPS still runs the previous version.
- **VPS uv path**: `uv` is at `~/.local/bin/uv` on `cctv-vps` and **not** on the default ssh `PATH`. Prefix any non-interactive command that calls `make test` / `uv run` with `export PATH=$HOME/.local/bin:$PATH &&` or it will fail with `make: uv: No such file or directory`.
- **VPS docker socket**: rootless docker on `cctv-vps` is broken (the user systemd unit fails on start); the system-wide dockerd at `/var/run/docker.sock` is what works, and user `mvp` is in the `docker` group. Every non-interactive `docker`/`docker compose` call must `export DOCKER_HOST=unix:///var/run/docker.sock` first or it errors with `Cannot connect to the Docker daemon at unix:///run/user/1000/docker.sock`.
- **VPS disk hygiene**: Docker build cache + dangling images silently fill GPU boxes (cctv-vps hit 100% twice; #19). `bash scripts/setup-docker-disk-cron.sh` installs an idempotent weekly prune (`docker system prune -f` + `docker builder prune -f --filter until=168h`, non-destructive — keeps running containers/`:latest`/named volumes). Auto-detects sudo: system `/etc/cron.weekly/docker-prune` when root, per-user crontab otherwise (e.g. cctv-vps-2, no sudo). Run it on every new GPU node (`git pull` then `bash scripts/setup-docker-disk-cron.sh`). Installed on cctv-vps + cctv-vps-2 (both per-user crontab, Sun 04:00 — neither has passwordless sudo). Note the socket differs: cctv-vps needs `DOCKER_HOST=unix:///var/run/docker.sock` baked in (rootless broken), cctv-vps-2 uses the default context; the installer auto-detects which and bakes the right one into the cron line.
- **VPS env files**: the GPU stack reads `.env.gpu` (real R2 credentials, used by `docker-compose.yml`) at the repo root. The client no longer runs on the VPS as a Docker container — it is bare-metal only and reads its config from `/etc/cctv-client/` on the appliance (`cameras.env` for RTSP creds, `platform.env` for `PLATFORM_URL`/`APPLIANCE_TOKEN` in platform mode). No R2 credentials live on the client anymore.
- **Local RTSP fake for testing recorder flows**: spin up `bluenviron/mediamtx` as an RTSP server, then push a sample MP4 with `ffmpeg -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://<mediamtx-host>:8554/stream`. Drive the recorder on the bare-metal appliance (there is no `cctv-client-agent` container anymore): run the appliance with `python -m client_agent.appliance` and reach the stream at `rtsp://<mediamtx-host>:8554/stream`. HTTP `POST /start` accepts a unified `duration_s` field with presets `{300, 900, 1800, 2700, 3600, 7200, 14400, 28800}` (5/15/30/45 min + 1/2/4/8 h). For sub-5-minute smoke tests bypass the route and call `Recorder.start(url=..., duration_s=30)` directly on the appliance host (e.g. `uv run python -c "..."`). The recorder is buffer-only now: it writes chunks to a local rolling buffer and leaves them on disk — no R2 upload.

## TODO (deferred)

- _(none currently)_ — the R2 upload/download retry item (issue #5 follow-up) is **done** in issue #61: the gpu-service `R2Client` retries every network method 3× with exponential backoff via `_with_retry` (SPEC §8.2), and the status-list walks share an ETag-keyed cache so an idle bucket costs zero `get_object` calls per poll. (The client-agent R2Client copy was removed in #29 — the client no longer touches R2 directly.)

## Architecture

```
client appliance (bare-metal, Flask :8080) → R2 (presigned URLs) → gpu-service (Docker+NVIDIA)
```

- **Pipeline** (`pipeline/`): frame extraction → YOLO-pose (person detection + displacement) → VLM or heuristic activity classification → HTML report
- **Client Agent** (`client-agent/`): Flask UI on :8080 for camera discovery, per-camera snapshots, the managed-cameras panel, `/test-connection`, `/stop`, and an RTSP recorder with ffmpeg stream-copy + segmented chunks (#8 ✅, e2e-validated on cctv-vps). The shared package `client_agent/` has a **single entrypoint**: `client_agent.appliance` (bare-metal via waitress; #23 ✅). `agent.py` is no longer an entrypoint — it survives only as the shared `build_app` Flask-app factory that the appliance imports. The recorder is buffer-only (local rolling buffer); in platform mode the appliance uploads via presigned URLs (PresignedUploader). The legacy on-site R2 routes (`/upload`, `/start`, `/jobs`, `/report`) now return 503 (#29).
- **Client Appliance** (`client-appliance/`): packaging-only target (#24 ✅) — systemd unit, idempotent `install.sh`, env templates, README. Zero Python; consumes the shared `client_agent` package. Operator runs `sudo ./client-appliance/install.sh` on a fresh Ubuntu/RPi mini-PC and gets a `systemctl enable --now`-ed Flask UI in LAN.
- **GPU Service** (`gpu-service/`): two run-modes share one Docker image:
  - `gpu_service.worker` (default ENTRYPOINT, :5000 dashboard): R2 polling worker for the SPEC §6 client-agent flow.
  - `gpu_service.rest_server` (override entrypoint, :5003 REST): URL-passthrough contract for the [gpu-exchange](https://github.com/KopalnieKrypto/gpu-exchange) `gpu-agent` (#25 ✅). Routes: `POST /analyze` (multi-chunk, presigned URLs, tenant-prefix defense-in-depth), `GET /healthz` (model+CUDA gate, 503 while warming), `GET /status/:id` (state only, no result payload).

## Stack

| Component | Technology |
|-----------|-----------|
| Pose model | YOLOv11n-pose ONNX, input `[1,3,640,640]`, output `[1,56,N]` |
| Pose inference | onnxruntime-gpu, CUDAExecutionProvider only |
| Activity classifier (VLM) | Qwen2.5-VL-3B-Instruct via transformers + PyTorch cu128 |
| Activity classifier (heuristic) | Geometric heuristics on COCO keypoints (legacy fallback) |
| Walking detection | Bbox displacement between frames (norm > 0.05 = walking) |
| Frame extraction | ffmpeg at 1 fps |
| Image processing | OpenCV + Pillow |
| Report | Jinja2 + vendored Chart.js (standalone HTML) |
| Client UI | Flask |
| R2 client | boto3 (S3-compat) |
| Docker base | nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 |
| Python deps | uv preferred, pip fallback |

## Must follow

- Python + onnxruntime-gpu for pose inference, PyTorch + transformers for VLM. No Node.js, no CPU fallback
- `ort.preload_dlls(cuda=True, cudnn=True)` must be called before `InferenceSession()` — onnxruntime-gpu wheel has no RPATH to site-packages/nvidia/. See `pipeline/pose_detector.py`.
- `nvidia-cublas-cu12` must be a direct dep — `onnxruntime-gpu[cuda,cudnn]` extras don't pull it but `libcublasLt.so.12` is required at runtime.
- Docker base: `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` — CUDA 12.8 required for Blackwell (sm_120) PyTorch support
- VLM classifier: `CLASSIFIER=vlm` env var or `--classifier vlm` CLI flag. Model loaded lazily on first frame with a person.
- Frame extraction: ffmpeg 1 fps, frame-by-frame processing, never full video in RAM
- YOLO-pose output `[1,56,N]` transposed: rows 0-3 bbox, row 4 conf, rows 5-55 keypoints (17×3)
- Preprocessing: PIL RGB → resize 640×640 → float32 /255 → CHW → batch dim
- Activity classes: sitting, standing, walking, running — no others
- Person tracking (issue #32) sits between pose detection and aggregation and is **on by default**: OSNet Re-ID cosine similarity is the association metric (never IoU — useless at 1 fps), and a track must be seen `MIN_TRACK_DETECTIONS` (3) times within `TRACK_WINDOW_FRAMES` (5) before it counts — real YOLO output flickers, so the advisory's strict-consecutive rule under-counted real people badly. `--no-tracker` reproduces pre-#32 numbers for baseline comparison. Requires `models/osnet_x0_25.onnx` from `setup-models.sh`.
- Tracking defaults favour splitting one person into two tracks over merging two people into one (`max_track_age_s` 120 s) — for person-minute reporting a merge silently corrupts the numbers, a split only shows an absence gap
- Reports: standalone HTML, zero external deps (vendored Chart.js, base64 images)
- R2 bucket: `surveillance-data`. Key: `surveillance-jobs/{job_id}/`
- Job coordination: `status.json` in R2, no database
- Confidence threshold: 0.25, NMS IoU: 0.45
- uv over pip for Python dependency management
- Client-agent single bare-metal target: the shared package `client_agent/` is served **only** by the appliance (`client-appliance/`, entrypoint `client_agent.appliance` via waitress). There is no Docker target for the client (removed in #29). `agent.py` remains only as the `build_app` Flask-app factory the appliance imports.

## Don't

- Don't add CPU inference fallback — breaks 1:1 processing guarantee
- Don't trust `ort.get_available_providers()` alone — also check `session.get_providers()` after init to catch silent CPU fallback (microsoft/onnxruntime#25145).
- Don't use `python:slim` Docker images for GPU workloads
- Don't add per-person *re-identification across videos or days* — within-video tracking is in (issue #32), cross-video identity is a separate product tier
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
├── pipeline/                  # Core AI pipeline (CLI: python -m pipeline.analyze)
│   ├── analyze.py             # full-video CLI entry point
│   ├── pose_detector.py       # ONNX session + CUDAExecutionProvider guard
│   ├── tracker.py             # PersonTracker — OSNet-similarity association → stable track_id (#32)
│   ├── reid.py                # OSNetEmbedder — bbox crop → L2-normalized appearance vector (#32)
│   ├── track_filter.py        # MinTrackLengthFilter — delay line, only proven tracks aggregate (#32)
│   ├── activity_classifier.py # heuristic classifier + ActivitySmoother (displacement)
│   ├── vlm_classifier.py      # Qwen2.5-VL-3B wrapper for VLM classification
│   ├── report_renderer.py     # Jinja2 → standalone HTML (vendored Chart.js)
│   ├── report_template.html
│   ├── frame_extractor.py / video_frames.py        # ffmpeg @ 1 fps streaming
│   ├── preprocessing.py / postprocessing.py        # PIL→CHW; transpose+NMS
│   ├── annotator.py / aggregator.py                # boxes+keypoints; person-minutes
│   └── vendor/                # vendored Chart.js for offline reports
├── gpu-service/               # R2 polling worker (#5) + investor dashboard (#6) + gpu-exchange REST contract (#25)
│   ├── Dockerfile             # nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 + PyTorch cu128, EXPOSE 5000 5003
│   └── gpu_service/
│       ├── worker.py, dashboard.py, r2_client.py     # R2 polling mode (default entrypoint)
│       ├── rest_server.py, rest_api.py               # gpu-agent REST mode (override entrypoint, :5003)
│       ├── task_runner.py, ffmpeg_concat.py          # download → concat → pipeline → upload
│       ├── http_client.py, http_retry.py             # urllib + 3-try exp-backoff (1s/2s)
│       └── tenant_url.py                             # defense-in-depth tenants/{tid}/results/{tid}/ check
├── client-agent/              # Shared client package (served by the bare-metal appliance only)
│   └── client_agent/          # web.py (Flask), recorder.py (ffmpeg, buffer-only), discovery.py (ONVIF/RTSP-scan), agent.py (shared build_app Flask-app factory, imported by appliance), appliance.py (bare-metal entrypoint via waitress, #23) (+tests)
├── client-appliance/          # Standalone-appliance packaging (#24) — zero Python, just packaging
│   ├── cctv-client.service    # systemd unit (Type=simple, EnvironmentFile cameras.env+platform.env, journald)
│   ├── install.sh             # idempotent root installer (creates cctv user + venv at /opt/cctv-client)
│   ├── cameras.env.example / platform.env.example  # operator-edited templates → /etc/cctv-client/ (perms 600)
│   ├── README.md              # install/update/troubleshooting + 5-min smoke runbook + ADR git-vs-tarball
│   └── tests/                 # unit_file/env_examples/install_script/readme contract tests
├── tests/                     # Repo-level meta tests (build_config_test.py)
├── test/                      # Legacy single-frame validation scripts (pre-#4)
├── docs/                      # Operator setup guides (SETUP_GPU.md, SETUP_CLIENT.md)
├── setup-models.sh            # curl + sha256 verify yolo11n-pose.onnx from GH release
├── models/                    # yolo11n-pose.onnx (gitignored, fetched by setup-models.sh)
├── scripts/                   # standalone test/benchmark scripts (test_vlm_classifier.py)
└── test-data/                 # sample MP4s (gitignored)
```

## Performance (RTX 5070, 1h video)

- **VLM hybrid** (recommended): ~20 min total (3:1 ratio), ~6GB VRAM. YOLO ~100ms + VLM ~270ms per frame.
- **Heuristic only**: ~7 min total (8:1 ratio), ~600MB VRAM. YOLO ~100ms/frame.
- VLM model (Qwen2.5-VL-3B) loaded lazily — first frame ~40s, subsequent ~0.27s/frame.
- Accuracy: VLM hybrid ~85% vs heuristic ~45% on ground truth test (sitting 89%, walking 96%).

## Platform integration (future, not now)

This is a standalone prototype. Integration into ML Compute Exchange (KopalnieKrypto/ml-compute-engine) planned for Phase 4: docker image reference in gpu-agent compose, `problem_type: 'surveillance_analysis'`, billing, tenant isolation.
