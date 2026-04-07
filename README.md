# CCTV GPU Engine

Batch surveillance video analysis powered by idle GPU infrastructure. Upload MP4 footage, get a standalone HTML activity report — no manual review needed.

**What it does:** Detects people in surveillance footage using YOLO-pose, classifies their activity (sitting, standing, walking, running), and generates a self-contained HTML report with charts, timeline, and annotated keyframes.

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

## Quick Start

### GPU Server (investor)

```bash
# Download the model
./setup-models.sh

# Configure R2 credentials
cp .env.gpu.example .env

# Run
docker compose up
```

### Client

```bash
# Configure R2 credentials
cp .env.client.example .env

# Run
docker compose -f docker-compose.client.yml up -d

# Open browser
open http://localhost:8080
```

### Local Pipeline (development)

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
