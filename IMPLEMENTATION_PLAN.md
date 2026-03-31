# Surveillance Video Activity Analysis — Standalone Prototype

## Context

Zleceniodawca chce zwalidować nowy use case: analiza wideo z monitoringu klientów na GPU infrastrukturze inwestorów. Klient udostępnia obraz z kamery → system generuje raport aktywności osób (siedzi/stoi/chodzi/biega z person-minutes). Budujemy standalone prototyp, docelowo nowy `problem_type` w platformie.

**Kluczowe decyzje z interview:**
- Custom pipeline (nie Frigate) — YOLO-pose + heurystyki
- Client-agent wzorem gpu-agent (outbound HTTPS, R2 jako mediator)
- MVP = analiza historyczna (batch), nie live monitoring
- Agregat per-frame (bez per-person trackingu)
- 1 fps sampling, ~1:1 processing ratio
- Python, standalone HTML raport z Chart.js
- RODO ignorowane na prototyp

## Architecture

```
┌─ Client LAN ──────────────┐        ┌─ R2 Bucket ─────────────┐        ┌─ GPU Server ─────────────┐
│                            │        │ surveillance-jobs/       │        │                          │
│  client-agent (Docker)     │        │   {job_id}/              │        │  surveillance-serve      │
│  Flask UI :8080            │───────>│     status.json          │<───────│  (Docker, NVIDIA)        │
│  ffmpeg RTSP → MP4         │ upload │     input/chunk_001.mp4  │  poll  │                          │
│  boto3 → R2                │        │     output/report.html   │        │  worker.py (poll loop)   │
│                            │<───────│                          │───────>│  analyze.py pipeline     │
│  Shows report in browser   │  poll  │                          │ upload │  YOLO-pose + heuristics  │
└────────────────────────────┘        └──────────────────────────┘        └──────────────────────────┘
```

R2 key convention = job coordination (no database/API needed for prototype).

## File Structure

```
infra/surveillance-prototype/
├── docker-compose.yml              # GPU service (surveillance-serve)
├── docker-compose.client.yml       # Client agent (:8080)
├── .env.example
├── setup-models.sh                 # Download yolo11n-pose.onnx
├── benchmark.sh
├── pipeline/                       # Phase 1: Core AI pipeline
│   ├── Dockerfile                  # nvidia/cuda base + python + ffmpeg
│   ├── requirements.txt
│   ├── analyze.py                  # CLI: python analyze.py input.mp4
│   ├── frame_extractor.py          # ffmpeg 1fps → JPEGs
│   ├── pose_detector.py            # YOLO-pose ONNX inference
│   ├── activity_classifier.py      # Keypoint heuristics (sit/stand/walk/run)
│   ├── report_generator.py         # Jinja2 → standalone HTML
│   ├── report_template.html        # Chart.js inline, base64 keyframes
│   ├── chartjs.min.js              # Vendored Chart.js (~200KB)
│   └── test_heuristics.py          # Unit tests for thresholds
├── gpu-service/                    # Phase 3: R2 polling worker
│   ├── Dockerfile                  # Extends pipeline image
│   ├── requirements.txt
│   ├── worker.py                   # Poll R2 → process → upload report
│   └── r2_client.py                # S3-compatible R2 operations
├── client-agent/                   # Phase 2: Client daemon
│   ├── Dockerfile                  # python:3.11-slim + ffmpeg
│   ├── requirements.txt
│   ├── agent.py                    # Flask app
│   ├── recorder.py                 # ffmpeg RTSP → MP4
│   ├── uploader.py                 # boto3 → R2
│   └── templates/index.html        # Plain HTML + vanilla JS
├── models/                         # .gitkeep (yolo11n-pose.onnx)
└── test-data/                      # .gitkeep (sample MP4s)
```

## Phase 1: Core AI Pipeline (prove AI works)

**Goal:** `python analyze.py sample.mp4 --output report.html`

### 1.1 Model setup
- `setup-models.sh`: export `yolo11n-pose.pt` → ONNX via ultralytics CLI
- Output shape: `[1, 56, num_boxes]` — 4 bbox + 1 conf + 51 keypoints (17×3: x,y,vis)

### 1.2 Frame extraction (`frame_extractor.py`)
- `ffmpeg -i input.mp4 -vf fps=1 -q:v 2 output/frame_%06d.jpg`
- 1fps = 3600 frames/hour, ~50-100KB/frame at 1080p

### 1.3 Pose detection (`pose_detector.py`)
- Port existing YOLO preprocessing from `infra/ai-test/yolo-serve/src/index.ts:58-81` (resize 640×640, HWC→CHW, normalize 0-1)
- Port NMS from `yolo-serve/src/index.ts:114-166`, extend for 51 keypoint values
- ONNX Runtime with CUDA EP, CPU fallback (same pattern as yolo-serve)

**COCO 17 keypoints used by heuristics:**
```
5,6: shoulders | 11,12: hips | 13,14: knees | 15,16: ankles
```

### 1.4 Activity classification (`activity_classifier.py`)

Rule-based decision tree per detected person:

```
1. Hips+knees+ankles visible (conf>0.5)?
   NO → bbox aspect ratio fallback (h/w < 1.5 → SITTING, else STANDING)

2. knee_angle = avg angle(hip,knee,ankle) both legs
3. hip_height_ratio = (bbox_bottom - hip_y) / bbox_height

4. knee_angle < 120° AND hip_height_ratio < 0.40 → SITTING

5. stride_ratio = |ankle_L_x - ankle_R_x| / |hip_L_x - hip_R_x|
6. torso_lean = angle from vertical (shoulders→hips)

7. stride_ratio > 2.0 AND torso_lean > 15° → RUNNING
8. stride_ratio > 1.3 → WALKING
9. else → STANDING
```

Thresholds as constants at top of file for easy tuning.

### 1.5 Aggregation (no tracking)

Per frame: `{ timestamp_s, person_count, activities: {Activity → count} }`
Total: sum person-frames per activity / 60 = **person-minutes**

### 1.6 Report generation (`report_generator.py`)

Standalone HTML (Jinja2 template + Chart.js vendored inline):
1. **Summary table** — duration, frames, peak/avg person count, dominant activity
2. **Pie chart** — % person-time per activity (green=standing, blue=sitting, orange=walking, red=running)
3. **Timeline stacked bar** — 1-minute bins, X=time, Y=person count, colored by activity
4. **5 annotated keyframes** — selected by max person count, min 2min spacing, bboxes + skeleton + activity labels, base64 PNG

### 1.7 Verification
- Run on test MP4, open report in browser
- Check: charts render, person count matches visual, activities plausible
- `test_heuristics.py`: synthetic keypoint arrays → verify sit/stand/walk/run thresholds
- Benchmark: confirm ~1:1 ratio (1h video ≈ 1h processing)

## Phase 2: Client Agent (prove delivery works)

**Goal:** Client opens `localhost:8080`, enters RTSP URL + duration, gets report back.

### 2.1 Flask web UI (`agent.py` + `templates/index.html`)
- `GET /` — form: RTSP URL, duration (or start/end time), "Start Recording" button
- `GET /jobs` — job list with status badges, auto-refresh 10s
- `GET /jobs/<id>/report` — serve downloaded HTML report
- "Test Connection" button (ffmpeg probe of RTSP URL)

### 2.2 Recording (`recorder.py`)
- `ffmpeg -rtsp_transport tcp -i {url} -c copy -t {duration_s} output.mp4`
- Stream copy (no re-encoding) → minimal CPU on client PC
- Long recordings: `-f segment -segment_time 3600` → 1h chunks

### 2.3 Upload (`uploader.py`)
- boto3 S3-compatible client → R2 (same credential pattern as DD-02)
- Direct R2 credentials (scoped API token), not presigned URLs
- Upload chunks + write `status.json` with `status: "pending"`

### 2.4 R2 coordination (no database)

```
surveillance-jobs/{job_id}/
  status.json              # { status: pending|processing|completed|failed, input_chunks, progress_pct, error }
  input/chunk_001.mp4
  output/report.html
```

Client-agent polls `status.json` every 15s → downloads report when completed.

### 2.5 Docker + install

```yaml
# docker-compose.client.yml
services:
  surveillance-agent:
    build: ./client-agent
    ports: ["8080:8080"]
    volumes: ["./recordings:/recordings"]
    env_file: .env
    restart: unless-stopped
```

Base: `python:3.11-slim` + ffmpeg (no GPU). Client install: Docker + clone + `.env` + `docker compose up`.

### 2.6 Verification
- Build, open UI, test RTSP connection
- Submit recording → verify MP4 in R2
- Manually set status=completed + dummy report → verify agent shows it

## Phase 3: GPU Processing Service (prove E2E)

**Goal:** Automated loop — poll R2, process, upload report.

### 3.1 Worker (`worker.py`)
```python
while True:
    job = poll_for_pending_job()  # S3 ListObjects → find status=pending
    if job:
        claim_job(job)            # Write status=processing + worker_id
        download_video(job)       # R2 → local temp
        run_pipeline(job)         # Import analyze.py as library
        upload_report(job)        # Local HTML → R2
        update_status(job, "completed")
    else:
        sleep(10)
```

Atomic claim MVP: last-writer-wins (acceptable at low concurrency). Post-MVP → platform's `SELECT FOR UPDATE SKIP LOCKED`.

### 3.2 Docker (GPU)

```yaml
# docker-compose.yml
services:
  surveillance-serve:
    build: ./pipeline
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${SURVEILLANCE_GPU:-0}"]
              capabilities: [gpu]
    volumes:
      - ${MODELS_DIR:-./models}:/models:ro
    env_file: .env
    command: ["python3", "gpu-service/worker.py"]
```

Pattern: identical to `infra/ai-test/docker-compose.yml`.

### 3.3 Verification (full E2E)
1. Start GPU service on test server
2. Submit recording from client-agent UI
3. Watch logs: job detected → processing → completed
4. Report appears in client-agent UI
5. Verify report quality matches Phase 1 local test

## Phase 4: Integration Path (future, not prototype)

When validated, integrate as new `problem_type: 'surveillance_analysis'`:
- New entry in `task_routing_rules` (model_id, vram 4096MB, est. duration 3600s)
- New entry in `models` table (yolo11n-pose, docker_image: surveillance-serve)
- Docker service added to gpu-agent docker-compose (port 5003)
- Client dashboard: new task form (upload MP4 or connect client-agent)
- R2 keys migrate to `tenants/{tenantId}/...` prefix isolation
- Client-agent becomes optional (direct upload via dashboard also works)
- Job claim via platform's atomic `SELECT FOR UPDATE SKIP LOCKED`
- Billing: GPU-seconds at surveillance rate (new `billing_rates` entry)

## Critical Files to Reference

| File | What to reuse |
|------|--------------|
| `infra/ai-test/docker-compose.yml` | NVIDIA runtime, device reservation, healthcheck pattern |
| `infra/ai-test/yolo-serve/src/index.ts:58-81` | YOLO preprocessing (resize, HWC→CHW, normalize) — port to Python |
| `infra/ai-test/yolo-serve/src/index.ts:114-166` | NMS postprocessing — port + extend for keypoints |
| `infra/ai-test/surrogate-serve/Dockerfile` | Python Docker service pattern |
| `docs/design/02-data-pipeline.md:143-162` | R2 S3-compatible client config |
| `infra/ai-test/setup-models.sh` | Model download/export script pattern |

## Key Libraries

| Library | Role | Component |
|---------|------|-----------|
| `ultralytics` | YOLO-pose model export | setup-models.sh |
| `onnxruntime-gpu` | ONNX inference CUDA EP | pipeline |
| `opencv-python-headless` | Frame annotation (bboxes, skeletons) | pipeline |
| `numpy` | Keypoint math | pipeline |
| `jinja2` | HTML report template | pipeline |
| `Pillow` | Image → base64 for report | pipeline |
| `flask` | Client web UI | client-agent |
| `boto3` | S3-compatible R2 access | client-agent, gpu-service |
| `ffmpeg` (system) | Frame extraction + RTSP recording | all |

## Resolved Questions

1. **yolo11n-pose.onnx na RTX 5070 (Blackwell)** — wymaga weryfikacji na test serwerze jako pierwszy krok Phase 1. Fallback: CPU EP (~10x wolniejsze). Jeśli CPU EP nieakceptowalny → rozważyć TensorRT lub PyTorch zamiast ONNX
2. **R2 bucket** — osobny `surveillance-data` (czystszy, izolacja od głównej platformy)
3. **Upload** — boto3 multipart upload (auto via `upload_file()`, R2 limit 5GB single / 5TB multipart). Trzeba zweryfikować bandwidth klienta w praktyce
4. **Client-agent recording** — MVP: "nagraj od teraz przez Xh". Scheduled recording (od-do) deferred
5. **Heurystyka accuracy** — target 80%. Przetestować na realnym nagraniu, tuningować thresholds. Jeśli <80% → rozważyć lekki model klasyfikacji aktywności zamiast heurystyk
