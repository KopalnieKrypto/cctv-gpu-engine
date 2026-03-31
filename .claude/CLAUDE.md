# CCTV GPU Engine

Batch surveillance video analysis: MP4 → YOLO-pose → activity classification → standalone HTML report. Full spec: `SPEC.md`.

## Quick reference

- CLI: `python pipeline/analyze.py input.mp4 --output report.html`
- Model: `models/yolo11n-pose.onnx` (download via `setup-models.sh`)
- Test: `cd test && ./run_test.sh`
- GPU service: `docker compose up` (polls R2 for pending jobs)
- Client agent: `docker compose -f docker-compose.client.yml up` (Flask UI :8080)

## Architecture

```
client-agent (Flask :8080) → R2 bucket (surveillance-data) → gpu-service (Docker+NVIDIA)
```

- **Pipeline** (`pipeline/`): frame extraction → YOLO-pose → activity heuristics → HTML report
- **Client Agent** (`client-agent/`): Flask UI, ffmpeg RTSP recording, boto3 upload to R2
- **GPU Service** (`gpu-service/`): R2 polling worker, downloads video, runs pipeline, uploads report

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
├── pipeline/                  # Core AI pipeline
│   ├── analyze.py             # CLI entry point
│   ├── pose_detector.py
│   ├── activity_classifier.py
│   └── report_generator.py
├── gpu-service/               # R2 polling worker
├── client-agent/              # Flask UI + ffmpeg recorder
├── test/                      # Validation scripts
├── models/                    # yolo11n-pose.onnx (gitignored)
└── test-data/                 # sample MP4s (gitignored)
```

## Performance (RTX 5070, 1h video)

- ~7 min total (8:1 ratio), ~600MB VRAM
- YOLO: ~100ms/frame, 3600 frames/hour at 1fps

## Platform integration (future, not now)

This is a standalone prototype. Integration into ML Compute Exchange (KopalnieKrypto/ml-compute-engine) planned for Phase 4: docker image reference in gpu-agent compose, `problem_type: 'surveillance_analysis'`, billing, tenant isolation.
