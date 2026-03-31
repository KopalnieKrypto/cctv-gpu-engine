#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# prereqs
for cmd in uv ffmpeg python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found"
        [ "$cmd" = "ffmpeg" ] && echo "  Install: sudo apt install -y ffmpeg"
        [ "$cmd" = "uv" ] && echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
done

MODEL_PATH="./yolo11n-pose.onnx"

# export pose model if needed
if [ ! -f "$MODEL_PATH" ]; then
    echo "=== Exporting yolo11n-pose to ONNX ==="
    uv run --with ultralytics -- python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n-pose.pt')
model.export(format='onnx', imgsz=640, simplify=True)
"
    mv yolo11n-pose.onnx "$MODEL_PATH" 2>/dev/null || true
    rm -f yolo11n-pose.pt
    echo "=== Model exported ==="
fi

echo "=== Running video pose test ==="
uv run --with onnxruntime-gpu --with numpy --with Pillow --with opencv-python-headless \
    -- python3 test_video_pose.py --model "$MODEL_PATH"

echo "=== Done. Check output/ for annotated keyframes ==="
