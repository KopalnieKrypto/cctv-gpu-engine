#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -f "yolo11n-pose.onnx" ]; then
    echo "yolo11n-pose.onnx already exists"
    exit 0
fi

echo "Exporting yolo11n-pose to ONNX..."
uv run --with ultralytics -- yolo export model=yolo11n-pose.pt format=onnx imgsz=640
mv yolo11n-pose.onnx ./yolo11n-pose.onnx 2>/dev/null || true
rm -f yolo11n-pose.pt
echo "Done: yolo11n-pose.onnx"
