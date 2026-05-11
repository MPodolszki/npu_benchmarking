#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
mkdir -p models logs

python - <<'PY'
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
out = model.export(
    format='onnx',
    imgsz=(1024, 1024),
    batch=1,
    simplify=True,
    opset=13,
    nms=False,
)
print(f'Export output: {out}')
PY

find . -maxdepth 3 -name '*.onnx' -print | tee logs/04_export_onnx_files.log

if [[ -f yolov8s.onnx ]]; then
  cp yolov8s.onnx models/yolov8s_1024.onnx
fi

if [[ ! -f models/yolov8s_1024.onnx ]]; then
  found_onnx=$(find . -maxdepth 3 -name '*.onnx' | head -n 1)
  if [[ -n "$found_onnx" ]]; then
    cp "$found_onnx" models/yolov8s_1024.onnx
  fi
fi

test -f models/yolov8s_1024.onnx

echo "ONNX export ready: models/yolov8s_1024.onnx"
