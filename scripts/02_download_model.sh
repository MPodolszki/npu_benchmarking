#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
mkdir -p models
python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8s.pt')
print('Downloaded yolov8s.pt')
PY

find "$HOME/.cache" -name 'yolov8s.pt' -print | head -n 5
