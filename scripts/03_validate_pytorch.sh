#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
mkdir -p results logs

yolo detect val \
  model=yolov8s.pt \
  data=data/coco/coco_val2017.yaml \
  imgsz=1024 \
  batch=1 \
  device=cpu \
  project=results \
  name=val_pytorch_1024 | tee logs/03_validate_pytorch.log
