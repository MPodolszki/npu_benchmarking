#!/usr/bin/env bash
set -euo pipefail

sudo pacman -Syu --needed \
  base-devel \
  git \
  wget \
  unzip \
  curl \
  rsync \
  python \
  python-pip \
  python-virtualenv

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements-host.txt

yolo checks
python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8s.pt')
print('Ultralytics import + yolov8s download: OK')
PY
