#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements-target.txt

python - <<'PY'
import onnxruntime as ort
print('onnxruntime available:', ort.__version__)
print('providers:', ort.get_available_providers())
PY
