#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python - <<'PY'
import onnx

m = onnx.load('models/yolov8s_1024.onnx')
onnx.checker.check_model(m)
print('ONNX model is valid')
print('IR version:', m.ir_version)
print('Number of graph nodes:', len(m.graph.node))
PY
