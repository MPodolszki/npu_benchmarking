#!/usr/bin/env bash
set -euo pipefail

# Export YOLOv8 model to TensorFlow Lite for Neutron conversion.
# Default export is INT8 because Neutron mapping on i.MX95 generally requires quantized ops.
# Set EXPORT_INT8=0 to force float32 export.

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

source .venv/bin/activate
mkdir -p models logs

MODEL_PT=${MODEL_PT:-"yolov8s.pt"}
IMG_SIZE=${IMG_SIZE:-1024}
EXPORT_INT8=${EXPORT_INT8:-1}
SAFE_MODE=${SAFE_MODE:-0}

if [[ "$SAFE_MODE" == "1" ]]; then
  # Keep host memory usage stable during export on desktop systems.
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
  export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
  export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
  export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-1}
  export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}
  export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
fi

python - <<'PY'
import os
from pathlib import Path

import yaml
from ultralytics import YOLO

model_pt = os.getenv("MODEL_PT", "yolov8s.pt")
img_size = int(os.getenv("IMG_SIZE", "1024"))
export_int8 = os.getenv("EXPORT_INT8", "0") == "1"

kwargs = {
    "format": "tflite",
    "imgsz": (img_size, img_size),
    "batch": 1,
    "nms": False,
}

if export_int8:
    kwargs["int8"] = True
    src_yaml = Path("data/coco/coco_val2017.yaml")
    calib_yaml = Path("data/coco/coco_int8_calib.yaml")
    data_cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    if "train" not in data_cfg:
        data_cfg["train"] = data_cfg.get("val", "val2017")
    if "val" not in data_cfg:
        data_cfg["val"] = data_cfg.get("train", "val2017")
    calib_yaml.write_text(yaml.safe_dump(data_cfg, sort_keys=False), encoding="utf-8")
    kwargs["data"] = str(calib_yaml)

model = YOLO(model_pt)
out = model.export(**kwargs)
print(f"Export output: {out}")
PY

find . -maxdepth 5 -name '*.tflite' -print | tee logs/04b_export_tflite_files.log

if [[ "$EXPORT_INT8" == "1" ]]; then
  target_name="models/yolov8s_1024_int8.tflite"
else
  target_name="models/yolov8s_1024_float32.tflite"
fi

# Avoid reusing stale files from previous runs.
rm -f "$target_name"

if [[ "$EXPORT_INT8" == "1" ]]; then
  found_tflite=$(find . -maxdepth 5 -name '*.tflite' \
    | grep -Ev '/models/neutron/' \
    | grep -Ev '^\./models/yolov8s_1024_int8\.tflite$' \
    | grep -E 'int8\.tflite$' \
    | head -n 1 || true)
else
  found_tflite=$(find . -maxdepth 5 -name '*.tflite' \
    | grep -Ev '/models/neutron/' \
    | grep -E 'float32\.tflite$' \
    | head -n 1 || true)
fi

if [[ -z "$found_tflite" && "$EXPORT_INT8" != "1" ]]; then
  found_tflite=$(find . -maxdepth 5 -name '*.tflite' \
    | grep -Ev '/models/neutron/' \
    | head -n 1 || true)
fi

if [[ -z "$found_tflite" ]]; then
  if [[ "$EXPORT_INT8" == "1" ]]; then
    echo "Kein INT8-TFLite-Export gefunden."
    echo "Bitte Export-Log pruefen: logs/04b_export_tflite_files.log"
    echo "Hinweis: Wenn nur float32/float16 erzeugt werden, hat der Export kein echtes INT8 geliefert."
  else
    echo "Kein TFLite-Export gefunden."
  fi
  echo "Falls TensorFlow fehlt:"
  echo "  source .venv/bin/activate && pip install tensorflow"
  exit 1
fi

if [[ "$(readlink -f "$found_tflite")" != "$(readlink -f "$target_name")" ]]; then
  cp "$found_tflite" "$target_name"
fi

if [[ "$EXPORT_INT8" == "1" ]]; then
  python - <<'PY'
import sys
import tensorflow as tf

model = "models/yolov8s_1024_int8.tflite"
interp = tf.lite.Interpreter(model_path=model)
interp.allocate_tensors()
in_dtypes = {d["dtype"].__name__ for d in interp.get_input_details()}
out_dtypes = {d["dtype"].__name__ for d in interp.get_output_details()}
ok = in_dtypes.issubset({"int8", "uint8"}) and out_dtypes.issubset({"int8", "uint8"})
if not ok:
    print(f"Kein echtes INT8-Modell: input dtypes={sorted(in_dtypes)}, output dtypes={sorted(out_dtypes)}")
    sys.exit(2)
print(f"INT8-Modell bestaetigt: input dtypes={sorted(in_dtypes)}, output dtypes={sorted(out_dtypes)}")
PY
fi

echo "TFLite export ready: $target_name"
