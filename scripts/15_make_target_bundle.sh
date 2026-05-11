#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${MODE:-smoke}
SMOKE_IMAGES=${SMOKE_IMAGES:-100}
DEST=${DEST:-"${ROOT}/dist/imx95_bundle_${MODE}"}

if [[ ! -f "${ROOT}/models/yolov8s_1024.onnx" ]]; then
  echo "Fehler: ${ROOT}/models/yolov8s_1024.onnx fehlt (erst ./scripts/04_export_onnx.sh ausfuehren)." >&2
  exit 1
fi

ANN_SRC="${ROOT}/data/coco/annotations/instances_val2017.json"
if [[ ! -f "$ANN_SRC" ]]; then
  ANN_SRC="${ROOT}/data/coco/instances_val2017.json"
fi
if [[ ! -f "$ANN_SRC" ]]; then
  echo "Fehler: COCO Annotationen fehlen." >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"/scripts "$DEST"/models "$DEST"/data/coco/val2017 "$DEST"/results

cp "${ROOT}/requirements-target.txt" "$DEST/"
cp "${ROOT}/models/yolov8s_1024.onnx" "$DEST/models/"
cp "${ROOT}/scripts/07_target_setup.sh" "$DEST/scripts/"
cp "${ROOT}/scripts/08_run_inference_imx95.py" "$DEST/scripts/"
cp "${ROOT}/scripts/09_eval_coco.py" "$DEST/scripts/"
cp "${ROOT}/scripts/10_collect_thermal.sh" "$DEST/scripts/"
cp "${ROOT}/scripts/11_generate_report.sh" "$DEST/scripts/"

if [[ "$MODE" == "full" ]]; then
  mkdir -p "$DEST/data/coco/annotations"
  cp "$ANN_SRC" "$DEST/data/coco/annotations/instances_val2017.json"
  cp -r "${ROOT}/data/coco/val2017/." "$DEST/data/coco/val2017/"
else
  python3 - <<PY
import json
from pathlib import Path

root = Path("${ROOT}")
dest = Path("${DEST}")
num_images = int("${SMOKE_IMAGES}")

ann_src = root / "data/coco/annotations/instances_val2017.json"
if not ann_src.exists():
    ann_src = root / "data/coco/instances_val2017.json"

with ann_src.open("r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco.get("images", [])[:num_images]
image_ids = {img["id"] for img in images}
image_names = {img["file_name"] for img in images}
annotations = [a for a in coco.get("annotations", []) if a.get("image_id") in image_ids]

for name in image_names:
    src = root / "data/coco/val2017" / name
    dst = dest / "data/coco/val2017" / name
    if src.exists():
        dst.write_bytes(src.read_bytes())

filtered = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "images": images,
    "annotations": annotations,
    "categories": coco.get("categories", []),
}

ann_out = dest / "data/coco/annotations"
ann_out.mkdir(parents=True, exist_ok=True)
with (ann_out / "instances_val2017.json").open("w", encoding="utf-8") as f:
    json.dump(filtered, f)

print(f"Smoke bundle created with {len(images)} images and {len(annotations)} annotations")
PY
fi

chmod +x "$DEST"/scripts/*.sh

DU=$(du -sh "$DEST" | awk '{print $1}')
echo "Bundle bereit: $DEST (${DU})"
