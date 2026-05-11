#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f env.sh ]]; then
  echo "env.sh fehlt. Bitte env.sh.example nach env.sh kopieren und anpassen."
  exit 1
fi

source env.sh

ssh "${TARGET_USER}@${TARGET_IP}" "mkdir -p ${TARGET_DIR}/models ${TARGET_DIR}/data/coco ${TARGET_DIR}/scripts ${TARGET_DIR}/results"

scp models/yolov8s_1024.onnx "${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}/models/"
scp -r data/coco/val2017 "${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}/data/coco/"
scp data/coco/annotations/instances_val2017.json "${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}/data/coco/"
scp requirements-target.txt "${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}/"
scp scripts/07_target_setup.sh "${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}/scripts/"
scp scripts/08_run_inference_imx95.py "${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}/scripts/"
scp scripts/09_eval_coco.py "${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}/scripts/"
scp scripts/10_collect_thermal.sh "${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}/scripts/"
scp scripts/11_generate_report.sh "${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}/scripts/"

echo "Transfer completed"
