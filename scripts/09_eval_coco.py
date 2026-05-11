#!/usr/bin/env python3
import json
import os
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(os.getenv("IMX95_BENCH_ROOT", "/opt/imx95-yolov8-benchmark"))
RESULT_TAG = os.getenv("RESULT_TAG", "imx95")

ann_file = ROOT / "data/coco/instances_val2017.json"
if not ann_file.exists():
    ann_file = ROOT / "data/coco/annotations/instances_val2017.json"
det_file = ROOT / f"results/coco_detections_{RESULT_TAG}.json"
out_file = ROOT / f"results/metrics_{RESULT_TAG}.json"

coco_gt = COCO(str(ann_file))
coco_dt = coco_gt.loadRes(str(det_file))

coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

stats = {
    "mAP50_95": float(coco_eval.stats[0]),
    "mAP50": float(coco_eval.stats[1]),
    "mAP75": float(coco_eval.stats[2]),
    "AP_small": float(coco_eval.stats[3]),
    "AP_medium": float(coco_eval.stats[4]),
    "AP_large": float(coco_eval.stats[5]),
}

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print(json.dumps(stats, indent=2))
