#!/usr/bin/env python3
import json
import os
import time
import gc
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO

ROOT = Path(os.getenv("IMX95_BENCH_ROOT", "/opt/imx95-yolov8-benchmark"))
IMAGES = ROOT / "data/coco/val2017"
ANN = ROOT / "data/coco/instances_val2017.json"
if not ANN.exists():
    ANN = ROOT / "data/coco/annotations/instances_val2017.json"
PT_MODEL = Path(os.getenv("PT_MODEL", str(ROOT / "yolov8s.pt")))
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

RESULT_TAG = os.getenv("RESULT_TAG", "host_pytorch_cuda")
DEVICE = os.getenv("PT_DEVICE", "0")
IMG_SIZE = int(os.getenv("IMG_SIZE", "1024"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
IOU_THRES = float(os.getenv("IOU_THRES", "0.7"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))
WARMUP = int(os.getenv("WARMUP", "10"))
PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY", "100"))
CLEAR_CUDA_CACHE_EVERY = int(os.getenv("CLEAR_CUDA_CACHE_EVERY", "100"))

COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def xyxy_to_coco_bbox(xyxy):
    x1, y1, x2, y2 = xyxy
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


class JsonArrayWriter:
    def __init__(self, path):
        self.path = path
        self._fh = None
        self._first = True

    def __enter__(self):
        self._fh = open(self.path, "w", encoding="utf-8")
        self._fh.write("[")
        return self

    def write_item(self, item):
        if not self._first:
            self._fh.write(",\n")
        else:
            self._fh.write("\n")
            self._first = False
        self._fh.write(json.dumps(item, ensure_ascii=False))

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            if not self._first:
                self._fh.write("\n")
            self._fh.write("]\n")
            self._fh.close()


def run_predict(model, img_path):
    result = model.predict(
        source=str(img_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        verbose=False,
    )[0]
    return result


def main():
    if not PT_MODEL.exists():
        raise FileNotFoundError(f"Model not found: {PT_MODEL}")

    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("PT_DEVICE ist nicht cpu, aber CUDA ist nicht verfuegbar.")

    coco = COCO(str(ANN))
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)
    if MAX_IMAGES > 0:
        imgs = imgs[:MAX_IMAGES]

    model = YOLO(str(PT_MODEL))

    if not imgs:
        raise RuntimeError("Keine Bilder gefunden.")

    warmup_img = IMAGES / imgs[0]["file_name"]
    for _ in range(WARMUP):
        _ = run_predict(model, warmup_img)

    timing_rows = []
    det_name = f"coco_detections_{RESULT_TAG}.json"
    det_path = RESULTS / det_name

    with JsonArrayWriter(det_path) as det_writer:
        for idx, meta in enumerate(imgs, start=1):
            img_path = IMAGES / meta["file_name"]
            if not img_path.exists():
                continue

            t0 = time.perf_counter()
            result = run_predict(model, img_path)
            t1 = time.perf_counter()

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                cls_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
                scores = boxes.conf.detach().cpu().numpy()

                for i in range(len(xyxy)):
                    cid = int(cls_ids[i])
                    coco_cat = COCO80_TO_91[cid] if 0 <= cid < len(COCO80_TO_91) else cid + 1
                    det_writer.write_item(
                        {
                            "image_id": meta["id"],
                            "category_id": int(coco_cat),
                            "bbox": xyxy_to_coco_bbox([float(v) for v in xyxy[i]]),
                            "score": float(scores[i]),
                        }
                    )

            timing_rows.append(
                {
                    "image_id": meta["id"],
                    "total_ms": (t1 - t0) * 1000.0,
                    "nms_included_in_postprocess": True,
                    "memory_transfer_included": False,
                    "pipeline": "ultralytics_predict",
                }
            )

            if PROGRESS_EVERY > 0 and idx % PROGRESS_EVERY == 0:
                print(f"progress: {idx}/{len(imgs)} images")

            if (
                CLEAR_CUDA_CACHE_EVERY > 0
                and idx % CLEAR_CUDA_CACHE_EVERY == 0
                and torch.cuda.is_available()
            ):
                gc.collect()
                torch.cuda.empty_cache()

    timing_csv_name = f"timing_breakdown_{RESULT_TAG}.csv"
    timing_json_name = f"metrics_{RESULT_TAG}_timing_only.json"

    df = pd.DataFrame(timing_rows)
    df.to_csv(RESULTS / timing_csv_name, index=False)

    avg_total_ms = float(df["total_ms"].mean()) if len(df) else 0.0
    fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

    cuda_name = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)

    summary = {
        "dataset": "COCO val2017",
        "imgsz": IMG_SIZE,
        "batch": 1,
        "images_evaluated": int(len(df)),
        "avg_latency_ms": avg_total_ms,
        "fps_from_avg_latency": fps,
        "timing_definition": {
            "includes_preprocess": True,
            "includes_inference": True,
            "includes_postprocess": True,
            "includes_nms": True,
            "includes_memory_transfers": False,
            "notes": "Ultralytics predict pipeline; timings are end-to-end per image.",
        },
        "runtime": {
            "framework": "pytorch_ultralytics",
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device": cuda_name,
            "requested_device": DEVICE,
        },
        "result_tag": RESULT_TAG,
        "artifacts": {
            "detections": det_name,
            "timing_csv": timing_csv_name,
            "timing_json": timing_json_name,
        },
    }

    with open(RESULTS / timing_json_name, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
