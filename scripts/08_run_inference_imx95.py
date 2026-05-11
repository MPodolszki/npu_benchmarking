#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from pycocotools.coco import COCO

ROOT = Path(os.getenv("IMX95_BENCH_ROOT", "/opt/imx95-yolov8-benchmark"))
IMAGES = ROOT / "data/coco/val2017"
ANN = ROOT / "data/coco/instances_val2017.json"
if not ANN.exists():
    ANN = ROOT / "data/coco/annotations/instances_val2017.json"
MODEL = ROOT / "models/yolov8s_1024.onnx"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = int(os.getenv("IMG_SIZE", "1024"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
IOU_THRES = float(os.getenv("IOU_THRES", "0.7"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))
WARMUP = int(os.getenv("WARMUP", "10"))
RESULT_TAG = os.getenv("RESULT_TAG", "imx95")
ORT_PROVIDERS_ENV = os.getenv("ORT_PROVIDERS", "").strip()
ORT_FAIL_IF_MISSING = os.getenv("ORT_FAIL_IF_MISSING", "0") == "1"
ORT_DISABLE_CPU_FALLBACK = os.getenv("ORT_DISABLE_CPU_FALLBACK", "0") == "1"
PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY", "100"))

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


def letterbox(im, new_shape=(1024, 1024), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess(img_path):
    im = cv2.imread(str(img_path))
    if im is None:
        raise RuntimeError(f"Failed to read {img_path}")
    im0 = im.copy()
    im, ratio, dwdh = letterbox(im, (IMG_SIZE, IMG_SIZE))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))[None]
    return im, im0, ratio, dwdh


def nms_numpy(boxes, scores, iou_thres):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def decode_predictions(pred):
    # expected YOLOv8 ONNX shape typically [1, 84, N]
    if pred.ndim != 3:
        raise RuntimeError(f"Unexpected output shape: {pred.shape}")
    if pred.shape[1] < pred.shape[2]:
        pred = np.squeeze(pred).T
    else:
        pred = np.squeeze(pred)
    if pred.shape[1] < 6:
        raise RuntimeError(f"Unexpected decoded shape: {pred.shape}")
    return pred


def postprocess(pred, im0_shape, ratio, dwdh):
    pred = decode_predictions(pred)
    boxes = pred[:, :4]
    cls_scores_all = pred[:, 4:]

    cls_ids = np.argmax(cls_scores_all, axis=1)
    cls_scores = cls_scores_all[np.arange(cls_scores_all.shape[0]), cls_ids]
    mask = cls_scores > CONF_THRES

    boxes = boxes[mask]
    cls_ids = cls_ids[mask]
    cls_scores = cls_scores[mask]
    if len(boxes) == 0:
        return []

    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    dw, dh = dwdh
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy /= ratio

    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, im0_shape[1])
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, im0_shape[0])

    dets = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep = nms_numpy(boxes_xyxy[idx], cls_scores[idx], IOU_THRES)
        for k in keep:
            j = idx[k]
            x1, y1, x2, y2 = boxes_xyxy[j]
            coco_cat = COCO80_TO_91[int(c)] if int(c) < len(COCO80_TO_91) else int(c) + 1
            dets.append(
                {
                    "category_id": int(coco_cat),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(cls_scores[j]),
                }
            )
    return dets


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


def main():
    coco = COCO(str(ANN))
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)
    if MAX_IMAGES > 0:
        imgs = imgs[:MAX_IMAGES]

    providers = ort.get_available_providers()
    if ORT_PROVIDERS_ENV:
        requested = [p.strip() for p in ORT_PROVIDERS_ENV.split(",") if p.strip()]
        missing = [p for p in requested if p not in providers]
        if missing and ORT_FAIL_IF_MISSING:
            raise RuntimeError(
                f"Requested ORT providers not available: {missing}. Available: {providers}"
            )
        enabled = [p for p in requested if p in providers]
        if not enabled:
            enabled = ["CPUExecutionProvider"]
    else:
        preferred = [
            "VitisAIExecutionProvider",
            "QNNExecutionProvider",
            "ACLExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        enabled = [p for p in preferred if p in providers] or ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    if ORT_DISABLE_CPU_FALLBACK:
        # Force ORT to fail instead of silently using CPU for unsupported ops.
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    sess = ort.InferenceSession(str(MODEL), sess_options=sess_options, providers=enabled)
    active_providers = sess.get_providers()
    if ORT_PROVIDERS_ENV and ORT_FAIL_IF_MISSING:
        requested = [p.strip() for p in ORT_PROVIDERS_ENV.split(",") if p.strip()]
        missing_active = [p for p in requested if p not in active_providers]
        if missing_active:
            raise RuntimeError(
                f"Requested ORT providers not active: {missing_active}. Active: {active_providers}"
            )
    input_name = sess.get_inputs()[0].name

    for _ in range(WARMUP):
        dummy = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        _ = sess.run(None, {input_name: dummy})

    timing_rows = []
    det_name = f"coco_detections_{RESULT_TAG}.json"
    det_path = RESULTS / det_name

    with JsonArrayWriter(det_path) as det_writer:
        for idx, meta in enumerate(imgs, start=1):
            img_path = IMAGES / meta["file_name"]

            t0 = time.perf_counter()
            inp, im0, ratio, dwdh = preprocess(img_path)
            t1 = time.perf_counter()

            t2 = time.perf_counter()
            pred = sess.run(None, {input_name: inp})[0]
            t3 = time.perf_counter()

            dets = postprocess(pred, im0.shape, ratio, dwdh)
            t4 = time.perf_counter()

            for d in dets:
                det_writer.write_item(
                    {
                        "image_id": meta["id"],
                        "category_id": d["category_id"],
                        "bbox": xyxy_to_coco_bbox(d["bbox_xyxy"]),
                        "score": d["score"],
                    }
                )

            timing_rows.append(
                {
                    "image_id": meta["id"],
                    "preprocess_ms": (t1 - t0) * 1000.0,
                    "inference_ms": (t3 - t2) * 1000.0,
                    "postprocess_ms": (t4 - t3) * 1000.0,
                    "total_ms": (t4 - t0) * 1000.0,
                    "nms_included_in_postprocess": True,
                    "memory_transfer_included": False,
                    "providers_enabled": ",".join(active_providers),
                }
            )

            if PROGRESS_EVERY > 0 and idx % PROGRESS_EVERY == 0:
                print(f"progress: {idx}/{len(imgs)} images")

    timing_csv_name = f"timing_breakdown_{RESULT_TAG}.csv"
    timing_json_name = f"metrics_{RESULT_TAG}_timing_only.json"

    df = pd.DataFrame(timing_rows)
    df.to_csv(RESULTS / timing_csv_name, index=False)

    avg_total_ms = float(df["total_ms"].mean()) if len(df) else 0.0
    fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

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
        },
        "runtime": {
            "available_providers": providers,
            "enabled_providers": active_providers,
            "cpu_fallback_disabled": ORT_DISABLE_CPU_FALLBACK,
        },
    }

    summary["result_tag"] = RESULT_TAG
    summary["artifacts"] = {
        "detections": det_name,
        "timing_csv": timing_csv_name,
        "timing_json": timing_json_name,
    }

    with open(RESULTS / timing_json_name, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
