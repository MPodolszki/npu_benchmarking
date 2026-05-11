#!/usr/bin/env python3
"""YOLOv8 inference on i.MX95 NPU via TFLite + Neutron delegate.

Environment variables (all optional, with defaults):
  IMX95_BENCH_ROOT   – root of the benchmark tree  (default /opt/imx95-yolov8-benchmark)
  MODEL              – path to Neutron-converted .tflite model
  DELEGATE_SO        – path to libneutron_delegate.so
  IMG_SIZE           – input resolution                          (default 1024)
  CONF_THRES         – detection confidence threshold           (default 0.25)
  IOU_THRES          – NMS IoU threshold                        (default 0.70)
  MAX_IMAGES         – limit number of images (0 = all 5000)    (default 0)
  WARMUP             – number of warm-up inferences             (default 5)
  RESULT_TAG         – suffix used for result filenames         (default npu_neutron)
  PROGRESS_EVERY     – print progress every N images            (default 100)
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

ROOT = Path(os.getenv("IMX95_BENCH_ROOT", "/opt/imx95-yolov8-benchmark"))
IMAGES = ROOT / "data/coco/val2017"
ANN = ROOT / "data/coco/instances_val2017.json"
if not ANN.exists():
    ANN = ROOT / "data/coco/annotations/instances_val2017.json"

_default_model = ROOT / "models/neutron/yolov8s_1024_imx95_neutron_int8.tflite"
MODEL = Path(os.getenv("MODEL", str(_default_model)))
DELEGATE_SO = os.getenv("DELEGATE_SO", "/usr/lib/libneutron_delegate.so")
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = int(os.getenv("IMG_SIZE",       "1024"))
CONF_THRES  = float(os.getenv("CONF_THRES",   "0.25"))
IOU_THRES   = float(os.getenv("IOU_THRES",    "0.70"))
MAX_IMAGES  = int(os.getenv("MAX_IMAGES",     "0"))
WARMUP      = int(os.getenv("WARMUP",         "5"))
RESULT_TAG  = os.getenv("RESULT_TAG",         "npu_neutron")
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


# ---------------------------------------------------------------------------
# Image pre-processing (letterbox → RGB → uint8 for INT8 TFLite)
# ---------------------------------------------------------------------------

def letterbox(im, new_shape=(1024, 1024), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess(img_path, input_scale, input_zero_point, input_dtype):
    im = cv2.imread(str(img_path))
    if im is None:
        raise RuntimeError(f"Failed to read {img_path}")
    im0 = im.copy()
    im, ratio, dwdh = letterbox(im, (IMG_SIZE, IMG_SIZE))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if np.issubdtype(input_dtype, np.integer):
        # INT8 / UINT8: quantize float [0,1] → quantized integer
        im_f = im.astype(np.float32) / 255.0
        im_q = np.round(im_f / input_scale + input_zero_point).astype(input_dtype)
        blob = im_q[None]  # [1, H, W, 3]
    else:
        im = im.astype(np.float32) / 255.0
        blob = im[None]

    return blob, im0, ratio, dwdh


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def nms_numpy(boxes, scores, iou_thres):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return keep


def postprocess(raw_output, output_scale, output_zero_point, output_dtype,
                im0_shape, ratio, dwdh):
    """Decode YOLOv8 TFLite output [1, N, 84] or [1, 84, N].

    The Neutron-converted INT8 TFLite model outputs coordinates normalized to
    [0, 1] (quantization scale ~0.004 → max ~1.0), unlike the ONNX model which
    outputs absolute pixel values.  We multiply by IMG_SIZE to restore pixels.
    """
    pred = np.squeeze(raw_output)  # → [N, 84] or [84, N]
    if pred.ndim == 1:
        return []
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # → [N, 84]

    # Dequantize if integer
    if np.issubdtype(output_dtype, np.integer):
        pred = (pred.astype(np.float32) - output_zero_point) * output_scale

    boxes_raw = pred[:, :4]   # cx, cy, w, h — normalized [0,1] for TFLite INT8
    cls_scores_all = pred[:, 4:]

    # Scale normalized coords back to pixel space of the letterboxed image
    boxes_raw = boxes_raw * IMG_SIZE

    cls_ids  = np.argmax(cls_scores_all, axis=1)
    cls_conf = cls_scores_all[np.arange(len(cls_ids)), cls_ids]
    mask     = cls_conf > CONF_THRES

    boxes_raw = boxes_raw[mask]
    cls_ids   = cls_ids[mask]
    cls_conf  = cls_conf[mask]

    if len(boxes_raw) == 0:
        return []

    # cx,cy,w,h → x1,y1,x2,y2
    bx = np.empty_like(boxes_raw)
    bx[:, 0] = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
    bx[:, 1] = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
    bx[:, 2] = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
    bx[:, 3] = boxes_raw[:, 1] + boxes_raw[:, 3] / 2

    # Remove letterbox padding and rescale to original image
    dw, dh = dwdh
    bx[:, [0, 2]] -= dw
    bx[:, [1, 3]] -= dh
    bx /= ratio
    bx[:, [0, 2]] = bx[:, [0, 2]].clip(0, im0_shape[1])
    bx[:, [1, 3]] = bx[:, [1, 3]].clip(0, im0_shape[0])

    dets = []
    for c in np.unique(cls_ids):
        idx  = np.where(cls_ids == c)[0]
        keep = nms_numpy(bx[idx], cls_conf[idx], IOU_THRES)
        for k in keep:
            j = idx[k]
            x1, y1, x2, y2 = bx[j]
            coco_cat = COCO80_TO_91[int(c)] if int(c) < len(COCO80_TO_91) else int(c) + 1
            dets.append({
                "category_id": int(coco_cat),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(cls_conf[j]),
            })
    return dets


def xyxy_to_coco_bbox(xyxy):
    x1, y1, x2, y2 = xyxy
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


# ---------------------------------------------------------------------------
# Streaming JSON writer (avoids building a huge list in RAM)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ----- Load COCO annotations -----
    try:
        from pycocotools.coco import COCO
        coco = COCO(str(ANN))
        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)
    except ImportError:
        # No pycocotools on target — build minimal image list from filesystem
        print("[warn] pycocotools not available; reading images directly from disk")
        coco = None
        img_files = sorted(IMAGES.glob("*.jpg"))
        imgs = [{"id": i, "file_name": p.name} for i, p in enumerate(img_files, 1)]

    if MAX_IMAGES > 0:
        imgs = imgs[:MAX_IMAGES]

    print(f"[info] Model:         {MODEL}")
    print(f"[info] Delegate:      {DELEGATE_SO}")
    print(f"[info] Images:        {len(imgs)}")
    print(f"[info] Result tag:    {RESULT_TAG}")

    # ----- Build interpreter -----
    if Path(DELEGATE_SO).exists():
        delegate = tflite.load_delegate(DELEGATE_SO)
        interp = tflite.Interpreter(model_path=str(MODEL),
                                    experimental_delegates=[delegate])
        delegate_active = True
        print(f"[info] Neutron delegate loaded")
    else:
        print(f"[warn] Delegate SO not found ({DELEGATE_SO}), running CPU-only")
        interp = tflite.Interpreter(model_path=str(MODEL))
        delegate_active = False

    interp.allocate_tensors()

    inp_det  = interp.get_input_details()[0]
    out_det  = interp.get_output_details()[0]

    input_scale      = inp_det["quantization"][0] if inp_det["quantization"][0] != 0 else 1.0
    input_zero_point = inp_det["quantization"][1]
    input_dtype      = inp_det["dtype"]
    output_scale     = out_det["quantization"][0] if out_det["quantization"][0] != 0 else 1.0
    output_zero_point= out_det["quantization"][1]
    output_dtype     = out_det["dtype"]

    print(f"[info] Input  dtype={input_dtype.__name__}  scale={input_scale:.6f}  zp={input_zero_point}")
    print(f"[info] Output dtype={output_dtype.__name__} scale={output_scale:.6f}  zp={output_zero_point}")

    # ----- Warm-up -----
    dummy_img = (IMAGES / imgs[0]["file_name"])
    for _ in range(WARMUP):
        blob, _, _, _ = preprocess(dummy_img, input_scale, input_zero_point, input_dtype)
        interp.set_tensor(inp_det["index"], blob)
        interp.invoke()

    # ----- Inference loop -----
    timing_rows = []
    det_path = RESULTS / f"coco_detections_{RESULT_TAG}.json"

    with JsonArrayWriter(det_path) as writer:
        for idx, meta in enumerate(imgs, start=1):
            img_path = IMAGES / meta["file_name"]

            t0 = time.perf_counter()
            blob, im0, ratio, dwdh = preprocess(img_path, input_scale,
                                                 input_zero_point, input_dtype)
            t1 = time.perf_counter()

            interp.set_tensor(inp_det["index"], blob)
            t2 = time.perf_counter()
            interp.invoke()
            t3 = time.perf_counter()
            raw = interp.get_tensor(out_det["index"])
            t4 = time.perf_counter()

            dets = postprocess(raw, output_scale, output_zero_point, output_dtype,
                                im0.shape, ratio, dwdh)
            t5 = time.perf_counter()

            for d in dets:
                writer.write_item({
                    "image_id": meta["id"],
                    "category_id": d["category_id"],
                    "bbox": xyxy_to_coco_bbox(d["bbox_xyxy"]),
                    "score": d["score"],
                })

            timing_rows.append({
                "image_id":       meta["id"],
                "preprocess_ms":  (t1 - t0) * 1000,
                "set_tensor_ms":  (t2 - t1) * 1000,
                "invoke_ms":      (t3 - t2) * 1000,
                "get_tensor_ms":  (t4 - t3) * 1000,
                "postprocess_ms": (t5 - t4) * 1000,
                "total_ms":       (t5 - t0) * 1000,
                "delegate_active": delegate_active,
            })

            if PROGRESS_EVERY > 0 and idx % PROGRESS_EVERY == 0:
                avg_invoke = sum(r["invoke_ms"] for r in timing_rows[-PROGRESS_EVERY:]) / PROGRESS_EVERY
                print(f"progress: {idx}/{len(imgs)}  avg_invoke={avg_invoke:.1f}ms")

    # ----- Summary -----
    invoke_ms  = [r["invoke_ms"]  for r in timing_rows]
    total_ms   = [r["total_ms"]   for r in timing_rows]
    avg_invoke = float(np.mean(invoke_ms))
    avg_total  = float(np.mean(total_ms))
    fps        = 1000.0 / avg_total if avg_total > 0 else 0.0

    summary = {
        "dataset":            "COCO val2017",
        "imgsz":              IMG_SIZE,
        "batch":              1,
        "images_evaluated":   len(timing_rows),
        "delegate_active":    delegate_active,
        "avg_invoke_ms":      round(avg_invoke, 2),
        "avg_total_ms":       round(avg_total,  2),
        "fps_from_avg_total": round(fps, 2),
        "result_tag":         RESULT_TAG,
        "detections_json":    str(det_path),
    }

    summary_path = RESULTS / f"metrics_{RESULT_TAG}_timing_only.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = RESULTS / f"timing_breakdown_{RESULT_TAG}.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=timing_rows[0].keys())
        w.writeheader()
        w.writerows(timing_rows)

    print(f"\n=== ERGEBNIS ===")
    print(f"Images evaluated : {summary['images_evaluated']}")
    print(f"Delegate active  : {delegate_active}")
    print(f"Avg invoke (NPU) : {avg_invoke:.1f} ms")
    print(f"Avg total        : {avg_total:.1f} ms  ({fps:.2f} FPS)")
    print(f"Detections JSON  : {det_path}")
    print(f"Timing CSV       : {csv_path}")
    print(f"Summary JSON     : {summary_path}")


if __name__ == "__main__":
    main()
