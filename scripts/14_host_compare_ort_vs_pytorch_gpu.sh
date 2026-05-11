#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Fehler: .venv fehlt. Bitte zuerst ./scripts/00_host_setup_manjaro.sh ausfuehren."
  exit 1
fi

source .venv/bin/activate
source scripts/ort_gpu_env.sh

export OMP_NUM_THREADS=${BENCH_OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}

MIN_AVAIL_GB=${BENCH_MIN_AVAIL_GB:-10}

check_mem_available() {
  local avail_kb need_kb
  avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  need_kb=$((MIN_AVAIL_GB * 1024 * 1024))
  if [[ -z "$avail_kb" ]]; then
    echo "Warnung: MemAvailable konnte nicht gelesen werden, laufe ohne RAM-Guard."
    return 0
  fi
  if (( avail_kb < need_kb )); then
    echo "Fehler: zu wenig freier RAM: ${avail_kb} KiB verfuegbar, mindestens ${need_kb} KiB benoetigt."
    echo "Tipp: BENCH_MIN_AVAIL_GB reduzieren oder andere speicherintensive Prozesse stoppen."
    exit 3
  fi
}

if [[ ! -f models/yolov8s_1024.onnx ]]; then
  echo "models/yolov8s_1024.onnx fehlt. Bitte zuerst ./scripts/04_export_onnx.sh ausfuehren."
  exit 1
fi

if [[ ! -f yolov8s.pt ]]; then
  echo "yolov8s.pt fehlt. Bitte zuerst ./scripts/02_download_model.sh ausfuehren."
  exit 1
fi

MAX_IMAGES=${COMPARE_MAX_IMAGES:-300}
WARMUP=${COMPARE_WARMUP:-3}

echo "[1/5] Pruefe ORT CUDA Provider"
CUDA_READY=$(python3 - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print('CUDAExecutionProvider' in providers)
print(','.join(providers))
PY
)
CUDA_OK=$(echo "$CUDA_READY" | head -n1)
PROVIDERS=$(echo "$CUDA_READY" | tail -n1)

echo "Verfuegbare ORT Provider: ${PROVIDERS}"

if [[ "$CUDA_OK" != "True" ]]; then
  echo "CUDAExecutionProvider ist nicht verfuegbar."
  echo "Installiere GPU-ORT im venv, z. B.:"
  echo "  source .venv/bin/activate"
  echo "  pip uninstall -y onnxruntime && pip install onnxruntime-gpu"
  exit 2
fi

echo "[2/5] Pruefe PyTorch CUDA"
python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('PyTorch CUDA ist nicht verfuegbar. Prüfe Treiber/CUDA.')
print('Torch CUDA ok:', torch.cuda.get_device_name(0))
PY

check_mem_available

echo "[3/5] ONNX Runtime CUDA Lauf"
IMX95_BENCH_ROOT="$ROOT" \
RESULT_TAG=host_ort_cuda \
ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
ORT_FAIL_IF_MISSING=1 \
MAX_IMAGES="$MAX_IMAGES" \
WARMUP="$WARMUP" \
PROGRESS_EVERY=${PROGRESS_EVERY:-50} \
python3 scripts/08_run_inference_imx95.py
IMX95_BENCH_ROOT="$ROOT" RESULT_TAG=host_ort_cuda python3 scripts/09_eval_coco.py

check_mem_available

echo "[4/5] PyTorch CUDA Lauf"
IMX95_BENCH_ROOT="$ROOT" \
RESULT_TAG=host_pytorch_cuda \
PT_DEVICE=0 \
MAX_IMAGES="$MAX_IMAGES" \
WARMUP="$WARMUP" \
PROGRESS_EVERY=${PROGRESS_EVERY:-50} \
CLEAR_CUDA_CACHE_EVERY=${CLEAR_CUDA_CACHE_EVERY:-50} \
python3 scripts/13_run_pytorch_inference_host.py
IMX95_BENCH_ROOT="$ROOT" RESULT_TAG=host_pytorch_cuda python3 scripts/09_eval_coco.py

echo "[5/5] Erzeuge Vergleichsreport"
python3 - <<'PY'
import json
from pathlib import Path

root = Path('.')
results = root / 'results'

ort_t = json.loads((results / 'metrics_host_ort_cuda_timing_only.json').read_text(encoding='utf-8'))
ort_a = json.loads((results / 'metrics_host_ort_cuda.json').read_text(encoding='utf-8'))
pt_t = json.loads((results / 'metrics_host_pytorch_cuda_timing_only.json').read_text(encoding='utf-8'))
pt_a = json.loads((results / 'metrics_host_pytorch_cuda.json').read_text(encoding='utf-8'))

comparison = {
    'onnxruntime_cuda': {
        'latency_ms': ort_t.get('avg_latency_ms'),
        'fps': ort_t.get('fps_from_avg_latency'),
        'mAP50': ort_a.get('mAP50'),
        'mAP50_95': ort_a.get('mAP50_95'),
        'runtime': ort_t.get('runtime', {}),
    },
    'pytorch_cuda': {
        'latency_ms': pt_t.get('avg_latency_ms'),
        'fps': pt_t.get('fps_from_avg_latency'),
        'mAP50': pt_a.get('mAP50'),
        'mAP50_95': pt_a.get('mAP50_95'),
        'runtime': pt_t.get('runtime', {}),
    },
}

ort_fps = comparison['onnxruntime_cuda']['fps'] or 0.0
pt_fps = comparison['pytorch_cuda']['fps'] or 0.0
ort_lat = comparison['onnxruntime_cuda']['latency_ms'] or 0.0
pt_lat = comparison['pytorch_cuda']['latency_ms'] or 0.0

comparison['pytorch_vs_ort_fps_x'] = (pt_fps / ort_fps) if ort_fps > 0 else None
comparison['pytorch_vs_ort_latency_delta_percent'] = ((ort_lat - pt_lat) / ort_lat * 100.0) if ort_lat > 0 else None

(results / 'host_ort_vs_pytorch_cuda.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')

lines = []
lines.append('# Host Vergleich: ONNX Runtime CUDA vs PyTorch CUDA')
lines.append('')
lines.append('## ONNX Runtime CUDA')
lines.append(f"- Avg Latency (ms): {comparison['onnxruntime_cuda']['latency_ms']}")
lines.append(f"- FPS: {comparison['onnxruntime_cuda']['fps']}")
lines.append(f"- mAP50: {comparison['onnxruntime_cuda']['mAP50']}")
lines.append(f"- mAP50-95: {comparison['onnxruntime_cuda']['mAP50_95']}")
lines.append(f"- Runtime: {comparison['onnxruntime_cuda']['runtime']}")
lines.append('')
lines.append('## PyTorch CUDA')
lines.append(f"- Avg Latency (ms): {comparison['pytorch_cuda']['latency_ms']}")
lines.append(f"- FPS: {comparison['pytorch_cuda']['fps']}")
lines.append(f"- mAP50: {comparison['pytorch_cuda']['mAP50']}")
lines.append(f"- mAP50-95: {comparison['pytorch_cuda']['mAP50_95']}")
lines.append(f"- Runtime: {comparison['pytorch_cuda']['runtime']}")
lines.append('')
lines.append('## Delta')
lines.append(f"- PyTorch vs ORT speedup (FPS x): {comparison['pytorch_vs_ort_fps_x']}")
lines.append(f"- PyTorch vs ORT latency delta (%): {comparison['pytorch_vs_ort_latency_delta_percent']}")

(results / 'host_ort_vs_pytorch_cuda.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(comparison, indent=2))
PY

echo "Vergleich fertig: results/host_ort_vs_pytorch_cuda.json und results/host_ort_vs_pytorch_cuda.md"
