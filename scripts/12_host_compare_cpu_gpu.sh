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

MAX_IMAGES=${COMPARE_MAX_IMAGES:-300}
WARMUP=${COMPARE_WARMUP:-3}

check_mem_available

echo "[1/4] CPU-Referenzlauf startet (MAX_IMAGES=${MAX_IMAGES})"
IMX95_BENCH_ROOT="$ROOT" \
RESULT_TAG=host_cpu \
ORT_PROVIDERS=CPUExecutionProvider \
ORT_FAIL_IF_MISSING=1 \
MAX_IMAGES="$MAX_IMAGES" \
WARMUP="$WARMUP" \
PROGRESS_EVERY=${PROGRESS_EVERY:-50} \
python3 scripts/08_run_inference_imx95.py

IMX95_BENCH_ROOT="$ROOT" RESULT_TAG=host_cpu python3 scripts/09_eval_coco.py

echo "[2/4] Pruefe CUDAExecutionProvider"
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
  echo "Danach Skript erneut starten."
  exit 2
fi

check_mem_available

echo "[3/4] RTX-GPU Lauf startet (MAX_IMAGES=${MAX_IMAGES})"
IMX95_BENCH_ROOT="$ROOT" \
RESULT_TAG=host_rtx5090 \
ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
ORT_FAIL_IF_MISSING=1 \
MAX_IMAGES="$MAX_IMAGES" \
WARMUP="$WARMUP" \
PROGRESS_EVERY=${PROGRESS_EVERY:-50} \
python3 scripts/08_run_inference_imx95.py

IMX95_BENCH_ROOT="$ROOT" RESULT_TAG=host_rtx5090 python3 scripts/09_eval_coco.py

echo "[4/4] Erzeuge Vergleichsreport"
python3 - <<'PY'
import json
from pathlib import Path

root = Path('.')
results = root / 'results'

cpu_t = json.loads((results / 'metrics_host_cpu_timing_only.json').read_text(encoding='utf-8'))
cpu_a = json.loads((results / 'metrics_host_cpu.json').read_text(encoding='utf-8'))
gpu_t = json.loads((results / 'metrics_host_rtx5090_timing_only.json').read_text(encoding='utf-8'))
gpu_a = json.loads((results / 'metrics_host_rtx5090.json').read_text(encoding='utf-8'))

comparison = {
    'cpu': {
        'latency_ms': cpu_t.get('avg_latency_ms'),
        'fps': cpu_t.get('fps_from_avg_latency'),
        'mAP50': cpu_a.get('mAP50'),
        'mAP50_95': cpu_a.get('mAP50_95'),
        'providers': cpu_t.get('runtime', {}).get('enabled_providers', []),
    },
    'rtx5090': {
        'latency_ms': gpu_t.get('avg_latency_ms'),
        'fps': gpu_t.get('fps_from_avg_latency'),
        'mAP50': gpu_a.get('mAP50'),
        'mAP50_95': gpu_a.get('mAP50_95'),
        'providers': gpu_t.get('runtime', {}).get('enabled_providers', []),
    },
}

cpu_fps = comparison['cpu']['fps'] or 0.0
gpu_fps = comparison['rtx5090']['fps'] or 0.0
cpu_lat = comparison['cpu']['latency_ms'] or 0.0
gpu_lat = comparison['rtx5090']['latency_ms'] or 0.0

comparison['speedup_fps_x'] = (gpu_fps / cpu_fps) if cpu_fps > 0 else None
comparison['latency_reduction_percent'] = ((cpu_lat - gpu_lat) / cpu_lat * 100.0) if cpu_lat > 0 else None

(results / 'host_cpu_vs_rtx5090.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')

lines = []
lines.append('# Host Vergleich: CPU vs RTX 5090 (ONNX Runtime)')
lines.append('')
lines.append('## CPU')
lines.append(f"- Avg Latency (ms): {comparison['cpu']['latency_ms']}")
lines.append(f"- FPS: {comparison['cpu']['fps']}")
lines.append(f"- mAP50: {comparison['cpu']['mAP50']}")
lines.append(f"- mAP50-95: {comparison['cpu']['mAP50_95']}")
lines.append(f"- Provider: {comparison['cpu']['providers']}")
lines.append('')
lines.append('## RTX 5090')
lines.append(f"- Avg Latency (ms): {comparison['rtx5090']['latency_ms']}")
lines.append(f"- FPS: {comparison['rtx5090']['fps']}")
lines.append(f"- mAP50: {comparison['rtx5090']['mAP50']}")
lines.append(f"- mAP50-95: {comparison['rtx5090']['mAP50_95']}")
lines.append(f"- Provider: {comparison['rtx5090']['providers']}")
lines.append('')
lines.append('## Delta')
lines.append(f"- Speedup (FPS x): {comparison['speedup_fps_x']}")
lines.append(f"- Latency reduction (%): {comparison['latency_reduction_percent']}")

(results / 'host_cpu_vs_rtx5090.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(comparison, indent=2))
PY

echo "Vergleich fertig: results/host_cpu_vs_rtx5090.json und results/host_cpu_vs_rtx5090.md"
