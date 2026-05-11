#!/usr/bin/env bash
set -euo pipefail

ROOT=${IMX95_BENCH_ROOT:-/opt/imx95-yolov8-benchmark}
mkdir -p "${ROOT}/results"

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.getenv("IMX95_BENCH_ROOT", "/opt/imx95-yolov8-benchmark"))
results = root / "results"

timing_file = results / "metrics_imx95_timing_only.json"
acc_file = results / "metrics_imx95.json"
out_file = results / "report.md"

timing = {}
acc = {}
if timing_file.exists():
    timing = json.loads(timing_file.read_text(encoding="utf-8"))
if acc_file.exists():
    acc = json.loads(acc_file.read_text(encoding="utf-8"))

lines = []
lines.append("# Benchmark Report: YOLOv8s on PHYTEC i.MX95")
lines.append("")
lines.append("## Accuracy")
lines.append(f"- mAP50: {acc.get('mAP50', 'n/a')}")
lines.append(f"- mAP50-95: {acc.get('mAP50_95', 'n/a')}")
lines.append("")
lines.append("## Performance")
lines.append(f"- Avg Latency (ms): {timing.get('avg_latency_ms', 'n/a')}")
lines.append(f"- FPS: {timing.get('fps_from_avg_latency', 'n/a')}")
lines.append(f"- Images evaluated: {timing.get('images_evaluated', 'n/a')}")
lines.append("")
lines.append("## Runtime")
runtime = timing.get("runtime", {})
lines.append(f"- Available providers: {runtime.get('available_providers', [])}")
lines.append(f"- Enabled providers: {runtime.get('enabled_providers', [])}")
lines.append("")
lines.append("## Timing definition")
defn = timing.get("timing_definition", {})
lines.append(f"- Includes preprocessing: {defn.get('includes_preprocess', 'n/a')}")
lines.append(f"- Includes inference: {defn.get('includes_inference', 'n/a')}")
lines.append(f"- Includes postprocessing: {defn.get('includes_postprocess', 'n/a')}")
lines.append(f"- Includes NMS: {defn.get('includes_nms', 'n/a')}")
lines.append(f"- Includes memory transfers: {defn.get('includes_memory_transfers', 'n/a')}")

out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {out_file}")
PY
