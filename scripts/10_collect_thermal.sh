#!/usr/bin/env bash
set -euo pipefail

ROOT=${IMX95_BENCH_ROOT:-/opt/imx95-yolov8-benchmark}
OUT="${ROOT}/results/environment.txt"

{
  date
  uname -a
  echo

  echo '=== CPU info ==='
  cat /proc/cpuinfo || true
  echo

  echo '=== Memory ==='
  free -h || true
  echo

  echo '=== Thermal zones ==='
  for z in /sys/class/thermal/thermal_zone*; do
    echo "ZONE: $z"
    cat "$z/type" 2>/dev/null || true
    cat "$z/temp" 2>/dev/null || true
    echo
  done
} > "$OUT"

echo "Wrote $OUT"
