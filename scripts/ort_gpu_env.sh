#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY="${ROOT}/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Fehler: Python venv nicht gefunden unter ${PY}" >&2
  exit 1
fi

CUDA_LIB_PATHS=$(
  "$PY" - <<'PY'
import glob
import os
import site

paths = []
for base in site.getsitepackages():
    paths.extend(glob.glob(os.path.join(base, "nvidia", "*", "lib")))

# Preserve order but remove duplicates.
seen = set()
ordered = []
for p in paths:
    if p not in seen:
        ordered.append(p)
        seen.add(p)

print(":".join(ordered))
PY
)

if [[ -n "${CUDA_LIB_PATHS}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_LIB_PATHS}:${LD_LIBRARY_PATH:-}"
fi
