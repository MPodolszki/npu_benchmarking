#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${MODE:-smoke}
SMOKE_IMAGES=${SMOKE_IMAGES:-100}
SSHFS_TARGET=${SSHFS_TARGET:-$HOME/sshfs/imx95-yolov8-benchmark}

MODE="$MODE" SMOKE_IMAGES="$SMOKE_IMAGES" "${ROOT}/scripts/15_make_target_bundle.sh"
BUNDLE_DIR="${ROOT}/dist/imx95_bundle_${MODE}"

mkdir -p "$SSHFS_TARGET"
rsync -a --delete \
	--exclude '.venv/' \
	--exclude 'results/' \
	"$BUNDLE_DIR"/ "$SSHFS_TARGET"/

echo "Deploy abgeschlossen: ${SSHFS_TARGET}"
