#!/usr/bin/env bash
set -euo pipefail

# Convert a TFLite model with NXP neutron-converter for i.MX95.
# IMPORTANT: neutron-converter expects TFLite input (not ONNX).

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SDK_ROOT=${SDK_ROOT:-"/home/podolszki/AI/NXP/eiq-neutron-sdk-linux-3.0.1"}
CONVERTER_BIN=${CONVERTER_BIN:-"$SDK_ROOT/bin/neutron-converter"}
NEUTRON_TARGET=${NEUTRON_TARGET:-"imx95"}
DEFAULT_INPUT_TFLITE="$ROOT/models/yolov8s_1024_float32.tflite"
if [[ -f "$ROOT/models/yolov8s_1024_int8.tflite" ]]; then
  DEFAULT_INPUT_TFLITE="$ROOT/models/yolov8s_1024_int8.tflite"
fi
INPUT_TFLITE=${INPUT_TFLITE:-"$DEFAULT_INPUT_TFLITE"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/models/neutron"}
OUTPUT_TFLITE=${OUTPUT_TFLITE:-"$OUTPUT_DIR/yolov8s_1024_imx95_neutron_int8.tflite"}
SAFE_MODE=${SAFE_MODE:-0}
LOG_FILE=${LOG_FILE:-"$ROOT/logs/18_host_convert_for_neutron.log"}
LOG_TAIL_LINES=${LOG_TAIL_LINES:-120}

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$INPUT_TFLITE" ]]; then
  echo "Input TFLite nicht gefunden: $INPUT_TFLITE"
  echo "Bitte zuerst TFLite exportieren, z. B.:"
  echo "  ./scripts/04b_export_tflite.sh"
  exit 1
fi

if [[ ! -d "$SDK_ROOT" ]]; then
  echo "SDK Root nicht gefunden: $SDK_ROOT"
  echo "Bitte zuerst: ./scripts/17_host_prepare_neutron_sdk.sh"
  exit 1
fi

if [[ ! -x "$CONVERTER_BIN" ]]; then
  echo "Konverter nicht gefunden oder nicht ausfuehrbar: $CONVERTER_BIN"
  exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")"

if [[ "$SAFE_MODE" == "1" ]]; then
  # Limit thread fan-out to keep host memory usage predictable.
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
  export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
  export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
  export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
fi

echo "Verwende Konverter: $CONVERTER_BIN"
echo "Target:    $NEUTRON_TARGET"
echo "Input:     $INPUT_TFLITE"
echo "Output:    $OUTPUT_TFLITE"

cmd=(
  "$CONVERTER_BIN"
  --input "$INPUT_TFLITE"
  --target "$NEUTRON_TARGET"
  --output "$OUTPUT_TFLITE"
  --dump-statistics
  --dump-statistics-file
)

if [[ "${USE_OOPT:-0}" == "1" ]]; then
  cmd+=(--optimization-level OOpt)
fi

if [[ "${DUMP_IR:-0}" == "1" ]]; then
  cmd+=(--dump-neutron-ir-file --dump-neutron-ir-final-file)
fi

if [[ "${RUN_SANITY:-0}" == "1" ]]; then
  cmd+=(--run-after-import --run-after-optimize --run-after-extract --run-after-generate)
fi

echo "Starte Konvertierung ..."
echo "Log: $LOG_FILE"

set +e
"${cmd[@]}" >"$LOG_FILE" 2>&1
rc=$?
set -e

echo
echo "--- converter log (last ${LOG_TAIL_LINES} lines) ---"
tail -n "$LOG_TAIL_LINES" "$LOG_FILE" || true
echo "--- end converter log ---"

if [[ $rc -ne 0 ]]; then
  echo "Konvertierung fehlgeschlagen (rc=$rc). Voller Log: $LOG_FILE"
  exit "$rc"
fi

if [[ ! -f "$OUTPUT_TFLITE" ]]; then
  echo "Konvertierung beendet, aber Output fehlt: $OUTPUT_TFLITE"
  exit 3
fi

echo
echo "Neutron-Konvertierung erfolgreich."
echo "Output: $OUTPUT_TFLITE"
echo "Log:    $LOG_FILE"
echo
echo "Naechster Schritt (Target-Test, full delegation):"
echo "  benchmark_model --graph=$OUTPUT_TFLITE --external_delegate_path=/usr/lib/libneutron_delegate.so --external_delegate_options=require_full_delegation=true"
