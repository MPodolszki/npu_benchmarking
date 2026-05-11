#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ ! -f "$ROOT/env.sh" ]]; then
  echo "env.sh fehlt. Bitte env.sh.example nach env.sh kopieren und anpassen."
  exit 1
fi

source "$ROOT/env.sh"

REMOTE_CANDIDATE_MODEL=${REMOTE_CANDIDATE_MODEL:-"$TARGET_DIR/models/neutron/yolov8s_1024_imx95_neutron_int8.tflite"}
NUM_RUNS_STRICT=${NUM_RUNS_STRICT:-1}
DELEGATE_SO=${DELEGATE_SO:-"/usr/lib/libneutron_delegate.so"}
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=12)
TARGET_SSH="${TARGET_USER}@${TARGET_IP}"

echo "[info] Target: ${TARGET_USER}@${TARGET_IP}"
echo "[info] Target dir: ${TARGET_DIR}"
echo "[info] Candidate model: ${REMOTE_CANDIDATE_MODEL}"

ssh "${SSH_OPTS[@]}" "$TARGET_SSH" "echo [info] ssh_ok && hostname" >/dev/null

benchmark_bin=$(ssh "${SSH_OPTS[@]}" "$TARGET_SSH" "for b in /usr/bin/tensorflow-lite-*/examples/benchmark_model /usr/bin/benchmark_model; do [ -x \"\$b\" ] && echo \"\$b\" && exit 0; done; command -v benchmark_model || true")
if [[ -z "$benchmark_bin" ]]; then
  echo "RESULT=NO_GO"
  echo "REASON=benchmark_model_missing"
  echo "[no-go] NPU benchmark nicht freigegeben (benchmark_model fehlt)."
  exit 2
fi

if ! ssh "${SSH_OPTS[@]}" "$TARGET_SSH" "test -e /dev/neutron0"; then
  echo "RESULT=NO_GO"
  echo "REASON=missing_/dev/neutron0"
  echo "[no-go] NPU benchmark nicht freigegeben (/dev/neutron0 fehlt)."
  exit 2
fi

if ! ssh "${SSH_OPTS[@]}" "$TARGET_SSH" "test -f '$DELEGATE_SO'"; then
  echo "RESULT=NO_GO"
  echo "REASON=missing_delegate_so"
  echo "[no-go] NPU benchmark nicht freigegeben (Delegate-SO fehlt)."
  exit 2
fi

echo "BENCHMARK_BIN=$benchmark_bin"
echo "DELEGATE_SO=$DELEGATE_SO"

ref_model=$(ssh "${SSH_OPTS[@]}" "$TARGET_SSH" "for m in /usr/bin/tensorflow-lite-*/examples/mobilenet_v1_1.0_224_quant.tflite '$TARGET_DIR/models/mobilenet_v1_1.0_224_quant.tflite'; do [ -f \"\$m\" ] && echo \"\$m\" && exit 0; done; true")

mode="converted_only"
if [[ -n "$ref_model" ]]; then
  echo "REF_MODEL=$ref_model"
  set +e
  ref_out=$(ssh "${SSH_OPTS[@]}" "$TARGET_SSH" "'$benchmark_bin' --graph='$ref_model' --external_delegate_path='$DELEGATE_SO' --require_full_delegation=true --num_runs='$NUM_RUNS_STRICT'" 2>&1)
  ref_rc=$?
  set -e
  ref_delegated=$(printf '%s\n' "$ref_out" | sed -n 's/.*delegate: \([0-9][0-9]*\) nodes delegated out of.*/\1/p' | tail -n 1)
  ref_delegated=${ref_delegated:-0}
  echo "REF_STRICT_RC=$ref_rc"
  echo "REF_DELEGATED_NODES=$ref_delegated"
  if [[ "$ref_rc" -eq 0 && "$ref_delegated" -gt 0 ]]; then
    mode="legacy_raw_tflite_ok"
  fi
fi

echo "STACK_MODE=$mode"

if ! ssh "${SSH_OPTS[@]}" "$TARGET_SSH" "test -f '$REMOTE_CANDIDATE_MODEL'"; then
  if [[ "$mode" == "legacy_raw_tflite_ok" ]]; then
    echo "RESULT=GO"
    echo "REASON=legacy_mode_raw_tflite_delegates"
    echo "[go] NPU benchmark freigegeben (legacy raw tflite path)."
    exit 0
  fi
  echo "RESULT=NO_GO"
  echo "REASON=converted_model_missing"
  echo "[no-go] NPU benchmark nicht freigegeben (konvertiertes Modell fehlt)."
  exit 2
fi

set +e
# Neutron-converted models contain NeutronGraph custom ops — the bulk of the
# network is fused into 1 NeutronGraph op (1 delegated node is correct).
# Do NOT use --require_full_delegation here; pre/post-processing ops run on CPU.
conv_out=$(ssh "${SSH_OPTS[@]}" "$TARGET_SSH" "'$benchmark_bin' --graph='$REMOTE_CANDIDATE_MODEL' --external_delegate_path='$DELEGATE_SO' --num_runs='$NUM_RUNS_STRICT'" 2>&1)
conv_rc=$?
set -e

conv_delegated=$(printf '%s\n' "$conv_out" | sed -n 's/.*delegate: \([0-9][0-9]*\) nodes delegated out of.*/\1/p' | tail -n 1)
conv_delegated=${conv_delegated:-0}
conv_avg_us=$(printf '%s\n' "$conv_out" | sed -n 's/.*Inference (avg): \([0-9][0-9]*\).*/\1/p' | tail -n 1)
conv_avg_us=${conv_avg_us:-0}

echo "CONVERTED_MODEL=$REMOTE_CANDIDATE_MODEL"
echo "CONVERTED_RC=$conv_rc"
echo "CONVERTED_DELEGATED_NODES=$conv_delegated"
echo "CONVERTED_INFERENCE_AVG_US=$conv_avg_us"

if [[ "$conv_rc" -eq 0 && "$conv_delegated" -gt 0 ]]; then
  result="GO"
  reason="neutron_delegate_ok_${conv_delegated}_nodes_delegated"
else
  result="NO_GO"
  if [[ "$conv_delegated" -eq 0 ]]; then
    reason="converted_model_no_delegation"
  else
    reason="converted_model_rc_nonzero"
  fi
fi

echo "RESULT=$result"
echo "REASON=$reason"

if [[ "$result" == "GO" ]]; then
  echo "[go] NPU benchmark freigegeben (${reason})."
  exit 0
fi

echo "[no-go] NPU benchmark nicht freigegeben (${reason})."
exit 2
