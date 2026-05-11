#!/usr/bin/env bash
set -euo pipefail

# Prepare/validate local Neutron SDK workspace.
# It supports both:
# 1) already unpacked SDK directory (default in this project setup)
# 2) unpacking an NXP ZIP via SDK_ZIP

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SDK_ROOT=${SDK_ROOT:-"/home/podolszki/AI/NXP/eiq-neutron-sdk-linux-3.0.1"}
SDK_ZIP=${SDK_ZIP:-""}

if [[ -n "$SDK_ZIP" ]]; then
  if [[ ! -f "$SDK_ZIP" ]]; then
    echo "SDK ZIP nicht gefunden: $SDK_ZIP"
    exit 1
  fi

  sdk_parent=$(dirname "$SDK_ROOT")
  mkdir -p "$sdk_parent"
  echo "Entpacke SDK nach: $sdk_parent"
  unzip -o "$SDK_ZIP" -d "$sdk_parent" >/dev/null

  if [[ ! -d "$SDK_ROOT" ]]; then
    guess=$(find "$sdk_parent" -maxdepth 1 -type d -name 'eiq-neutron-sdk-linux-*' | sort | tail -n 1 || true)
    if [[ -n "$guess" ]]; then
      SDK_ROOT="$guess"
    fi
  fi
fi

if [[ ! -d "$SDK_ROOT" ]]; then
  echo "SDK Root nicht gefunden: $SDK_ROOT"
  echo "Setze SDK_ROOT oder entpacke per SDK_ZIP, z. B.:"
  echo "  SDK_ZIP=~/Downloads/EIQ-NEUTRON-SDK-3.0.1-LIN.zip ./scripts/17_host_prepare_neutron_sdk.sh"
  exit 1
fi

CONVERTER_BIN="$SDK_ROOT/bin/neutron-converter"
if [[ ! -x "$CONVERTER_BIN" ]]; then
  echo "Konverter nicht gefunden oder nicht ausfuehrbar: $CONVERTER_BIN"
  exit 1
fi

echo "SDK bereit: $SDK_ROOT"
echo "Konverter: $CONVERTER_BIN"
echo
echo "Verfuegbare Neutron Targets:"
"$CONVERTER_BIN" --show-targets

echo

echo "SDK vorbereitet. Naechster Schritt:"
echo "  ./scripts/18_host_convert_for_neutron.sh"
