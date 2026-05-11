# YOLOv8s Benchmark: Manjaro Host -> PHYTEC i.MX95

Dieses Projekt trennt den Ablauf sauber in zwei Phasen:

1. Entwicklung und Validierung auf x86-Manjaro (Host)
2. Deployment und Messung auf PHYTEC i.MX95 (Target)

Ziel: belastbare Werte fuer Accuracy und Performance auf dem i.MX95 inkl. NPU-Runtime.

## Zielmetriken

- mAP50
- mAP50-95
- FPS
- Avg-Latenz (ms)
- Transparenz, ob NMS und Memory-Transfers enthalten sind

## Projektstruktur

- scripts/: End-to-end Skripte (Host + Target)
- data/coco/: COCO val2017 + Annotationen
- models/: exportierte Modelle
- results/: Mess- und Auswertungsartefakte
- logs/: Logdateien

## Voraussetzungen (Host: Manjaro x86)

- Internetzugang fuer COCO/Modelldownload
- Python 3.10+ empfohlen
- ausreichend Speicherplatz fuer COCO val2017

## Schnellstart Host

```bash
chmod +x scripts/*.sh
./scripts/00_host_setup_manjaro.sh
source .venv/bin/activate
./scripts/01_download_coco.sh
./scripts/02_download_model.sh
./scripts/03_validate_pytorch.sh
./scripts/04_export_onnx.sh
./scripts/05_check_onnx.sh
```

## RTX 5090 Vergleich auf dem Host

Fuer einen direkten CPU-vs-GPU Vergleich (ONNX Runtime) auf deinem Manjaro-x86 Host:

```bash
source .venv/bin/activate
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
./scripts/12_host_compare_cpu_gpu.sh
```

Optionale Parameter:

```bash
COMPARE_MAX_IMAGES=2000 COMPARE_WARMUP=20 ./scripts/12_host_compare_cpu_gpu.sh
```

Stabilere Defaults gegen OOM sind jetzt aktiv (`COMPARE_MAX_IMAGES=300`, `COMPARE_WARMUP=3`).
Fuer grosse Laeufe kannst du schrittweise hochdrehen.

Weitere Hardening-Parameter:

```bash
BENCH_MIN_AVAIL_GB=10 BENCH_OMP_NUM_THREADS=4 PROGRESS_EVERY=50 ./scripts/12_host_compare_cpu_gpu.sh
```

Erzeugte Vergleichsartefakte:

- results/host_cpu_vs_rtx5090.json
- results/host_cpu_vs_rtx5090.md

## ONNX Runtime CUDA vs. PyTorch CUDA

Zusaetzlicher GPU-Vergleich auf derselben Karte (z. B. RTX 5090):

```bash
source .venv/bin/activate
./scripts/14_host_compare_ort_vs_pytorch_gpu.sh
```

Optionale Parameter:

```bash
COMPARE_MAX_IMAGES=2000 COMPARE_WARMUP=20 ./scripts/14_host_compare_ort_vs_pytorch_gpu.sh
```

Stabilere Defaults gegen OOM sind jetzt aktiv (`COMPARE_MAX_IMAGES=300`, `COMPARE_WARMUP=3`).

Weitere Hardening-Parameter:

```bash
BENCH_MIN_AVAIL_GB=10 BENCH_OMP_NUM_THREADS=4 PROGRESS_EVERY=50 CLEAR_CUDA_CACHE_EVERY=50 ./scripts/14_host_compare_ort_vs_pytorch_gpu.sh
```

Erzeugte Vergleichsartefakte:

- results/host_ort_vs_pytorch_cuda.json
- results/host_ort_vs_pytorch_cuda.md

Optional Transfer zum Target:

```bash
cp env.sh.example env.sh
# env.sh mit IP/User anpassen
./scripts/06_transfer_to_target.sh
```

## Schnellstart Target (i.MX95)

```bash
cd /opt/imx95-yolov8-benchmark
chmod +x scripts/*.sh
./scripts/07_target_setup.sh
source .venv/bin/activate
python3 scripts/08_run_inference_imx95.py
python3 scripts/09_eval_coco.py
bash scripts/10_collect_thermal.sh
bash scripts/11_generate_report.sh
```

## Wichtiger NPU-Hinweis

`onnxruntime` ist nur Referenzpfad. Fuer finale i.MX95-NPU-Werte muss die BSP-konforme
NPU-Toolchain verwendet werden (z. B. NXP/PHYTEC Runtime + Provider/Delegate). Das Skript
`08_run_inference_imx95.py` ist so gebaut, dass Laufzeitdetails in den Ergebnisdateien
sichtbar sind.

## i.MX95 NPU-Konvertierung (Host)

Fuer i.MX95 mit eIQ Neutron NPU wird der `neutron-converter` aus dem eIQ Neutron SDK benoetigt.
Wichtig: dieser Konverter erwartet TFLite als Eingabe (nicht ONNX direkt).

Relevante Referenzen:

- PHYTEC Anleitung (i.MX8MP, aehnlicher NPU-Flow):
	https://www.phytec.de/cdocuments/?doc=lpPSPw
- ExecuTorch NXP Backend Tutorial:
	https://docs.pytorch.org/executorch/1.1/backends/nxp/tutorials/nxp-basic-tutorial.html
- NXP UG10166 (Compute Backends / Delegates):
	https://docs.nxp.com/bundle/UG10166/page/topics/compute_backends_and_delegates.html

NXP Download-Pakete:

- eIQ Neutron SDK 3.0.1 (Linux): `EIQ-NEUTRON-SDK-3.0.1-LIN` (bevorzugt fuer BSP 6.12.34-2.1.0 / ALPHA2)
- eIQ Neutron SDK 3.1.0 (Linux): `EIQ-NEUTRON-SDK-3.1.0-LIN` (nur fuer passende neuere BSP-Linien)

Hinweis: Die Downloads sind account-/lizenzpflichtig auf NXP.

### Workflow auf dem x86-Host

1. Lokales SDK validieren (oder via `SDK_ZIP` entpacken):

```bash
./scripts/17_host_prepare_neutron_sdk.sh
# optional mit ZIP:
# SDK_ZIP=~/Downloads/EIQ-NEUTRON-SDK-3.0.1-LIN.zip ./scripts/17_host_prepare_neutron_sdk.sh
```

2. YOLO als TFLite exportieren:

```bash
./scripts/04b_export_tflite.sh
```

3. Neutron-Konvertierung fuer i.MX95 ausfuehren:

```bash
./scripts/18_host_convert_for_neutron.sh
```

4. Das erzeugte NPU-TFLite-Modell auf dem Target mit strict delegation testen:

```bash
benchmark_model --graph=/pfad/zum/modell.tflite \
	--external_delegate_path=/usr/lib/libneutron_delegate.so \
	--external_delegate_options=require_full_delegation=true
```

Wenn dieser Test fehlschlaegt, ist das Modell weiterhin nicht voll NPU-kompatibel
und muss mit angepassten Export-/Quantisierungsparametern neu erzeugt werden.

## NPU GO/NO-GO Check (vor jedem Benchmark)

Um sicherzustellen, dass keine CPU-Fallback-Messungen als NPU-Werte gewertet werden,
gibt es einen automatischen Freigabe-Check:

```bash
chmod +x scripts/19_npu_gonogo_check.sh
./scripts/19_npu_gonogo_check.sh
```

Optionales Zielmodell (Remote-Pfad auf dem Target):

```bash
REMOTE_CANDIDATE_MODEL=/opt/imx95-yolov8-benchmark/models/neutron/yolov8s_1024_imx95_neutron_int8.tflite \
./scripts/19_npu_gonogo_check.sh
```

Der Check liefert am Ende klar:

- `RESULT=GO`: NPU-Benchmark freigegeben
- `RESULT=NO_GO`: erst Stack/Modell korrigieren, dann benchmarken

## Minimalset fuer i.MX95 (kleiner Transfer)

Wenn du den kompletten Workspace kopierst, wird es schnell gross (typisch: `.venv` + COCO Daten).
Stattdessen kannst du ein minimales Bundle bauen.

Smoke-Bundle (100 Bilder, fuer schnellen Funktionstest):

```bash
MODE=smoke SMOKE_IMAGES=100 ./scripts/15_make_target_bundle.sh
```

Voll-Bundle (komplette val2017, fuer volle Metriken):

```bash
MODE=full ./scripts/15_make_target_bundle.sh
```

Direkt auf gemountetes `~/sshfs` deployen:

```bash
MODE=smoke SMOKE_IMAGES=100 SSHFS_TARGET=~/sshfs/imx95-yolov8-benchmark ./scripts/16_deploy_bundle_to_sshfs.sh
```

Enthalten sind nur notwendige Dateien: Modell, Target-Skripte, Target-Requirements und COCO-Daten gemaess Modus.

---

## Schnellstart: NPU-Benchmark auf PHYTEC phyFLEX i.MX95 (verifizierter Pfad)

Diese Schritte sind der getestete, reproduzierbare Weg fuer das BSP `phytec-vision-image` mit
Kernel 6.12.34 und eIQ Neutron SDK 3.1.0.

### Voraussetzungen

| Was | Wo herunterladen |
|---|---|
| eIQ Neutron SDK 3.1.0 | NXP.com → `EIQ-NEUTRON-SDK-3.1.0-LIN.zip` (Account erforderlich) |
| PHYTEC BSP ALPHA2 | `phytec-vision-image` auf dem Board geflasht |
| COCO val2017 | `./scripts/01_download_coco.sh` |
| YOLOv8s TFLite INT8 | Bereits unter `models/neutron/yolov8s_1024_imx95_neutron_int8_sdk310.tflite` |

### Schritt 1 — env.sh anpassen

```bash
cp env.sh.example env.sh
# TARGET_IP, TARGET_USER, TARGET_DIR setzen
```

### Schritt 2 — Neutron-Runtime auf das Board deployen

```bash
SDK=/pfad/zu/eiq-neutron-sdk-linux-3.1.0
TARGET_IP=<BOARD_IP>

scp "${SDK}/target/imx95/imx95/NeutronFirmware.elf"      root@${TARGET_IP}:/lib/firmware/
scp "${SDK}/target/imx95/imx95/libNeutronDriver.so"       root@${TARGET_IP}:/usr/lib/
scp "${SDK}/target/imx95/delegate/libneutron_delegate.so" root@${TARGET_IP}:/usr/lib/
ssh root@${TARGET_IP} ldconfig
```

### Schritt 3 — GO/NO_GO pruefen

```bash
./scripts/19_npu_gonogo_check.sh
# Erwartetes Ergebnis: RESULT=GO
```

### Schritt 4 — COCO-Bilder + Skripte uebertragen

```bash
TARGET_DIR=/opt/imx95-yolov8-benchmark
rsync -a data/coco/val2017/ root@${TARGET_IP}:${TARGET_DIR}/data/coco/val2017/
scp data/coco/annotations/instances_val2017.json root@${TARGET_IP}:${TARGET_DIR}/data/coco/
scp scripts/08b_run_inference_npu_tflite.py       root@${TARGET_IP}:${TARGET_DIR}/scripts/
scp scripts/09_eval_coco.py                       root@${TARGET_IP}:${TARGET_DIR}/scripts/
scp models/neutron/yolov8s_1024_imx95_neutron_int8_sdk310.tflite \
    root@${TARGET_IP}:${TARGET_DIR}/models/neutron/yolov8s_1024_imx95_neutron_int8.tflite
```

### Schritt 5 — Benchmark ausfuehren

```bash
ssh root@${TARGET_IP} \
  "cd ${TARGET_DIR} && WARMUP=5 RESULT_TAG=npu_full python3 scripts/08b_run_inference_npu_tflite.py"
```

Laeuft ca. 25 Minuten. Fortschrittsausgabe alle 100 Bilder.

### Schritt 6 — Ergebnisse holen und mAP berechnen

```bash
mkdir -p results/npu_full
scp "root@${TARGET_IP}:${TARGET_DIR}/results/coco_detections_npu_full.json" results/npu_full/
scp "root@${TARGET_IP}:${TARGET_DIR}/results/timing_breakdown_npu_full.csv"  results/npu_full/

# mAP auf dem Host auswerten (pycocotools benoetigt):
python3 -m pip install pycocotools -q
IMX95_BENCH_ROOT=. RESULT_TAG=npu_full python3 scripts/09_eval_coco.py
```

### Benchmark-Ergebnisse (verifiziert, 11. Mai 2026)

| Metrik | Wert |
|---|---|
| mAP@50:95 | **0.335** |
| mAP@50 | **0.502** |
| Avg NPU invoke | **145.5 ms** |
| FPS (end-to-end) | **3.38** |
| Images | 5000 (COCO val2017) |
| Board | PHYTEC phyFLEX i.MX95, Kernel 6.12.34 |
| Delegate | NeutronDelegate v1.0.0-d98743a7 |
| SDK | eIQ Neutron 3.1.0 |
