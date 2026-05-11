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
