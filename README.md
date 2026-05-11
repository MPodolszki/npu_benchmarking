# YOLOv8s NPU-Benchmark: PHYTEC phyFLEX i.MX95

YOLOv8s INT8 auf dem NXP i.MX95 Neutron-S NPU — vollständige COCO val2017 Auswertung (5000 Bilder).

## Ergebnisse

| Metrik | Wert |
|---|---|
| mAP@50:95 | **0.335** |
| mAP@50 | **0.502** |
| Avg NPU invoke | **145.5 ms** |
| FPS (end-to-end) | **3.38** |
| Board | PHYTEC phyFLEX i.MX95, Kernel 6.12.34 |
| Delegate | NeutronDelegate v1.0.0-d98743a7 |
| SDK | eIQ Neutron 3.1.0 |
| Modell | YOLOv8s INT8 1024×1024 |

Vollständige Ergebnisse: [`results/npu_full/metrics_npu_full.json`](results/npu_full/metrics_npu_full.json)

---

## Benchmark auf dem Board ausführen

### Voraussetzungen

- PHYTEC phyFLEX i.MX95 mit `phytec-vision-image` (BSP ALPHA2, Kernel 6.12.34)
- eIQ Neutron SDK 3.1.0 — Download auf [nxp.com](https://www.nxp.com) (Account erforderlich, `EIQ-NEUTRON-SDK-3.1.0-LIN.zip`)
- COCO val2017 lokal vorhanden (→ `./scripts/01_download_coco.sh`)

### Schritt 1 — Verbindung konfigurieren

```bash
cp env.sh.example env.sh
# TARGET_IP, TARGET_USER und TARGET_DIR in env.sh setzen
```

### Schritt 2 — Neutron-Runtime auf das Board deployen

```bash
SDK=/pfad/zu/eiq-neutron-sdk-linux-3.1.0
source env.sh

scp "${SDK}/target/imx95/imx95/NeutronFirmware.elf"      root@${TARGET_IP}:/lib/firmware/
scp "${SDK}/target/imx95/imx95/libNeutronDriver.so"       root@${TARGET_IP}:/usr/lib/
scp "${SDK}/target/imx95/delegate/libneutron_delegate.so" root@${TARGET_IP}:/usr/lib/
ssh root@${TARGET_IP} ldconfig
```

### Schritt 3 — GO/NO_GO prüfen

```bash
./scripts/19_npu_gonogo_check.sh
# Erwartetes Ergebnis: RESULT=GO
```

### Schritt 4 — Daten und Skripte übertragen

```bash
source env.sh

ssh root@${TARGET_IP} "mkdir -p ${TARGET_DIR}/data/coco/val2017 ${TARGET_DIR}/models/neutron ${TARGET_DIR}/scripts ${TARGET_DIR}/results"

rsync -a data/coco/val2017/                    root@${TARGET_IP}:${TARGET_DIR}/data/coco/val2017/
scp data/coco/annotations/instances_val2017.json   root@${TARGET_IP}:${TARGET_DIR}/data/coco/
scp scripts/08b_run_inference_npu_tflite.py        root@${TARGET_IP}:${TARGET_DIR}/scripts/
scp scripts/09_eval_coco.py                        root@${TARGET_IP}:${TARGET_DIR}/scripts/
scp models/neutron/yolov8s_1024_imx95_neutron_int8_sdk310.tflite \
    root@${TARGET_IP}:${TARGET_DIR}/models/neutron/yolov8s_1024_imx95_neutron_int8.tflite
```

### Schritt 5 — Benchmark ausführen (~25 Min.)

```bash
source env.sh
ssh root@${TARGET_IP} \
  "cd ${TARGET_DIR} && WARMUP=5 RESULT_TAG=npu_full python3 scripts/08b_run_inference_npu_tflite.py"
```

### Schritt 6 — Ergebnisse holen und mAP berechnen

```bash
source env.sh
mkdir -p results/npu_full
scp "root@${TARGET_IP}:${TARGET_DIR}/results/coco_detections_npu_full.json" results/npu_full/
scp "root@${TARGET_IP}:${TARGET_DIR}/results/timing_breakdown_npu_full.csv"  results/npu_full/

python3 -m pip install pycocotools -q
IMX95_BENCH_ROOT=. RESULT_TAG=npu_full python3 scripts/09_eval_coco.py
```

---

## Skript-Referenz

| Skript | Zweck |
|---|---|
| `00_host_setup_manjaro.sh` | Richtet die Python-Virtualenv und alle Host-Abhängigkeiten ein |
| `01_download_coco.sh` | Lädt COCO val2017 (Bilder + Annotationen) herunter |
| `02_download_model.sh` | Lädt das YOLOv8s PyTorch-Modell herunter |
| `03_validate_pytorch.sh` | Validiert das Modell kurz mit PyTorch auf dem Host |
| `04_export_onnx.sh` | Exportiert YOLOv8s als ONNX (für ONNX-Runtime-Pfad) |
| `04b_export_tflite.sh` | Exportiert YOLOv8s als TFLite INT8 (Eingabe für Neutron-Converter) |
| `05_check_onnx.sh` | Prüft das ONNX-Modell auf Korrektheit (Input/Output-Shapes) |
| `06_transfer_to_target.sh` | Überträgt ONNX-Modell, Daten und Skripte via SCP auf das Board |
| `07_target_setup.sh` | Installiert Python-Abhängigkeiten direkt auf dem Board |
| `08_run_inference_imx95.py` | ONNX-Runtime-Inferenz auf dem Board (CPU-Referenzpfad) |
| `08b_run_inference_npu_tflite.py` | **NPU-Inferenz** via TFLite + Neutron-Delegate auf dem Board |
| `09_eval_coco.py` | Berechnet mAP@50:95 aus den Detection-JSON-Ergebnissen (Host) |
| `10_collect_thermal.sh` | Liest CPU/SoC-Temperatur während eines Laufs auf dem Board aus |
| `11_generate_report.sh` | Erstellt einen Textreport aus allen vorhandenen Ergebnisdateien |
| `12_host_compare_cpu_gpu.sh` | ONNX-Runtime CPU vs. GPU Vergleich auf dem Host |
| `13_run_pytorch_inference_host.py` | PyTorch-Inferenz auf dem Host (CPU oder CUDA) |
| `14_host_compare_ort_vs_pytorch_gpu.sh` | ONNX-Runtime-GPU vs. PyTorch-CUDA Vergleich auf dem Host |
| `15_make_target_bundle.sh` | Packt ein minimales Deployment-Bundle (smoke=100 Bilder / full=5000) |
| `16_deploy_bundle_to_sshfs.sh` | Kopiert das Bundle direkt in ein gemountetes SSHFS-Verzeichnis |
| `17_host_prepare_neutron_sdk.sh` | Entpackt und validiert das eIQ Neutron SDK auf dem Host |
| `18_host_convert_for_neutron.sh` | Konvertiert ein TFLite INT8-Modell mit dem Neutron-Converter für i.MX95 |
| `19_npu_gonogo_check.sh` | Prüft ob die NPU korrekt delegiert — liefert `RESULT=GO` oder `RESULT=NO_GO` |
