# AnalysisApp — RF Signal Auto-Identification Service

Subscribes to IQ streams from **SdrResourceManager**, performs five-layer automatic modulation recognition (AMR), and publishes signal identifications over AMQP 1.0.

## Architecture

```
SdrResourceManager ──UDP IQ──► AnalysisApp ──AMQP──► rf.analysis topic
                                    │
                  FeatureExtractor (FFT, cumulants, OFDM CP, FHSS)
                  ModulationClassifier (analog / digital / structure)
                  ProtocolMapper (59-entry protocol database)
                  OnnxClassifier (optional — RadioML-trained CNN)
```

## Supported Modulations & Protocols

### Analog modulations

| ID | Description |
|---|---|
| `FM_WB` | FM Broadcast — deviation > 50 kHz |
| `FM_NB` | Narrow-band FM — land mobile, PMR |
| `AM_DSB_LC` | AM double-sideband with carrier |
| `AM_DSB_SC` | AM double-sideband suppressed carrier |
| `SSB_USB` / `SSB_LSB` | Single sideband — HF voice |
| `PM` | Phase modulation |
| `CW` | Morse / OOK burst |

### Digital modulations

| ID | Description |
|---|---|
| `BPSK` | Binary PSK — GPS, satellite links |
| `QPSK` | Quad PSK — DVB, Iridium, Meteor-M |
| `8PSK` | 8-phase PSK — DVB-S2, APCO P25 |
| `QAM16` / `QAM32` / `QAM64` / `QAM256` | QAM — cable, LTE downlink |
| `FSK` | Binary / M-ary FSK — POCSAG, AIS |
| `MSK/GMSK` | Minimum shift keying — GSM, Bluetooth |
| `OFDM` | Multi-carrier — LTE, Wi-Fi, DAB |
| `CSS` | Chirp spread spectrum — LoRa |
| `FHSS` | Frequency hopping — Bluetooth Classic |
| `M-ASK/OOK` | Amplitude shift keying — TPMS, key fobs |

### Structure detectors

These run on top of modulation and feed into the protocol mapper:

- **Burst / TDMA** — duty-cycle and period estimation
- **FHSS** — hop-rate detection from centroid variance
- **OFDM cyclic-prefix** — FFT-size and CP-ratio estimation
- **Chirp** — linear frequency sweep rate (CSS / FMCW radar)

### Protocol hypotheses (59 entries)

Matched by centre frequency, modulation, bandwidth, and structure flags:

| Category | Systems |
|---|---|
| Broadcast | FM Broadcast, AM Broadcast, DAB, DVB-T |
| Aviation | VHF voice, ADS-B (1090 MHz), ACARS, VOR/ILS, ATIS |
| Marine | AIS (161/162 MHz), DSC, Marine VHF |
| Amateur | SSB HF, WSPR, FT8, APRS, DMR, D-STAR, CW |
| Public safety | TETRA, APCO P25, DMR, MPT1327, NXDN |
| Cellular | GSM, UMTS/WCDMA, LTE, 5G NR |
| ISM / IoT | LoRa, Wi-Fi 2.4/5 GHz, Bluetooth, Zigbee, Z-Wave, TPMS, ANT+ |
| Satellite | GPS L1, NOAA APT, Meteor LRPT, Iridium, Inmarsat AERO |
| Radar | FMCW 77 GHz, Pulse radar, WSR-88D weather radar |

### Accuracy under hardware impairments

Tested with five hardware profiles, 5 independent noise realisations per signal
(rule-based path only, no ONNX model):

| Profile | Exact | Family |
|---|---|---|
| PlutoSDR (GPSDO / good IQ) | **100%** | **100%** |
| HackRF One | **97%** | **97%** |
| RTL-SDR dongle | **97%** | **100%** |
| Clean lab | 91% | 91% |
| Over-the-Air (multipath) | 71% | 71% |

Per-signal breakdown:

| Signal | Clean | RTL-SDR | HackRF | PlutoSDR | OTA |
|---|---|---|---|---|---|
| BPSK | 80% | **100%** | 80% | **100%** | 0% |
| QPSK | 60% | 80% | **100%** | **100%** | 0% |
| FM_NB | **100%** | **100%** | **100%** | **100%** | **100%** |
| AM_DSB_LC | **100%** | **100%** | **100%** | **100%** | **100%** |
| OFDM | **100%** | **100%** | **100%** | **100%** | **100%** |
| FSK | **100%** | **100%** | **100%** | **100%** | **100%** |
| CSS/LoRa | **100%** | **100%** | **100%** | **100%** | **100%** |

SNR sensitivity (RTL-SDR impairments + AWGN, 3 realisations × 7 signals):

| SNR | Exact |
|---|---|
| −10 dB | 10% |
| −5 dB | 10% |
| 0 dB | 10% |
| 5 dB | 29% |
| 10 dB | 62% |
| 15 dB | 81% |
| 20 dB | 90% |
| 30 dB | 90% |

**Known limitation**: BPSK and QPSK under OTA multipath drop to 0% on the
rule-based path — this is the gap the ONNX fallback is designed to fill.
FM/AM/OFDM/FSK/CSS are 100% across all profiles including OTA.

Latency: 5–7 ms per classification window (Python harness, single-threaded).
C++ engine is ~1–3 ms on the same hardware.

---

## System Requirements

| Dependency | Version | CentOS/RHEL | Ubuntu |
|---|---|---|---|
| CMake | ≥ 3.20 | `dnf install cmake` | `apt install cmake` |
| GCC | ≥ 12 (C++20) | `dnf install gcc-c++` | `apt install g++` |
| FFTW3 (single) | ≥ 3.3 | `dnf install fftw-devel` | `apt install libfftw3-dev` |
| qpid-proton-cpp | ≥ 0.39 | `dnf install qpid-proton-cpp-devel` | `apt install libqpid-proton-cpp12-dev` |
| tinyxml2 | ≥ 9 | `dnf install tinyxml2-devel` | `apt install libtinyxml2-dev` |
| libuuid | any | `dnf install libuuid-devel` | `apt install uuid-dev` |
| nlohmann/json | ≥ 3.11 | auto-fetched by CMake | `apt install nlohmann-json3-dev` |
| spdlog | ≥ 1.12 | auto-fetched by CMake | `apt install libspdlog-dev` |
| **SdrTaskApi** | sibling | `git clone https://github.com/BMichaud7/SdrTaskApi.git ../SdrTaskApi` | same |

### Optional: ONNX Runtime (ML classification)

Download a release from https://github.com/microsoft/onnxruntime/releases:

```bash
# CPU variant
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/\
onnxruntime-linux-x64-1.17.3.tgz | tar xz -C /opt

# GPU/CUDA variant
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/\
onnxruntime-linux-x64-gpu-1.17.3.tgz | tar xz -C /opt
```

### Python test harness (tools/test_harness)

```bash
pip install numpy scipy matplotlib h5py tqdm scikit-learn
```

### ML training (tools/ml)

```bash
pip install -r tools/ml/requirements.txt   # torch, onnx, onnxruntime-gpu, …
```

---

## Testing

Run the 27 unit tests in a container — no broker, SDR, or local deps needed:

```bash
podman build --target test -t sdr-analysis:test .
```

The `Containerfile` is multi-stage (CentOS Stream 10). `--target test` builds the binary and runs `ctest` inside the builder, exiting 0 on success.

Tests cover: `FeatureExtractor` (cumulant values, OFDM CP detector, SNR, chirp), `ModulationClassifier` (all modulation branches, PSK discriminants), `ProtocolMapper` (hypothesis ranking, frequency database).

Run the Python harness against hardware impairment profiles:

```bash
cd tools/test_harness
python verify_realworld.py              # all 5 profiles
python verify_realworld.py --stress-snr # SNR sweep -10 to +30 dB
```

---

## Build from Source

```bash
# 1. Clone with SdrTaskApi sibling
git clone https://github.com/BMichaud7/AnalysisApp.git
git clone https://github.com/BMichaud7/SdrTaskApi.git

# 2. Build (Release)
cd AnalysisApp
./build.sh

# 3. With ONNX Runtime ML support
cmake -B build -DWITH_ONNX=ON \
      -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.17.3
cmake --build build --parallel

# 4. Run tests
./build.sh --tests
```

Binary: `build/sdr_analysis`

---

## Container

### Build

```bash
podman build -t sdr-analysis:2.3.0 .
# or
docker build -t sdr-analysis:2.3.0 .
```

### Run locally

```bash
podman run --rm \
  -v ./config/analysis.xml:/etc/sdr-analysis/analysis.xml:ro \
  -p 20000-20099:20000-20099/udp \
  -e SDR_LOG_LEVEL=debug \
  sdr-analysis:2.3.0
```

### Push to registry

```bash
podman tag sdr-analysis:2.3.0 ghcr.io/BMichaud7/sdr-analysis:2.3.0
podman push ghcr.io/BMichaud7/sdr-analysis:2.3.0
```

---

## systemd

### Install

```bash
# 1. Install binary and config
sudo cmake --install build
# or manually:
sudo cp build/sdr_analysis /usr/local/bin/
sudo mkdir -p /etc/sdr-analysis
sudo cp config/analysis.xml /etc/sdr-analysis/

# 2. Create service user
sudo groupadd -r sdranalysis
sudo useradd -r -g sdranalysis -s /sbin/nologin sdranalysis
sudo mkdir -p /var/lib/sdr-analysis
sudo chown sdranalysis:sdranalysis /var/lib/sdr-analysis

# 3. Install and enable unit
sudo cp deploy/systemd/sdr-analysis.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sdr-analysis
```

### Environment overrides

Create `/etc/sdr-analysis/env`:

```
SDR_LOG_LEVEL=debug
```

Then `sudo systemctl restart sdr-analysis`.

### Logs

```bash
journalctl -u sdr-analysis -f
```

---

## k3s / Kubernetes

### Prerequisites

- k3s cluster with `sdr-system` namespace (created by SdrResourceManager manifests)
- Container image pushed to registry
- ActiveMQ broker running (`activemq-service.sdr-system` in-cluster)

### Deploy

```bash
# 1. Update the image tag in deploy/k8s/deployment.yaml
#    image: ghcr.io/BMichaud7/sdr-analysis:2.3.0

# 2. Set AMQP credentials
kubectl create secret generic analysis-amqp-credentials \
  -n sdr-system \
  --from-literal=username=sdr_ctrl \
  --from-literal=password=YOUR_PASSWORD

# 3. Apply manifests
kubectl apply -f deploy/k8s/deployment.yaml

# 4. Verify
kubectl get pods -n sdr-system -l app=sdr-analysis
kubectl logs -n sdr-system -l app=sdr-analysis -f
```

### Update config

```bash
kubectl create configmap analysis-config -n sdr-system \
  --from-file=analysis.xml=config/analysis.xml \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deployment/sdr-analysis -n sdr-system
```

### Scaling

AnalysisApp is stateless — scale replicas freely:

```bash
kubectl scale deployment sdr-analysis -n sdr-system --replicas=4
```

Each pod announces its own `POD_IP` to SdrResourceManager; tasks are distributed across pods via AMQP.

---

## Configuration (`config/analysis.xml`)

| Field | Default | Description |
|---|---|---|
| `scanner_id` | `analysis-0` | Instance identifier (must be unique per pod) |
| `streaming_ip` | `127.0.0.1` | IP to bind UDP IQ receive socket (`0.0.0.0` in containers) |
| `amqp/url` | — | AMQP 1.0 broker URL |
| `collector/analysis_sample_rate_sps` | `2000000` | IQ sample rate expected from SdrResourceManager |
| `collector/collect_samples` | `1000000` | Samples per analysis window |
| `collector/analysis_timeout_ms` | `5000` | Max wait for IQ collection |
| `engine/fft_size` | `4096` | FFT size for spectral features |
| `engine/snr_threshold_db` | `5.0` | Minimum SNR to attempt classification |
| `engine/rank` | `1` | Task priority rank when requesting IQ from SdrResourceManager |
| `engine/onnx/model_path` | *(empty)* | Path to `.onnx` model file — omit to disable ONNX |
| `engine/onnx/classes_path` | *(empty)* | Path to `.classes.json` label file |
| `engine/onnx/use_gpu` | `true` | Try CUDA execution provider, fall back to CPU |
| `engine/onnx/input_len` | `1024` | IQ samples per inference window |
| `engine/onnx/fallback_confidence_threshold` | `0.60` | Run ONNX when rule confidence is below this value |
| `engine/onnx/fallback_on_unknown` | `true` | Always run ONNX when rule-based returns UNKNOWN |

---

## ML Training

Train a RadioML model and export to ONNX for C++ inference:

```bash
cd tools/ml

# RadioML 2016.10a (11 modulations, 128-sample windows)
python train.py --data /data/RML2016.10a.hdf5 \
                --model resnet --epochs 50 \
                --out models/radioml2016_resnet.onnx

# RadioML 2018.01a with CUDA (24 modulations, 1024-sample windows)
python train.py --data /data/RML2018.01a.hdf5 \
                --model fusion --epochs 100 --cuda \
                --out models/radioml2018_fusion.onnx
```

The exported `.onnx` file and companion `.classes.json` are loaded by `OnnxClassifier` in the C++ service when built with `-DWITH_ONNX=ON`.

---

## Classification Path

```
                    ┌─────────────────────────────┐
IQ buffer ─────────►│  Rule-based classifier       │ always runs (~1–3 ms)
                    │  FM / AM / OFDM / FSK / CSS  │
                    └────────────┬────────────────-┘
                                 │ rule_confidence
                    ┌────────────▼────────────────-┐
                    │  < threshold OR "UNKNOWN"?    │ configurable per XML
                    └────────────┬────────────────-┘
                          yes    │    no
               ┌────────────────┘    └──────────────────────┐
               ▼                                            ▼
    ┌─────────────────────┐                    ┌───────────────────────┐
    │  ONNX classifier    │ (~0.5–2 ms GPU)    │  Rule result used     │
    │  (RadioML CNN)      │                    │  onnx_used = false    │
    └─────────┬───────────┘                    └───────────────────────┘
              │ modulation, onnx_confidence
              ▼
    ┌─────────────────────┐
    │  Protocol mapper    │
    │  (59-entry DB)      │
    └─────────────────────┘
```

Rule confidence thresholds by modulation type:

| Modulation | Rule confidence | Typical ONNX trigger |
|---|---|---|
| OFDM, CSS | 0.95 | Never (structural CP/chirp detector) |
| FHSS | 0.90 | Never (hop detector) |
| FM_WB/NB | 0.90 | Never |
| AM, SSB, CW | 0.85 | Never |
| FSK, MSK | 0.80 | Never |
| BPSK, QPSK, 8PSK | 0.65 | When threshold > 0.65 |
| QAM | 0.60 | When threshold > 0.60 |
| UNKNOWN | 0.00 | Always (if `fallback_on_unknown=true`) |

ONNX results and the rule path that fired are reported in every published JSON:
```json
"classification_path": {
  "rule_confidence": 0.65,
  "onnx_used": true,
  "onnx_confidence": 0.91
}
```

---

## Test Harness

Validate the feature extractor and classifier against synthetic signals:

```bash
cd tools/test_harness
python harness.py
```

Validate against hardware impairment profiles and SNR sweep:

```bash
# All 5 hardware profiles (Clean, RTL-SDR, HackRF, PlutoSDR, OTA)
python verify_realworld.py

# Single profile
python verify_realworld.py --profile rtl_sdr

# Add SNR sweep from −10 to +30 dB
python verify_realworld.py --stress-snr
```

Supported test signals: BPSK, QPSK, FM_NB, AM_DSB_LC, OFDM, FSK, CSS/LoRa.
