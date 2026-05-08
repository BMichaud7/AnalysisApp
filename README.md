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

## Test Harness

Validate the feature extractor and classifier against synthetic signals:

```bash
cd tools/test_harness
python harness.py
```

Supported test signals: BPSK, QPSK, FM_WB, AM_DSB_LC, OFDM, CSS/LoRa, BFSK, AIS.
