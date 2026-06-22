# AnalysisApp — RF Signal Auto-Identification Service

Subscribes to IQ streams from **SdrResourceManager**, performs five-layer automatic modulation recognition (AMR), and publishes signal identifications over AMQP 1.0.

## Architecture

```
AcquisitionApp ──RF_DETECTION (+ IQ snapshot b64)──► rf.detections
                                                            │
                        ┌───────────────────────────────────┘
                        │
              ┌─────────▼──────────────────────────────────────────────┐
              │  AnalysisService — 2 parallel worker threads            │
              │                                                          │
              │  ① Fast path  (ONNX loaded + snapshot present, ~5 ms)  │
              │    ONNX on embedded 1 024-sample IQ snapshot             │
              │    → publish immediately if confidence ≥ 75%             │
              │                                                          │
              │  ② Slow path  (~80 ms, persistent IqTaskChannel)        │
              │    IqTaskChannel.exchange() → NARROWBAND task            │
              │    (persistent AMQP conn — no per-call reconnect cost)   │
              │    → collect 65 536 samples at 2 MSPS (~32 ms)          │
              │    → FeatureExtractor + ONNX (parallel)                  │
              │    → persist to PostgreSQL analysis_results              │
              │    → publish result                                       │
              └────────────────────────────────────────────────────────-┘
                        │
               AMQP ──► rf.analysis topic
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

Latency (C++ engine, 2 worker threads):

| Path | Trigger | Total latency |
|------|---------|---------------|
| Fast (ONNX from snapshot, confidence ≥ 75%) | ~65% of signals | **~5 ms** |
| Slow (full collection + features) | ~35% of signals | **~80 ms** |

The fast path fires before any SDR re-acquisition — ONNX runs on the 1 024-sample IQ snapshot embedded in the `RF_DETECTION` AMQP message. The slow path uses a persistent `IqTaskChannel` (connects once, reused across all collections) to submit a NARROWBAND task in ~16 ms, then collects 65 536 samples at 2 MSPS (~32 ms) and runs feature extraction + ONNX in parallel. Two workers run slow-path collections concurrently, doubling throughput within the 3 s analysis window.

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
| spdlog | ≥ 1.14.1 | auto-fetched by CMake (v1.14.1) | `apt install libspdlog-dev` |
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

The ML pipeline produces three ONNX artefacts:

| File | Description |
|---|---|
| `models/dae_iq.pt` / `.onnx` | Denoising autoencoder — removes hardware impairments from raw IQ |
| `models/amr_cnn_24class.onnx` | 28-class AMR ResNet classifier (opset 17) — macro val_acc 0.743, QAM256 recall 53% |
| `models/amr_low_snr_denoised.onnx` | DAE → classifier chain; used when SNR < 5 dB |

**Training the classifier:**

```bash
# Generate QAM-focused data with post-AFC impairments + multipath
python generate_qam.py --out data/qam_heavy.npz --n 8000

# Fine-tune on GPU with focal loss (γ=2) and 4× QAM oversampling
python train.py --npz data/synth_v3.npz data/qam_heavy.npz \
    --model resnet --epochs 20 --batch 512 --lr 2e-4 --cuda \
    --focal 2.0 --boost-qam 4 \
    --resume models/amr_cnn_24class.best.pt \
    --out models/amr_cnn_24class.onnx
```

**Rebuilding the DAE chain** (run after retraining the classifier):

```bash
python - << 'EOF'
import onnx
from onnx import compose, version_converter
dae = onnx.load("models/dae_iq.onnx")
clf = onnx.load("models/amr_cnn_24class.onnx")
clf_opset = clf.opset_import[0].version
if dae.opset_import[0].version != clf_opset:
    dae = version_converter.convert_version(dae, clf_opset)
merged = compose.merge_models(
    dae, compose.add_prefix(clf, prefix="clf_", rename_edges=True),
    io_map=[("iq_clean", "clf_iq_input")])
onnx.save(merged, "models/amr_low_snr_denoised.onnx")
EOF
```

Or use `run_qam_fix.sh` which generates data, trains, and rebuilds the chain automatically.

---

## Testing

Run the 33 unit tests in a container — no broker, SDR, or local deps needed:

```bash
podman build --target test -t sdr-analysis:test .
```

The `Containerfile` is multi-stage (CentOS Stream 10). `--target test` builds the binary and runs `ctest` inside the builder, exiting 0 on success.

Tests cover: `FeatureExtractor` (cumulant values, OFDM CP detector, SNR, chirp), `ModulationClassifier` (all modulation branches, PSK discriminants), `ProtocolMapper` (hypothesis ranking, frequency database), `AnalysisEngine` (fast-path `analyzeSnapshot()`, `onnxLoaded()` when no model configured).

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

# 3. With ONNX Runtime ML support + PostgreSQL persistence
cmake -B build -DWITH_ONNX=ON \
      -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.17.3 \
      -DWITH_DB=ON
cmake --build build --parallel

# PostgreSQL schema (run once):
psql -U sdr -d sdr_scanner -f schema/init.sql

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

### Deploy (full stack)

```bash
cd ../SdrResourceManager
./k8s/deploy.sh            # prompts for passwords interactively
./k8s/deploy.sh --dry-run  # preview

# Or non-interactive:
AMQP_PASSWORD=s3cr3t DB_PASSWORD=s3cr3t ./k8s/deploy.sh
```

The script creates the `sdr-credentials` Secret, then applies Namespace →
PostgreSQL → Artemis + controller → AcquisitionApp → AnalysisApp → signal-logger.

### Apply just this manifest

Requires `sdr-credentials` Secret to exist first
(see `SdrResourceManager/k8s/secrets.yaml`):

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl get pods -n sdr-system -l app=sdr-analysis
kubectl logs -n sdr-system -l app=sdr-analysis -f
```

### How credentials work

An initContainer renders `analysis.xml` from the ConfigMap template, substituting
`${AMQP_PASSWORD}` and `${DB_PASSWORD}` from the `sdr-credentials` Secret into an
emptyDir volume. The main container reads from the emptyDir — the ConfigMap never
contains real passwords.

### Update config only

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

Each pod announces its own `POD_IP`; tasks are distributed across pods via AMQP.
Each pod runs 2 internal worker threads, so 4 replicas = 8 concurrent slow-path collections.

---

## Configuration (`config/analysis.xml`)

| Field | Default | Description |
|---|---|---|
| `scanner_id` | `analysis-0` | Instance identifier (must be unique per pod) |
| `streaming_ip` | `127.0.0.1` | IP to bind UDP IQ receive socket (`0.0.0.0` in containers) |
| `amqp/url` | — | AMQP 1.0 broker URL |
| `collector/analysis_sample_rate_sps` | `2000000` | IQ sample rate expected from SdrResourceManager |
| `collector/collect_samples` | `65536` | Samples per slow-path collection window (32 ms at 2 MSPS). The fast path uses the embedded `iq_snapshot` instead and does not collect. |
| `collector/analysis_timeout_ms` | `5000` | Max wait for IQ collection |
| `engine/fft_size` | `4096` | FFT size for spectral features |
| `engine/snr_threshold_db` | `5.0` | Minimum SNR to attempt classification |
| `engine/rank` | `1` | Task priority rank when requesting IQ from SdrResourceManager |
| `engine/onnx/model_path` | *(empty)* | Path to `.onnx` model file — omit to disable ONNX |
| `engine/onnx/classes_path` | *(empty)* | Path to `.classes.json` label file |
| `engine/onnx/use_gpu` | `true` | Try CUDA execution provider, fall back to CPU |
| `engine/onnx/use_tensorrt` | `true` | Try TensorRT EP before CUDA EP (3× faster on RTX; requires `use_gpu=true`) |
| `engine/onnx/tensorrt_fp16` | `true` | Use fp16 Tensor Core kernels (RTX 2060+) |
| `engine/onnx/tensorrt_cache_mb` | `128` | TRT engine compile cache size (MB) |
| `engine/onnx/input_len` | `1024` | IQ samples per inference window |
| `engine/onnx/max_batch` | `8` | Max signals batched per `classifyBatch()` call |
| `engine/onnx/fallback_confidence_threshold` | `0.60` | Run ONNX in slow path when rule confidence is below this value |
| `engine/onnx/fallback_on_unknown` | `true` | Always run ONNX slow path when rule-based returns UNKNOWN |
| `database/host` | `localhost` | PostgreSQL host — omit `<database>` block to disable persistence |
| `database/port` | `5432` | PostgreSQL port |
| `database/name` | `sdr_scanner` | Database name (must have `analysis_results` table from `schema/init.sql`) |
| `database/user` | `sdr` | PostgreSQL user |
| `database/password` | — | PostgreSQL password |

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

### Getting trained models without training them yourself

Trained models aren't tracked in git (`.pt` checkpoints are gitignored;
`.onnx` files are large binaries too) and there's no automatic deploy step
that pushes them anywhere — they live in `tools/ml/models/` on whichever
machine trained them until someone explicitly publishes a snapshot.

```bash
# Publish everything currently in tools/ml/models/ as a new release:
tools/ml/publish_models_release.sh

# Download the latest published snapshot instead of training:
gh release list --repo OpenRFStack/AnalysisApp | grep ^models-
gh release download <tag> --repo OpenRFStack/AnalysisApp --dir tools/ml/models/
```

Releases are tagged `models-<timestamp>` — deliberately separate from
`latest-main`, which is the CI-built RPM binaries the product containers
pull at startup (see `SdrScripts/deploy/*/entrypoint.sh`), so a model drop
never collides with or gets mistaken for a binary build. There's no fixed
versioning scheme beyond the timestamp yet; check the release notes (a
file listing with sizes) or `tools/ml/*.classes.json` to see what's in a
given snapshot.

---

## Classification Path

Two paths depending on whether the incoming `RF_DETECTION` message carries an `iq_snapshot`:

```
RF_DETECTION received
        │
        ├─ iq_snapshot present AND ONNX loaded?
        │         │
        │         ▼ FAST PATH (~5 ms total)
        │   ┌─────────────────────┐
        │   │  ONNX on snapshot   │  0.3 ms TRT / 2 ms CUDA EP
        │   │  (1 024 samples)    │
        │   └────────┬────────────┘
        │            │ confidence ≥ 75%?
        │            ├── yes → publish immediately
        │            └── no  → fall through to slow path
        │
        └─ SLOW PATH (~165 ms total)
                  │
                  ▼
        ┌─────────────────────────────────────────────────┐
        │  Collect 65 536 samples at 2 MSPS (32 ms)       │
        └─────────────────────────┬───────────────────────┘
                                  │  (ONNX also runs in
                                  │   parallel on this IQ)
                    ┌─────────────▼─────────────┐
                    │  Rule-based classifier     │ ~20–50 ms
                    │  FM / AM / OFDM / FSK / CSS│
                    └──────────────┬────────────-┘
                                   │ rule_confidence
                    ┌──────────────▼────────────-┐
                    │  < threshold OR "UNKNOWN"?  │
                    └──────────────┬────────────-┘
                          yes      │       no
               ┌──────────────────┘   └─────────────────────┐
               ▼                                            ▼
    ┌─────────────────────┐                    ┌───────────────────────┐
    │  ONNX result        │ (already done,     │  Rule result used     │
    │  (parallel future)  │  just .get())      │  onnx_used = false    │
    └──────────┬──────────┘                    └───────────────────────┘
               │
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
| QAM16/32/256 | 0.55 | When threshold > 0.55 (QAM256 recall ~53%) |
| QAM64 | 0.45 | When threshold > 0.45 (hard to separate from QAM32/256) |
| UNKNOWN | 0.00 | Always (if `fallback_on_unknown=true`) |

ONNX results and the rule path that fired are reported in every published JSON:
```json
"classification_path": {
  "rule_confidence": 0.65,
  "onnx_used": true,
  "onnx_confidence": 0.91,
  "fast_path": true
}
```

`fast_path: true` means the result came from the embedded IQ snapshot (~5 ms) without re-acquiring the SDR. This field is also stored in the `fast_path` column of the `analysis_results` PostgreSQL table and exposed in the `classification_path_stats` view.

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
