# AnalysisApp — ML Training Pipeline

Tools for training, evaluating, and deploying the 28-class automatic modulation recognition (AMR) models used by AnalysisApp's ONNX fast-path classifier.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt    # torch, onnx, onnxruntime-gpu, scikit-learn, …
```

GPU training requires CUDA 11+ and a matching `torch` wheel (already in `requirements.txt`).

## Models

| File | Input | Output | Notes |
|---|---|---|---|
| `models/amr_cnn_24class.best.pt` | — | — | PyTorch checkpoint (28 classes despite filename) |
| `models/amr_cnn_24class.onnx` | `(B,2,512)` float32 IQ | `(B,28)` logits | Deployed to AnalysisApp |
| `models/dae_iq.pt` / `.onnx` | `(B,2,512)` noisy IQ | `(B,2,512)` clean IQ | Denoising autoencoder |
| `models/amr_low_snr_denoised.onnx` | `(B,2,512)` noisy IQ | `(B,28)` logits | DAE → classifier chain for SNR < 5 dB |

## Quickstart: fix QAM classes (GPU)

```bash
# 1. Generate QAM-focused data (post-AFC impairments + multipath, ~10 min CPU)
python generate_qam.py --out data/qam_heavy.npz --n 8000

# 2. Fine-tune + rebuild DAE chain (all-in-one, ~2 hours on RTX 2060 Super)
./run_qam_fix.sh
```

`run_qam_fix.sh` blocks until `qam_heavy.npz` exists, then runs `train.py` with focal loss and 4× QAM oversampling, then rebuilds `amr_low_snr_denoised.onnx`.

## Scripts

### `generate_v3.py` — 28-class synthetic dataset

```bash
python generate_v3.py --out data/synth_v3.npz --n 2500 --len 512
# 28 classes × 9 SNR steps × 2500 = 630 000 samples
```

Generates RRC pulse-shaped digital modes, FM/AM variants, OFDM, CSS, LFM, TONE with PlutoSDR-realistic impairments (IQ imbalance, DC offset, frequency offset ±2% SR, phase noise).

### `generate_qam.py` — QAM-focused dataset

```bash
python generate_qam.py --out data/qam_heavy.npz --n 8000 --snr-min -5
# QAM16/32/64/256 × 8 SNR steps × 8000 = 256 000 samples
```

Key difference from `generate_v3`: frequency offset is ±0.2% SR (post-AFC residual, 10× smaller) so the constellation stays coherent over 512 samples. Also adds multipath (1–3 taps) whose ISI signature is QAM-order-specific.

### `train.py` — Train / fine-tune a ResNet classifier

```bash
python train.py \
    --npz data/synth_v3.npz data/qam_heavy.npz \
    --max-per-class 30000 \
    --model resnet --epochs 20 --batch 512 --lr 2e-4 \
    --cuda \
    --focal 2.0 \           # focal loss γ=2 (focuses on hard QAM confusions)
    --boost-qam 4 \         # 4× QAM oversampling via WeightedRandomSampler
    --resume models/amr_cnn_24class.best.pt \
    --out models/amr_cnn_24class.onnx
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--focal GAMMA` | off | Use focal loss instead of cross-entropy |
| `--boost-qam N` | off | Oversample QAM classes N× via `WeightedRandomSampler` |
| `--resume PT` | none | Load checkpoint; shape-mismatched layers (e.g. head on class-count change) are re-initialised |
| `--model` | `resnet` | `cnn` / `resnet` / `fusion` |

### `train_dae.py` — Train the denoising autoencoder

```bash
python train_dae.py --npz data/synth_v3.npz --epochs 30 --cuda
```

Outputs `models/dae_iq.pt` and `models/dae_iq.onnx`.

### `evaluate.py` / `eval_holdout.py` — Evaluation

```bash
python evaluate.py --model models/amr_cnn_24class.onnx \
    --npz data/v3_holdout.npz data/v3_holdout_impaired.npz
```

### `watch_finetune.sh` — Post-training automation

Polls a training log for completion, then rebuilds the DAE chain and optionally restarts `sdr-analysis`. Run in the background before kicking off a long training job:

```bash
nohup ./watch_finetune.sh >> /tmp/watch_finetune.log 2>&1 &
```

## DAE chain rebuild

After retraining the classifier, rebuild the DAE → classifier merged model:

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
print("Chain OK")
EOF
```

The `watch_finetune.sh` and `run_qam_fix.sh` scripts do this automatically.

## Class list (28 classes)

`16ASK`, `16PSK`, `32PSK`, `4ASK`, `4FSK`, `8FSK`, `8PSK`, `AM_DSB`, `AM_DSB_SC`, `AM_SSB_LSB`, `AM_SSB_USB`, `BPSK`, `CSS`, `FM_NB`, `FM_WB`, `FSK`, `GFSK`, `GMSK`, `LFM`, `MSK`, `OFDM`, `OOK`, `QAM16`, `QAM256`, `QAM32`, `QAM64`, `QPSK`, `TONE`
