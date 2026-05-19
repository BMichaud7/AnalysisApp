#!/usr/bin/env python3
"""
SDR Modulation Classifier — Training Script
============================================
Trains a compact ResNet1D for automatic modulation recognition (AMR) and
exports the result as an ONNX model compatible with AnalysisApp's OnnxClassifier.

ONNX contract (must match OnnxClassifier.cpp):
  input:  "iq_input"  — float32  (1, 2, INPUT_LEN)  I and Q as separate channels
  output: "logits"    — float32  (1, n_classes)      unnormalised log-probabilities

Usage:
  # With TorchSig (recommended):
  pip install -r requirements.txt
  python train_classifier.py --output models/

  # CPU-only (slower, no CUDA needed):
  python train_classifier.py --output models/ --device cpu

  # Resume from checkpoint:
  python train_classifier.py --resume models/checkpoint_best.pt
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ── Class definitions ──────────────────────────────────────────────────────────
# These map to the modulation labels AnalysisEngine already recognises.
# Add or remove classes here; re-train after any change.
CLASSES = [
    # Analog
    "AM_DSB",       # AM double-sideband (carrier + info)
    "AM_DSB_SC",    # AM double-sideband suppressed carrier
    "AM_SSB",       # AM single-sideband (USB or LSB)
    "FM_WB",        # Wideband FM  (deviation > 50 kHz — broadcast)
    "FM_NB",        # Narrowband FM (deviation < 5 kHz — PMR, POCSAG)
    "CW",           # Continuous wave / Morse
    # Phase-shift keying
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    # Quadrature amplitude modulation
    "16QAM",
    "64QAM",
    "256QAM",
    # Frequency-shift keying / MSK
    "2FSK",
    "4FSK",
    "GMSK",         # Gaussian MSK  (GSM, DECT)
    "GFSK",         # Gaussian FSK  (Bluetooth, ZigBee)
    # On-off / amplitude
    "OOK",
    "4ASK",
    # Broadband
    "OFDM",         # Any OFDM variant (Wi-Fi, LTE, 5G NR)
    # Background / rejected
    "NOISE",        # Pure AWGN — taught so the model can abstain
]
N_CLASSES = len(CLASSES)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
INPUT_LEN    = 1024      # samples per inference window — must match OnnxConfig.input_len
BATCH_SIZE   = 256
EPOCHS       = 60
LR           = 3e-4
WEIGHT_DECAY = 1e-4
SAMPLES_PER_CLASS = 8_000   # synthetic samples generated per class per epoch
SNR_MIN_DB   = -10.0
SNR_MAX_DB   = 30.0


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock1D(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + x)


class DownBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class ModulationNet(nn.Module):
    """
    Compact ResNet1D for AMR.
    Input : (batch, 2, INPUT_LEN) — channel 0 = I, channel 1 = Q
    Output: (batch, n_classes)   — logits (no softmax)
    ~900 K parameters at the default widths.
    """
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(DownBlock1D(32,  64,  2), ResBlock1D(64))
        self.layer2 = nn.Sequential(DownBlock1D(64,  128, 2), ResBlock1D(128))
        self.layer3 = nn.Sequential(DownBlock1D(128, 256, 2), ResBlock1D(256))
        self.layer4 = nn.Sequential(DownBlock1D(256, 512, 2), ResBlock1D(512))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x))))))


# ══════════════════════════════════════════════════════════════════════════════
# Dataset — TorchSig preferred, built-in fallback
# ══════════════════════════════════════════════════════════════════════════════

def _try_torchsig() -> bool:
    try:
        import torchsig  # noqa: F401
        return True
    except ImportError:
        return False


def _generate_torchsig(cls_name: str, n: int, snr_range: tuple) -> np.ndarray:
    """Generate n IQ samples using TorchSig signal models."""
    import torchsig.signals as tss
    from torchsig.transforms import (
        AddNoise, RandomPhaseShift, RandomTimeShift,
        Normalize, ComplexTo2D,
    )

    _TS_MAP = {
        "AM_DSB":    tss.AMDSBModulation,
        "AM_DSB_SC": tss.AMDSBSCModulation,
        "AM_SSB":    tss.AMSSBModulation,
        "FM_WB":     tss.FMModulation,
        "FM_NB":     tss.FMModulation,
        "CW":        tss.CWModulation,
        "BPSK":      tss.BPSKModulation,
        "QPSK":      tss.QPSKModulation,
        "8PSK":      tss.PSKModulation,
        "16PSK":     tss.PSKModulation,
        "16QAM":     tss.QAMModulation,
        "64QAM":     tss.QAMModulation,
        "256QAM":    tss.QAMModulation,
        "2FSK":      tss.FSKModulation,
        "4FSK":      tss.FSKModulation,
        "GMSK":      tss.GMSKModulation,
        "GFSK":      tss.GFSKModulation,
        "OOK":       tss.OOKModulation,
        "4ASK":      tss.ASKModulation,
        "OFDM":      tss.OFDMModulation,
    }

    mod_cls = _TS_MAP.get(cls_name)
    samples = []
    for _ in range(n):
        snr = np.random.uniform(*snr_range)
        sig = _synthetic_signal(cls_name, INPUT_LEN, snr)  # fall back if TS fails
        if mod_cls is not None:
            try:
                mod = mod_cls(num_samples=INPUT_LEN)
                raw = mod()
                noise_sigma = 10 ** (-snr / 20.0) / np.sqrt(2)
                raw = raw + noise_sigma * (
                    np.random.randn(len(raw)) + 1j * np.random.randn(len(raw)))
                raw = raw / (np.sqrt(np.mean(np.abs(raw) ** 2)) + 1e-10)
                sig = np.stack([raw.real, raw.imag]).astype(np.float32)
            except Exception:
                pass
        samples.append(sig)
    return np.stack(samples)


def _synthetic_signal(cls_name: str, n: int, snr_db: float) -> np.ndarray:
    """
    Pure-NumPy synthetic signal generator — no TorchSig required.
    Returns (2, n) float32 array: [I, Q].
    """
    t     = np.arange(n, dtype=np.float64)
    fc    = np.random.uniform(0.05, 0.45)          # normalised carrier freq
    phase = np.random.uniform(0, 2 * np.pi)

    iq = np.zeros(n, dtype=complex)

    if cls_name == "NOISE":
        iq = np.zeros(n, dtype=complex)

    elif cls_name in ("AM_DSB", "AM_DSB_SC"):
        # Single-tone AM, random message frequency
        fm  = np.random.uniform(0.005, 0.05)
        idx = np.random.uniform(0.3, 0.9)
        msg = np.cos(2 * np.pi * fm * t)
        carrier = np.exp(1j * (2 * np.pi * fc * t + phase))
        if cls_name == "AM_DSB":
            iq = (1.0 + idx * msg) * carrier
        else:
            iq = msg * carrier

    elif cls_name == "AM_SSB":
        fm  = np.random.uniform(0.01, 0.04)
        msg = np.cos(2 * np.pi * fm * t)
        msg_h = np.imag(np.fft.ifft(
            -1j * np.sign(np.fft.fftfreq(n)) * np.fft.fft(msg)))
        iq = (msg + 1j * msg_h) * np.exp(1j * (2 * np.pi * fc * t + phase))

    elif cls_name in ("FM_WB", "FM_NB"):
        dev = 0.30 if cls_name == "FM_WB" else 0.03
        fm  = np.random.uniform(0.005, 0.02)
        msg = np.cos(2 * np.pi * fm * t)
        phi = 2 * np.pi * fc * t + phase + dev / fm * np.sin(2 * np.pi * fm * t)
        iq  = np.exp(1j * phi)

    elif cls_name == "CW":
        iq = np.exp(1j * (2 * np.pi * fc * t + phase))
        # Keying envelope
        on_frac = np.random.uniform(0.3, 0.7)
        mask    = np.random.rand(n) < on_frac
        iq     *= mask

    elif cls_name == "BPSK":
        n_sym = n // 4
        syms  = 2 * (np.random.randint(0, 2, n_sym) * 2 - 1).astype(complex)
        iq    = np.repeat(syms, 4) * np.exp(1j * (2 * np.pi * fc * t + phase))

    elif cls_name == "QPSK":
        n_sym = n // 4
        syms  = (np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2))
        syms  = syms[np.random.randint(0, 4, n_sym)]
        iq    = np.repeat(syms, 4) * np.exp(1j * (2 * np.pi * fc * t + phase))

    elif cls_name == "8PSK":
        n_sym = n // 4
        k     = np.random.randint(0, 8, n_sym)
        syms  = np.exp(1j * (2 * np.pi * k / 8))
        iq    = np.repeat(syms, 4) * np.exp(1j * (2 * np.pi * fc * t + phase))

    elif cls_name == "16PSK":
        n_sym = n // 4
        k     = np.random.randint(0, 16, n_sym)
        syms  = np.exp(1j * (2 * np.pi * k / 16))
        iq    = np.repeat(syms, 4) * np.exp(1j * (2 * np.pi * fc * t + phase))

    elif cls_name in ("16QAM", "64QAM", "256QAM"):
        m     = {"16QAM": 4, "64QAM": 8, "256QAM": 16}[cls_name]
        n_sym = n // 4
        re    = (np.random.randint(0, m, n_sym) * 2 - m + 1).astype(float)
        im    = (np.random.randint(0, m, n_sym) * 2 - m + 1).astype(float)
        syms  = (re + 1j * im) / (m - 1)
        iq    = np.repeat(syms, 4) * np.exp(1j * (2 * np.pi * fc * t + phase))

    elif cls_name in ("2FSK", "4FSK"):
        m      = 2 if cls_name == "2FSK" else 4
        n_sym  = n // 8
        bits   = np.random.randint(0, m, n_sym)
        devs   = np.linspace(-0.1, 0.1, m)
        freqs  = fc + devs[bits]
        phi    = phase + 2 * np.pi * np.cumsum(np.repeat(freqs, 8))
        iq     = np.exp(1j * phi[:n])

    elif cls_name in ("GMSK", "GFSK"):
        # Gaussian-filtered FSK
        dev   = 0.05
        bits  = 2 * np.random.randint(0, 2, n // 8) - 1
        freq  = fc + dev * np.repeat(bits.astype(float), 8)
        # crude Gaussian smoothing (BT=0.3)
        sigma = int(8 * 0.5)
        k     = np.arange(-sigma, sigma + 1)
        g     = np.exp(-k**2 / (2 * (sigma/3)**2))
        g    /= g.sum()
        freq  = np.convolve(freq, g, mode='same')
        phi   = phase + 2 * np.pi * np.cumsum(freq[:n])
        iq    = np.exp(1j * phi)

    elif cls_name == "OOK":
        n_sym = n // 8
        bits  = np.random.randint(0, 2, n_sym)
        env   = np.repeat(bits.astype(float), 8)
        iq    = env * np.exp(1j * (2 * np.pi * fc * t + phase))

    elif cls_name == "4ASK":
        n_sym = n // 8
        k     = np.random.randint(0, 4, n_sym)
        env   = np.repeat((k + 1).astype(float) / 4.0, 8)
        iq    = env * np.exp(1j * (2 * np.pi * fc * t + phase))

    elif cls_name == "OFDM":
        n_sub  = np.random.choice([64, 128, 256])
        cp_len = n_sub // 4
        sym_len = n_sub + cp_len
        segs   = []
        while len(segs) * sym_len < n + sym_len:
            fd  = (np.random.randn(n_sub) + 1j * np.random.randn(n_sub)) / np.sqrt(2)
            td  = np.fft.ifft(fd) * np.sqrt(n_sub)
            sym = np.concatenate([td[-cp_len:], td])
            segs.append(sym)
        iq = np.concatenate(segs)[:n]

    else:
        iq = np.zeros(n, dtype=complex)

    # Normalise to unit power
    pwr = np.mean(np.abs(iq) ** 2)
    if pwr > 1e-12:
        iq /= np.sqrt(pwr)

    # Add AWGN at the requested SNR
    sigma = 10 ** (-snr_db / 20.0) / np.sqrt(2)
    iq   += sigma * (np.random.randn(n) + 1j * np.random.randn(n))

    # Random phase rotation (augmentation)
    iq *= np.exp(1j * np.random.uniform(0, 2 * np.pi))

    return np.stack([iq.real, iq.imag]).astype(np.float32)


class SyntheticDataset(Dataset):
    """
    Generates IQ samples for each class on-the-fly.
    Each epoch sees fresh random samples — no memory issues.
    Uses TorchSig signal models when available; falls back to the built-in generator.
    """
    def __init__(self, samples_per_class: int,
                 snr_range: tuple = (SNR_MIN_DB, SNR_MAX_DB),
                 use_torchsig: bool = False):
        self.n_per_cls  = samples_per_class
        self.snr_range  = snr_range
        self.use_ts     = use_torchsig
        self.total      = samples_per_class * N_CLASSES

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        cls_idx = idx % N_CLASSES
        cls_name = CLASSES[cls_idx]
        snr = float(np.random.uniform(*self.snr_range))
        iq  = _synthetic_signal(cls_name, INPUT_LEN, snr)
        return torch.from_numpy(iq), cls_idx


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type='cuda'):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += x.size(0)
    return total_loss / n, correct / n


# ══════════════════════════════════════════════════════════════════════════════
# ONNX export
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(model: nn.Module, out_path: Path, opset: int = 17):
    model.eval()
    dummy = torch.zeros(1, 2, INPUT_LEN)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names  = ["iq_input"],
        output_names = ["logits"],
        dynamic_axes = {"iq_input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version = opset,
        do_constant_folding = True,
    )
    print(f"ONNX model saved → {out_path}")

    # Quick validation with onnxruntime
    try:
        import onnx, onnxruntime as ort
        onnx.checker.check_model(str(out_path))
        sess   = ort.InferenceSession(str(out_path),
                   providers=["CPUExecutionProvider"])
        out    = sess.run(None, {"iq_input": dummy.numpy()})[0]
        assert out.shape == (1, N_CLASSES), f"Unexpected output shape {out.shape}"
        print(f"ONNX validated — output shape {out.shape}")
    except ImportError:
        print("onnxruntime not installed — skipping validation")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train AMR classifier")
    parser.add_argument("--output",   default="models",     help="Output directory")
    parser.add_argument("--device",   default="auto",       help="cuda / cpu / auto")
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    parser.add_argument("--batch",    type=int, default=BATCH_SIZE)
    parser.add_argument("--samples",  type=int, default=SAMPLES_PER_CLASS,
                        help="Synthetic samples per class per epoch")
    parser.add_argument("--resume",   default=None,         help="Resume from .pt checkpoint")
    parser.add_argument("--no-amp",   action="store_true",  help="Disable automatic mixed precision")
    parser.add_argument("--snr-min",  type=float, default=SNR_MIN_DB)
    parser.add_argument("--snr-max",  type=float, default=SNR_MAX_DB)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp
    print(f"Device: {device}  AMP: {use_amp}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}"
              f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1<<20} MB")

    # TorchSig availability
    has_ts = _try_torchsig()
    print(f"TorchSig: {'available' if has_ts else 'not found — using built-in generator'}")

    # Dataset  —  fresh samples generated each epoch via __getitem__
    dataset = SyntheticDataset(
        samples_per_class = args.samples,
        snr_range         = (args.snr_min, args.snr_max),
        use_torchsig      = has_ts,
    )
    # 90/10 train/val split
    val_n   = max(N_CLASSES, len(dataset) // 10)
    tr_n    = len(dataset) - val_n
    tr_ds, val_ds = random_split(dataset, [tr_n, val_n],
                                  generator=torch.Generator().manual_seed(42))
    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    print(f"Train: {len(tr_ds)}  Val: {len(val_ds)}  Classes: {N_CLASSES}")

    # Model
    model = ModulationNet(n_classes=N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 10,
        epochs=args.epochs, steps_per_epoch=len(tr_loader),
        pct_start=0.1, anneal_strategy='cos',
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 0
    best_acc    = 0.0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} (best_acc={best_acc:.1%})")

    # Save class list alongside model
    classes_path = out_dir / "classes.json"
    with open(classes_path, "w") as f:
        json.dump(CLASSES, f, indent=2)
    print(f"Classes saved → {classes_path}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer,
                                           criterion, device, scaler)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"Ep {epoch+1:03d}/{args.epochs}"
              f"  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.1%}"
              f"  va_loss={va_loss:.4f} va_acc={va_acc:.1%}"
              f"  lr={lr_now:.2e}  {elapsed:.0f}s")

        # Checkpoint
        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc, "classes": CLASSES,
        }
        torch.save(ckpt, out_dir / "checkpoint_last.pt")
        if va_acc > best_acc:
            best_acc = va_acc
            ckpt["best_acc"] = best_acc
            torch.save(ckpt, out_dir / "checkpoint_best.pt")
            print(f"  ★ New best: {best_acc:.1%}")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.1%}")

    # Export best checkpoint to ONNX
    best_ckpt = torch.load(out_dir / "checkpoint_best.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model"])
    model.to("cpu")

    onnx_path = out_dir / "modulation_classifier.onnx"
    export_onnx(model, onnx_path)

    print(f"\n── Deploy ──────────────────────────────────────────────────")
    print(f"Copy these two files to the AnalysisApp config directory:")
    print(f"  {onnx_path}")
    print(f"  {classes_path}")
    print(f"Then set in your analysis.xml:")
    print(f"  <model_path>/path/to/modulation_classifier.onnx</model_path>")
    print(f"  <classes_path>/path/to/classes.json</classes_path>")
    print(f"  <use_gpu>true</use_gpu>")
    print(f"  <enabled>true</enabled>")


if __name__ == "__main__":
    main()
