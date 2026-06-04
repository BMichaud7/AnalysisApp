#!/usr/bin/env python3
# ========================================================================
# Project: OpenRFStack
# Author:  Brendan Michaud
# Year:    2026
# Part of OpenRFStack (https://github.com/OpenRFStack)
#
# Licensed under the Personal Use License.
# Do not use for commercial, organizational, or military purposes.
# Contact author for permission: https://github.com/OpenRFStack
# ========================================================================

"""
train_dae.py — Train a 1-D U-Net denoising autoencoder for IQ signals.

Takes high-SNR examples from an existing NPZ as "clean" targets, corrupts
them with AWGN during training, and learns to recover the clean IQ.

Output: dae_iq.pt  (checkpoint) + dae_iq.onnx  (standalone denoiser)

The denoiser can be chained with a classifier via chain_dae_classifier.py.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


# ── Model ─────────────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, pad=None):
        pad = pad if pad is not None else k // 2
        super().__init__(
            nn.Conv1d(cin, cout, k, stride=stride, padding=pad),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
        )

class IqDenoisingUNet(nn.Module):
    """
    Conservative 1-D U-Net: preserves phase/amplitude structure via skip
    connections. Output is a residual correction added to the noisy input,
    so the model only needs to learn the noise component (easier task).

    Input/output: (B, 2, 512) float32 — interleaved I/Q channels.
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        # Encoder
        self.enc1 = _ConvBnRelu(2,   c,   k=7)          # (B, c,   512)
        self.enc2 = _ConvBnRelu(c,   c*2, k=3, stride=2) # (B, c*2, 256)
        self.enc3 = _ConvBnRelu(c*2, c*4, k=3, stride=2) # (B, c*4, 128)
        # Bottleneck
        self.bot  = nn.Sequential(
            _ConvBnRelu(c*4, c*4, k=3),
            _ConvBnRelu(c*4, c*4, k=3),
        )
        # Decoder — input is concat(up, skip)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(c*4 + c*4, c*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c*2), nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(c*2 + c*2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c), nn.ReLU(inplace=True),
        )
        # Residual output: predict the noise correction, not the clean signal
        self.out = nn.Conv1d(c + c, 2, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bot(e3)
        d3 = self.dec3(torch.cat([b,  e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        residual = self.out(torch.cat([d2, e1], dim=1))
        return x + residual   # learn noise correction


# ── Noise augmentation ────────────────────────────────────────────────────────

def add_awgn(x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """Add AWGN to a batch at per-sample SNR levels."""
    sig_pow  = x.pow(2).mean(dim=(1, 2), keepdim=True)
    noise_pow = sig_pow / (10.0 ** (snr_db.view(-1, 1, 1) / 10.0))
    noise = torch.randn_like(x) * noise_pow.sqrt()
    return x + noise


# ── Training ──────────────────────────────────────────────────────────────────

def train(args) -> None:
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load clean data from all NPZ files — use only high-SNR examples as targets
    X_parts = []
    for npz_path in args.npz:
        print(f"Loading {npz_path} …")
        d    = np.load(npz_path, allow_pickle=True)
        X_f  = d["X"].astype(np.float32)
        snrs = d["snrs"].astype(np.float32) if "snrs" in d else np.zeros(len(X_f), np.float32)
        mask = snrs >= args.clean_snr_min
        X_parts.append(X_f[mask])
        print(f"  {mask.sum()} / {len(X_f)} examples kept (SNR ≥ {args.clean_snr_min} dB)")
    X = np.concatenate(X_parts, axis=0)
    print(f"Total clean targets: {len(X)}")

    if len(X) < 1000:
        print("ERROR: too few clean examples. Lower --clean-snr-min.")
        sys.exit(1)

    X_t   = torch.from_numpy(X)
    ds    = TensorDataset(X_t)
    n_val = max(512, int(len(ds) * 0.1))
    n_tr  = len(ds) - n_val
    tr_ds, val_ds = random_split(ds, [n_tr, n_val],
                                 generator=torch.Generator().manual_seed(42))

    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,
                            num_workers=2 if device.type == "cuda" else 0,
                            pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=2 if device.type == "cuda" else 0,
                            pin_memory=(device.type == "cuda"))

    model = IqDenoisingUNet(base_ch=args.base_ch).to(device)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_p:,}")

    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(ep):
        wu = max(1, args.epochs // 10)
        if ep < wu:
            return float(ep + 1) / wu
        prog = (ep - wu) / max(1, args.epochs - wu)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit  = nn.MSELoss()

    best_loss  = float("inf")
    best_state = None
    ckpt_path  = args.out.replace(".onnx", ".pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for (clean,) in tqdm(tr_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            clean = clean.to(device)
            # Random target SNR for each sample in the batch
            snr = torch.empty(len(clean), device=device).uniform_(
                args.corrupt_snr_min, args.corrupt_snr_max)
            noisy = add_awgn(clean, snr)
            opt.zero_grad()
            denoised = model(noisy)
            loss = crit(denoised, clean)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(clean)

        tr_loss /= len(tr_loader.dataset)
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (clean,) in val_loader:
                clean = clean.to(device)
                snr   = torch.full((len(clean),), 0.0, device=device)  # fixed 0 dB for val
                noisy = add_awgn(clean, snr)
                val_loss += crit(model(noisy), clean).item() * len(clean)
        val_loss /= len(val_loader.dataset)

        lr_now = opt.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}  train_mse={tr_loss:.6f}  val_mse={val_loss:.6f}  "
              f"lr={lr_now:.2e}")

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)

    print(f"\nBest val MSE: {best_loss:.6f}")
    model.load_state_dict(best_state)
    model.eval().cpu()

    # Export ONNX
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 2, 512)
    torch.onnx.export(
        model, dummy, args.out,
        opset_version=15,
        input_names=["iq_noisy"],
        output_names=["iq_clean"],
        dynamic_axes={"iq_noisy": {0: "batch"}, "iq_clean": {0: "batch"}},
        dynamo=False,
    )
    print(f"DAE exported → {args.out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Train IQ denoising autoencoder")
    ap.add_argument("--npz", nargs="+",  default=["data/synthetic_large.npz"],
                    metavar="FILE")
    ap.add_argument("--out",            default="models/dae_iq.onnx")
    ap.add_argument("--clean-snr-min",  type=float, default=12.0,
                    help="Min SNR (dB) for examples used as clean targets")
    ap.add_argument("--corrupt-snr-min",type=float, default=-15.0,
                    help="Min SNR (dB) added during corruption")
    ap.add_argument("--corrupt-snr-max",type=float, default=8.0,
                    help="Max SNR (dB) added during corruption")
    ap.add_argument("--base-ch",        type=int,   default=32)
    ap.add_argument("--epochs",         type=int,   default=50)
    ap.add_argument("--batch",          type=int,   default=256)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--cuda",           action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
