#!/usr/bin/env python3
"""Generate a hard-class booster dataset for v11 training.

Two passes combined into one npz:
  Pass 1 — full SNR range (-10..+35 dB, n_per_snr=10000): gives 2× the
            baseline synth coverage for the 11 hardest classes.
  Pass 2 — high-SNR only (+15..+35 dB, n_per_snr=15000): extra oversampling
            of the collapse region where these classes fall from their mid-SNR
            peak to near 0%.

Both passes use --multi-sdr impairments (seed 4444 / 5555) so the booster
is consistent with the v9 synthetic data.
Saves to data/synth_v11_hardboost_1024.npz.
"""
import generate_v3 as g
import numpy as np

HARD_CLASSES = [
    "QAM64", "QAM32", "16PSK",
    "RTTY", "FM_WB", "TONE",
    "ACARS", "EAS_SAME", "DTMF",
    "P25_C4FM", "AM_DSB",
]

OUT = "data/synth_v11_hardboost_1024.npz"

print("Pass 1: full SNR range (-10..+35 dB, n_per_snr=10000)...", flush=True)
X1, y1, s1, classes = g.generate(
    n_per_snr=10000, length=1024,
    snr_min=-10.0, snr_max=35.0, snr_step=5.0,
    seed=4444, impair=True, multi_sdr=True,
    filter_classes=HARD_CLASSES,
)
print(f"  {X1.shape[0]:,} samples", flush=True)

print("Pass 2: high-SNR only (+15..+35 dB, n_per_snr=15000)...", flush=True)
X2, y2, s2, _ = g.generate(
    n_per_snr=15000, length=1024,
    snr_min=15.0, snr_max=35.0, snr_step=5.0,
    seed=5555, impair=True, multi_sdr=True,
    filter_classes=HARD_CLASSES,
)
print(f"  {X2.shape[0]:,} samples", flush=True)

X    = np.concatenate([X1, X2], axis=0)
y    = np.concatenate([y1, y2], axis=0)
snrs = np.concatenate([s1, s2], axis=0)

print(f"Total: {X.shape[0]:,} samples across {len(classes)} classes", flush=True)
np.savez_compressed(OUT, X=X, y=y, snrs=snrs, classes=np.array(classes))
print(f"Saved {OUT}", flush=True)
