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
prep_real_v8.py — Merge the real-capture datasets (512-sample windows) into
a single 1024-sample npz compatible with the v6/v7 synthetic training data,
for use as an extra --npz source in the v8 fine-tune.

real_fresh.npz and real_ils.npz each cover the same 7 classes (FM_WB,
AM_DSB, FSK, FM_NB, GMSK, OFDM, GFSK) but with different per-file class
index orderings, so labels are remapped by name before merging. real.npz
is a strict subset (FM_WB/AM_DSB only, lower max SNR) of real_fresh and is
skipped.

Each 512-sample IQ window is upsampled to 1024 samples via FFT resampling
(scipy.signal.resample) to match the v6/v7 window length.

Usage:
    .venv/bin/python3 prep_real_v8.py --out data/real_combined_1024.npz
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.signal import resample

SOURCES = ["data/real_fresh.npz", "data/real_ils.npz"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_combined_1024.npz")
    ap.add_argument("--target-len", type=int, default=1024)
    args = ap.parse_args()

    X_list, y_list, snr_list = [], [], []
    class_names: list[str] = []

    for path in SOURCES:
        d = np.load(path, allow_pickle=True)
        X, y, snrs = d["X"], d["y"], d["snrs"]
        classes = [str(c) for c in d["classes"]]

        for cls in classes:
            if cls not in class_names:
                class_names.append(cls)

        # Remap this file's label indices onto the unified class_names list.
        remap = np.array([class_names.index(c) for c in classes], dtype=np.int64)
        y_remapped = remap[y]

        # Upsample 512 -> target_len via FFT resampling.
        X_resampled = resample(X, args.target_len, axis=-1).astype(np.float32)

        print(f"{path}: {X.shape} -> {X_resampled.shape}, classes {classes}")
        X_list.append(X_resampled)
        y_list.append(y_remapped)
        snr_list.append(snrs.astype(np.float32))

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    snr_all = np.concatenate(snr_list, axis=0)

    print(f"\nMerged: X={X_all.shape}, classes={class_names}")
    for i, c in enumerate(class_names):
        print(f"  {c}: {(y_all == i).sum()}")

    np.savez(args.out, X=X_all, y=y_all, snrs=snr_all,
             classes=np.array(class_names))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
