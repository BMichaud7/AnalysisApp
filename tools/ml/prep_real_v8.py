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
prep_real_v8.py — Merge all real-capture datasets into a single 1024-sample
npz for use as --npz in training.

Original sources (7 classes: FM_WB, AM_DSB, FSK, FM_NB, GMSK, OFDM, GFSK):
  data/real_fresh.npz
  data/real_ils.npz

v11 expansion sources (new classes from updated capture_real.py TARGETS):
  data/real_v11_*.npz  — any file matching this glob is merged in

Each file may have different window lengths and per-file class index
orderings. Labels are remapped by name. Windows shorter than target_len
are FFT-upsampled; longer windows are centre-cropped.

Usage:
    .venv/bin/python3 prep_real_v8.py --out data/real_combined_1024.npz
    .venv/bin/python3 prep_real_v8.py --out data/real_combined_1024.npz \\
        --extra data/real_v11_p25.npz data/real_v11_acars.npz
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
from scipy.signal import resample

SOURCES = ["data/real_fresh.npz", "data/real_ils.npz"]


def _merge_file(path: str, target_len: int,
                X_list: list, y_list: list, snr_list: list,
                class_names: list[str]) -> None:
    d = np.load(path, allow_pickle=True)
    X, y, snrs = d["X"], d["y"], d["snrs"]
    classes = [str(c) for c in d["classes"]]

    for cls in classes:
        if cls not in class_names:
            class_names.append(cls)

    remap = np.array([class_names.index(c) for c in classes], dtype=np.int64)
    y_remapped = remap[y]

    cur_len = X.shape[-1]
    if cur_len != target_len:
        # FFT resample to target length (handles both up and down)
        X = resample(X, target_len, axis=-1).astype(np.float32)
    else:
        X = X.astype(np.float32)

    per_class = {c: int((y_remapped == class_names.index(c)).sum()) for c in classes}
    print(f"{path}: {cur_len}→{target_len} samples, "
          + ", ".join(f"{c}:{n}" for c, n in per_class.items()))
    X_list.append(X)
    y_list.append(y_remapped)
    snr_list.append(snrs.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_combined_1024.npz")
    ap.add_argument("--target-len", type=int, default=1024)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Additional npz files to merge (e.g. data/real_v11_*.npz). "
                         "Glob patterns are expanded automatically.")
    args = ap.parse_args()

    X_list, y_list, snr_list = [], [], []
    class_names: list[str] = []

    # Base sources
    all_sources = list(SOURCES)
    # Expand any globs in --extra
    for pat in args.extra:
        matched = sorted(glob.glob(pat))
        if not matched:
            print(f"WARNING: --extra pattern '{pat}' matched no files")
        all_sources.extend(matched)
    # Auto-discover data/real_v11_*.npz even without --extra
    for p in sorted(glob.glob("data/real_v11_*.npz")):
        if p not in all_sources:
            all_sources.append(p)

    for path in all_sources:
        try:
            _merge_file(path, args.target_len, X_list, y_list, snr_list, class_names)
        except FileNotFoundError:
            print(f"  (skip — not found: {path})")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    snr_all = np.concatenate(snr_list, axis=0)

    print(f"\nMerged total: {X_all.shape[0]:,} samples, {len(class_names)} classes")
    for i, c in enumerate(class_names):
        n = int((y_all == i).sum())
        print(f"  {c:15s}: {n:6,}")

    np.savez_compressed(args.out, X=X_all, y=y_all, snrs=snr_all,
                        classes=np.array(class_names))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
