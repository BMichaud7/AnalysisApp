#!/usr/bin/env python3
"""
regen_dstar.py — Patch DSTAR into training and holdout NPZs.

Root cause: generate_v3._dstar() was fixed from dev=SR*0.024=4800 Hz to
dev=1600 Hz (correct GMSK deviation for D-STAR), but synth_v10 and the
holdout were never regenerated. The model trained at 4800 Hz deviation
and the holdout tests at 1600 Hz → 0% DSTAR accuracy.

This script:
  PART 1: Replace DSTAR rows in synth_v10_47class_1024.npz
          → synth_v12_47class_1024.npz (DSTAR at correct dev=1600 Hz)
  PART 2: Replace DSTAR rows in v6_holdout_v2_1024.npz
          → v7_holdout_1024.npz
"""
import sys
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from generate_v3 import generate, CLASS_NAMES

PATCH_CLASS = ["DSTAR"]
LEN         = 1024
SNR_MIN, SNR_MAX, SNR_STEP = -10.0, 30.0, 5.0

# Training: match synth_v9/v10 density (5000 × 9 SNR steps = 45000)
N_TRAIN_PER_SNR = 5000
TRAIN_SEED      = 3311   # new seed; v9=2222, v10=2224

# Holdout: match v6_holdout_v2 density (34 × 9 = 306/class)
N_HOLD_PER_SNR  = 34
HOLD_SEED       = 3312

SRC_V10     = Path("data/synth_v10_47class_1024.npz")
OUT_V12     = Path("data/synth_v12_47class_1024.npz")

SRC_HOLD    = Path("data/v6_holdout_v2_1024.npz")
OUT_HOLD    = Path("data/v7_holdout_1024.npz")


def log_counts(label: str, y: np.ndarray, classes: list) -> None:
    counts = Counter(y.tolist())
    idx = classes.index("DSTAR")
    print(f"{label}: DSTAR (id={idx}) → {counts.get(idx, 0)} samples")


def patch_npz(src_path: Path, out_path: Path,
              n_per_snr: int, seed: int, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"{label}: {src_path} → {out_path}")
    print(f"{'='*60}")

    src = np.load(src_path)
    all_classes = src["classes"].tolist()
    label_of = {c: i for i, c in enumerate(all_classes)}
    dstar_id = label_of["DSTAR"]
    n_old = int((src["y"] == dstar_id).sum())
    print(f"Source: {len(src['X']):,} samples, {len(all_classes)} classes")
    print(f"Removing {n_old:,} old DSTAR rows (id={dstar_id})")

    print(f"Generating DSTAR [seed={seed}, n_per_snr={n_per_snr}] ...")
    X_new, y_new, snrs_new, gen_cls = generate(
        n_per_snr=n_per_snr, length=LEN,
        snr_min=SNR_MIN, snr_max=SNR_MAX, snr_step=SNR_STEP,
        seed=seed, impair=True, multi_sdr=True,
        filter_classes=PATCH_CLASS,
    )
    local_to_global = {i: label_of[c] for i, c in enumerate(gen_cls)}
    y_new_global = np.array([local_to_global[int(v)] for v in y_new], dtype=y_new.dtype)
    print(f"Generated {len(X_new):,} new DSTAR samples")

    keep = src["y"] != dstar_id
    X_out    = np.concatenate([src["X"][keep],    X_new],         axis=0)
    y_out    = np.concatenate([src["y"][keep],    y_new_global],  axis=0)
    snrs_out = np.concatenate([src["snrs"][keep], snrs_new],      axis=0)

    perm = np.random.default_rng(seed + 1000).permutation(len(X_out))
    X_out = X_out[perm]; y_out = y_out[perm]; snrs_out = snrs_out[perm]

    print(f"Saving {out_path} ({len(X_out):,} samples) ...")
    np.savez_compressed(out_path, X=X_out, y=y_out, snrs=snrs_out,
                        classes=np.array(all_classes))
    log_counts("After patch", y_out, all_classes)
    print(f"Saved → {out_path}")


def main() -> None:
    patch_npz(SRC_V10, OUT_V12, N_TRAIN_PER_SNR, TRAIN_SEED, "PART 1: Training patch")
    patch_npz(SRC_HOLD, OUT_HOLD, N_HOLD_PER_SNR, HOLD_SEED, "PART 2: Holdout patch")
    print("\nDone. Next: run run_v14_train.sh")


if __name__ == "__main__":
    main()
