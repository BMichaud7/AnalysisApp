#!/usr/bin/env python3
"""
regen_fixed_classes.py

PART 1 — Training data patch:
  Regenerate 5 generator-fixed classes (4FSK, 8FSK, GFSK, RTTY, NAVTEX) and
  patch into synth_v10_47class_1024.npz (copy of v9 with those 5 replaced).
  Other 42 classes kept from synth_v9.

PART 2 — Holdout regeneration:
  Regenerate ALL 47 classes fresh for v6_holdout_v2_1024.npz.
  Fixes holdout-quality issues: QAM256/QAM64/P25_C4FM/DMR confusions were
  artifacts of old generators used to build v6_holdout_1024.npz.
"""
import sys
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from generate_v3 import generate, CLASS_NAMES

# ── Training patch params (matches synth_v9: 5000 × 9 SNR steps = 45000/class) ──
FIXED_CLASSES   = ["4FSK", "8FSK", "GFSK", "RTTY", "NAVTEX"]
N_PER_SNR       = 5000
LEN             = 1024
SNR_MIN, SNR_MAX, SNR_STEP = -10.0, 30.0, 5.0
TRAIN_SEED      = 2224   # v9 used 2222, v6 used 2223 — new seed avoids overlap

SRC_V9          = Path("data/synth_v9_47class_1024.npz")
OUT_V10         = Path("data/synth_v10_47class_1024.npz")

# ── Holdout params (matches v6_holdout: 34 × 9 SNR steps = 306/class × 47 = 14382) ──
N_HOLD_PER_SNR  = 34
HOLD_SEED       = 9997   # v6_holdout used different seed; 9997 avoids overlap
HOLDOUT_SRC     = Path("data/v6_holdout_1024.npz")
HOLDOUT_OUT     = Path("data/v6_holdout_v2_1024.npz")


def log_counts(label: str, y: np.ndarray, classes: list[str],
               highlight: list[str] | None = None) -> None:
    counts = Counter(y.tolist())
    print(f"\n{label} ({len(y):,} total):")
    for i, c in enumerate(classes):
        if highlight is None or c in highlight:
            print(f"  {c}: {counts.get(i, 0)}")


def main() -> None:
    # ── Load source synth_v9 to get class list & label map ───────────────────
    print("=" * 60)
    print("PART 1: Patch training data (5 fixed classes)")
    print("=" * 60)
    src = np.load(SRC_V9)
    all_classes = src["classes"].tolist()
    label_of = {c: i for i, c in enumerate(all_classes)}
    fixed_ids = [label_of[c] for c in FIXED_CLASSES]
    print(f"Source: {SRC_V9}  ({len(src['X']):,} samples, {len(all_classes)} classes)")
    print(f"Replacing: {list(zip(FIXED_CLASSES, fixed_ids))}")

    # Generate new samples for 5 fixed classes
    print(f"\nGenerating {FIXED_CLASSES}  [seed={TRAIN_SEED}, multi-sdr=True] ...")
    X_new, y_new, snrs_new, gen_cls = generate(
        n_per_snr=N_PER_SNR, length=LEN,
        snr_min=SNR_MIN, snr_max=SNR_MAX, snr_step=SNR_STEP,
        seed=TRAIN_SEED, impair=True, multi_sdr=True,
        filter_classes=FIXED_CLASSES,
    )
    # Remap local y (0..4) → global label IDs
    local_to_global = {i: label_of[c] for i, c in enumerate(gen_cls)}
    y_new_global = np.array([local_to_global[int(v)] for v in y_new], dtype=y_new.dtype)
    print(f"Generated {len(X_new):,} new training samples")

    # Remove old fixed-class rows from v9, append new
    keep = ~np.isin(src["y"], fixed_ids)
    X_out    = np.concatenate([src["X"][keep], X_new], axis=0)
    y_out    = np.concatenate([src["y"][keep], y_new_global], axis=0)
    snrs_out = np.concatenate([src["snrs"][keep], snrs_new], axis=0)

    # Shuffle
    perm = np.random.default_rng(1234).permutation(len(X_out))
    X_out    = X_out[perm];  y_out = y_out[perm];  snrs_out = snrs_out[perm]

    print(f"Saving {OUT_V10} ...")
    np.savez_compressed(OUT_V10, X=X_out, y=y_out, snrs=snrs_out,
                        classes=np.array(all_classes))
    log_counts("Training patch", y_out, all_classes,
               highlight=FIXED_CLASSES + ["QAM256", "QAM64", "P25_C4FM", "DMR"])
    print(f"\nTraining patch saved → {OUT_V10}")

    # ── PART 2: Regenerate full holdout ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 2: Regenerate full holdout (all 47 classes)")
    print("=" * 60)
    hold_src = np.load(HOLDOUT_SRC)
    hold_cls = hold_src["classes"].tolist()
    print(f"Source holdout: {HOLDOUT_SRC}  ({len(hold_src['X']):,} samples, {len(hold_cls)} classes)")

    print(f"\nGenerating all {len(hold_cls)} classes  [seed={HOLD_SEED}, n_per_snr={N_HOLD_PER_SNR}] ...")
    X_h, y_h, snrs_h, h_cls = generate(
        n_per_snr=N_HOLD_PER_SNR, length=LEN,
        snr_min=SNR_MIN, snr_max=SNR_MAX, snr_step=SNR_STEP,
        seed=HOLD_SEED, impair=True, multi_sdr=True,
        filter_classes=None,
    )
    print(f"Generated {len(X_h):,} holdout samples ({len(set(h_cls))} classes)")

    # Remap if class ordering differs
    hold_label_of = {c: i for i, c in enumerate(hold_cls)}
    h_local_to_global = {i: hold_label_of[c] for i, c in enumerate(h_cls) if c in hold_label_of}
    mask = np.array([int(v) in h_local_to_global for v in y_h])
    X_h    = X_h[mask]
    snrs_h = snrs_h[mask]
    y_h_global = np.array([h_local_to_global[int(v)] for v in y_h if int(v) in h_local_to_global],
                           dtype=y_h.dtype)

    perm2 = np.random.default_rng(5678).permutation(len(X_h))
    X_h    = X_h[perm2];  y_h_global = y_h_global[perm2];  snrs_h = snrs_h[perm2]

    print(f"Saving {HOLDOUT_OUT} ...")
    np.savez_compressed(HOLDOUT_OUT, X=X_h, y=y_h_global, snrs=snrs_h,
                        classes=np.array(hold_cls))
    log_counts("New holdout", y_h_global, hold_cls,
               highlight=FIXED_CLASSES + ["QAM256", "QAM64", "P25_C4FM", "DMR"])
    print(f"\nHoldout saved → {HOLDOUT_OUT}")
    print("\nAll done.")


if __name__ == "__main__":
    main()
