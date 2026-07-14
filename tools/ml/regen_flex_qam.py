#!/usr/bin/env python3
"""
regen_flex_qam.py — Patch FLEX, QAM64, QAM256, 16PSK, 32PSK into synth_v12
                    → synth_v13_47class_1024.npz

Root causes fixed in generate_v3.py (this session):
  FLEX:   sync insertion was 2 dibits / 50% → too weak a cue for 8-symbol windows.
          Fixed to always insert 4 A1-sync dibits + random tail for diversity.
  QAM64/QAM256/16PSK/32PSK: rolloff ranges overlapped, creating identical training
          samples for adjacent orders. Fixed to non-overlapping ranges.

v16 should use synth_v13 instead of synth_v12 for these 5 classes.
"""
import sys, numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from generate_v3 import generate, CLASS_NAMES

PATCH_CLASSES = ["FLEX", "QAM64", "QAM256", "16PSK", "32PSK"]
LEN = 1024
SNR_MIN, SNR_MAX, SNR_STEP = -10.0, 30.0, 5.0
N_PER_SNR   = 5000   # 45000 per class (same density as synth_v9/v10/v12)
SEED        = 4401
SRC = Path("data/synth_v12_47class_1024.npz")
OUT = Path("data/synth_v13_47class_1024.npz")


def main() -> None:
    print(f"Loading {SRC} ...")
    src = np.load(SRC)
    all_classes = src["classes"].tolist()
    label_of = {c: i for i, c in enumerate(all_classes)}

    X_out    = src["X"].copy()
    y_out    = src["y"].copy()
    snrs_out = src["snrs"].copy()

    for cls in PATCH_CLASSES:
        cid = label_of[cls]
        old_n = int((y_out == cid).sum())
        print(f"\nPatching {cls} (id={cid}, removing {old_n:,}) ...")

        X_new, y_new, snrs_new, gen_cls = generate(
            n_per_snr=N_PER_SNR, length=LEN,
            snr_min=SNR_MIN, snr_max=SNR_MAX, snr_step=SNR_STEP,
            seed=SEED + PATCH_CLASSES.index(cls) * 100,
            impair=True, multi_sdr=True,
            filter_classes=[cls],
        )
        local_to_global = {i: label_of[c] for i, c in enumerate(gen_cls)}
        y_new_global = np.array([local_to_global[int(v)] for v in y_new], dtype=y_new.dtype)
        print(f"  Generated {len(X_new):,} new samples")

        keep = y_out != cid
        X_out    = np.concatenate([X_out[keep],    X_new],          axis=0)
        y_out    = np.concatenate([y_out[keep],    y_new_global],   axis=0)
        snrs_out = np.concatenate([snrs_out[keep], snrs_new],       axis=0)

    perm = np.random.default_rng(SEED + 9999).permutation(len(X_out))
    X_out = X_out[perm]; y_out = y_out[perm]; snrs_out = snrs_out[perm]

    print(f"\nSaving {OUT} ({len(X_out):,} samples) ...")
    np.savez_compressed(OUT, X=X_out, y=y_out, snrs=snrs_out,
                        classes=np.array(all_classes))
    c = Counter(y_out.tolist())
    for cls in PATCH_CLASSES:
        cid = label_of[cls]
        print(f"  {cls}: {c[cid]:,}")
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
