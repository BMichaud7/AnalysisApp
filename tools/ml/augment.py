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
Augment a synthetic .npz dataset with hardware impairments.

For each sample in the input .npz the script applies a randomly-selected
impairment profile (PlutoSDR, HackRF, RTL-SDR, over-the-air) and emits a
new .npz with the same class labels.  Training on synthetic + augmented data
forces the model to learn modulation-discriminating features that survive
frequency offset, IQ imbalance, phase noise, DC offset, and multipath —
the same impairments present on real SDR hardware.

Usage
-----
    python augment.py \
        --npz  data/synthetic.npz \
        --out  data/augmented.npz \
        --copies 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Locate test_harness relative to this file
sys.path.insert(0, str(Path(__file__).parent.parent / "test_harness"))
from impairments import add_impairments, ImpairmentProfile, PROFILES


# Profile weights: over-the-air is rarer but important for robustness
_PROFILES = [
    ImpairmentProfile.PLUTO_SDR,
    ImpairmentProfile.HACKRF,
    ImpairmentProfile.RTL_SDR,
    ImpairmentProfile.OVER_THE_AIR,
    ImpairmentProfile.CLEAN,
]
_WEIGHTS = np.array([0.30, 0.20, 0.25, 0.15, 0.10])
_WEIGHTS /= _WEIGHTS.sum()


def augment(npz_path: Path, out_path: Path, copies: int, seed: int) -> None:
    print(f"Loading {npz_path} …")
    data    = np.load(npz_path, allow_pickle=True)
    X_in    = data["X"].astype(np.float32)      # (N, 2, L)
    y_in    = data["y"].astype(np.int64)
    snrs_in = data["snrs"].astype(np.float32) if "snrs" in data else \
              np.full(len(y_in), 0.0, dtype=np.float32)
    classes = list(data["classes"])

    N, _, L = X_in.shape
    rng     = np.random.default_rng(seed)

    print(f"  {N} input samples, {len(classes)} classes, {copies} impaired copies")

    X_out_list = [X_in]
    y_out_list = [y_in]
    snr_out_list = [snrs_in]

    for copy_idx in range(copies):
        print(f"  Generating impaired copy {copy_idx + 1}/{copies} …")
        X_copy  = np.empty_like(X_in)
        snr_copy = np.empty(N, dtype=np.float32)

        chosen = rng.choice(len(_PROFILES), size=N, p=_WEIGHTS)
        # Sample rate for impairments — use 1 MHz as a generic reference
        sr = 1e6

        for i in range(N):
            profile = _PROFILES[int(chosen[i])]
            cfg     = PROFILES[profile]

            # Reconstruct complex IQ from (2, L) layout
            iq = (X_in[i, 0] + 1j * X_in[i, 1]).astype(np.complex64)

            # Randomise SNR slightly around the profile's default
            snr_jitter = rng.uniform(-3, 3)
            from dataclasses import replace
            cfg_j = replace(cfg, snr_db=cfg.snr_db + snr_jitter)

            iq_imp = add_impairments(iq, sr, cfg=cfg_j, rng=rng)

            # Normalise power after impairments
            pwr = np.mean(np.abs(iq_imp) ** 2)
            if pwr > 0:
                iq_imp = (iq_imp / np.sqrt(pwr)).astype(np.complex64)

            X_copy[i, 0] = iq_imp.real
            X_copy[i, 1] = iq_imp.imag
            snr_copy[i]  = float(cfg_j.snr_db)

        X_out_list.append(X_copy)
        y_out_list.append(y_in)
        snr_out_list.append(snr_copy)

    X_out   = np.concatenate(X_out_list, axis=0)
    y_out   = np.concatenate(y_out_list, axis=0)
    snr_out = np.concatenate(snr_out_list, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X_out, y=y_out, snrs=snr_out,
        classes=np.array(classes),
    )
    print(f"Saved {len(X_out)} samples → {out_path}  "
          f"(shape {X_out.shape})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply hardware impairments to synthetic .npz")
    ap.add_argument("--npz",     required=True, help="Input .npz file")
    ap.add_argument("--out",     required=True, help="Output .npz file")
    ap.add_argument("--copies",  type=int, default=3,
                    help="Number of impaired copies to create (default 3)")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    augment(Path(args.npz), Path(args.out), args.copies, args.seed)


if __name__ == "__main__":
    main()
