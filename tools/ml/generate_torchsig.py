#!/usr/bin/env python3
"""
Generate a narrowband AMR training dataset from TorchSig (57 signal classes).

TorchSig normally targets wideband scene generation.  This script configures
it in single-signal narrowband mode: each sample is one signal that fills the
entire IQ window.  Hardware impairments (via our own impairments.py) are then
applied to produce training-ready data.

Usage
-----
    python generate_torchsig.py \
        --out data/torchsig_amr.npz \
        --n 500 \
        --len 512 \
        --snr-min 0 --snr-max 30 \
        --impair-copies 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "test_harness"))

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST

from impairments import add_impairments, ImpairmentProfile, PROFILES


_PROFILES = [
    ImpairmentProfile.PLUTO_SDR,
    ImpairmentProfile.HACKRF,
    ImpairmentProfile.RTL_SDR,
    ImpairmentProfile.OVER_THE_AIR,
    ImpairmentProfile.CLEAN,
]
_WEIGHTS = np.array([0.30, 0.20, 0.25, 0.15, 0.10])
_WEIGHTS /= _WEIGHTS.sum()


def build_dataset(n: int, length: int, snr_min: float, snr_max: float,
                  seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate n samples per class from TorchSig (57 classes).

    Returns (X, y, class_names):
      X: (N_total, 2, length)  float32
      y: (N_total,)            int64
    """
    sr = 10_000_000       # 10 MHz effective sample rate
    bw_max = int(sr * 0.45)  # ~4.5 MHz — stays within Nyquist

    meta = TorchSigDefaults().default_dataset_metadata
    meta.update({
        "num_iq_samples_dataset": length,
        "num_signals_min": 1,
        "num_signals_max": 1,
        "fft_size": max(32, length // 32),
        "fft_stride": max(32, length // 32),
        "signal_duration_in_samples_min": int(length * 0.85),
        "signal_duration_in_samples_max": length,
        "bandwidth_min": int(sr * 0.10),
        "bandwidth_max": bw_max,
        "signal_center_freq_min": -int(sr * 0.20),
        "signal_center_freq_max":  int(sr * 0.20),
        "frequency_min": -int(sr * 0.20),
        "frequency_max":  int(sr * 0.20),
        "snr_db_min": snr_min,
        "snr_db_max": snr_max,
        "noise_power_db": -40.0,
    })

    class_names = sorted(SIGNALS_SHARED_LIST)
    label_map   = {name: i for i, name in enumerate(class_names)}
    n_classes   = len(class_names)

    X_list: list[np.ndarray] = []
    y_list: list[int]        = []

    counts: dict[str, int] = {c: 0 for c in class_names}
    done = False

    ds = TorchSigIterableDataset(
        signal_generators="all",
        validate_init=False,
        metadata=meta,
        seed=seed,
    )

    total_needed = n * n_classes
    collected    = 0

    print(f"Generating {n} samples × {n_classes} classes = {total_needed} total …")
    print(f"  IQ window: {length} samples @ {sr/1e6:.0f} MHz")

    for sample in ds:
        if done:
            break

        # Extract class from the first component signal
        comps = sample.component_signals
        if not comps:
            continue
        class_name = comps[0].metadata.get("class_name", "")
        if class_name not in label_map:
            continue
        if counts[class_name] >= n:
            continue

        iq = sample.data.astype(np.complex64)
        if len(iq) != length:
            continue

        # Normalise
        pwr = np.mean(np.abs(iq) ** 2)
        if pwr > 0:
            iq = iq / np.sqrt(pwr)

        x = np.stack([iq.real, iq.imag], axis=0).astype(np.float32)
        X_list.append(x)
        y_list.append(label_map[class_name])
        counts[class_name] += 1
        collected += 1

        if collected % 1000 == 0:
            done_classes = sum(1 for v in counts.values() if v >= n)
            print(f"  {collected}/{total_needed} samples, "
                  f"{done_classes}/{n_classes} classes complete")

        if all(v >= n for v in counts.values()):
            done = True

    missing = {k: n - v for k, v in counts.items() if v < n}
    if missing:
        print(f"  Warning: {len(missing)} classes under-sampled: "
              + ", ".join(f"{k}:{v}" for k, v in missing.items()))

    return (
        np.stack(X_list, axis=0),
        np.array(y_list, dtype=np.int64),
        class_names,
    )


def apply_impairments(X: np.ndarray, y: np.ndarray,
                      class_names: list[str],
                      copies: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Add `copies` impaired versions of X and stack with the originals."""
    if copies == 0:
        return X, y

    N, _, L = X.shape
    sr = 1e6  # reference rate for impairments

    from dataclasses import replace

    X_aug_list = [X]
    y_aug_list = [y]

    for copy_idx in range(copies):
        print(f"  Generating impaired copy {copy_idx + 1}/{copies} …")
        X_copy  = np.empty_like(X)
        chosen  = rng.choice(len(_PROFILES), size=N, p=_WEIGHTS)

        for i in range(N):
            profile = _PROFILES[int(chosen[i])]
            cfg     = PROFILES[profile]
            snr_j   = rng.uniform(-3, 3)
            cfg_j   = replace(cfg, snr_db=cfg.snr_db + snr_j)

            iq = (X[i, 0] + 1j * X[i, 1]).astype(np.complex64)
            iq = add_impairments(iq, sr, cfg=cfg_j, rng=rng)
            pwr = np.mean(np.abs(iq) ** 2)
            if pwr > 0:
                iq = iq / np.sqrt(pwr)

            X_copy[i, 0] = iq.real.astype(np.float32)
            X_copy[i, 1] = iq.imag.astype(np.float32)

        X_aug_list.append(X_copy)
        y_aug_list.append(y)

    return np.concatenate(X_aug_list, axis=0), np.concatenate(y_aug_list, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate TorchSig narrowband AMR dataset (57 classes)"
    )
    ap.add_argument("--out",          required=True, help="Output .npz path")
    ap.add_argument("--n",            type=int, default=500,
                    help="Samples per class (default 500)")
    ap.add_argument("--len",          type=int, default=512,
                    help="IQ window length in samples (default 512)")
    ap.add_argument("--snr-min",      type=float, default=0.0,
                    help="Minimum SNR dB (default 0)")
    ap.add_argument("--snr-max",      type=float, default=30.0,
                    help="Maximum SNR dB (default 30)")
    ap.add_argument("--impair-copies", type=int, default=2,
                    help="Number of impaired copies to add (default 2)")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    X, y, class_names = build_dataset(
        n=args.n, length=args.len,
        snr_min=args.snr_min, snr_max=args.snr_max,
        seed=args.seed,
    )
    print(f"\nBase dataset: {len(X)} samples, {len(class_names)} classes, "
          f"shape {X.shape}")

    if args.impair_copies > 0:
        print("Applying hardware impairments …")
        rng = np.random.default_rng(args.seed + 1)
        X, y = apply_impairments(X, y, class_names, args.impair_copies, rng)
        print(f"Augmented: {len(X)} samples total")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    snrs = np.full(len(y), 15.0, dtype=np.float32)  # approximate mid-range SNR
    np.savez_compressed(out, X=X, y=y, snrs=snrs, classes=np.array(class_names))
    print(f"Saved → {out}  (shape {X.shape})")


if __name__ == "__main__":
    main()
