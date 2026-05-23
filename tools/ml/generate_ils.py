#!/usr/bin/env python3
"""
generate_ils.py — Generate synthetic ILS (Instrument Landing System) IQ data.

ILS localizer/glideslope: AM_DSB with carrier + 90 Hz + 150 Hz audio tones.
Optionally includes 1020 Hz Morse ident bursts.

Output class label: AM_DSB  (ILS is technically AM_DSB_LC with specific tones)

Usage:
    python generate_ils.py --out data/ils_synthetic.npz --n 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _gen_ils_window(n: int, sr: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one ILS-like IQ window (baseband, complex64).

    Signal model:
        s(t) = (1 + m90*sin(2π*90*t + φ90) + m150*sin(2π*150*t + φ150)
                  + m_id*sin(2π*1020*t + φ_id) * gate(t)) * exp(jθ)

    θ   : random carrier phase offset
    m90, m150 : modulation depths 0.15–0.40 each (ICAO spec 0.20 ±20%)
    m_id      : optional ident tone 0.0 or 0.05–0.15
    gate(t)   : on/off burst for ident (50% duty cycle, random phase)
    """
    t = np.arange(n, dtype=np.float32) / sr

    m90  = rng.uniform(0.15, 0.40)
    m150 = rng.uniform(0.15, 0.40)
    φ90  = rng.uniform(0, 2 * np.pi)
    φ150 = rng.uniform(0, 2 * np.pi)

    audio = m90  * np.sin(2 * np.pi * 90  * t + φ90) \
          + m150 * np.sin(2 * np.pi * 150 * t + φ150)

    # Ident tone present ~40% of the time
    if rng.random() < 0.4:
        m_id = rng.uniform(0.05, 0.15)
        φ_id = rng.uniform(0, 2 * np.pi)
        # Simple on/off gate at ~12.5 Hz (Morse dot period)
        gate_freq = rng.uniform(8.0, 20.0)
        gate = (np.sin(2 * np.pi * gate_freq * t) > 0).astype(np.float32)
        audio += m_id * np.sin(2 * np.pi * 1020 * t + φ_id) * gate

    # AM_DSB with large carrier
    envelope = (1.0 + audio).astype(np.float32)
    envelope = np.clip(envelope, 0.0, None)   # no phase reversal

    # Random carrier phase offset (simulates LO phase uncertainty)
    θ = rng.uniform(0, 2 * np.pi)
    iq = (envelope * np.exp(1j * θ)).astype(np.complex64)

    # Normalise power
    rms = np.sqrt(np.mean(np.abs(iq) ** 2))
    return (iq / (rms + 1e-9)).astype(np.complex64)


def _add_awgn(iq: np.ndarray, snr_db: float) -> np.ndarray:
    sig_pow   = np.mean(np.abs(iq) ** 2)
    noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    noise     = (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))) \
                * np.sqrt(noise_pow / 2)
    return (iq + noise).astype(np.complex64)


def generate(n_per_snr: int, snr_min: float, snr_max: float,
             snr_step: float, sample_len: int, seed: int) -> tuple:
    rng    = np.random.default_rng(seed)
    snrs   = np.arange(snr_min, snr_max + 0.1, snr_step)
    sr     = 500_000.0   # 500 kHz — sufficient for ILS tones up to 1020 Hz

    X_list, snr_list = [], []
    for snr in snrs:
        for _ in range(n_per_snr):
            clean = _gen_ils_window(sample_len, sr, rng)
            noisy = _add_awgn(clean, snr)
            X_list.append(np.stack([noisy.real, noisy.imag], axis=0))
            snr_list.append(float(snr))

    X    = np.stack(X_list).astype(np.float32)
    snrs_arr = np.array(snr_list, dtype=np.float32)
    # Single class: AM_DSB  (ILS is AM_DSB technically)
    y    = np.zeros(len(X), dtype=np.int64)
    cls  = np.array(["AM_DSB"])
    return X, y, snrs_arr, cls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",        default="data/ils_synthetic.npz")
    ap.add_argument("--n",          type=int,   default=200,
                    help="Examples per SNR step")
    ap.add_argument("--snr-min",    type=float, default=-10.0)
    ap.add_argument("--snr-max",    type=float, default=30.0)
    ap.add_argument("--snr-step",   type=float, default=2.0)
    ap.add_argument("--len",        type=int,   default=512)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    X, y, snrs, cls = generate(
        n_per_snr=args.n,
        snr_min=args.snr_min, snr_max=args.snr_max, snr_step=args.snr_step,
        sample_len=args.len, seed=args.seed,
    )
    print(f"Generated {len(X)} ILS windows  ({len(np.unique(snrs))} SNR steps × {args.n})")
    print(f"Class: {cls[0]}  Shape: {X.shape}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, snrs=snrs, classes=cls)
    print(f"Saved → {args.out}  ({X.nbytes/1e6:.1f} MB uncompressed)")


if __name__ == "__main__":
    main()
