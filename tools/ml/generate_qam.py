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
generate_qam.py — QAM-focused dataset generator.

Key difference from generate_v3.py:
  - Frequency offset is ±0.2% SR (post-AFC simulation) instead of ±2% SR.
    At ±2% of 200 kHz SR over 512 samples the carrier rotates >10 full turns —
    the constellation is completely destroyed and QAM order is undetectable.
    Real receivers run AFC so residual error is ~10× smaller.
  - Multipath channel (1–3 taps, delay 1–12 samples, gain 0.05–0.35).
    Multipath creates ISI whose statistical signature is QAM-order-dependent —
    a key discriminator the CNN can learn.
  - Timing offset (½ symbol or less) adds realistic inter-symbol interference.
  - SNR range -5 to 30 dB (below -5 dB QAM256 is unclassifiable noise).

Usage:
    python generate_qam.py --out data/qam_heavy.npz --n 8000 --len 512
    # 4 classes × 9 SNR steps × 8000 = 288 000 samples, ~7 min on CPU
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

SR = 200_000.0

CLASSES = ["QAM16", "QAM32", "QAM64", "QAM256"]


# ── DSP helpers ───────────────────────────────────────────────────────────────

def _norm(iq: np.ndarray) -> np.ndarray:
    iq = np.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.mean(np.abs(iq) ** 2)
    return (iq / np.sqrt(p) if p > 1e-12 else iq).astype(np.complex64)


def _awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sp = np.mean(np.abs(iq) ** 2)
    np_ = sp / (10 ** (snr_db / 10))
    n = rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
    return (iq + n.astype(np.complex64) * math.sqrt(np_ / 2)).astype(np.complex64)


def _rrc(sps: int, rolloff: float = 0.35, span: int = 10) -> np.ndarray:
    N = span * sps
    t = np.arange(-N // 2, N // 2 + 1, dtype=float) / sps
    h = np.zeros(len(t))
    alpha = rolloff
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1 - alpha + 4 * alpha / math.pi
        elif abs(ti) == 1.0 / (2 * alpha):
            h[i] = (alpha / math.sqrt(2)) * (
                (1 + 2 / math.pi) * math.sin(math.pi / (4 * alpha))
                + (1 - 2 / math.pi) * math.cos(math.pi / (4 * alpha))
            )
        else:
            num = math.sin(math.pi * ti * (1 - alpha)) + 4 * alpha * ti * math.cos(math.pi * ti * (1 + alpha))
            den = math.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / den
    h /= math.sqrt(np.sum(h ** 2))
    return h.astype(np.float32)


def _apply_rrc(symbols: np.ndarray, sps: int, rolloff: float,
               timing_offset: float = 0.0) -> np.ndarray:
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    h = _rrc(sps, rolloff)
    real = lfilter(h, [1.0], upsampled.real)
    imag = lfilter(h, [1.0], upsampled.imag)
    iq = (real + 1j * imag).astype(np.complex64)
    # Sub-sample timing offset: shift starting sample within first symbol
    offset = int(round(timing_offset * sps)) % sps
    return iq[offset:]


def _constellation(M: int) -> np.ndarray:
    side = int(math.sqrt(M))
    if side * side == M:
        levels = np.arange(-(side - 1), side, 2, dtype=float)
        pts = np.array([x + 1j * y for x in levels for y in levels], dtype=np.complex64)
    else:
        # cross QAM (QAM32)
        side2 = int(math.sqrt(M * 1.5))
        levels = np.arange(-(side2 - 1), side2, 2, dtype=float)
        pts = np.array([x + 1j * y for x in levels for y in levels], dtype=np.complex64)
        pts = pts[np.abs(pts) <= side2 * 0.9][:M]
    pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts.astype(np.complex64)


CONSTS = {cls: _constellation(int(cls[3:])) for cls in CLASSES}


# ── QAM-aware impairment ──────────────────────────────────────────────────────

def _qam_impair(iq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Hardware impairments tuned for QAM recognisability:
      - DC offset, IQ imbalance, phase skew: same as generate_v3
      - Frequency offset: ±0.2% SR (post-AFC, 10× smaller than generate_v3)
        → <0.2 radians/sample; constellation stays coherent over 512 samples
      - Phase noise: mild (0.003 std dev per sample)
      - Multipath: 1–3 taps, random delay/gain/phase
        → ISI pattern encodes QAM order → helps CNN learn order-specific features
    """
    # DC offset
    dc = rng.uniform(-0.02, 0.02) + 1j * rng.uniform(-0.02, 0.02)
    iq = iq + dc

    # IQ amplitude imbalance (±0.5 dB)
    amp = 10 ** (rng.uniform(-0.05, 0.05) / 20)
    iq = iq.real * amp + 1j * iq.imag / amp

    # IQ phase skew (±3°)
    phi = rng.uniform(-0.05, 0.05)
    iq = iq.real + iq.imag * (math.sin(phi) + 1j * math.cos(phi))

    # Frequency offset: ±0.2% SR (post-AFC residual)
    fo = rng.uniform(-0.002, 0.002) * SR
    t = np.arange(len(iq)) / SR
    iq = (iq * np.exp(1j * 2 * math.pi * fo * t)).astype(np.complex64)

    # Phase noise
    pn = np.cumsum(rng.standard_normal(len(iq)) * 0.003).astype(np.float32)
    iq = (iq * np.exp(1j * pn)).astype(np.complex64)

    # Multipath: 1–3 reflected paths
    n_paths = int(rng.integers(1, 4))
    out = iq.astype(np.complex128)
    for _ in range(n_paths):
        delay = int(rng.integers(1, 13))
        gain  = float(rng.uniform(0.05, 0.35))
        phase = float(rng.uniform(0.0, 2 * math.pi))
        if delay < len(iq):
            out[delay:] += gain * np.exp(1j * phase) * iq[:-delay]

    return _norm(out.astype(np.complex64))


# ── Generator ─────────────────────────────────────────────────────────────────

def _gen_qam(cls: str, n: int, rng: np.random.Generator) -> np.ndarray:
    const  = CONSTS[cls]
    sps    = int(rng.integers(4, 13))
    ro     = float(rng.uniform(0.20, 0.50))
    timing = float(rng.uniform(0.0, 0.5))   # timing offset as fraction of symbol

    n_syms = math.ceil(n / sps) + 20
    syms   = rng.choice(const, n_syms)
    iq     = _apply_rrc(syms, sps, ro, timing)

    # Strip RRC filter delay
    delay = (len(_rrc(sps, ro)) - 1) // 2
    start = delay + int(round(timing * sps)) % sps
    iq    = iq[start:start + n]
    if len(iq) < n:
        iq = np.pad(iq, (0, n - len(iq)))

    return _norm(iq[:n])


def generate(n_per_snr: int, length: int,
             snr_min: float, snr_max: float, snr_step: float,
             seed: int) -> tuple:
    rng  = np.random.default_rng(seed)
    snrs = np.arange(snr_min, snr_max + snr_step / 2, snr_step)
    total = len(CLASSES) * len(snrs) * n_per_snr
    X = np.empty((total, 2, length), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    s = np.empty(total, dtype=np.float32)
    label = {c: i for i, c in enumerate(CLASSES)}

    idx = 0
    for ci, cls in enumerate(CLASSES):
        for snr in snrs:
            for _ in range(n_per_snr):
                iq = _gen_qam(cls, length, rng)
                iq = _qam_impair(iq, rng)
                iq = _awgn(iq, float(snr), rng)
                X[idx, 0] = iq.real
                X[idx, 1] = iq.imag
                y[idx]    = label[cls]
                s[idx]    = float(snr)
                idx += 1
        print(f"  {ci+1}/{len(CLASSES)}  {cls:<8}  {idx:,} samples", flush=True)

    return X, y, s, CLASSES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",      default="data/qam_heavy.npz")
    ap.add_argument("--n",        type=int,   default=8000,
                    help="Samples per class per SNR step (default 8000)")
    ap.add_argument("--len",      type=int,   default=512)
    ap.add_argument("--snr-min",  type=float, default=-5.0)
    ap.add_argument("--snr-max",  type=float, default=30.0)
    ap.add_argument("--snr-step", type=float, default=5.0)
    ap.add_argument("--seed",     type=int,   default=1234)
    args = ap.parse_args()

    n_snr = len(np.arange(args.snr_min, args.snr_max + args.snr_step / 2, args.snr_step))
    total = len(CLASSES) * n_snr * args.n
    print(f"Generating {args.n} × {len(CLASSES)} classes × {n_snr} SNR steps = {total:,} samples")
    print(f"SNR range: {args.snr_min} to {args.snr_max} dB, step {args.snr_step} dB")
    print(f"Impairments: IQ imbalance, DC offset, phase noise, multipath,")
    print(f"             freq offset ±0.2% SR (post-AFC, 10× tighter than synth_v3)")

    X, y, snrs, cls = generate(
        n_per_snr=args.n, length=args.len,
        snr_min=args.snr_min, snr_max=args.snr_max, snr_step=args.snr_step,
        seed=args.seed,
    )
    print(f"\nTotal: {len(X):,}  shape={X.shape}  ({X.nbytes/1e9:.2f} GB)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, snrs=snrs, classes=np.array(cls))
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
