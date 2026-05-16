#!/usr/bin/env python3
"""
Generate an extended AMR training dataset with 28 signal classes.

Covers the full modulation families used in TorchSig/RadioML research:
  - ASK family:   OOK, 4ASK, 16ASK
  - FSK family:   2FSK, 4FSK, 8FSK, MSK, GFSK, GMSK
  - PSK family:   BPSK, QPSK, 8PSK, 16PSK, 32PSK
  - QAM family:   QAM16, QAM32, QAM64, QAM256
  - Analog:       FM_NB, FM_WB, AM_DSB, AM_DSB_SC, AM_SSB
  - Multi-carrier: OFDM
  - Spread/Chirp: CSS, LFM
  - Special:      TONE

Usage
-----
    python generate_extended.py \
        --out   data/extended.npz \
        --n     1000 \
        --len   512  \
        --impair-copies 2
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "test_harness"))
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


# ── Signal generators ─────────────────────────────────────────────────────────

def _norm(iq: np.ndarray) -> np.ndarray:
    pwr = np.mean(np.abs(iq) ** 2)
    return (iq / np.sqrt(pwr) if pwr > 0 else iq).astype(np.complex64)


def _awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_pwr  = np.mean(np.abs(iq) ** 2)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    noise = (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))
    return (iq + noise.astype(np.complex64) * math.sqrt(noise_pwr / 2)).astype(np.complex64)


def _ask(order: int, n: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    levels = np.linspace(0, 1, order, dtype=np.float32)
    syms = rng.choice(levels, n // sps + 1)
    iq = np.repeat(syms, sps)[:n].astype(np.complex64)
    return _norm(iq)


def _psk(order: int, n: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    angles = np.arange(order) * 2 * np.pi / order
    const  = np.exp(1j * angles).astype(np.complex64)
    syms   = rng.choice(const, n // sps + 1)
    return _norm(np.repeat(syms, sps)[:n])


def _qam(order: int, n: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    k    = int(round(math.sqrt(order)))
    vals = np.arange(-(k - 1), k, 2, dtype=np.float32)
    I, Q = np.meshgrid(vals, vals)
    const = (I.ravel() + 1j * Q.ravel()).astype(np.complex64)[:order]
    syms  = rng.choice(const, n // sps + 1)
    return _norm(np.repeat(syms, sps)[:n])


def _fsk(order: int, n: int, sr: float, sps: int, dev: float,
         rng: np.random.Generator) -> np.ndarray:
    levels = np.linspace(-dev, dev, order)
    bits   = rng.integers(0, order, n // sps + 1)
    freqs  = np.repeat(levels[bits], sps).astype(np.float32)
    phase  = 2 * np.pi / sr * np.cumsum(freqs[:n])
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _gfsk(order: int, n: int, sr: float, sps: int, dev: float, bt: float,
          rng: np.random.Generator) -> np.ndarray:
    levels = np.linspace(-dev, dev, order)
    bits   = rng.integers(0, order, n // sps + 1)
    raw    = np.repeat(levels[bits], sps).astype(np.float32)[:n]
    # Gaussian filter (approximated by a short window)
    taps   = max(sps, int(sps / bt))
    t_g    = np.linspace(-taps // 2, taps // 2, taps)
    sigma  = sps / (2 * math.pi * bt)
    g      = np.exp(-0.5 * (t_g / sigma) ** 2).astype(np.float32)
    g     /= g.sum()
    filtered = np.convolve(raw, g, mode="same")
    phase  = 2 * np.pi / sr * np.cumsum(filtered)
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _fm(dev: float, n: int, sr: float, rng: np.random.Generator) -> np.ndarray:
    audio = rng.standard_normal(n).astype(np.float32)
    audio /= max(np.abs(audio).max(), 1e-9)
    phase  = 2 * np.pi * dev / sr * np.cumsum(audio)
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _am(depth: float, suppressed: bool, n: int,
        rng: np.random.Generator) -> np.ndarray:
    audio = (rng.standard_normal(n) * 0.5).astype(np.float32)
    if suppressed:
        return _norm(audio.astype(np.complex64))
    return _norm((1.0 + depth * audio).astype(np.complex64))


def _ssb(n: int, sr: float, upper: bool, rng: np.random.Generator) -> np.ndarray:
    audio = rng.standard_normal(n).astype(np.float32)
    analytic = np.fft.fft(audio)
    freqs    = np.fft.fftfreq(n)
    if upper:
        analytic[freqs <= 0] = 0
    else:
        analytic[freqs >= 0] = 0
    return _norm(np.fft.ifft(analytic).astype(np.complex64))


def _ofdm(n: int, nfft: int, cp: int, rng: np.random.Generator) -> np.ndarray:
    syms = []
    qpsk = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64) / math.sqrt(2)
    while len(syms) * (nfft + cp) < n:
        fd = rng.choice(qpsk, nfft)
        td = np.fft.ifft(fd).astype(np.complex64)
        syms.append(np.concatenate([td[-cp:], td]))
    return _norm(np.concatenate(syms)[:n])


def _css(n: int, sr: float, bw: float, rng: np.random.Generator) -> np.ndarray:
    cl  = max(int(sr / bw), 1)
    t   = np.arange(cl) / sr
    c   = np.exp(1j * 2 * np.pi * (-bw / 2 * t + bw / (2 * cl / sr) * t**2))
    # randomise start phase
    start = rng.integers(0, max(cl, 1))
    base  = np.tile(c, n // cl + 2)
    return _norm(base[start:start + n].astype(np.complex64))


def _lfm(n: int, sr: float, bw: float, rng: np.random.Generator) -> np.ndarray:
    # LFM sweep direction randomised
    direction = rng.choice([-1, 1])
    t  = np.arange(n) / sr
    ph = 2 * np.pi * (direction * (-bw / 2) * t + direction * bw / (2 * n / sr) * t**2)
    return _norm(np.exp(1j * ph).astype(np.complex64))


def _tone(n: int, sr: float, rng: np.random.Generator) -> np.ndarray:
    freq = rng.uniform(-sr * 0.3, sr * 0.3)
    t    = np.arange(n) / sr
    return _norm(np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64))


# ── Class definitions ─────────────────────────────────────────────────────────

SR  = 200_000.0   # reference sample rate used by generators
SPS = 8           # samples per symbol

GENERATORS: dict[str, callable] = {
    # ASK family
    "OOK":      lambda n, r: _ask(2,   n, SPS, r),
    "4ASK":     lambda n, r: _ask(4,   n, SPS, r),
    "16ASK":    lambda n, r: _ask(16,  n, SPS // 2, r),
    # FSK family
    "FSK":      lambda n, r: _fsk(2,   n, SR, SPS * 4, SR * 0.05, r),
    "4FSK":     lambda n, r: _fsk(4,   n, SR, SPS * 2, SR * 0.04, r),
    "8FSK":     lambda n, r: _fsk(8,   n, SR, SPS,     SR * 0.03, r),
    "MSK":      lambda n, r: _fsk(2,   n, SR, SPS * 2, SR * 0.025, r),
    "GFSK":     lambda n, r: _gfsk(2,  n, SR, SPS * 4, SR * 0.05, 0.5, r),
    "GMSK":     lambda n, r: _gfsk(2,  n, SR, SPS * 2, SR * 0.025, 0.3, r),
    # PSK family
    "BPSK":     lambda n, r: _psk(2,   n, SPS, r),
    "QPSK":     lambda n, r: _psk(4,   n, SPS, r),
    "8PSK":     lambda n, r: _psk(8,   n, SPS, r),
    "16PSK":    lambda n, r: _psk(16,  n, SPS // 2, r),
    "32PSK":    lambda n, r: _psk(32,  n, SPS // 2, r),
    # QAM family
    "QAM16":    lambda n, r: _qam(16,  n, SPS, r),
    "QAM32":    lambda n, r: _qam(32,  n, SPS, r),
    "QAM64":    lambda n, r: _qam(64,  n, SPS // 2, r),
    "QAM256":   lambda n, r: _qam(256, n, SPS // 2, r),
    # Analog
    "FM_NB":    lambda n, r: _fm(SR * 0.1,  n, SR, r),
    "FM_WB":    lambda n, r: _fm(SR * 0.35, n, SR, r),
    "AM_DSB":   lambda n, r: _am(0.85, False, n, r),
    "AM_DSB_SC":lambda n, r: _am(1.00, True,  n, r),
    "AM_SSB_USB":lambda n, r: _ssb(n, SR, True,  r),
    "AM_SSB_LSB":lambda n, r: _ssb(n, SR, False, r),
    # Multi-carrier
    "OFDM":     lambda n, r: _ofdm(n, 64, 16, r),
    # Chirp / spread
    "CSS":      lambda n, r: _css(n, SR, SR * 0.25, r),
    "LFM":      lambda n, r: _lfm(n, SR, SR * 0.40, r),
    # Special
    "TONE":     lambda n, r: _tone(n, SR, r),
}

CLASS_NAMES = sorted(GENERATORS.keys())


# ── Generation ────────────────────────────────────────────────────────────────

def generate(n: int, length: int,
             snr_min: float, snr_max: float, snr_step: float,
             seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng  = np.random.default_rng(seed)
    snrs = np.arange(snr_min, snr_max + snr_step / 2, snr_step)
    n_classes = len(CLASS_NAMES)
    label_map = {c: i for i, c in enumerate(CLASS_NAMES)}

    total = n_classes * len(snrs) * n
    X = np.empty((total, 2, length), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    s = np.empty(total, dtype=np.float32)

    idx = 0
    for cls in CLASS_NAMES:
        gen = GENERATORS[cls]
        label = label_map[cls]
        for snr in snrs:
            for _ in range(n):
                iq = gen(length, rng)
                iq = _awgn(iq, float(snr), rng)
                pwr = np.mean(np.abs(iq) ** 2)
                if pwr > 0:
                    iq = iq / np.sqrt(pwr)
                X[idx, 0] = iq.real
                X[idx, 1] = iq.imag
                y[idx] = label
                s[idx] = float(snr)
                idx += 1

    return X[:idx], y[:idx], CLASS_NAMES


def apply_impairments(X: np.ndarray, y: np.ndarray, snrs: np.ndarray,
                      copies: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if copies == 0:
        return X, y, snrs
    N, _, L = X.shape
    X_list, y_list, s_list = [X], [y], [snrs]

    for c in range(copies):
        print(f"  Impaired copy {c + 1}/{copies} …")
        Xc = np.empty_like(X)
        sc = np.empty(N, dtype=np.float32)
        chosen = rng.choice(len(_PROFILES), size=N, p=_WEIGHTS)
        for i in range(N):
            profile = _PROFILES[int(chosen[i])]
            cfg     = PROFILES[profile]
            jitter  = rng.uniform(-3, 3)
            cfg_j   = replace(cfg, snr_db=cfg.snr_db + jitter)
            iq = (X[i, 0] + 1j * X[i, 1]).astype(np.complex64)
            iq = add_impairments(iq, 1e6, cfg=cfg_j, rng=rng)
            pwr = np.mean(np.abs(iq) ** 2)
            if pwr > 0:
                iq = (iq / np.sqrt(pwr)).astype(np.complex64)
            Xc[i, 0] = iq.real.astype(np.float32)
            Xc[i, 1] = iq.imag.astype(np.float32)
            sc[i]    = float(cfg_j.snr_db)
        X_list.append(Xc)
        y_list.append(y)
        s_list.append(sc)

    return (np.concatenate(X_list, axis=0),
            np.concatenate(y_list, axis=0),
            np.concatenate(s_list, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate extended 29-class AMR training dataset"
    )
    ap.add_argument("--out",           required=True)
    ap.add_argument("--n",             type=int,   default=200,
                    help="Samples per (class, SNR) bucket (default 200)")
    ap.add_argument("--len",           type=int,   default=512)
    ap.add_argument("--snr-min",       type=float, default=0.0)
    ap.add_argument("--snr-max",       type=float, default=20.0)
    ap.add_argument("--snr-step",      type=float, default=4.0)
    ap.add_argument("--impair-copies", type=int,   default=2)
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    n_snrs = len(np.arange(args.snr_min, args.snr_max + args.snr_step/2, args.snr_step))
    total_base = len(CLASS_NAMES) * n_snrs * args.n
    print(f"Generating {len(CLASS_NAMES)} classes × {n_snrs} SNRs × {args.n} samples = {total_base} base samples")
    print(f"  IQ length: {args.len} samples")

    X, y, class_names = generate(
        n=args.n, length=args.len,
        snr_min=args.snr_min, snr_max=args.snr_max, snr_step=args.snr_step,
        seed=args.seed,
    )
    snrs = np.full(len(y), (args.snr_min + args.snr_max) / 2, dtype=np.float32)

    print(f"Base dataset: {len(X)} samples")

    if args.impair_copies > 0:
        rng = np.random.default_rng(args.seed + 99)
        X, y, snrs = apply_impairments(X, y, snrs, args.impair_copies, rng)
        print(f"Augmented: {len(X)} total samples")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, snrs=snrs, classes=np.array(class_names))
    print(f"Saved → {out}  shape={X.shape}  classes={len(class_names)}")


if __name__ == "__main__":
    main()
