#!/usr/bin/env python3
"""
generate_v3.py — High-fidelity 28-class AMR dataset generator.

Key improvements over generate_extended.py:
  - Root-raised-cosine (RRC) pulse shaping for all digital modes
  - Variable samples-per-symbol (4–16) per sample instance
  - FM_WB includes 19 kHz stereo pilot + 38 kHz DSB-SC L-R channel
  - Multiple OFDM variants (32/64/128 subcarriers, different CP)
  - Variable symbol rates / tone spacing for FSK variants
  - PlutoSDR-realistic impairments: IQ imbalance, DC offset, phase noise, freq offset

Usage:
    python generate_v3.py --out data/synth_v3.npz --n 2500 --len 512
    # n = samples per class per SNR step; with 9 SNR steps → 2500×9×28 = 630k total
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

# ── Constants ─────────────────────────────────────────────────────────────────

SR  = 200_000.0   # reference sample rate for all generators

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
    """Root-raised cosine FIR filter (matched filter pair)."""
    N = span * sps
    t = np.arange(-N // 2, N // 2 + 1, dtype=float) / sps
    h = np.zeros(len(t))
    alpha = rolloff
    T = 1.0
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = (1 - alpha + 4 * alpha / math.pi)
        elif abs(ti) == T / (2 * alpha):
            h[i] = (alpha / math.sqrt(2)) * (
                (1 + 2 / math.pi) * math.sin(math.pi / (4 * alpha))
                + (1 - 2 / math.pi) * math.cos(math.pi / (4 * alpha))
            )
        else:
            num = math.sin(math.pi * ti * (1 - alpha) / T) + 4 * alpha * ti / T * math.cos(math.pi * ti * (1 + alpha) / T)
            den = math.pi * ti / T * (1 - (4 * alpha * ti / T) ** 2)
            h[i] = num / den
    h /= math.sqrt(np.sum(h ** 2))
    return h.astype(np.float32)


def _apply_rrc(symbols: np.ndarray, sps: int, rolloff: float = 0.35) -> np.ndarray:
    """Upsample + apply RRC pulse shaping."""
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    h = _rrc(sps, rolloff)
    real = lfilter(h, [1.0], upsampled.real)
    imag = lfilter(h, [1.0], upsampled.imag)
    return (real + 1j * imag).astype(np.complex64)


def _impair(iq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply PlutoSDR-realistic hardware impairments."""
    # DC offset
    dc = rng.uniform(-0.02, 0.02) + 1j * rng.uniform(-0.02, 0.02)
    iq = iq + dc
    # IQ amplitude imbalance (±0.5 dB)
    amp = 10 ** (rng.uniform(-0.05, 0.05) / 20)
    iq = iq.real * amp + 1j * iq.imag / amp
    # IQ phase skew (±3°)
    phi = rng.uniform(-0.05, 0.05)
    iq = iq.real * (1 + 0j) + iq.imag * (math.sin(phi) + 1j * math.cos(phi))
    # Frequency offset (±2% of SR)
    fo = rng.uniform(-0.02, 0.02) * SR
    t  = np.arange(len(iq)) / SR
    iq = iq * np.exp(1j * 2 * math.pi * fo * t).astype(np.complex64)
    # Phase noise
    pn = np.cumsum(rng.standard_normal(len(iq)) * 0.005).astype(np.float32)
    iq = iq * np.exp(1j * pn)
    return iq.astype(np.complex64)


# ── Signal generators ─────────────────────────────────────────────────────────

def _constellation(M: int, kind: str = "qam") -> np.ndarray:
    """Generate PSK or QAM constellation points."""
    if kind == "psk":
        angles = np.arange(M) * 2 * math.pi / M + math.pi / M
        return (np.cos(angles) + 1j * np.sin(angles)).astype(np.complex64)
    # Gray-coded rectangular QAM
    side = int(math.sqrt(M))
    if side * side == M:  # square QAM
        levels = np.arange(-(side - 1), side, 2, dtype=float)
        pts = np.array([x + 1j * y for x in levels for y in levels], dtype=np.complex64)
    else:  # cross QAM (QAM32)
        # Approximate: use a larger square and trim corners
        side2 = int(math.sqrt(M * 1.5))
        levels = np.arange(-(side2 - 1), side2, 2, dtype=float)
        pts = np.array([x + 1j * y for x in levels for y in levels], dtype=np.complex64)
        pts = pts[np.abs(pts) <= side2 * 0.9][:M]
    pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts.astype(np.complex64)


def _digital(constellation: np.ndarray, n: int, rng: np.random.Generator,
             sps: int = 8, rolloff: float = 0.35) -> np.ndarray:
    n_syms = math.ceil(n / sps) + 20  # extra for filter delay
    syms   = rng.choice(constellation, n_syms)
    iq     = _apply_rrc(syms, sps, rolloff)
    # Strip filter delay (half filter length)
    delay  = (len(_rrc(sps, rolloff)) - 1) // 2
    iq     = iq[delay:delay + n]
    return _norm(iq)


def _ask(M: int, n: int, rng: np.random.Generator, sps: int = 8) -> np.ndarray:
    levels = np.arange(1, M + 1, dtype=np.float32) / M * 2 - 1
    levels = (levels / np.sqrt(np.mean(levels ** 2))).astype(np.complex64)
    return _digital(levels, n, rng, sps)


def _fsk(M: int, n: int, rng: np.random.Generator,
         sps: int = 8, dev: float = None) -> np.ndarray:
    if dev is None:
        dev = SR * 0.04 / max(M - 1, 1)
    freqs = (np.arange(M) - (M - 1) / 2) * dev * 2
    n_syms = math.ceil(n / sps) + 1
    sym_idx = rng.integers(0, M, n_syms)
    phase = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sym = sym_idx[min(i // sps, n_syms - 1)]
        phase[i] = 2 * math.pi * freqs[sym] / SR
    return _norm(np.exp(1j * np.cumsum(phase)).astype(np.complex64))


def _gfsk(M: int, n: int, rng: np.random.Generator,
          sps: int = 8, dev: float = None, bt: float = 0.5) -> np.ndarray:
    freqs = (np.arange(M) - (M - 1) / 2)
    if dev is None:
        dev = SR * 0.025
    n_syms = math.ceil(n / sps) + 5
    sym_idx = rng.integers(0, M, n_syms)
    # Gaussian filter for BT product
    t = np.arange(-3 * sps, 3 * sps + 1) / sps
    g = np.exp(-2 * math.pi ** 2 * bt ** 2 * t ** 2 / math.log(2))
    g /= g.sum()
    freq_seq = np.repeat(freqs[sym_idx] * dev * 2, sps)[:n + len(g)]
    smoothed = lfilter(g, [1.0], freq_seq)[:n]
    phase = np.cumsum(2 * math.pi * smoothed / SR)
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _fm_wb(n: int, rng: np.random.Generator) -> np.ndarray:
    """FM broadcast: mono + 19 kHz stereo pilot + 38 kHz DSB-SC L-R."""
    t      = np.arange(n) / SR
    # Mono audio (bandlimited to ~15 kHz, simulated as filtered noise)
    audio  = rng.standard_normal(n).astype(np.float32)
    # Low-pass envelope
    fc     = 15000.0 / SR
    b      = np.sinc(2 * fc * np.arange(-32, 33)).astype(np.float32)
    b     /= b.sum()
    audio  = lfilter(b, [1.0], audio).astype(np.float32)
    audio /= max(np.abs(audio).max(), 1e-9)

    # Stereo pilot at 19 kHz
    pilot_amp = rng.uniform(0.08, 0.12)
    pilot     = pilot_amp * np.sin(2 * math.pi * 19000 * t)

    # L-R at 38 kHz (DSB-SC with audio modulation), ~40% of random fraction of mono
    lr_amp    = rng.uniform(0.05, 0.15)
    lr_audio  = (audio * rng.uniform(-1, 1)).astype(np.float32)
    lr_sub    = lr_amp * lr_audio * np.cos(2 * math.pi * 38000 * t)

    composite = audio + pilot + lr_sub
    composite /= max(np.abs(composite).max(), 1e-9)

    # FM modulate: 75 kHz deviation
    dev   = 75_000.0
    phase = 2 * math.pi * dev / SR * np.cumsum(composite)
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _fm_nb(n: int, rng: np.random.Generator) -> np.ndarray:
    """Narrowband FM: 5 kHz deviation, voice-like audio."""
    audio = rng.standard_normal(n).astype(np.float32)
    # Bandpass to 300-3400 Hz (voice)
    b = np.sinc(2 * 3400 / SR * np.arange(-32, 33)) - np.sinc(2 * 300 / SR * np.arange(-32, 33))
    b = b.astype(np.float32); b /= b.sum() if b.sum() > 0 else 1
    audio = lfilter(b, [1.0], audio).astype(np.float32)
    audio /= max(np.abs(audio).max(), 1e-9)
    dev   = rng.uniform(3000, 6000)
    phase = 2 * math.pi * dev / SR * np.cumsum(audio)
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _am_dsb(n: int, rng: np.random.Generator) -> np.ndarray:
    audio = rng.standard_normal(n).astype(np.float32)
    depth = rng.uniform(0.5, 0.95)
    env   = (1.0 + depth * audio / max(np.abs(audio).max(), 1e-9)).astype(np.float32)
    env   = np.clip(env, 0.0, None)
    phase = rng.uniform(0, 2 * math.pi)
    return _norm((env * np.exp(1j * phase)).astype(np.complex64))


def _am_dsb_sc(n: int, rng: np.random.Generator) -> np.ndarray:
    audio = rng.standard_normal(n).astype(np.float32)
    audio /= max(np.abs(audio).max(), 1e-9)
    return _norm(audio.astype(np.complex64))


def _am_ssb(n: int, upper: bool, rng: np.random.Generator) -> np.ndarray:
    audio  = rng.standard_normal(n).astype(np.float32)
    spec   = np.fft.fft(audio)
    freqs  = np.fft.fftfreq(n)
    spec[freqs <= 0 if upper else freqs >= 0] = 0
    return _norm(np.fft.ifft(spec).astype(np.complex64))


def _ofdm(n: int, nfft: int, cp: int, rng: np.random.Generator) -> np.ndarray:
    consts = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64) / math.sqrt(2)
    sym_len = nfft + cp
    n_sym   = math.ceil(n / sym_len) + 1
    syms    = []
    for _ in range(n_sym):
        fd   = rng.choice(consts, nfft)
        td   = np.fft.ifft(fd).astype(np.complex64)
        cyc  = np.concatenate([td[-cp:], td])
        syms.append(cyc)
    return _norm(np.concatenate(syms)[:n])


def _ofdm_variant(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random OFDM parameters to increase intra-class diversity."""
    nfft = rng.choice([32, 64, 128])
    cp   = rng.choice([nfft // 8, nfft // 4])
    return _ofdm(n, int(nfft), int(cp), rng)


def _css(n: int, rng: np.random.Generator) -> np.ndarray:
    bw  = rng.uniform(SR * 0.15, SR * 0.40)
    cl  = max(int(SR / bw), 8)
    t   = np.arange(cl) / SR
    c   = np.exp(1j * 2 * math.pi * (-bw / 2 * t + bw / (2 * cl / SR) * t ** 2))
    start = rng.integers(0, cl)
    base  = np.tile(c, n // cl + 2)
    return _norm(base[start:start + n].astype(np.complex64))


def _lfm(n: int, rng: np.random.Generator) -> np.ndarray:
    bw  = rng.uniform(SR * 0.20, SR * 0.45)
    d   = rng.choice([-1, 1])
    t   = np.arange(n) / SR
    ph  = 2 * math.pi * (d * (-bw / 2) * t + d * bw / (2 * n / SR) * t ** 2)
    return _norm(np.exp(1j * ph).astype(np.complex64))


def _tone(n: int, rng: np.random.Generator) -> np.ndarray:
    freq = rng.uniform(-SR * 0.35, SR * 0.35)
    t    = np.arange(n) / SR
    return _norm(np.exp(1j * 2 * math.pi * freq * t).astype(np.complex64))


def _ook(n: int, rng: np.random.Generator, sps: int = 8) -> np.ndarray:
    levels = np.array([0.0 + 0j, 1.0 + 0j], dtype=np.complex64)
    return _digital(levels, n, rng, sps)


# ── Class registry ────────────────────────────────────────────────────────────

PSK_CONSTS = {
    "BPSK":  _constellation(2,   "psk"),
    "QPSK":  _constellation(4,   "psk"),
    "8PSK":  _constellation(8,   "psk"),
    "16PSK": _constellation(16,  "psk"),
    "32PSK": _constellation(32,  "psk"),
}
QAM_CONSTS = {
    "QAM16":  _constellation(16,  "qam"),
    "QAM32":  _constellation(32,  "qam"),
    "QAM64":  _constellation(64,  "qam"),
    "QAM256": _constellation(256, "qam"),
}

CLASS_NAMES = sorted([
    "OOK", "4ASK", "16ASK",
    "FSK", "4FSK", "8FSK", "MSK", "GFSK", "GMSK",
    "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "QAM16", "QAM32", "QAM64", "QAM256",
    "FM_NB", "FM_WB", "AM_DSB", "AM_DSB_SC", "AM_SSB_LSB", "AM_SSB_USB",
    "OFDM", "CSS", "LFM", "TONE",
])


def _gen_one(cls: str, n: int, rng: np.random.Generator) -> np.ndarray:
    sps = int(rng.integers(4, 13))  # 4–12 sps for variety
    ro  = float(rng.uniform(0.20, 0.50))  # RRC rolloff 0.20–0.50
    if cls == "OOK":       return _ook(n, rng, sps)
    if cls == "4ASK":      return _ask(4,   n, rng, sps)
    if cls == "16ASK":     return _ask(16,  n, rng, max(sps, 6))
    if cls == "FSK":       return _fsk(2,   n, rng, sps)
    if cls == "4FSK":      return _fsk(4,   n, rng, sps)
    if cls == "8FSK":      return _fsk(8,   n, rng, max(sps, 6))
    if cls == "MSK":       return _fsk(2,   n, rng, sps, SR * 0.025)
    if cls == "GFSK":      return _gfsk(2,  n, rng, sps, SR * 0.05, float(rng.uniform(0.3, 0.6)))
    if cls == "GMSK":      return _gfsk(2,  n, rng, sps, SR * 0.025, 0.3)
    if cls in PSK_CONSTS:  return _digital(PSK_CONSTS[cls], n, rng, sps, ro)
    if cls in QAM_CONSTS:  return _digital(QAM_CONSTS[cls], n, rng, max(sps, 5), ro)
    if cls == "FM_WB":     return _fm_wb(n, rng)
    if cls == "FM_NB":     return _fm_nb(n, rng)
    if cls == "AM_DSB":    return _am_dsb(n, rng)
    if cls == "AM_DSB_SC": return _am_dsb_sc(n, rng)
    if cls == "AM_SSB_USB": return _am_ssb(n, True,  rng)
    if cls == "AM_SSB_LSB": return _am_ssb(n, False, rng)
    if cls == "OFDM":      return _ofdm_variant(n, rng)
    if cls == "CSS":       return _css(n, rng)
    if cls == "LFM":       return _lfm(n, rng)
    if cls == "TONE":      return _tone(n, rng)
    raise ValueError(f"Unknown class: {cls}")


# ── Dataset generation ────────────────────────────────────────────────────────

def generate(n_per_snr: int, length: int,
             snr_min: float, snr_max: float, snr_step: float,
             seed: int, impair: bool = True,
             filter_classes: list[str] | None = None) -> tuple:
    classes = [c for c in CLASS_NAMES if filter_classes is None or c in filter_classes]
    if filter_classes:
        unknown = set(filter_classes) - set(CLASS_NAMES)
        if unknown:
            raise ValueError(f"Unknown classes: {sorted(unknown)}")
    rng   = np.random.default_rng(seed)
    snrs  = np.arange(snr_min, snr_max + snr_step / 2, snr_step)
    total = len(classes) * len(snrs) * n_per_snr
    X     = np.empty((total, 2, length), dtype=np.float32)
    y     = np.empty(total, dtype=np.int64)
    s     = np.empty(total, dtype=np.float32)
    label = {c: i for i, c in enumerate(classes)}

    idx = 0
    for ci, cls in enumerate(classes):
        for snr in snrs:
            for _ in range(n_per_snr):
                iq = _gen_one(cls, length, rng)
                if impair:
                    iq = _impair(iq, rng)
                iq = _awgn(iq, float(snr), rng)
                X[idx, 0] = iq.real
                X[idx, 1] = iq.imag
                y[idx]    = label[cls]
                s[idx]    = float(snr)
                idx += 1
        print(f"  {ci+1:2d}/{len(classes)}  {cls:<14}  {idx:,} samples", flush=True)

    return X, y, s, classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",       default="data/synth_v3.npz")
    ap.add_argument("--n",         type=int,   default=2500,
                    help="Samples per class per SNR step")
    ap.add_argument("--len",       type=int,   default=512)
    ap.add_argument("--snr-min",   type=float, default=-10.0)
    ap.add_argument("--snr-max",   type=float, default=30.0)
    ap.add_argument("--snr-step",  type=float, default=5.0)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--no-impair",      action="store_true")
    ap.add_argument("--filter-classes", nargs="+",  metavar="CLS",
                    help="Only generate these classes (subset of the 28)")
    args = ap.parse_args()

    filter_cls = args.filter_classes or None
    n_cls = len(filter_cls) if filter_cls else len(CLASS_NAMES)
    print(f"Generating {args.n} × {n_cls} classes × SNR {args.snr_min}:{args.snr_step}:{args.snr_max} dB")
    X, y, snrs, cls = generate(
        n_per_snr=args.n, length=args.len,
        snr_min=args.snr_min, snr_max=args.snr_max, snr_step=args.snr_step,
        seed=args.seed, impair=not args.no_impair,
        filter_classes=filter_cls,
    )
    print(f"\nTotal: {len(X):,}  shape={X.shape}  ({X.nbytes/1e9:.2f} GB)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, snrs=snrs, classes=np.array(cls))
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
