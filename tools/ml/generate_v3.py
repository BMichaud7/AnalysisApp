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
    # Frequency offset (±0.3% of SR — post-AFC residual).
    # ±2% SR caused 10+ full carrier rotations over 512 samples, destroying
    # all constellation structure and making all modulations look identical.
    # ±0.3% SR ≈ ±1.5 rotations — preserves constellation while staying robust.
    fo = rng.uniform(-0.003, 0.003) * SR
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


# ── Protocol-specific generators ──────────────────────────────────────────────

def _p25_c4fm(n: int, rng: np.random.Generator) -> np.ndarray:
    """P25 Phase 1 C4FM: 4-level FSK, 4800 sym/s, ±600/±1800 Hz deviation.
    Includes occasional P25 frame sync pattern for realism."""
    # At SR=200kHz: 4800 sym/s → ~41.7 samples/sym; use fixed 40 sps
    sps = 40
    # Normalised deviations (Hz / SR)
    dev = np.array([-1800, -600, 600, 1800], dtype=float) / SR
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 4, n_syms)
    # Insert P25 frame sync (0x5575F5FF77FF) dibits with 40% probability
    if rng.random() > 0.6 and n_syms > 24:
        sync_dibits = [1,3,1,3,3,3,1,3,3,3,3,3,3,3,3,3,1,3,1,3,3,3,1,3]
        syms[:min(24, n_syms)] = sync_dibits[:min(24, n_syms)]
    freq_seq = np.repeat([dev[s] for s in syms], sps)[:n].astype(np.float64)
    phase = np.cumsum(2 * math.pi * freq_seq)
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _dmr(n: int, rng: np.random.Generator) -> np.ndarray:
    """DMR (Digital Mobile Radio): 4FSK, 4800 sym/s, ±648/±1944 Hz deviation.
    TDMA 2-slot, uses AMBE+2 voice. Common in commercial/public safety."""
    sps = 40  # 200kHz/4800 ≈ 41 samples/sym
    dev = np.array([-1944, -648, 648, 1944], dtype=float) / SR
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 4, n_syms)
    # DMR voice sync: 0x755FD7DF75F7 (dibits: 1,3,1,1,3,3,1,3,1,3,3,3,3,1,3,...)
    if rng.random() > 0.5 and n_syms > 24:
        syms[:12] = [1, 3, 1, 1, 3, 3, 1, 3, 1, 3, 3, 3]
    freq_seq = np.repeat([dev[s] for s in syms], sps)[:n].astype(np.float64)
    phase = np.cumsum(2 * math.pi * freq_seq)
    # DMR is TDMA — add burst envelope (on/off pattern for 2 slots)
    burst_period = int(SR / 50)   # 50 Hz burst rate
    env = np.ones(n, dtype=np.float32)
    for start in range(0, n, burst_period):
        end = min(start + burst_period // 2, n)
        env[end:min(start + burst_period, n)] = 0.05  # guard interval
    return _norm((np.exp(1j * phase) * env).astype(np.complex64))


def _nxdn(n: int, rng: np.random.Generator) -> np.ndarray:
    """NXDN: 4FSK, 4800 sym/s, ±1050/±3150 Hz (12.5 kHz BW).
    FDMA digital voice used by Icom/Kenwood, common in industrial US."""
    sps = 40
    dev = np.array([-3150, -1050, 1050, 3150], dtype=float) / SR
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 4, n_syms)
    # NXDN Frame Sync (hex: 0xAA or pattern-specific)
    if rng.random() > 0.5 and n_syms > 8:
        syms[:4] = [2, 2, 2, 2]  # preamble pattern
    freq_seq = np.repeat([dev[s] for s in syms], sps)[:n].astype(np.float64)
    phase = np.cumsum(2 * math.pi * freq_seq)
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _dstar(n: int, rng: np.random.Generator) -> np.ndarray:
    """D-STAR: GMSK, 4800 bps, BT=0.5. Amateur digital voice."""
    return _gfsk(2, n, rng, sps=40, dev=SR * 0.024, bt=0.5)


def _tetra(n: int, rng: np.random.Generator) -> np.ndarray:
    """TETRA: π/4-DQPSK, 18000 sym/s (36 kbps), 25 kHz channel.
    European public safety trunked radio standard."""
    # π/4-QPSK: rotate constellation by π/4 each symbol
    sps = int(rng.integers(4, 8))
    angles = np.array([1, 3, 5, 7], dtype=float) * math.pi / 4
    n_syms = math.ceil(n / sps) + 20
    sym_idx = rng.integers(0, 4, n_syms)
    # Apply π/4 rotation
    phase_shifts = angles[sym_idx]
    phases = np.cumsum(phase_shifts)
    syms = np.exp(1j * phases).astype(np.complex64)
    iq = _apply_rrc(syms, sps, rolloff=0.35)
    delay = (len(_rrc(sps, 0.35)) - 1) // 2
    return _norm(iq[delay:delay + n])


def _ais(n: int, rng: np.random.Generator) -> np.ndarray:
    """AIS (Automatic Identification System): GMSK, 9600 bps, BT=0.4.
    Marine VHF channels 87B (161.975 MHz) and 88B (162.025 MHz)."""
    # 9600 baud at 200kHz SR → ~20 sps
    return _gfsk(2, n, rng, sps=20, dev=SR * 0.024, bt=0.4)


def _pocsag(n: int, rng: np.random.Generator) -> np.ndarray:
    """POCSAG: 2-FSK, 512/1200/2400 bps, ±4500 Hz deviation.
    Paging protocol still used by hospitals, fire departments."""
    baud = rng.choice([512, 1200, 2400])
    sps = max(4, int(SR / baud))
    dev = 4500.0 / SR
    freqs = np.array([-dev, dev])
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 2, n_syms)
    # POCSAG preamble: alternating 1/0 for 576 bits
    if rng.random() > 0.3:
        pre_len = min(32, n_syms)
        syms[:pre_len] = [i % 2 for i in range(pre_len)]
    freq_seq = np.repeat([freqs[s] for s in syms], sps)[:n]
    phase = np.cumsum(2 * math.pi * freq_seq)
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _acars(n: int, rng: np.random.Generator) -> np.ndarray:
    """ACARS: AM-modulated 2400 bps FSK (2400 Hz / 1200 Hz tones).
    Aircraft communications on VHF 129.125, 136.900 MHz etc."""
    t = np.arange(n) / SR
    # 2-FSK subcarrier at 1200 (space) or 2400 (mark) Hz
    baud = 2400.0
    sps = max(4, int(SR / baud))
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 2, n_syms)
    # ACARS prekey: 1 second of 2400 Hz idle
    if rng.random() > 0.5:
        syms[:min(12, n_syms)] = 1
    # FSK: 1=2400 Hz (mark), 0=1200 Hz (space)
    tones = np.array([1200.0, 2400.0])
    freq_seq = np.repeat([tones[s] for s in syms], sps)[:n]
    audio = np.sin(2 * math.pi * np.cumsum(freq_seq / SR)).astype(np.float32)
    audio /= max(np.abs(audio).max(), 1e-9)
    # AM modulate: depth 0.85, carrier at ~20% of SR
    carrier_freq = rng.uniform(0.10, 0.20) * SR
    carrier = np.sin(2 * math.pi * carrier_freq * t).astype(np.float32)
    depth = rng.uniform(0.7, 0.95)
    am = ((1.0 + depth * audio) * carrier).astype(np.float32)
    phase_angle = rng.uniform(0, 2 * math.pi)
    iq = am * np.exp(1j * phase_angle).astype(np.complex64)
    return _norm(iq)



def _flex(n: int, rng: np.random.Generator) -> np.ndarray:
    """FLEX paging: 4-FSK, 1600 baud, ±1600 Hz deviation. Includes sync 0xA8C9."""
    sps = max(4, int(SR / 1600.0))
    dev = np.array([-1600, -600, 600, 1600], dtype=float) / SR
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 4, n_syms)
    # Insert FLEX sync word 0xA8C9 as dibits
    sync_dibits = [2,2,3,1,2,2,3,1]  # 0xA8C9 approximated as 4-level
    syms[:min(8,n_syms)] = sync_dibits[:min(8,n_syms)]
    freq_seq = np.repeat([dev[s] for s in syms], sps)[:n]
    phase = np.cumsum(2 * math.pi * freq_seq)
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _mdc1200(n: int, rng: np.random.Generator) -> np.ndarray:
    """MDC-1200: 2-FSK 1200 baud, ±1200 Hz. Motorola PTT ID."""
    sps = max(4, int(SR / 1200.0))
    dev = 1200.0 / SR
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 2, n_syms)
    # Preamble: pre-tone at 1200 Hz then data
    pre_len = min(16, n_syms)
    syms[:pre_len] = 1  # idle at mark
    freq_seq = np.repeat([dev if s else -dev for s in syms], sps)[:n]
    phase = np.cumsum(2 * math.pi * freq_seq)
    return _norm(np.exp(1j * phase).astype(np.complex64))


def _dtmf(n: int, rng: np.random.Generator) -> np.ndarray:
    """DTMF: dual-tone pairs on an AM carrier. Random digit sequence."""
    rows = [697.0, 770.0, 852.0, 941.0]
    cols = [1209.0, 1336.0, 1477.0, 1633.0]
    t = np.arange(n) / SR
    signal = np.zeros(n, dtype=np.float32)
    # Generate 3-6 random DTMF digits
    n_digits = rng.integers(3, 7)
    dig_samples = n // (n_digits * 2)
    for i in range(n_digits):
        row = rng.integers(0, 4)
        col = rng.integers(0, 4)
        start = i * dig_samples * 2
        end = min(start + dig_samples, n)
        if start >= n: break
        t_seg = t[start:end]
        signal[start:end] = (np.sin(2*math.pi*rows[row]*t_seg) +
                              np.sin(2*math.pi*cols[col]*t_seg)).astype(np.float32)
    # AM modulate onto a carrier
    carrier_hz = rng.uniform(0.05, 0.15) * SR
    signal /= max(np.abs(signal).max(), 1e-9)
    carrier = np.sin(2*math.pi*carrier_hz*t).astype(np.float32)
    return _norm(((1.0 + 0.85*signal)*carrier).astype(np.complex64))


def _eas_same(n: int, rng: np.random.Generator) -> np.ndarray:
    """EAS/SAME: AFSK 520 baud, mark=2083 Hz, space=1563 Hz over AM."""
    t = np.arange(n) / SR
    baud = 520.833
    mark_hz, space_hz = 2083.3, 1562.5
    sps = max(4, int(SR / baud))
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 2, n_syms)
    # Preamble: 16×0xAB = alternating 1/0
    pre = [1 if (i//4)%2==0 else 0 for i in range(min(64,n_syms))]
    syms[:len(pre)] = pre
    audio_phase = 0.0
    audio = np.zeros(n, np.float32)
    for i in range(n):
        tone = mark_hz if syms[min(i//sps,n_syms-1)] else space_hz
        audio_phase += 2*math.pi*tone/SR
        audio[i] = math.sin(audio_phase)
    carrier_hz = rng.uniform(0.1, 0.25) * SR
    carrier = np.sin(2*math.pi*carrier_hz*t).astype(np.float32)
    return _norm(((1.0 + 0.85*audio)*carrier).astype(np.complex64))


def _rtty(n: int, rng: np.random.Generator) -> np.ndarray:
    """RTTY: 2-FSK 45.45 baud, ±85 Hz (170 Hz shift). Baudot ITA-2."""
    sps = max(4, int(SR / 45.45))
    dev = 85.0 / SR
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 2, n_syms)
    freq_seq = np.repeat([dev if s else -dev for s in syms], sps)[:n]
    phase = np.cumsum(2*math.pi*freq_seq)
    return _norm(np.exp(1j*phase).astype(np.complex64))


def _p25_phase2(n: int, rng: np.random.Generator) -> np.ndarray:
    """P25 Phase 2: π/4-DQPSK, 12000 sym/s, 12.5 kHz channel."""
    sps = max(4, int(SR / 12000.0))
    const = _constellation(4, "psk")  # use QPSK constellation
    syms = rng.choice(const, math.ceil(n/sps)+20)
    # Apply π/4 rotation per symbol
    for i in range(1, len(syms)):
        syms[i] *= np.exp(1j * math.pi / 4)
    iq = _apply_rrc(syms, sps, 0.2)
    delay = (len(_rrc(sps, 0.2))-1)//2
    return _norm(iq[delay:delay+n])


def _adsb(n: int, rng: np.random.Generator) -> np.ndarray:
    """ADS-B: Pulse-Position Modulation at 1090 MHz. Simulated PPM bursts."""
    # PPM: 1 MHz bit rate → 1μs per bit at 200kHz SR = 0.2 samples/bit → upsample
    # Simulate as OOK burst with 1μs pulses
    us = max(1, int(SR / 1e6))  # samples per microsecond
    signal = np.zeros(n, np.float32)
    # Preamble pulses at 0, 1, 3.5, 4.5 μs
    for pulse_us in [0, 1, 3.5, 4.5]:
        start = min(int(pulse_us * us), n-us)
        signal[start:min(start+us,n)] = 1.0
    # Random data bits (PPM encoded)
    n_bits = min(56, (n-8*us)//(2*us))
    for bit_i in range(int(n_bits)):
        bit = rng.integers(0, 2)
        pos = 8*us + bit_i*2*us + bit*us
        if pos+us < n: signal[pos:pos+us] = 1.0
    # Carrier
    carrier_hz = rng.uniform(0.3, 0.45) * SR
    t = np.arange(n) / SR
    carrier = (np.cos(2*math.pi*carrier_hz*t) + 1j*np.sin(2*math.pi*carrier_hz*t)).astype(np.complex64)
    return _norm((signal.astype(np.complex64) * carrier))


def _dsc(n: int, rng: np.random.Generator) -> np.ndarray:
    """DSC Digital Selective Calling: FSK 1200 baud, ±400 Hz, marine VHF."""
    sps = max(4, int(SR / 1200.0))
    dev = 400.0 / SR
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 2, n_syms)
    # DSC phasing sequence 0x7B7B
    phase_dibits = [1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]
    syms[:min(16,n_syms)] = phase_dibits[:min(16,n_syms)]
    freq_seq = np.repeat([dev if s else -dev for s in syms], sps)[:n]
    phase = np.cumsum(2*math.pi*freq_seq)
    return _norm(np.exp(1j*phase).astype(np.complex64))


def _navtex(n: int, rng: np.random.Generator) -> np.ndarray:
    """NAVTEX: FSK 100 baud, ±150 Hz, SITOR-B, maritime safety 518 kHz."""
    sps = max(4, int(SR / 100.0))
    dev = 150.0 / SR
    n_syms = math.ceil(n / sps) + 5
    syms = rng.integers(0, 2, n_syms)
    # NAVTEX starts with ZCZC — encode as alternating bits
    syms[:min(16,n_syms)] = [1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0]
    freq_seq = np.repeat([dev if s else -dev for s in syms], sps)[:n]
    phase = np.cumsum(2*math.pi*freq_seq)
    return _norm(np.exp(1j*phase).astype(np.complex64))


def _vdl2(n: int, rng: np.random.Generator) -> np.ndarray:
    """VDL Mode 2: D8PSK 10500 sym/s, 31.5 kbps, aviation 136 MHz."""
    sps = max(4, int(SR / 10500.0))
    const = _constellation(8, "psk")  # 8-PSK
    syms = rng.choice(const, math.ceil(n/sps)+20)
    iq = _apply_rrc(syms, sps, 0.6)
    delay = (len(_rrc(sps, 0.6))-1)//2
    return _norm(iq[delay:delay+n])


def _psk31(n: int, rng: np.random.Generator) -> np.ndarray:
    """PSK31: BPSK 31.25 baud, extremely narrow ~31 Hz BW. HF amateur."""
    sps = max(4, int(SR / 31.25))
    const = np.array([1.0+0j, -1.0+0j], dtype=np.complex64)  # BPSK
    n_syms = math.ceil(n / sps) + 5
    syms = rng.choice(const, n_syms)
    iq = _apply_rrc(syms, sps, 0.2)
    delay = (len(_rrc(sps, 0.2))-1)//2
    return _norm(iq[delay:delay+n])


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
    # Original 28 classes
    "OOK", "4ASK", "16ASK",
    "FSK", "4FSK", "8FSK", "MSK", "GFSK", "GMSK",
    "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "QAM16", "QAM32", "QAM64", "QAM256",
    "FM_NB", "FM_WB", "AM_DSB", "AM_DSB_SC", "AM_SSB_LSB", "AM_SSB_USB",
    "OFDM", "CSS", "LFM", "TONE",
    # Protocol-specific classes (36 total)
    "P25_C4FM",   # P25 Phase 1 control/voice channel
    "DMR",        # Digital Mobile Radio (TDMA 4FSK)
    "NXDN",       # NXDN FDMA digital voice
    "DSTAR",      # D-STAR GMSK amateur digital
    "TETRA",      # TETRA π/4-QPSK trunked
    "AIS",        # AIS marine GMSK 9600 bps
    "POCSAG",     # POCSAG 2-FSK paging
    "ACARS",      # ACARS AM-FSK aircraft comm
    "FLEX",        "MDC_1200",   "DTMF",
    "EAS_SAME",    "RTTY",       "P25_PHASE2",
    "ADS_B",       "DSC",        "NAVTEX",
    "VDL2",        "PSK31",
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
    # Protocol-specific classes
    if cls == "P25_C4FM":  return _p25_c4fm(n, rng)
    if cls == "DMR":       return _dmr(n, rng)
    if cls == "NXDN":      return _nxdn(n, rng)
    if cls == "DSTAR":     return _dstar(n, rng)
    if cls == "TETRA":     return _tetra(n, rng)
    if cls == "AIS":       return _ais(n, rng)
    if cls == "POCSAG":    return _pocsag(n, rng)
    if cls == "ACARS":     return _acars(n, rng)
    if cls == "FLEX":      return _flex(n, rng)
    if cls == "MDC_1200":  return _mdc1200(n, rng)
    if cls == "DTMF":      return _dtmf(n, rng)
    if cls == "EAS_SAME":  return _eas_same(n, rng)
    if cls == "RTTY":      return _rtty(n, rng)
    if cls == "P25_PHASE2":return _p25_phase2(n, rng)
    if cls == "ADS_B":     return _adsb(n, rng)
    if cls == "DSC":       return _dsc(n, rng)
    if cls == "NAVTEX":    return _navtex(n, rng)
    if cls == "VDL2":      return _vdl2(n, rng)
    if cls == "PSK31":     return _psk31(n, rng)
    if cls == "FLEX":      return _flex(n, rng)
    if cls == "MDC_1200":  return _mdc1200(n, rng)
    if cls == "DTMF":      return _dtmf(n, rng)
    if cls == "EAS_SAME":  return _eas_same(n, rng)
    if cls == "RTTY":      return _rtty(n, rng)
    if cls == "P25_PHASE2":return _p25_phase2(n, rng)
    if cls == "ADS_B":     return _adsb(n, rng)
    if cls == "DSC":       return _dsc(n, rng)
    if cls == "NAVTEX":    return _navtex(n, rng)
    if cls == "VDL2":      return _vdl2(n, rng)
    if cls == "PSK31":     return _psk31(n, rng)

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
