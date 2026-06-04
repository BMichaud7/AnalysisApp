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
Hardware impairment models for real-world IQ fidelity testing.

These impairments are present in all real SDR hardware to varying degrees:
  - AWGN           : thermal noise floor
  - Frequency offset: LO/reference oscillator error (RTL-SDR: ±50 ppm)
  - IQ imbalance   : amplitude and phase mismatch between I and Q ADC paths
  - Phase noise    : LO phase noise causing spectral spreading
  - DC offset      : LNA/mixer DC bias leaking into baseband
  - Timing jitter  : sample clock instability
  - Multipath      : for over-the-air captures

Usage:
    from impairments import add_impairments, ImpairmentProfile
    iq_impaired = add_impairments(iq_clean, sample_rate, profile=ImpairmentProfile.RTL_SDR)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass
class ImpairmentConfig:
    """Parameters controlling each impairment model."""
    # AWGN
    snr_db: float = 20.0

    # Frequency offset (fractional ppm of centre freq, or Hz absolute)
    freq_offset_ppm: float = 0.0          # applied as fraction of sample_rate
    freq_offset_hz: float = 0.0           # Hz absolute (overrides ppm if nonzero)

    # IQ imbalance
    iq_amplitude_imbalance_db: float = 0.0   # dB — Q amplitude error
    iq_phase_imbalance_deg: float = 0.0       # degrees — Q phase error

    # Phase noise (Lorentzian model, Hz linewidth at -3 dB)
    phase_noise_linewidth_hz: float = 0.0

    # DC offset (fraction of signal RMS, complex)
    dc_offset_i: float = 0.0
    dc_offset_q: float = 0.0

    # Timing jitter (fraction of sample period, sigma)
    timing_jitter_sigma: float = 0.0

    # Multipath (list of (delay_samples, amplitude, phase_deg) tuples)
    multipath_taps: list[tuple[float, float, float]] = field(default_factory=list)


class ImpairmentProfile(Enum):
    """Pre-defined profiles matching real hardware characteristics."""
    CLEAN         = "clean"         # minimal impairments (lab conditions)
    RTL_SDR       = "rtl_sdr"       # RTL-SDR dongle (worst-case consumer)
    HACKRF        = "hackrf"        # HackRF One (mid-range)
    PLUTO_SDR     = "pluto_sdr"     # ADALM-PlutoSDR (good IQ balance)
    OVER_THE_AIR  = "ota"           # adds multipath + Doppler-like drift


PROFILES: dict[ImpairmentProfile, ImpairmentConfig] = {
    ImpairmentProfile.CLEAN: ImpairmentConfig(
        snr_db=40,
        freq_offset_ppm=0.5,
        iq_amplitude_imbalance_db=0.1,
        iq_phase_imbalance_deg=0.5,
        phase_noise_linewidth_hz=10,
        dc_offset_i=0.002,
        dc_offset_q=0.001,
    ),
    ImpairmentProfile.RTL_SDR: ImpairmentConfig(
        snr_db=25,
        freq_offset_ppm=30,          # RTL-SDR: up to ±50 ppm without correction
        iq_amplitude_imbalance_db=0.5,
        iq_phase_imbalance_deg=3.0,
        phase_noise_linewidth_hz=500,
        dc_offset_i=0.02,
        dc_offset_q=0.015,
        timing_jitter_sigma=0.002,
    ),
    ImpairmentProfile.HACKRF: ImpairmentConfig(
        snr_db=30,
        freq_offset_ppm=5,
        iq_amplitude_imbalance_db=0.3,
        iq_phase_imbalance_deg=1.5,
        phase_noise_linewidth_hz=200,
        dc_offset_i=0.005,
        dc_offset_q=0.003,
        timing_jitter_sigma=0.001,
    ),
    ImpairmentProfile.PLUTO_SDR: ImpairmentConfig(
        snr_db=35,
        freq_offset_ppm=1,
        iq_amplitude_imbalance_db=0.15,
        iq_phase_imbalance_deg=0.8,
        phase_noise_linewidth_hz=50,
        dc_offset_i=0.003,
        dc_offset_q=0.002,
    ),
    ImpairmentProfile.OVER_THE_AIR: ImpairmentConfig(
        snr_db=20,
        freq_offset_ppm=10,
        iq_amplitude_imbalance_db=0.4,
        iq_phase_imbalance_deg=2.0,
        phase_noise_linewidth_hz=300,
        dc_offset_i=0.01,
        dc_offset_q=0.008,
        timing_jitter_sigma=0.003,
        multipath_taps=[
            (2.1,  0.4,  45.0),   # 2 samples delayed, 40% amplitude, 45° phase
            (5.3,  0.2, -30.0),   # 5 samples delayed, 20% amplitude, -30° phase
            (11.7, 0.08, 90.0),
        ],
    ),
}


def add_impairments(iq: np.ndarray,
                    sample_rate: float,
                    cfg: ImpairmentConfig | None = None,
                    profile: ImpairmentProfile = ImpairmentProfile.RTL_SDR,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Apply hardware impairments to a complex IQ array.

    Parameters
    ----------
    iq          : complex64 input signal, unit power assumed
    sample_rate : samples per second
    cfg         : explicit ImpairmentConfig (overrides profile if given)
    profile     : hardware profile enum (used when cfg is None)
    rng         : numpy random Generator for reproducibility

    Returns
    -------
    complex64 array with impairments applied
    """
    if cfg is None:
        cfg = PROFILES[profile]

    if rng is None:
        rng = np.random.default_rng()

    x = iq.astype(np.complex128)
    n = len(x)

    # ── Multipath ──────────────────────────────────────────────────────────────
    if cfg.multipath_taps:
        y = x.copy()
        for delay, amp, phase_deg in cfg.multipath_taps:
            ph = np.exp(1j * np.radians(phase_deg))
            d_int = int(delay)
            d_frac = delay - d_int
            # Integer delay
            delayed = np.zeros_like(x)
            if d_int < n:
                delayed[d_int:] = x[:n - d_int]
            # Fractional delay (linear interpolation)
            if d_frac > 0 and d_int + 1 < n:
                delayed2 = np.zeros_like(x)
                delayed2[d_int + 1:] = x[:n - d_int - 1]
                delayed = (1 - d_frac) * delayed + d_frac * delayed2
            y += amp * ph * delayed
        x = y

    # ── Frequency offset ───────────────────────────────────────────────────────
    if cfg.freq_offset_hz != 0.0:
        f_off = cfg.freq_offset_hz
    elif cfg.freq_offset_ppm != 0.0:
        f_off = cfg.freq_offset_ppm * 1e-6 * sample_rate
    else:
        f_off = 0.0

    if f_off != 0.0:
        t = np.arange(n) / sample_rate
        x = x * np.exp(1j * 2 * np.pi * f_off * t)

    # ── Phase noise (Lorentzian — random walk in phase) ────────────────────────
    if cfg.phase_noise_linewidth_hz > 0:
        phase_noise_sigma = np.sqrt(
            2 * np.pi * cfg.phase_noise_linewidth_hz / sample_rate
        )
        phase_walk = np.cumsum(rng.normal(0, phase_noise_sigma, n))
        x = x * np.exp(1j * phase_walk)

    # ── IQ imbalance ───────────────────────────────────────────────────────────
    if cfg.iq_amplitude_imbalance_db != 0.0 or cfg.iq_phase_imbalance_deg != 0.0:
        alpha = 10 ** (cfg.iq_amplitude_imbalance_db / 20.0)   # Q amplitude scale
        phi   = np.radians(cfg.iq_phase_imbalance_deg)          # Q phase error
        I = x.real
        Q = x.imag * alpha * np.cos(phi) + x.real * np.sin(phi)
        x = (I + 1j * Q).astype(np.complex128)

    # ── DC offset ──────────────────────────────────────────────────────────────
    pwr = np.mean(np.abs(x) ** 2)
    rms = np.sqrt(pwr) if pwr > 0 else 1.0
    x = x + (cfg.dc_offset_i + 1j * cfg.dc_offset_q) * rms

    # ── Timing jitter (resample at perturbed time points) ─────────────────────
    if cfg.timing_jitter_sigma > 0:
        t_ideal  = np.arange(n, dtype=np.float64)
        t_jitter = t_ideal + rng.normal(0, cfg.timing_jitter_sigma, n)
        t_jitter = np.clip(t_jitter, 0, n - 1)
        idx_lo = np.floor(t_jitter).astype(int)
        idx_hi = np.minimum(idx_lo + 1, n - 1)
        frac   = t_jitter - idx_lo
        x = x[idx_lo] * (1 - frac) + x[idx_hi] * frac

    # ── AWGN ───────────────────────────────────────────────────────────────────
    if cfg.snr_db < 100:
        sig_pwr    = np.mean(np.abs(x) ** 2)
        noise_pwr  = sig_pwr / (10 ** (cfg.snr_db / 10))
        noise      = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        noise     *= np.sqrt(noise_pwr / 2)
        x = x + noise

    return x.astype(np.complex64)

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
