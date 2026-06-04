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
Real-world IQ verification for AnalysisApp.

Applies four hardware impairment profiles (Clean, RTL-SDR, HackRF, OTA) to
the same synthetic signals used in harness.py and reports classifier accuracy
under each profile. This validates robustness to real hardware artefacts
without needing a physical radio or RadioML dataset.

Usage:
    python verify_realworld.py
    python verify_realworld.py --profile rtl_sdr --stress-snr
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from impairments import add_impairments, ImpairmentProfile, PROFILES
from features import FeatureExtractor
from classifier import ModulationClassifier, ProtocolMapper
from datasets import IqSample, normalise


# ── Synthetic signal generators (same as harness.py but consolidated here) ───

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def gen_bpsk(n: int = 8192, sr: float = 200e3, sps: int = 8, rng=None) -> np.ndarray:
    rng  = rng or _rng(1)
    bits = rng.integers(0, 2, n // sps)
    syms = (2 * bits - 1).astype(np.float32)
    iq   = np.repeat(syms, sps).astype(np.complex64)
    return normalise(iq[:n])


def gen_qpsk(n: int = 8192, sr: float = 200e3, sps: int = 8, rng=None) -> np.ndarray:
    rng  = rng or _rng(2)
    syms = rng.choice(np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64), n // sps)
    iq   = np.repeat(syms, sps)
    return normalise(iq[:n])


def gen_fm_nb(n: int = 8192, sr: float = 200e3, rng=None) -> np.ndarray:
    # NBFM: noise audio with max normalization (σ≈0.3 effective), 20 kHz deviation.
    # Noise audio has no periodic Bessel sidebands, so the symbol_rate detector
    # returns 0 → classifier takes the no-symbol-rate FM branch.
    # The h>3 guard in the classifier handles the case where a spurious symbol
    # rate is detected.
    rng   = rng or _rng(3)
    audio = rng.standard_normal(n).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-9
    dev   = 20e3
    phase = 2 * np.pi * dev / sr * np.cumsum(audio)
    return normalise(np.exp(1j * phase).astype(np.complex64))


def gen_am(n: int = 8192, sr: float = 200e3, rng=None) -> np.ndarray:
    # Two-tone AM DSB-LC: 800 Hz + 1200 Hz audio, 85% modulation depth
    # Two tones create four symmetric sidebands (±800, ±1200 Hz) ensuring the
    # spectral_symmetry computation is not skewed by a single-sided peak.
    t      = np.arange(n) / sr
    audio  = 0.5 * np.sin(2 * np.pi * 800  * t) + \
             0.5 * np.sin(2 * np.pi * 1200 * t)
    audio  = audio.astype(np.float32)
    return normalise((1.0 + 0.85 * audio).astype(np.complex64))


def gen_ofdm(n: int = 8192, sr: float = 1e6, nfft: int = 64,
             cp_len: int = 16, rng=None) -> np.ndarray:
    rng  = rng or _rng(5)
    syms = []
    while len(syms) * (nfft + cp_len) < n:
        fd   = rng.choice([1+1j, 1-1j, -1+1j, -1-1j], nfft).astype(np.complex64)
        td   = np.fft.ifft(fd).astype(np.complex64)
        syms.append(np.concatenate([td[-cp_len:], td]))
    iq = np.concatenate(syms)[:n]
    return normalise(iq)


def gen_fsk(n: int = 8192, sr: float = 200e3, sps: int = 20, rng=None) -> np.ndarray:
    rng  = rng or _rng(6)
    bits = rng.integers(0, 2, n // sps)
    freq = np.repeat((2 * bits - 1) * 8e3, sps).astype(np.float32)
    phase = 2 * np.pi / sr * np.cumsum(freq)
    return normalise(np.exp(1j * phase[:n]).astype(np.complex64))


def gen_css_lora(n: int = 8192, sr: float = 500e3, bw: float = 125e3, rng=None) -> np.ndarray:
    # Use 512-sample chirp sweep — chirp detector needs sufficient sweep length to
    # fit a reliable linear regression on the instantaneous frequency vs time.
    chirp_len = int(sr / bw * 512)
    t     = np.arange(chirp_len) / sr
    f0    = -bw / 2
    chirp = np.exp(1j * 2 * np.pi * (f0 * t + bw / (2 * chirp_len / sr) * t ** 2))
    reps  = (n // chirp_len) + 1
    iq    = np.tile(chirp, reps)[:n].astype(np.complex64)
    return normalise(iq)


SIGNALS: list[tuple[str, callable, float]] = [
    # (label, generator_fn, sample_rate)
    ("BPSK",     gen_bpsk,    200e3),
    ("QPSK",     gen_qpsk,    200e3),
    ("FM_NB",    gen_fm_nb,   200e3),
    ("AM_DSB_LC",gen_am,      200e3),
    ("OFDM",     gen_ofdm,    1e6),
    ("FSK",      gen_fsk,     200e3),
    ("CSS/LoRa", gen_css_lora,500e3),
]


# ── Classification helper ─────────────────────────────────────────────────────

EXPECTED: dict[str, str] = {
    "BPSK":      "BPSK",
    "QPSK":      "QPSK",
    "FM_NB":     "FM_NB",
    "AM_DSB_LC": "AM_DSB_LC",
    "OFDM":      "OFDM",
    "FSK":       "FSK",
    "CSS/LoRa":  "CSS",
}

FAMILY: dict[str, str] = {
    "BPSK": "PSK", "QPSK": "PSK",
    "FM_WB": "FM", "FM_NB": "FM",
    "AM_DSB_LC": "AM", "SSB": "AM",
    "OFDM": "OFDM", "FSK": "FSK", "MSK/GMSK": "FSK",
    "CSS": "CSS", "FHSS": "FHSS",
}


def classify_one(iq: np.ndarray, sr: float, cf: float,
                 extractor: FeatureExtractor,
                 clf: ModulationClassifier,
                 mapper: ProtocolMapper) -> tuple[str, float]:
    """Return (detected_mod, latency_ms)."""
    t0       = time.perf_counter()
    features = extractor.extract(normalise(iq), sr, cf)
    result   = clf.classify(features)
    if result.classified:
        result.hypotheses = mapper.map(features, result)
    ms = (time.perf_counter() - t0) * 1000
    mod = result.analog_modulation or result.digital_modulation or "UNCLASSIFIED"
    return mod, ms


def family_match(detected: str, expected: str) -> bool:
    return FAMILY.get(detected, detected) == FAMILY.get(expected, expected)


# ── Profile sweep ─────────────────────────────────────────────────────────────

def run_profile(profile: ImpairmentProfile,
                extractor: FeatureExtractor,
                clf: ModulationClassifier,
                mapper: ProtocolMapper,
                repeats: int = 5) -> dict:
    """
    Run all signals through the given impairment profile `repeats` times
    (different noise realisations) and return accuracy stats.
    """
    rng_base = np.random.default_rng(99)

    rows = []
    for label, gen_fn, sr in SIGNALS:
        iq_clean = gen_fn()
        expected = EXPECTED[label]
        exact_hits = 0
        fam_hits   = 0
        latencies  = []

        for rep in range(repeats):
            rng_imp = np.random.default_rng(int(rng_base.integers(0, 2**31)))
            iq_imp  = add_impairments(iq_clean, sr, profile=profile, rng=rng_imp)
            detected, ms = classify_one(iq_imp, sr, 0.0, extractor, clf, mapper)
            exact_hits += int(detected == expected)
            fam_hits   += int(family_match(detected, expected))
            latencies.append(ms)

        rows.append({
            "signal":     label,
            "expected":   expected,
            "exact_rate": exact_hits / repeats,
            "fam_rate":   fam_hits  / repeats,
            "latency_ms": sum(latencies) / len(latencies),
        })

    overall_exact = sum(r["exact_rate"] for r in rows) / len(rows)
    overall_fam   = sum(r["fam_rate"]   for r in rows) / len(rows)
    return {"rows": rows, "overall_exact": overall_exact, "overall_fam": overall_fam}


def print_profile_result(name: str, result: dict) -> None:
    print(f"\n{'═'*62}")
    print(f"  Profile: {name}")
    print(f"  Overall  Exact: {result['overall_exact']:.0%}   Family: {result['overall_fam']:.0%}")
    print(f"{'─'*62}")
    print(f"  {'Signal':<14} {'Expected':<14} {'Exact':>6}  {'Family':>7}  {'ms':>6}")
    print(f"{'─'*62}")
    for r in result["rows"]:
        ok_mark = "✓" if r["exact_rate"] == 1.0 else ("~" if r["fam_rate"] == 1.0 else "✗")
        print(f"  {r['signal']:<14} {r['expected']:<14} "
              f"{r['exact_rate']:>5.0%}  {r['fam_rate']:>6.0%}  "
              f"{r['latency_ms']:>5.1f}   {ok_mark}")


# ── SNR sweep ─────────────────────────────────────────────────────────────────

def run_snr_sweep(profile_base: ImpairmentProfile,
                  extractor: FeatureExtractor,
                  clf: ModulationClassifier,
                  mapper: ProtocolMapper) -> None:
    from impairments import ImpairmentConfig, PROFILES
    import copy

    snr_levels = [-10, -5, 0, 5, 10, 15, 20, 30]
    base_cfg   = PROFILES[profile_base]

    print(f"\n── SNR Sweep ({profile_base.value} impairments + AWGN) ──")
    print(f"  {'SNR':>5}  {'Exact':>7}  {'Family':>8}")

    for snr in snr_levels:
        cfg = ImpairmentConfig(
            snr_db=snr,
            freq_offset_ppm=base_cfg.freq_offset_ppm,
            iq_amplitude_imbalance_db=base_cfg.iq_amplitude_imbalance_db,
            iq_phase_imbalance_deg=base_cfg.iq_phase_imbalance_deg,
            phase_noise_linewidth_hz=base_cfg.phase_noise_linewidth_hz,
            dc_offset_i=base_cfg.dc_offset_i,
            dc_offset_q=base_cfg.dc_offset_q,
        )
        exact_hits = fam_hits = total = 0
        rng_base = np.random.default_rng(42)

        for label, gen_fn, sr in SIGNALS:
            iq_clean = gen_fn()
            expected = EXPECTED[label]
            for _ in range(3):
                rng_imp = np.random.default_rng(int(rng_base.integers(0, 2**31)))
                iq_imp  = add_impairments(iq_clean, sr, cfg=cfg, rng=rng_imp)
                detected, _ = classify_one(iq_imp, sr, 0.0, extractor, clf, mapper)
                exact_hits += int(detected == expected)
                fam_hits   += int(family_match(detected, expected))
                total      += 1

        print(f"  {snr:>4} dB  {exact_hits/total:>6.0%}   {fam_hits/total:>7.0%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Real-world IQ impairment verification")
    ap.add_argument("--profile", default="all",
                    choices=["all","clean","rtl_sdr","hackrf","pluto_sdr","ota"],
                    help="Hardware profile to test (default: all)")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Noise realisations per signal per profile (default 5)")
    ap.add_argument("--stress-snr", action="store_true",
                    help="Run SNR sweep across -10 to 30 dB (RTL-SDR impairments)")
    args = ap.parse_args()

    extractor = FeatureExtractor(fft_size=4096)
    clf       = ModulationClassifier()
    mapper    = ProtocolMapper()

    profiles_to_run: list[tuple[str, ImpairmentProfile]] = []
    if args.profile == "all":
        profiles_to_run = [
            ("Clean (lab)",          ImpairmentProfile.CLEAN),
            ("RTL-SDR (consumer)",   ImpairmentProfile.RTL_SDR),
            ("HackRF (mid-range)",   ImpairmentProfile.HACKRF),
            ("PlutoSDR (good IQ)",   ImpairmentProfile.PLUTO_SDR),
            ("Over-the-Air (OTA)",   ImpairmentProfile.OVER_THE_AIR),
        ]
    else:
        name_map = {
            "clean":     ("Clean (lab)",         ImpairmentProfile.CLEAN),
            "rtl_sdr":   ("RTL-SDR",             ImpairmentProfile.RTL_SDR),
            "hackrf":    ("HackRF",              ImpairmentProfile.HACKRF),
            "pluto_sdr": ("PlutoSDR",            ImpairmentProfile.PLUTO_SDR),
            "ota":       ("Over-the-Air (OTA)",  ImpairmentProfile.OVER_THE_AIR),
        }
        profiles_to_run = [name_map[args.profile]]

    print("AnalysisApp — Real-World IQ Verification")
    print(f"Signals: {len(SIGNALS)}   Repeats per signal: {args.repeats}")

    all_results = []
    for name, profile in profiles_to_run:
        result = run_profile(profile, extractor, clf, mapper, repeats=args.repeats)
        all_results.append((name, result))
        print_profile_result(name, result)

    # Summary table
    print(f"\n{'═'*62}")
    print(f"  SUMMARY")
    print(f"{'─'*62}")
    print(f"  {'Profile':<28}  {'Exact':>7}  {'Family':>8}")
    for name, result in all_results:
        print(f"  {name:<28}  {result['overall_exact']:>6.0%}   {result['overall_fam']:>7.0%}")
    print(f"{'═'*62}")

    if args.stress_snr:
        run_snr_sweep(ImpairmentProfile.RTL_SDR, extractor, clf, mapper)


if __name__ == "__main__":
    main()
