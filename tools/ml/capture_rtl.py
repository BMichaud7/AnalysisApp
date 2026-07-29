#!/usr/bin/env python3
# ========================================================================
# Project: OpenRFStack
# Author:  Brendan Michaud
# Year:    2026
# Part of OpenRFStack (https://github.com/OpenRFStack)
#
# Licensed under the Personal Use License.
# Do not use for commercial, ordinary, or military purposes.
# Contact author for permission: https://github.com/OpenRFStack
# ========================================================================
"""
capture_rtl.py — Capture labeled IQ data from RTL-SDR using rtl_sdr CLI.

Tunes to known signal frequencies, captures raw IQ via subprocess pipe,
slices into 1024-sample windows, filters by SNR, and saves a .npz dataset
compatible with train.py / real_combined_v*.npz.

RTL-SDR covers ~25 MHz – 1.75 GHz. Targets are a subset of capture_real.py
limited to that range.

Usage:
    python3 capture_rtl.py [--out data/real_pi_20260715.npz] \
                            [--gain 35] [--snr-min 3] [--max-per-class 3000]
"""

import argparse
import collections
import subprocess
import sys
import pathlib
import numpy as np

WINDOW = 1024
STRIDE = 256
RTL_SAMPLE_RATE = 2_048_000   # default 2.048 Msps — good for most targets

# (label, center_hz, sample_rate_sps, dwell_s, max_windows)
# Only frequencies accessible by RTL-SDR (< ~1.75 GHz)
TARGETS = [
    # FM broadcast — always strong
    ("FM_WB",  89.1e6,  2_048_000, 12, 2000),
    ("FM_WB",  93.1e6,  2_048_000, 12, 2000),
    ("FM_WB", 100.8e6,  2_048_000, 12, 2000),
    ("FM_WB", 102.3e6,  2_048_000, 12, 2000),
    ("FM_WB", 106.4e6,  2_048_000, 12, 2000),

    # Aviation AM (guard / ATC)
    ("AM_DSB", 121.5e6,  1_024_000, 20, 1500),
    ("AM_DSB", 123.45e6, 1_024_000, 20, 1500),
    ("AM_DSB", 132.7e6,  1_024_000, 20, 1500),

    # ILS Localizer AM
    ("AM_DSB", 108.1e6, 512_000, 20, 1500),
    ("AM_DSB", 109.9e6, 512_000, 20, 1500),
    ("AM_DSB", 111.1e6, 512_000, 20, 1500),

    # APRS FSK 1200-baud
    ("FSK", 144.390e6, 1_024_000, 40, 2000),

    # NOAA Weather Radio FM_NB
    ("FM_NB", 162.400e6, 1_024_000, 15, 2000),
    ("FM_NB", 162.425e6, 1_024_000, 15, 2000),
    ("FM_NB", 162.450e6, 1_024_000, 15, 2000),
    ("FM_NB", 162.500e6, 1_024_000, 15, 2000),
    ("FM_NB", 162.550e6, 1_024_000, 15, 2000),

    # Maritime VHF guard
    ("FM_NB", 156.800e6, 1_024_000, 15, 1500),

    # Paging (GMSK / POCSAG)
    ("GMSK", 152.000e6, 1_024_000, 20, 1500),
    ("GMSK", 157.450e6, 1_024_000, 20, 1500),

    # ISM 915 MHz (LoRa / FSK)
    ("FSK", 903.0e6, 2_048_000, 20, 2000),
    ("FSK", 915.0e6, 2_048_000, 20, 2000),
    ("FSK", 925.0e6, 2_048_000, 20, 1500),

    # ADS-B 1090 MHz — always active near airports / aircraft overhead
    ("ADS_B", 1090.0e6, 2_048_000, 40, 3000),

    # ACARS aviation VHF
    ("ACARS", 129.125e6, 512_000, 40, 2000),
    ("ACARS", 130.025e6, 512_000, 40, 1500),
    ("ACARS", 136.900e6, 512_000, 40, 2000),

    # LTE downlinks (within RTL-SDR range)
    ("OFDM",  739.0e6, 2_048_000, 15, 2000),
    ("OFDM",  751.0e6, 2_048_000, 15, 2000),
    ("OFDM",  881.0e6, 2_048_000, 15, 2000),
]


def estimate_snr(window: np.ndarray) -> float:
    """Rough SNR estimate: signal power vs noise floor (in dB)."""
    power = np.mean(np.abs(window) ** 2)
    if power <= 0:
        return -99.0
    # sort magnitudes; lower 20% is noise floor estimate
    mags = np.sort(np.abs(window))
    n_noise = max(1, len(mags) // 5)
    noise_floor = np.mean(mags[:n_noise] ** 2) + 1e-12
    snr_db = 10 * np.log10(power / noise_floor)
    return float(snr_db)


def capture_target(label: str, freq_hz: float, sr_sps: int, dwell_s: float,
                   gain: float, verbose: bool) -> np.ndarray:
    """Capture dwell_s seconds of IQ from rtl_sdr, return complex64 array."""
    n_samples = int(sr_sps * dwell_s)

    cmd = [
        "rtl_sdr",
        "-f", str(int(freq_hz)),
        "-s", str(sr_sps),
        "-g", str(gain),
        "-n", str(n_samples),   # rtl_sdr -n takes sample count, not byte count
        "-",
    ]
    if verbose:
        print(f"  $ {' '.join(cmd)}", flush=True)

    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=dwell_s + 10
        )
        if result.returncode != 0:
            if verbose:
                print(f"  rtl_sdr error: {result.stderr.decode(errors='replace')[:200]}", flush=True)
            return np.array([], dtype=np.complex64)

        raw = np.frombuffer(result.stdout, dtype=np.uint8)
        if len(raw) < 2:
            return np.array([], dtype=np.complex64)

        # Trim to even length
        raw = raw[: (len(raw) // 2) * 2]
        iq = raw.astype(np.float32) - 127.5
        iq /= 127.5
        return (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"  rtl_sdr timed out for {label} @ {freq_hz/1e6:.3f} MHz", flush=True)
        return np.array([], dtype=np.complex64)
    except FileNotFoundError:
        print("ERROR: rtl_sdr not found in PATH", file=sys.stderr)
        sys.exit(1)


def slice_windows(iq: np.ndarray, snr_min: float) -> list[np.ndarray]:
    """Slice IQ into WINDOW-length chunks, filter by SNR."""
    windows = []
    for start in range(0, len(iq) - WINDOW + 1, STRIDE):
        w = iq[start: start + WINDOW]
        if estimate_snr(w) >= snr_min:
            windows.append(w)
    return windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_pi.npz")
    ap.add_argument("--gain", type=float, default=35.0,
                    help="RTL-SDR gain in dB (0 = auto)")
    ap.add_argument("--snr-min", type=float, default=3.0)
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"RTL-SDR Real IQ Capture  →  {out_path}", flush=True)
    print(f"gain={args.gain} dB  snr-min={args.snr_min} dB  max-per-class={args.max_per_class}", flush=True)
    print(f"{len(TARGETS)} targets", flush=True)

    buckets: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    counts: dict[str, int] = collections.defaultdict(int)

    for label, freq_hz, sr_sps, dwell_s, max_win in TARGETS:
        if counts[label] >= args.max_per_class:
            print(f"[{label}]  class full, skipping {freq_hz/1e6:.3f} MHz", flush=True)
            continue

        print(f"[{label}]  {freq_hz/1e6:.3f} MHz  sr={sr_sps/1e6:.3f} Msps  {dwell_s}s ...",
              end="  ", flush=True)

        iq = capture_target(label, freq_hz, sr_sps, dwell_s, args.gain, args.verbose)
        if len(iq) == 0:
            print("no data", flush=True)
            continue

        windows = slice_windows(iq, args.snr_min)
        room = args.max_per_class - counts[label]
        windows = windows[:min(len(windows), room, max_win)]

        for w in windows:
            buckets[label].append(
                np.stack([w.real, w.imag], axis=0).astype(np.float32)  # (2, WINDOW)
            )
        counts[label] += len(windows)
        print(f"+{len(windows)} windows  (total {counts[label]})", flush=True)

    if not buckets:
        print("No data captured — check RTL-SDR connection.", file=sys.stderr)
        sys.exit(1)

    # Build X (N, 2, WINDOW) and y arrays compatible with train.py / capture_real.py
    all_labels = sorted(buckets.keys())
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    X_list, y_list = [], []
    for label, wins in buckets.items():
        idx = label_to_idx[label]
        for w in wins:
            X_list.append(w)        # already (2, WINDOW) float32
            y_list.append(idx)

    X = np.stack(X_list, axis=0)   # (N, 2, WINDOW)
    y = np.array(y_list, dtype=np.int64)

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        classes=np.array(all_labels),
        snrs=np.zeros(len(y_list), dtype=np.float32),   # placeholder
    )

    print(f"\nSaved {len(y)} windows, {len(all_labels)} classes → {out_path}", flush=True)
    for lbl in all_labels:
        print(f"  {lbl}: {counts[lbl]}", flush=True)


if __name__ == "__main__":
    main()
