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
AnalysisApp test harness — runs the rule-based classifier against real IQ datasets
and produces a structured results table + failure analysis.

Usage:
    python harness.py --radioml-2016  /path/to/RML2016.10a.hdf5
    python harness.py --radioml-2018  /path/to/RML2018.01a.hdf5
    python harness.py --sigmf         /path/to/*.sigmf-meta
    python harness.py --raw           /path/to/signal.iq --sr 2e6 --cf 162e6 --label "AIS"
    python harness.py --all-dir       /path/to/iq_dir/
    python harness.py --help
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

# ── Local modules ─────────────────────────────────────────────────────────────
from datasets import (
    IqSample, normalise,
    load_radioml_2016, load_radioml_2018,
    load_sigmf, load_raw_iq,
)
from features import FeatureExtractor
from classifier import ModulationClassifier, ProtocolMapper, ClassificationResult


# ── Normalised ground-truth aliases ───────────────────────────────────────────
# Map RadioML label → canonical modulation family for scoring purposes

GT_ALIASES: dict[str, str] = {
    # RadioML 2016 names
    "AM-DSB":  "AM_DSB_LC",
    "AM-SSB":  "SSB",
    "WBFM":    "FM_WB",
    "CPFSK":   "FSK",
    "GFSK":    "FSK",   # GFSK ≈ Gaussian FSK
    "PAM4":    "M-ASK/OOK",
    "QAM16":   "QAM16",
    "QAM64":   "QAM64",
    "8PSK":    "8PSK",
    "BPSK":    "BPSK",
    "QPSK":    "QPSK",
    # RadioML 2018 names
    "OOK":          "M-ASK/OOK",
    "4ASK":         "M-ASK/OOK",
    "8ASK":         "M-ASK/OOK",
    "16PSK":        "PSK",
    "32PSK":        "PSK",
    "16APSK":       "PSK",
    "32APSK":       "PSK",
    "64APSK":       "PSK",
    "128APSK":      "PSK",
    "16QAM":        "QAM16",
    "32QAM":        "QAM32",
    "128QAM":       "QAM64",
    "256QAM":       "QAM256",
    "AM-SSB-WC":    "SSB",
    "AM-SSB-SC":    "AM_DSB_SC",
    "AM-DSB-WC":    "AM_DSB_LC",
    "AM-DSB-SC":    "AM_DSB_SC",
    "FM":           "FM_WB",
    "GMSK":         "MSK/GMSK",
    "OQPSK":        "QPSK",
}

MODULATION_FAMILIES: dict[str, str] = {
    "BPSK": "PSK", "QPSK": "PSK", "8PSK": "PSK", "PSK": "PSK",
    "QAM16": "QAM", "QAM32": "QAM", "QAM64": "QAM", "QAM256": "QAM",
    "MSK/GMSK": "FSK", "FSK": "FSK",
    "FM_WB": "FM", "FM_NB": "FM",
    "AM_DSB_LC": "AM", "AM_DSB_SC": "AM", "SSB": "AM", "SSB_USB": "AM", "SSB_LSB": "AM",
    "OFDM": "OFDM",
    "CSS": "CSS",
    "M-ASK/OOK": "ASK",
    "FHSS": "FHSS",
}


def canonical(label: str) -> str:
    return GT_ALIASES.get(label, label)


def family(mod: str) -> str:
    return MODULATION_FAMILIES.get(mod, mod)


# ── Single-sample classification ──────────────────────────────────────────────

def classify_sample(sample:    IqSample,
                    extractor: FeatureExtractor,
                    clf:       ModulationClassifier,
                    mapper:    ProtocolMapper) -> dict:
    """Run the full pipeline on one IQ sample. Return a result row dict."""

    iq = normalise(sample.iq)

    t0       = time.perf_counter()
    features = extractor.extract(iq, sample.sample_rate, sample.center_freq)
    t_feat   = time.perf_counter() - t0

    t0       = time.perf_counter()
    result   = clf.classify(features)
    t_clf    = time.perf_counter() - t0

    if result.classified:
        result.hypotheses = mapper.map(features, result)

    t_total  = t_feat + t_clf

    # Determine detected modulation string
    detected_mod = result.analog_modulation or result.digital_modulation or "UNCLASSIFIED"

    # Compare to ground truth
    gt_raw      = sample.ground_truth
    gt_canon    = canonical(gt_raw)
    det_canon   = canonical(detected_mod)
    exact_match = (gt_canon.upper() == det_canon.upper())
    fam_match   = (family(gt_canon) == family(det_canon))

    # Top protocol hypothesis
    top_protocol  = result.hypotheses[0].system     if result.hypotheses else ""
    top_conf      = result.hypotheses[0].confidence if result.hypotheses else 0.0
    top_reasoning = result.hypotheses[0].reasoning  if result.hypotheses else ""

    return {
        "source_file":       Path(sample.source_file).name,
        "segment":           sample.segment_idx,
        "ground_truth":      gt_raw,
        "gt_canonical":      gt_canon,
        "detected_mod":      detected_mod,
        "exact_match":       exact_match,
        "family_match":      fam_match,
        "snr_db":            round(sample.snr_db, 1),
        "bandwidth_hz":      round(features.bandwidth_hz, 0),
        "symbol_rate_sps":   round(features.symbol_rate_sps, 0),
        "is_burst":          features.is_burst,
        "ofdm_detected":     features.ofdm_detected,
        "fhss_detected":     features.fhss_detected,
        "chirp_detected":    features.chirp_detected,
        "c40":               round(features.c40_real, 3),
        "c42":               round(features.c42, 3),
        "env_var_norm":      round(features.envelope_variance_norm, 3),
        "fm_dev_hz":         round(features.fm_deviation_hz, 0),
        "top_protocol":      top_protocol,
        "top_confidence":    round(top_conf, 3),
        "top_reasoning":     top_reasoning,
        "classified":        result.classified,
        "reject_reason":     result.reject_reason,
        "latency_ms":        round(t_total * 1000, 2),
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(samples:   list[IqSample],
              extractor: FeatureExtractor,
              clf:       ModulationClassifier,
              mapper:    ProtocolMapper,
              label:     str = "") -> pd.DataFrame:

    rows = []
    total = len(samples)
    for i, s in enumerate(samples):
        try:
            row = classify_sample(s, extractor, clf, mapper)
        except Exception as e:
            row = {"source_file": Path(s.source_file).name, "segment": s.segment_idx,
                   "ground_truth": s.ground_truth, "classified": False,
                   "reject_reason": str(e)}
        rows.append(row)
        if (i+1) % 50 == 0 or (i+1) == total:
            done  = sum(1 for r in rows if r.get("exact_match"))
            print(f"  {label} [{i+1}/{total}]  exact_acc={done/(i+1):.1%}")

    return pd.DataFrame(rows)


# ── Analysis & reports ────────────────────────────────────────────────────────

def analyse(df: pd.DataFrame) -> None:
    classified = df[df.get("classified", pd.Series([True]*len(df))).fillna(True)]
    n = len(classified)
    if n == 0:
        print("No classified samples.")
        return

    exact_acc  = classified["exact_match"].mean()
    family_acc = classified["family_match"].mean()

    print("\n" + "═"*70)
    print(f"  OVERALL  Exact: {exact_acc:.1%}   Family: {family_acc:.1%}   N={n}")
    print("═"*70)

    # ── Per-modulation breakdown ──────────────────────────────────────────────
    print("\n── Per Ground-Truth Modulation ──")
    breakdown = (
        classified
        .groupby("gt_canonical")
        .agg(N=("exact_match","count"),
             exact_pct=("exact_match","mean"),
             family_pct=("family_match","mean"),
             mean_snr=("snr_db","mean"))
        .sort_values("exact_pct")
    )
    breakdown["exact_pct"]  = breakdown["exact_pct"].apply(lambda x: f"{x:.0%}")
    breakdown["family_pct"] = breakdown["family_pct"].apply(lambda x: f"{x:.0%}")
    breakdown["mean_snr"]   = breakdown["mean_snr"].apply(lambda x: f"{x:.1f} dB")
    print(breakdown.to_string())

    # ── SNR-stratified accuracy ───────────────────────────────────────────────
    if "snr_db" in classified.columns and classified["snr_db"].notna().any():
        print("\n── Accuracy by SNR Tier ──")
        bins = [-30, -10, 0, 10, 20, 40]
        labels = ["<-10", "-10–0", "0–10", "10–20", ">20"]
        classified = classified.copy()
        classified["snr_tier"] = pd.cut(classified["snr_db"], bins=bins, labels=labels)
        snr_acc = (
            classified.groupby("snr_tier", observed=True)
            .agg(N=("exact_match","count"), exact_pct=("exact_match","mean"))
        )
        snr_acc["exact_pct"] = snr_acc["exact_pct"].apply(lambda x: f"{x:.0%}")
        print(snr_acc.to_string())

    # ── Confusion analysis ────────────────────────────────────────────────────
    print("\n── Confusion Matrix (top misclassifications) ──")
    misclassified = classified[~classified["exact_match"]]
    if len(misclassified) > 0:
        conf = (
            misclassified
            .groupby(["gt_canonical","detected_mod"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(15)
        )
        print(conf.to_string(index=False))
    else:
        print("  (no misclassifications)")

    # ── Failure mode analysis ─────────────────────────────────────────────────
    print("\n── Failure Mode Analysis ──")
    if len(misclassified) > 0:
        print(f"  Total misclassified: {len(misclassified)}")
        ofdm_err = misclassified["ofdm_detected"].sum()
        fhss_err = misclassified["fhss_detected"].sum()
        burst_err = misclassified["is_burst"].sum()
        low_snr  = (misclassified["snr_db"] < 0).sum() if "snr_db" in misclassified else 0
        print(f"  of which: OFDM={ofdm_err}  FHSS={fhss_err}  burst={burst_err}  SNR<0dB={low_snr}")

        # Modulations with worst accuracy
        worst = (
            classified.groupby("gt_canonical")["exact_match"]
            .mean()
            .sort_values()
            .head(5)
        )
        print("\n  Hardest modulations (lowest exact accuracy):")
        for mod, acc in worst.items():
            print(f"    {mod:20s}  {acc:.0%}")

    # ── Protocol mapping accuracy ─────────────────────────────────────────────
    # (Only meaningful for real-world captures with known protocol ground truth)

    # ── Latency stats ─────────────────────────────────────────────────────────
    if "latency_ms" in df.columns:
        lat = df["latency_ms"].describe()
        print(f"\n── Latency  mean={lat['mean']:.1f} ms  "
              f"p50={lat['50%']:.1f} ms  p95={lat['75%']:.1f} ms")

    # ── Recommendations ───────────────────────────────────────────────────────
    print("\n── Recommendations ──")
    _print_recommendations(classified, exact_acc, family_acc)


def _print_recommendations(df: pd.DataFrame,
                            exact_acc: float,
                            family_acc: float) -> None:
    recs = []

    # Check PSK confusion
    psk_df = df[df["gt_canonical"].isin(["BPSK","QPSK","8PSK"])]
    if len(psk_df) > 0 and psk_df["exact_match"].mean() < 0.70:
        recs.append("PSK sub-type confusion is high. Consider adding ML constellation "
                    "recovery (CNN on received constellation scatter plot) to disambiguate "
                    "BPSK/QPSK/8PSK — cumulants alone are unreliable at SNR < 10 dB.")

    # Check QAM confusion
    qam_df = df[df["gt_canonical"].str.startswith("QAM", na=False)]
    if len(qam_df) > 0 and qam_df["exact_match"].mean() < 0.60:
        recs.append("QAM order confusion is high. ML classifier trained on RadioML 2018 "
                    "(CNN with spectrogram input) consistently outperforms cumulant-based "
                    "decision trees for QAM16/32/64/256 at low SNR.")

    # Check FM confusion with FSK
    fm_df = df[df["gt_canonical"].str.startswith("FM", na=False)]
    if len(fm_df) > 0 and fm_df["exact_match"].mean() < 0.75:
        recs.append("FM/FSK confusion: tune FM_DEV_FRACTION threshold or add bandwidth-"
                    "normalised deviation ratio as a feature.")

    if family_acc > 0.80 and exact_acc < 0.65:
        recs.append("Family-level accuracy is good but exact modulation is poor. "
                    "The rule engine correctly identifies the modulation family; a small "
                    "per-family sub-classifier (e.g. symbol-count estimation via cyclic "
                    "spectrum) would improve exact accuracy.")

    if exact_acc < 0.50:
        recs.append("Overall accuracy is below 50%. Strongly recommend adding an ONNX "
                    "inference model (see tools/ml/train.py) trained on RadioML 2018 as "
                    "a drop-in replacement for the cumulant decision tree.")

    if not recs:
        recs.append("Classifier is performing well. Consider adding the ONNX-based ML "
                    "stage (tools/ml/) for additional robustness at SNR < 0 dB.")

    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r}")


def print_sample_table(df: pd.DataFrame, n: int = 30) -> None:
    cols = ["ground_truth", "detected_mod", "symbol_rate_sps",
            "is_burst", "top_protocol", "top_confidence", "exact_match", "snr_db"]
    cols = [c for c in cols if c in df.columns]
    print(f"\n── Sample Results (first {n} rows) ──")
    print(df[cols].head(n).to_string(index=False))


# ── Stress tests ──────────────────────────────────────────────────────────────

def stress_test_low_snr(base_samples: list[IqSample],
                        extractor:    FeatureExtractor,
                        clf:          ModulationClassifier,
                        mapper:       ProtocolMapper,
                        snr_levels_db: list[float] | None = None) -> pd.DataFrame:
    """
    Add artificial AWGN at specified SNR levels to base_samples.
    Returns a DataFrame comparing accuracy vs SNR.
    """
    if snr_levels_db is None:
        snr_levels_db = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

    rows = []
    rng = np.random.default_rng(42)

    for snr_db in snr_levels_db:
        for s in base_samples[:50]:   # limit for speed
            iq = normalise(s.iq.copy())
            # Add AWGN
            noise_pwr = 10 ** (-snr_db / 10)
            noise = rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
            noise = noise * math.sqrt(noise_pwr / 2)
            iq_noisy = iq + noise.astype(np.complex64)

            noisy_sample = IqSample(
                iq=iq_noisy, sample_rate=s.sample_rate, center_freq=s.center_freq,
                ground_truth=s.ground_truth, snr_db=snr_db,
                source_file=s.source_file, segment_idx=s.segment_idx,
            )
            row = classify_sample(noisy_sample, extractor, clf, mapper)
            row["added_snr_db"] = snr_db
            rows.append(row)

    df = pd.DataFrame(rows)
    snr_acc = (
        df.groupby("added_snr_db")["exact_match"]
        .agg(["mean","count"])
        .rename(columns={"mean":"exact_pct","count":"N"})
    )
    snr_acc["exact_pct"] = snr_acc["exact_pct"].apply(lambda x: f"{x:.0%}")
    print("\n── Stress Test: Accuracy vs Added AWGN ──")
    print(snr_acc.to_string())
    return df


def stress_test_partial_burst(samples: list[IqSample],
                               extractor: FeatureExtractor,
                               clf: ModulationClassifier,
                               mapper: ProtocolMapper) -> None:
    """Test on truncated burst signals (first 10%, 25%, 50% of samples)."""
    fractions = [0.10, 0.25, 0.50, 1.0]
    print("\n── Stress Test: Partial Burst Detection ──")
    print(f"{'Fraction':>10}  {'N':>5}  {'Exact':>7}")
    for frac in fractions:
        truncated = []
        for s in samples[:50]:
            n = max(64, int(len(s.iq) * frac))
            ts = IqSample(iq=s.iq[:n], sample_rate=s.sample_rate,
                          center_freq=s.center_freq, ground_truth=s.ground_truth,
                          snr_db=s.snr_db, source_file=s.source_file,
                          segment_idx=s.segment_idx)
            truncated.append(ts)
        df = run_batch(truncated, extractor, clf, mapper, label=f"{frac:.0%}")
        exact = df.get("exact_match", pd.Series([False])).mean()
        print(f"{frac:>10.0%}  {len(df):>5}  {exact:>7.0%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="AnalysisApp rule-based classifier test harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--radioml-2016",  metavar="HDF5",
                    help="Path to RML2016.10a.hdf5")
    ap.add_argument("--radioml-2018",  metavar="HDF5",
                    help="Path to RML2018.01a.hdf5")
    ap.add_argument("--sigmf",         metavar="GLOB",
                    help="Glob for .sigmf-meta files (e.g. 'data/*.sigmf-meta')")
    ap.add_argument("--raw",           metavar="FILE",
                    help="Path to a raw binary IQ file")
    ap.add_argument("--sr",            type=float, default=2e6,
                    help="Sample rate for --raw (default 2e6)")
    ap.add_argument("--cf",            type=float, default=0.0,
                    help="Centre freq Hz for --raw (default 0)")
    ap.add_argument("--dtype",         default="float32",
                    choices=["float32","int16","uint8"],
                    help="Sample dtype for --raw")
    ap.add_argument("--label",         default="",
                    help="Ground-truth label for --raw")
    ap.add_argument("--all-dir",       metavar="DIR",
                    help="Scan a directory for .sigmf-meta and raw .iq files")
    ap.add_argument("--snr-min",       type=float, default=-6,
                    help="Minimum SNR for RadioML samples (default -6 dB)")
    ap.add_argument("--max-per-class", type=int, default=100,
                    help="Max samples per modulation class (default 100)")
    ap.add_argument("--fft-size",      type=int, default=4096,
                    help="FFT size for feature extraction (default 4096)")
    ap.add_argument("--stress-snr",    action="store_true",
                    help="Run SNR stress test")
    ap.add_argument("--stress-burst",  action="store_true",
                    help="Run partial-burst stress test")
    ap.add_argument("--out-csv",       metavar="CSV",
                    help="Save results table to CSV")
    ap.add_argument("--out-json",      metavar="JSON",
                    help="Save results table to JSON")
    args = ap.parse_args()

    extractor = FeatureExtractor(fft_size=args.fft_size)
    clf       = ModulationClassifier()
    mapper    = ProtocolMapper()

    all_samples: list[IqSample] = []
    all_dfs: list[pd.DataFrame] = []

    # ── Load datasets ─────────────────────────────────────────────────────────
    if args.radioml_2016:
        print(f"Loading RadioML 2016.10a from {args.radioml_2016} …")
        samples = load_radioml_2016(
            args.radioml_2016,
            snr_min_db=args.snr_min,
            max_per_class=args.max_per_class,
        )
        print(f"  {len(samples)} samples loaded")
        all_samples.extend(samples)
        df = run_batch(samples, extractor, clf, mapper, label="RadioML-2016")
        all_dfs.append(df)
        print("\n══ RadioML 2016.10a Results ══")
        print_sample_table(df)
        analyse(df)

    if args.radioml_2018:
        print(f"\nLoading RadioML 2018.01a from {args.radioml_2018} …")
        samples = load_radioml_2018(
            args.radioml_2018,
            snr_min_db=args.snr_min,
            max_per_class=args.max_per_class,
        )
        print(f"  {len(samples)} samples loaded")
        all_samples.extend(samples)
        df = run_batch(samples, extractor, clf, mapper, label="RadioML-2018")
        all_dfs.append(df)
        print("\n══ RadioML 2018.01a Results ══")
        print_sample_table(df)
        analyse(df)

    if args.sigmf:
        print(f"\nLoading SigMF from {args.sigmf} …")
        for meta_file in glob.glob(args.sigmf, recursive=True):
            try:
                samples = load_sigmf(meta_file)
                all_samples.extend(samples)
                df = run_batch(samples, extractor, clf, mapper,
                               label=Path(meta_file).stem)
                all_dfs.append(df)
                print(f"\n══ {Path(meta_file).name} ══")
                print_sample_table(df, n=5)
                analyse(df)
            except Exception as e:
                print(f"  [skip] {meta_file}: {e}")

    if args.raw:
        print(f"\nLoading raw IQ from {args.raw} …")
        samples = load_raw_iq(
            args.raw, sample_rate=args.sr, center_freq=args.cf,
            ground_truth=args.label, dtype=args.dtype,
        )
        all_samples.extend(samples)
        df = run_batch(samples, extractor, clf, mapper, label="raw")
        all_dfs.append(df)
        print_sample_table(df)
        analyse(df)

    if args.all_dir:
        d = Path(args.all_dir)
        for f in sorted(d.glob("*.sigmf-meta")):
            try:
                samples = load_sigmf(f)
                all_samples.extend(samples)
                df = run_batch(samples, extractor, clf, mapper, label=f.stem)
                all_dfs.append(df)
            except Exception as e:
                print(f"  [skip] {f.name}: {e}")
        for f in sorted(d.glob("*.iq")):
            samples = load_raw_iq(f, sample_rate=2e6, center_freq=0)
            all_samples.extend(samples)
            df = run_batch(samples, extractor, clf, mapper, label=f.stem)
            all_dfs.append(df)

    if not all_dfs:
        print("No datasets loaded. Use --radioml-2016, --sigmf, --raw, or --all-dir.")
        ap.print_help()
        sys.exit(0)

    # ── Combined report ───────────────────────────────────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True)

    if len(all_dfs) > 1:
        print("\n\n══ COMBINED RESULTS ══")
        print_sample_table(combined)
        analyse(combined)

    # ── Stress tests ─────────────────────────────────────────────────────────
    if args.stress_snr and all_samples:
        stress_test_low_snr(all_samples, extractor, clf, mapper)

    if args.stress_burst and all_samples:
        stress_test_partial_burst(all_samples, extractor, clf, mapper)

    # ── Export ────────────────────────────────────────────────────────────────
    if args.out_csv:
        combined.to_csv(args.out_csv, index=False)
        print(f"\nResults saved → {args.out_csv}")

    if args.out_json:
        combined.to_json(args.out_json, orient="records", indent=2)
        print(f"Results saved → {args.out_json}")


if __name__ == "__main__":
    main()

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
