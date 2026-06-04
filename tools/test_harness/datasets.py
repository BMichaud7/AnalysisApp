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
Dataset loaders for the AnalysisApp test harness.

Supported datasets:
  - RadioML 2016.10a  (DeepSig, HDF5, 11 modulation classes)
  - RadioML 2018.01a  (DeepSig, HDF5, 24 modulation classes)
  - Synthetic         (generated on-the-fly, no download required)
  - HuggingFace       (any dataset via `datasets` library)
  - Numpy .npz        (X shape (N,2,L), y shape (N,), classes list)
  - SigMF files       (any .sigmf-data + .sigmf-meta pair)
  - Raw binary IQ     (float32 or int16, interleaved I/Q)

Kaggle downloads (requires `pip install kaggle` + ~/.kaggle/kaggle.json):
  python -m datasets download radioml2016   # → RML2016.10a.hdf5
  python -m datasets download radioml2018   # → RML2018.01a.hdf5
"""

from __future__ import annotations

import math
import os
import json
import struct
import subprocess
import urllib.request
import ssl
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator

import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import sigmf
    from sigmf import SigMFFile
    SIGMF_AVAILABLE = True
except ImportError:
    SIGMF_AVAILABLE = False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class IqSample:
    """A chunk of IQ samples with ground-truth metadata."""
    iq:            np.ndarray        # complex64, shape (N,)
    sample_rate:   float             # samples per second
    center_freq:   float             # Hz (0 if unknown)
    ground_truth:  str               # e.g. "QPSK", "FM Broadcast", ""
    snr_db:        float             # dB (NaN if unknown)
    source_file:   str               # provenance
    segment_idx:   int = 0
    extra:         dict = field(default_factory=dict)


# ── RadioML 2016.10a ──────────────────────────────────────────────────────────

RADIOML_2016_URL = (
    "https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2"
)

RADIOML_2016_MODS = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
    "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]

RADIOML_2016_SNRS = list(range(-20, 20, 2))   # -20 to +18 dB


def load_radioml_2016(path: str | Path,
                      snr_min_db: float = -6,
                      max_per_class: int = 100) -> list[IqSample]:
    """
    Load RadioML 2016.10a from its HDF5 file.

    Parameters
    ----------
    path         : path to RML2016.10a.hdf5 (download if not present)
    snr_min_db   : only include samples at or above this SNR
    max_per_class: cap on samples per (modulation, snr) combination
    """
    if not HDF5_AVAILABLE:
        raise RuntimeError("h5py not installed: pip install h5py")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"RadioML 2016.10a not found at {path}.\n"
            f"Download from: {RADIOML_2016_URL}\n"
            f"Then extract the .hdf5 file."
        )

    samples: list[IqSample] = []
    with h5py.File(path, "r") as f:
        # File structure: dict key = (mod_string, snr_int), value = (N, 2, 128)
        for key in f.keys():
            # keys are like "('QAM16', 6)"
            try:
                mod, snr = eval(key)
            except Exception:
                continue
            if snr < snr_min_db:
                continue

            data = f[key][:]    # (N, 2, 128): N examples, 2 channels (I/Q), 128 samples
            data = data[:max_per_class]

            for i, ex in enumerate(data):
                iq = ex[0] + 1j * ex[1]    # complex64
                samples.append(IqSample(
                    iq=iq.astype(np.complex64),
                    sample_rate=200e3,       # RadioML uses ~200 kHz effective SR
                    center_freq=0.0,
                    ground_truth=mod,
                    snr_db=float(snr),
                    source_file=str(path),
                    segment_idx=i,
                    extra={"dataset": "RadioML2016.10a"},
                ))
    return samples


# ── RadioML 2018.01a ──────────────────────────────────────────────────────────

RADIOML_2018_MODS = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]


def load_radioml_2018(path: str | Path,
                      snr_min_db: float = 0,
                      max_per_class: int = 50) -> list[IqSample]:
    """Load RadioML 2018.01a from HDF5."""
    if not HDF5_AVAILABLE:
        raise RuntimeError("h5py not installed")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RadioML 2018.01a not found at {path}")

    samples: list[IqSample] = []
    with h5py.File(path, "r") as f:
        labels  = f["Y"][:]    # one-hot (N, 24)
        snrs    = f["Z"][:]    # (N,) dB
        data    = f["X"][:]    # (N, 1024, 2) I/Q interleaved

        class_names = RADIOML_2018_MODS
        for idx in range(min(len(data), max_per_class * len(class_names))):
            snr = float(snrs[idx])
            if snr < snr_min_db:
                continue
            mod_idx = int(np.argmax(labels[idx]))
            mod_name = class_names[mod_idx] if mod_idx < len(class_names) else "UNKNOWN"
            iq_raw = data[idx]           # (1024, 2)
            iq = (iq_raw[:, 0] + 1j * iq_raw[:, 1]).astype(np.complex64)

            samples.append(IqSample(
                iq=iq,
                sample_rate=1e6,
                center_freq=0.0,
                ground_truth=mod_name,
                snr_db=snr,
                source_file=str(path),
                segment_idx=idx,
                extra={"dataset": "RadioML2018.01a"},
            ))
    return samples


# ── SigMF files ───────────────────────────────────────────────────────────────

def load_sigmf(meta_path: str | Path,
               segment_len: int = 65536,
               max_segments: int = 10) -> list[IqSample]:
    """
    Load IQ from a SigMF file pair (.sigmf-meta + .sigmf-data).
    Falls back to manual JSON parse if the sigmf library is unavailable.
    """
    meta_path = Path(meta_path)
    data_path = meta_path.with_suffix(".sigmf-data")

    # Parse metadata
    with open(meta_path) as f:
        meta = json.load(f)

    global_meta = meta.get("global", {})
    sample_rate = float(global_meta.get("core:sample_rate", 0))
    center_freq = 0.0
    dtype_str   = global_meta.get("core:datatype", "cf32_le")
    description = global_meta.get("core:description", "")

    # Annotations may give center freq
    for ann in meta.get("annotations", []):
        if "core:freq_lower_edge" in ann and "core:freq_upper_edge" in ann:
            center_freq = (ann["core:freq_lower_edge"] + ann["core:freq_upper_edge"]) / 2
            break
    captures = meta.get("captures", [])
    if captures:
        center_freq = float(captures[0].get("core:frequency", center_freq))

    # Ground truth from description or label annotations
    ground_truth = description
    for ann in meta.get("annotations", []):
        if "core:label" in ann:
            ground_truth = ann["core:label"]
            break

    # Load binary IQ
    if "cf32" in dtype_str:
        raw = np.fromfile(data_path, dtype=np.float32)
        iq  = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    elif "ci16" in dtype_str or "cs16" in dtype_str:
        raw = np.fromfile(data_path, dtype=np.int16)
        iq  = ((raw[0::2] / 32768.0) + 1j * (raw[1::2] / 32768.0)).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported SigMF datatype: {dtype_str}")

    samples = []
    for seg_idx in range(min(max_segments, len(iq) // segment_len)):
        chunk = iq[seg_idx * segment_len : (seg_idx + 1) * segment_len]
        samples.append(IqSample(
            iq=chunk,
            sample_rate=sample_rate,
            center_freq=center_freq,
            ground_truth=ground_truth,
            snr_db=float("nan"),
            source_file=str(meta_path),
            segment_idx=seg_idx,
            extra={"dataset": "SigMF"},
        ))
    return samples


# ── Raw binary IQ ─────────────────────────────────────────────────────────────

def load_raw_iq(path: str | Path,
                sample_rate: float,
                center_freq: float,
                ground_truth: str = "",
                dtype: str = "float32",
                segment_len: int = 65536,
                max_segments: int = 10) -> list[IqSample]:
    """
    Load a raw binary IQ file (e.g. RTL-SDR .bin, SDRplay .iq).

    dtype: "float32" (CF32), "int16" (CS16), "uint8" (RTL-SDR raw)
    """
    path = Path(path)
    if dtype == "float32":
        raw = np.fromfile(path, dtype=np.float32)
        iq  = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    elif dtype == "int16":
        raw = np.fromfile(path, dtype=np.int16)
        iq  = ((raw[0::2] / 32768.0) + 1j * (raw[1::2] / 32768.0)).astype(np.complex64)
    elif dtype == "uint8":
        raw = np.fromfile(path, dtype=np.uint8).astype(np.float32)
        raw = (raw - 127.5) / 127.5
        iq  = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    samples = []
    for seg_idx in range(min(max_segments, len(iq) // segment_len)):
        chunk = iq[seg_idx * segment_len : (seg_idx + 1) * segment_len]
        samples.append(IqSample(
            iq=chunk,
            sample_rate=sample_rate,
            center_freq=center_freq,
            ground_truth=ground_truth,
            snr_db=float("nan"),
            source_file=str(path),
            segment_idx=seg_idx,
            extra={"dataset": "raw_binary"},
        ))
    return samples


# ── IQEngine public signals ───────────────────────────────────────────────────

IQENGINE_CATALOG_URL = "https://raw.githubusercontent.com/IQEngine/IQEngine/main/public/sample-recordings/catalog.json"

def list_iqengine_signals(catalog_path: str | Path | None = None) -> list[dict]:
    """Return the IQEngine public signal catalog entries."""
    if catalog_path and Path(catalog_path).exists():
        with open(catalog_path) as f:
            return json.load(f)
    try:
        with urllib.request.urlopen(IQENGINE_CATALOG_URL, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"[IQEngine] Could not fetch catalog: {e}")
        return []


# ── Utility: normalise IQ ─────────────────────────────────────────────────────

def normalise(iq: np.ndarray) -> np.ndarray:
    """Unit-power normalisation."""
    pwr = np.mean(np.abs(iq) ** 2)
    if pwr > 0:
        iq = iq / np.sqrt(pwr)
    return iq.astype(np.complex64)


# ── Synthetic dataset generator ───────────────────────────────────────────────
# Generates a RadioML-compatible training dataset entirely from code — no
# downloads required. Each modulation uses the same generator as
# verify_realworld.py and adds AWGN across a configurable SNR range.
#
# Usage:
#   samples = generate_synthetic(n_per_class=500, snr_range=(-20, 30))
#   # → list[IqSample] covering 11 modulations × SNR steps

SYNTHETIC_MODS = [
    "BPSK", "QPSK", "8PSK", "QAM16", "QAM64",
    "FM_NB", "FM_WB", "AM_DSB_LC", "AM_DSB_SC",
    "OFDM", "FSK", "MSK", "CSS",
]


def _awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_pwr   = np.mean(np.abs(iq) ** 2)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    noise     = rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
    noise     = (noise * math.sqrt(noise_pwr / 2)).astype(np.complex64)
    return (iq + noise).astype(np.complex64)


def _gen_psk(order: int, n: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    angles = np.arange(order) * 2 * np.pi / order
    const  = np.exp(1j * angles).astype(np.complex64)
    syms   = rng.choice(const, n // sps + 1)
    return normalise(np.repeat(syms, sps)[:n])


def _gen_qam(order: int, n: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    k    = int(math.sqrt(order))
    vals = np.arange(-(k-1), k, 2, dtype=np.float32)
    I, Q = np.meshgrid(vals, vals)
    const = (I.ravel() + 1j * Q.ravel()).astype(np.complex64)
    syms  = rng.choice(const, n // sps + 1)
    return normalise(np.repeat(syms, sps)[:n])


def _gen_fm(dev_hz: float, n: int, sr: float, rng: np.random.Generator) -> np.ndarray:
    audio = rng.standard_normal(n).astype(np.float32)
    audio /= (np.max(np.abs(audio)) + 1e-9)
    phase = 2 * np.pi * dev_hz / sr * np.cumsum(audio)
    return normalise(np.exp(1j * phase).astype(np.complex64))


def _gen_am(mod_depth: float, suppress_carrier: bool,
            n: int, sr: float, rng: np.random.Generator) -> np.ndarray:
    audio  = rng.standard_normal(n).astype(np.float32)
    audio  = audio * 0.5   # σ=0.5 → rarely exceeds ±1
    if suppress_carrier:
        return normalise((audio).astype(np.complex64))
    return normalise((1.0 + mod_depth * audio).astype(np.complex64))


def _gen_ofdm(n: int, nfft: int, cp: int, rng: np.random.Generator) -> np.ndarray:
    syms = []
    while len(syms) * (nfft + cp) < n:
        fd = rng.choice(np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64), nfft)
        td = np.fft.ifft(fd).astype(np.complex64)
        syms.append(np.concatenate([td[-cp:], td]))
    return normalise(np.concatenate(syms)[:n])


def _gen_fsk(n: int, sr: float, sps: int, dev: float,
             rng: np.random.Generator) -> np.ndarray:
    bits  = rng.integers(0, 2, n // sps + 1)   # +1 ensures len >= n after repeat
    freqs = np.repeat((2 * bits - 1) * dev, sps).astype(np.float32)
    phase = 2 * np.pi / sr * np.cumsum(freqs[:n])
    return normalise(np.exp(1j * phase).astype(np.complex64))


def _gen_css(n: int, sr: float, bw: float,
             n_chirp: int, rng: np.random.Generator) -> np.ndarray:
    cl = int(sr / bw * n_chirp)
    t  = np.arange(cl) / sr
    c  = np.exp(1j * 2 * np.pi * (-bw/2 * t + bw/(2*cl/sr) * t**2))
    return normalise(np.tile(c, n//cl+1)[:n].astype(np.complex64))


_SYNTH_GENERATORS: dict[str, tuple] = {
    # (fn, sr, extra_kwargs)
    "BPSK":      ("psk",  200e3, {"order": 2,  "sps": 8}),
    "QPSK":      ("psk",  200e3, {"order": 4,  "sps": 8}),
    "8PSK":      ("psk",  200e3, {"order": 8,  "sps": 8}),
    "QAM16":     ("qam",  200e3, {"order": 16, "sps": 8}),
    "QAM64":     ("qam",  200e3, {"order": 64, "sps": 8}),
    "FM_NB":     ("fm",   200e3, {"dev_hz": 20e3}),
    "FM_WB":     ("fm",   800e3, {"dev_hz": 75e3}),
    "AM_DSB_LC": ("am",   200e3, {"mod_depth": 0.85, "suppress": False}),
    "AM_DSB_SC": ("am",   200e3, {"mod_depth": 1.00, "suppress": True}),
    "OFDM":      ("ofdm", 1e6,   {"nfft": 64,  "cp": 16}),
    "FSK":       ("fsk",  200e3, {"sps": 20,   "dev": 8e3}),
    "MSK":       ("fsk",  200e3, {"sps": 10,   "dev": 5e3}),
    "CSS":       ("css",  500e3, {"bw": 125e3, "n_chirp": 512}),
}


def generate_synthetic(
        n_per_class: int      = 500,
        snr_range: tuple      = (-20, 30),
        snr_step_db: float    = 2.0,
        sample_len: int       = 1024,
        modulations: list[str] | None = None,
        seed: int             = 0) -> list[IqSample]:
    """
    Generate a synthetic AMR training/test dataset.

    Parameters
    ----------
    n_per_class : examples per (modulation, SNR) bucket
    snr_range   : (min_db, max_db) inclusive
    snr_step_db : SNR grid spacing
    sample_len  : IQ samples per example
    modulations : subset of SYNTHETIC_MODS (default: all)
    seed        : numpy RNG seed for reproducibility

    Returns
    -------
    list[IqSample] — length = n_mods × n_snr_steps × n_per_class
    """
    mods    = modulations or SYNTHETIC_MODS
    snrs    = np.arange(snr_range[0], snr_range[1] + snr_step_db/2, snr_step_db)
    rng     = np.random.default_rng(seed)
    samples: list[IqSample] = []
    n       = sample_len

    for mod in mods:
        if mod not in _SYNTH_GENERATORS:
            print(f"[synthetic] Unknown modulation {mod!r}, skipping")
            continue
        kind, sr, kw = _SYNTH_GENERATORS[mod]

        for snr in snrs:
            for _ in range(n_per_class):
                if kind == "psk":
                    iq = _gen_psk(kw["order"], n, kw["sps"], rng)
                elif kind == "qam":
                    iq = _gen_qam(kw["order"], n, kw["sps"], rng)
                elif kind == "fm":
                    iq = _gen_fm(kw["dev_hz"], n, sr, rng)
                elif kind == "am":
                    iq = _gen_am(kw["mod_depth"], kw["suppress"], n, sr, rng)
                elif kind == "ofdm":
                    iq = _gen_ofdm(n, kw["nfft"], kw["cp"], rng)
                elif kind == "fsk":
                    iq = _gen_fsk(n, sr, kw["sps"], kw["dev"], rng)
                elif kind == "css":
                    iq = _gen_css(n, sr, kw["bw"], kw["n_chirp"], rng)
                else:
                    continue

                iq = _awgn(iq, snr, rng)
                samples.append(IqSample(
                    iq=iq, sample_rate=sr, center_freq=0.0,
                    ground_truth=mod, snr_db=float(snr),
                    source_file="synthetic", segment_idx=len(samples),
                    extra={"dataset": "synthetic"},
                ))
    return samples


def save_synthetic_npz(path: str | Path, samples: list[IqSample]) -> None:
    """Save a list of IqSamples to a numpy .npz file for ML training."""
    path     = Path(path)
    classes  = sorted(set(s.ground_truth for s in samples))
    label_map = {m: i for i, m in enumerate(classes)}
    X = np.stack([np.stack([s.iq.real, s.iq.imag], axis=0) for s in samples])
    y = np.array([label_map[s.ground_truth] for s in samples], dtype=np.int64)
    snrs = np.array([s.snr_db for s in samples], dtype=np.float32)
    np.savez_compressed(path, X=X, y=y, snrs=snrs,
                        classes=np.array(classes))
    print(f"Saved {len(samples)} samples → {path}  "
          f"({len(classes)} classes, shape {X.shape})")


# ── Numpy .npz loader ─────────────────────────────────────────────────────────

def load_numpy_npz(path: str | Path,
                   snr_min_db: float = -6,
                   max_per_class: int = 5000) -> list[IqSample]:
    """
    Load IQ samples from a .npz file produced by save_synthetic_npz or
    any compatible array file.

    Expected keys:
      X      : (N, 2, L) float32  — channel 0 = I, channel 1 = Q
      y      : (N,) int64         — class indices
      classes: (C,) str array     — class name for each index
      snrs   : (N,) float32       — optional SNR per sample
    """
    data     = np.load(path, allow_pickle=True)
    X        = data["X"].astype(np.float32)
    y        = data["y"].astype(np.int64)
    classes  = list(data["classes"])
    snrs     = data["snrs"].astype(np.float32) if "snrs" in data else \
               np.full(len(y), float("nan"), dtype=np.float32)

    counts: dict[int, int] = {}
    samples: list[IqSample] = []
    for idx in range(len(X)):
        snr    = float(snrs[idx])
        if snr < snr_min_db:
            continue
        cls_i = int(y[idx])
        if counts.get(cls_i, 0) >= max_per_class:
            continue
        counts[cls_i] = counts.get(cls_i, 0) + 1
        iq = (X[idx, 0] + 1j * X[idx, 1]).astype(np.complex64)
        samples.append(IqSample(
            iq=iq,
            sample_rate=0.0,
            center_freq=0.0,
            ground_truth=classes[cls_i] if cls_i < len(classes) else str(cls_i),
            snr_db=snr,
            source_file=str(path),
            segment_idx=idx,
            extra={"dataset": "npz"},
        ))
    return samples


# ── HuggingFace datasets loader ───────────────────────────────────────────────

def load_hf_dataset(dataset_id: str,
                    split: str        = "train",
                    iq_col: str       = "iq",
                    label_col: str    = "label",
                    snr_col: str      = "snr",
                    sample_rate: float = 0.0,
                    snr_min_db: float  = -6,
                    max_per_class: int = 5000,
                    cache_dir: str | None = None) -> list[IqSample]:
    """
    Load an AMR dataset from HuggingFace Hub via the `datasets` library.

    The dataset must have columns for IQ samples, labels, and optionally SNR.
    IQ column may be:
      - A list/array of float32 interleaved [I0, Q0, I1, Q1, ...]
      - A dict {"I": [...], "Q": [...]}
      - A (2, L) nested list

    Install: pip install datasets

    Example:
        samples = load_hf_dataset("username/radio-amr", split="train")
    """
    try:
        from datasets import load_dataset as hf_load  # type: ignore
    except ImportError:
        raise RuntimeError(
            "HuggingFace datasets not installed.\n"
            "  pip install datasets"
        )

    print(f"[HuggingFace] Loading {dataset_id} ({split}) …")
    ds = hf_load(dataset_id, split=split, cache_dir=cache_dir)

    counts: dict[str, int] = {}
    samples: list[IqSample] = []

    for row in ds:
        snr = float(row.get(snr_col, float("nan")))
        if snr < snr_min_db:
            continue
        label = str(row.get(label_col, ""))
        if counts.get(label, 0) >= max_per_class:
            continue

        raw = row.get(iq_col)
        if raw is None:
            continue

        # Decode IQ — handle interleaved, dict, or (2,L) shapes
        if isinstance(raw, dict):
            iq = (np.array(raw["I"], dtype=np.float32) +
                  1j * np.array(raw["Q"], dtype=np.float32)).astype(np.complex64)
        else:
            arr = np.array(raw, dtype=np.float32).ravel()
            if arr.ndim == 1 and len(arr) % 2 == 0:
                iq = (arr[0::2] + 1j * arr[1::2]).astype(np.complex64)
            elif arr.shape[0] == 2:
                iq = (arr[0] + 1j * arr[1]).astype(np.complex64)
            else:
                continue

        counts[label] = counts.get(label, 0) + 1
        samples.append(IqSample(
            iq=iq, sample_rate=sample_rate, center_freq=0.0,
            ground_truth=label, snr_db=snr,
            source_file=f"hf:{dataset_id}", segment_idx=len(samples),
            extra={"dataset": dataset_id, "split": split},
        ))

    print(f"[HuggingFace] Loaded {len(samples)} samples, "
          f"{len(set(s.ground_truth for s in samples))} classes")
    return samples


# ── Kaggle download helper ────────────────────────────────────────────────────

KAGGLE_DATASETS = {
    "radioml2016": {
        "slug":     "pinxau1000/radioml201610a",
        "filename": "RML2016.10a.hdf5",
        "desc":     "RadioML 2016.10a (11 mods, 128 samples, -20..+18 dB)",
    },
    "radioml2018": {
        "slug":     "pinxau1000/radioml2018",
        "filename": "RML2018.01A_dict.hdf5",
        "desc":     "RadioML 2018.01a (24 mods, 1024 samples, -20..+30 dB)",
    },
}


def download_kaggle(name: str, dest_dir: str = ".") -> Path:
    """
    Download a RadioML dataset from Kaggle using the kaggle CLI.

    Prerequisites:
        pip install kaggle
        # Place API token at ~/.kaggle/kaggle.json
        # Get token from https://www.kaggle.com/account → API → Create New Token

    Parameters
    ----------
    name     : "radioml2016" or "radioml2018"
    dest_dir : directory to save the downloaded file

    Returns
    -------
    Path to the downloaded HDF5 file
    """
    if name not in KAGGLE_DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {list(KAGGLE_DATASETS)}")

    info = KAGGLE_DATASETS[name]
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    out_file = dest / info["filename"]
    if out_file.exists():
        print(f"[kaggle] {out_file} already exists, skipping download")
        return out_file

    print(f"[kaggle] Downloading {info['desc']} …")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", info["slug"],
             "--path", str(dest), "--unzip"],
            check=True, capture_output=True, text=True,
        )
        print(result.stdout)
    except FileNotFoundError:
        raise RuntimeError(
            "kaggle CLI not found.\n"
            "  pip install kaggle\n"
            "  # then place ~/.kaggle/kaggle.json (from kaggle.com → Account → API Token)"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"kaggle download failed:\n{e.stderr}")

    if not out_file.exists():
        # Some Kaggle datasets extract to subdirs — search for the file
        found = list(dest.rglob(info["filename"]))
        if found:
            return found[0]
        raise FileNotFoundError(
            f"Download succeeded but {info['filename']} not found in {dest}.\n"
            f"Contents: {list(dest.iterdir())}"
        )
    return out_file


# ── CLI: download datasets ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(description="Dataset download / generation utilities")
    sub = ap.add_subparsers(dest="cmd")

    dl = sub.add_parser("download", help="Download a dataset via Kaggle")
    dl.add_argument("name", choices=list(KAGGLE_DATASETS))
    dl.add_argument("--dest", default=".", help="Download directory")

    gen = sub.add_parser("generate", help="Generate synthetic training data")
    gen.add_argument("--out",      required=True, help="Output .npz path")
    gen.add_argument("--n",        type=int, default=200,
                     help="Examples per (mod, SNR) bucket (default 200)")
    gen.add_argument("--snr-min",  type=float, default=-20)
    gen.add_argument("--snr-max",  type=float, default=30)
    gen.add_argument("--snr-step", type=float, default=2)
    gen.add_argument("--len",      type=int, default=1024, dest="sample_len",
                     help="Samples per IQ window (default 1024)")
    gen.add_argument("--mods",     nargs="*", default=None,
                     help="Modulation subset (default: all 13)")

    args = ap.parse_args()

    if args.cmd == "download":
        path = download_kaggle(args.name, args.dest)
        print(f"Ready: {path}")

    elif args.cmd == "generate":
        print(f"Generating synthetic dataset …")
        samples = generate_synthetic(
            n_per_class=args.n,
            snr_range=(args.snr_min, args.snr_max),
            snr_step_db=args.snr_step,
            sample_len=args.sample_len,
            modulations=args.mods,
        )
        save_synthetic_npz(args.out, samples)

    else:
        ap.print_help()
        sys.exit(1)

# ========================================================================
# End of file — OpenRFStack
# Subject to Personal Use License
# https://github.com/OpenRFStack
# ========================================================================
