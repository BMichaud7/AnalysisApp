"""
Dataset loaders for the AnalysisApp test harness.

Supported datasets:
  - RadioML 2016.10a  (DeepSig, HDF5, 11 modulation classes)
  - RadioML 2018.01a  (DeepSig, HDF5, 24 modulation classes)
  - SigMF files       (any .sigmf-data + .sigmf-meta pair)
  - Raw binary IQ     (float32 or int16, interleaved I/Q)
  - IQEngine library  (JSON manifest with download URLs)
"""

from __future__ import annotations

import os
import json
import struct
import urllib.request
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
