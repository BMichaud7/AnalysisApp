# ========================================================================
# Project: OpenRFStack — Modulation family groupings for hierarchical AMR
# ========================================================================
"""
Maps each of the 47 AMR classes to a coarse modulation family.
Used for:
  1. Auxiliary family loss during training (regularises the model to organise
     features by family before fine-grained discrimination).
  2. Hierarchical inference at eval time (family-gated argmax reduces the
     effective search space from 47 → ~5 classes per family).
"""
from __future__ import annotations

# class_name → family_name
CLASS_FAMILY: dict[str, str] = {
    # ── Quadrature Amplitude Modulation ──────────────────────────────────────
    "QAM16":     "QAM",
    "QAM32":     "QAM",
    "QAM64":     "QAM",
    "QAM256":    "QAM",
    # ── Phase-Shift Keying ───────────────────────────────────────────────────
    "BPSK":      "PSK",
    "QPSK":      "PSK",
    "8PSK":      "PSK",
    "16PSK":     "PSK",
    "32PSK":     "PSK",
    "PSK31":     "PSK",
    # ── Amplitude-Shift Keying / OOK ─────────────────────────────────────────
    "4ASK":      "ASK",
    "16ASK":     "ASK",
    "OOK":       "ASK",
    # ── Narrowband FSK / paging / utility ────────────────────────────────────
    "FSK":       "FSK",
    "GFSK":      "FSK",
    "GMSK":      "FSK",
    "MSK":       "FSK",
    "4FSK":      "FSK",
    "8FSK":      "FSK",
    "RTTY":      "FSK",
    "NAVTEX":    "FSK",
    "DSC":       "FSK",
    "MDC_1200":  "FSK",
    "FLEX":      "FSK",
    "POCSAG":    "FSK",
    # ── Digital voice / land mobile radio ────────────────────────────────────
    # Merged into FSK: all six use FSK-derived modulation (C4FM, GMSK, 4FSK)
    # and the router could not distinguish them from raw FSK (DIGI_VOICE→FSK
    # was the #1 router confusion). The FSK specialist now covers all 18 classes.
    "P25_C4FM":  "FSK",
    "P25_PHASE2":"FSK",
    "DMR":       "FSK",
    "NXDN":      "FSK",
    "DSTAR":     "FSK",
    "TETRA":     "FSK",
    # ── Aviation / maritime data protocols ───────────────────────────────────
    "ACARS":     "PROTOCOL",
    "ADS_B":     "PROTOCOL",
    "AIS":       "PROTOCOL",
    "VDL2":      "PROTOCOL",
    "EAS_SAME":  "PROTOCOL",
    "DTMF":      "PROTOCOL",
    # ── Amplitude Modulation ─────────────────────────────────────────────────
    "AM_DSB":    "AM",
    "AM_DSB_SC": "AM",
    "AM_SSB_LSB":"AM",
    "AM_SSB_USB":"AM",
    # ── Frequency Modulation ─────────────────────────────────────────────────
    "FM_NB":     "FM",
    "FM_WB":     "FM",
    # ── Wideband / spread / special ──────────────────────────────────────────
    "OFDM":      "SPECIAL",
    "CSS":       "SPECIAL",
    "LFM":       "SPECIAL",
    "TONE":      "SPECIAL",
}

FAMILY_NAMES: list[str] = sorted(set(CLASS_FAMILY.values()))
FAMILY_INDEX: dict[str, int] = {f: i for i, f in enumerate(FAMILY_NAMES)}
NUM_FAMILIES: int = len(FAMILY_NAMES)


def class_to_family_idx(class_names: list[str]) -> list[int]:
    """Return family index for each class in class_names (unmapped → -1)."""
    return [FAMILY_INDEX.get(CLASS_FAMILY.get(c, ""), -1) for c in class_names]


def family_projection_matrix(class_names: list[str]) -> "torch.Tensor":
    """
    Returns a (num_classes, NUM_FAMILIES) float32 tensor P where
    P[c, f] = 1 if class c belongs to family f, else 0.

    logits (B, C) @ P  →  (B, F) family logits (sum within each family).
    """
    import torch
    C = len(class_names)
    F = NUM_FAMILIES
    P = torch.zeros(C, F)
    for ci, cls in enumerate(class_names):
        fi = FAMILY_INDEX.get(CLASS_FAMILY.get(cls, ""), -1)
        if fi >= 0:
            P[ci, fi] = 1.0
    return P
