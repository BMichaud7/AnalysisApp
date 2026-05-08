"""
Python mirror of ModulationClassifier + ProtocolMapper.

Operates on SignalFeatures to produce a classification result dict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from features import SignalFeatures


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class Hypothesis:
    system:     str
    category:   str
    confidence: float
    reasoning:  str


@dataclass
class ClassificationResult:
    # Layer 1
    analog_modulation: str   = ""
    analog_index:      float = 0.0

    # Layer 2
    digital_modulation: str   = ""
    symbol_rate_sps:    float = 0.0
    m_ary:              int   = 0
    is_ofdm:            bool  = False

    # Layer 3
    is_burst:      bool  = False
    burst_duty:    float = 1.0
    is_tdma:       bool  = False
    is_fhss:       bool  = False
    is_dsss:       bool  = False

    # Layer 4
    bit_rate_bps:    float = 0.0
    fec_detected:    bool  = False
    has_sync_pattern: bool = False

    # Layer 5
    hypotheses: list[Hypothesis] = field(default_factory=list)

    classified: bool = False
    reject_reason: str = ""


# ── Modulation classifier ─────────────────────────────────────────────────────

class ModulationClassifier:
    """Rule-based modulation classifier using signal features."""

    SNR_MIN_DB          = 5.0
    CONST_ENV_THRESH    = 0.08    # envelope_variance_norm threshold
    AM_ENV_THRESH       = 0.20
    FM_DEV_FRACTION     = 0.10

    def classify(self, f: SignalFeatures) -> ClassificationResult:
        r = ClassificationResult()

        if f.snr_db < self.SNR_MIN_DB:
            r.reject_reason = f"SNR too low: {f.snr_db:.1f} dB < {self.SNR_MIN_DB} dB"
            return r

        r.classified     = True
        r.is_burst       = f.is_burst
        r.burst_duty     = f.burst_duty_cycle
        r.is_fhss        = f.fhss_detected
        r.is_tdma        = f.is_burst and f.burst_period_ms > 0

        # ── Layer 2/3: structural fast-paths ──────────────────────────────────
        if f.chirp_detected:
            r.digital_modulation = "CSS"
            r.symbol_rate_sps    = abs(f.chirp_rate_hz_s / f.sample_rate_sps) if f.sample_rate_sps else 0
            return r

        if f.ofdm_detected:
            r.digital_modulation   = "OFDM"
            r.is_ofdm              = True
            if f.ofdm_fft_size_est > 0 and f.sample_rate_sps > 0:
                r.symbol_rate_sps = f.sample_rate_sps / f.ofdm_fft_size_est
            return r

        if f.fhss_detected:
            r.digital_modulation = "FHSS"
            return r

        # ── Layer 1: Analog vs digital ────────────────────────────────────────
        if self._is_analog(f):
            r.analog_modulation, r.analog_index = self._classify_analog(f)
            return r

        # ── Layer 2: Digital carrier modulation ───────────────────────────────
        r.digital_modulation, r.m_ary = self._classify_digital(f)
        r.symbol_rate_sps = f.symbol_rate_sps

        if r.m_ary > 1 and f.symbol_rate_sps > 0:
            r.bit_rate_bps = f.symbol_rate_sps * math.log2(r.m_ary)

        return r

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_analog(self, f: SignalFeatures) -> bool:
        """
        Analog decision:
          - High envelope variance (AM) OR
          - High FM deviation relative to BW (FM) OR
          - Constant envelope with significant phase variance (PM)
        """
        if f.envelope_variance_norm > self.AM_ENV_THRESH:
            return True
        if f.fm_deviation_hz > f.bandwidth_hz * self.FM_DEV_FRACTION and \
                f.envelope_variance_norm < self.CONST_ENV_THRESH:
            return True
        return False

    def _classify_analog(self, f: SignalFeatures) -> tuple[str, float]:
        """Returns (modulation_string, index_or_deviation)."""
        bw = f.bandwidth_hz if f.bandwidth_hz > 0 else 1e6

        # FM decision first — constant envelope + high deviation
        if f.envelope_variance_norm < self.CONST_ENV_THRESH:
            if f.fm_deviation_hz > 50_000:
                return "FM_WB", f.fm_deviation_hz   # broadcast FM
            if f.fm_deviation_hz > 3_000:
                return "FM_NB", f.fm_deviation_hz
            if f.inst_phase_std > 0.3:
                return "PM", f.inst_phase_std
            return "FM_NB", f.fm_deviation_hz

        # AM family
        if f.spectral_symmetry < 0.4 and f.am_index > 0.05:
            # Asymmetric spectrum → SSB
            if f.inst_freq_mean_hz > 0:
                return "SSB_USB", f.am_index
            else:
                return "SSB_LSB", f.am_index

        if f.am_index < 0.05:
            return "AM_DSB_SC", f.am_index   # suppressed carrier
        if f.is_burst and f.burst_duty_cycle < 0.5:
            return "CW", f.am_index           # OOK / Morse

        return "AM_DSB_LC", f.am_index

    def _classify_digital(self, f: SignalFeatures) -> tuple[str, int]:
        """
        Returns (modulation_string, m_ary) using normalized cumulants.

        Theoretical cumulant reference:
            BPSK:   c40 ≈ -2.0, c42 ≈  2.0
            QPSK:   c40 ≈ -2.0, c42 ≈  0.0
            8PSK:   c40 ≈  0.0, c42 ≈  0.0
            16QAM:  c40 ≈ -0.68, c42 ≈ -0.68
            64QAM:  c40 ≈ -0.62, c42 ≈ -0.62
            FSK:    c40 ≈  0.0, c42 ≈ -2.0  (constant envelope, 2 freq)
            GMSK:   c40 ≈  0.0, c42 ≈ -2.0
        """
        c40  = f.c40_real
        c42  = f.c42
        ev   = f.envelope_variance_norm
        const_env = ev < self.CONST_ENV_THRESH

        def near(x, target, tol):
            return abs(x - target) < tol

        # ── Constant envelope family ──────────────────────────────────────────
        if const_env:
            if near(c42, -2.0, 0.7) and near(c40, 0.0, 0.5):
                # FSK / MSK / GMSK
                if f.symbol_rate_sps > 0 and f.bandwidth_hz > 0:
                    mod_idx = f.fm_deviation_hz / (f.symbol_rate_sps / 2.0) \
                              if f.symbol_rate_sps > 0 else 1.0
                    if abs(mod_idx - 0.5) < 0.15:
                        return "MSK/GMSK", 2
                return "FSK", 2

            if near(c40, -2.0, 0.5) and near(c42, 2.0, 0.6):
                return "BPSK", 2
            if near(c40, -2.0, 0.6) and near(c42, 0.0, 0.4):
                return "QPSK", 4
            if near(c40, 0.0, 0.4) and near(c42, 0.0, 0.4):
                return "8PSK", 8

            return "PSK", 2   # fallback

        # ── Variable amplitude family ─────────────────────────────────────────
        if near(c40, -2.0, 0.5) and near(c42, 2.0, 0.6):
            return "BPSK", 2
        if near(c40, -2.0, 0.6) and near(c42, 0.0, 0.4):
            return "QPSK", 4
        if near(c40, -0.68, 0.15) and near(c42, -0.68, 0.15):
            return "QAM16", 16
        if near(c40, -0.62, 0.10) and near(c42, -0.62, 0.10):
            return "QAM64", 64
        if near(c40, -0.50, 0.15) and near(c42, -0.50, 0.15):
            return "QAM32", 32
        if near(c40, -0.46, 0.12) and near(c42, -0.46, 0.12):
            return "QAM256", 256

        # M-ASK (amplitude steps, variable envelope)
        if ev > 0.3 and abs(c42) > 1.0:
            return "M-ASK/OOK", 2

        return "UNKNOWN", 0


# ── Protocol mapper ───────────────────────────────────────────────────────────

class ProtocolMapper:
    """
    Maps classification result + signal features → ranked protocol hypotheses.

    Each signature is a dict with the following optional keys:
      name, category, freq_bands_mhz, modulations, bw_min_hz, bw_max_hz,
      sr_min_sps, sr_max_sps, burst, ofdm, fhss, dsss
    Score = weighted sum of component matches.
    """

    def __init__(self):
        self._db = _build_protocol_db()

    def map(self, f: SignalFeatures, r: ClassificationResult) \
            -> list[Hypothesis]:

        scores: list[tuple[float, dict]] = []
        cf_mhz = f.center_freq_hz / 1e6

        for sig in self._db:
            s = self._score(sig, f, r, cf_mhz)
            if s > 0.05:
                scores.append((s, sig))

        scores.sort(key=lambda x: -x[0])
        return [
            Hypothesis(
                system=sig["name"],
                category=sig["category"],
                confidence=float(min(score, 1.0)),
                reasoning=self._reasoning(sig, f, r),
            )
            for score, sig in scores[:3]
        ]

    @staticmethod
    def _score(sig: dict, f: SignalFeatures,
               r: ClassificationResult, cf_mhz: float) -> float:

        weight_freq  = 0.40
        weight_mod   = 0.30
        weight_bw    = 0.15
        weight_sr    = 0.15
        score        = 0.0

        # ── Frequency match ───────────────────────────────────────────────────
        bands = sig.get("freq_bands_mhz", [])
        if cf_mhz > 0 and bands:
            in_band = any(lo <= cf_mhz <= hi for lo, hi in bands)
            score += weight_freq * (1.0 if in_band else 0.0)
        else:
            score += weight_freq * 0.5   # no freq info → neutral

        # ── Modulation match ──────────────────────────────────────────────────
        mods = sig.get("modulations", [])
        if mods:
            detected = (r.analog_modulation or r.digital_modulation).upper()
            if any(detected.startswith(m.upper()) for m in mods):
                score += weight_mod * 1.0
            elif any(m.upper() in detected or detected in m.upper() for m in mods):
                score += weight_mod * 0.5
        else:
            score += weight_mod * 0.4

        # ── Bandwidth match ───────────────────────────────────────────────────
        bw_min = sig.get("bw_min_hz", 0)
        bw_max = sig.get("bw_max_hz", 0)
        if bw_min > 0 and bw_max > 0 and f.bandwidth_hz > 0:
            if bw_min <= f.bandwidth_hz <= bw_max:
                score += weight_bw * 1.0
            else:
                # Gaussian penalty outside range
                mid   = (bw_min + bw_max) / 2
                sigma = (bw_max - bw_min) / 2 + 1
                g     = math.exp(-((f.bandwidth_hz - mid) ** 2) / (2 * sigma ** 2))
                score += weight_bw * g
        else:
            score += weight_bw * 0.4

        # ── Symbol rate match ─────────────────────────────────────────────────
        sr_min = sig.get("sr_min_sps", 0)
        sr_max = sig.get("sr_max_sps", 0)
        if sr_min > 0 and sr_max > 0 and f.symbol_rate_sps > 0:
            if sr_min <= f.symbol_rate_sps <= sr_max:
                score += weight_sr * 1.0
            else:
                mid   = (sr_min + sr_max) / 2
                sigma = (sr_max - sr_min) / 2 + 1
                g     = math.exp(-((f.symbol_rate_sps - mid) ** 2) / (2 * sigma ** 2))
                score += weight_sr * g
        else:
            score += weight_sr * 0.4

        # ── Structural modifiers ──────────────────────────────────────────────
        if sig.get("burst") is True  and not r.is_burst:  score *= 0.5
        if sig.get("burst") is False and r.is_burst:      score *= 0.7
        if sig.get("ofdm")  is True  and not r.is_ofdm:   score *= 0.4
        if sig.get("fhss")  is True  and not r.is_fhss:   score *= 0.4

        return float(score)

    @staticmethod
    def _reasoning(sig: dict, f: SignalFeatures, r: ClassificationResult) -> str:
        parts = []
        mod = r.analog_modulation or r.digital_modulation
        if mod:
            parts.append(f"modulation={mod}")
        if f.symbol_rate_sps > 0:
            parts.append(f"sr={f.symbol_rate_sps/1e3:.1f} ksps")
        if f.bandwidth_hz > 0:
            parts.append(f"bw={f.bandwidth_hz/1e3:.1f} kHz")
        if f.center_freq_hz > 0:
            parts.append(f"cf={f.center_freq_hz/1e6:.3f} MHz")
        if r.is_burst:
            parts.append(f"burst duty={r.burst_duty:.0%}")
        return "; ".join(parts)


# ── Protocol database ─────────────────────────────────────────────────────────

def _build_protocol_db() -> list[dict]:
    """
    Returns list of protocol signature dicts.
    freq_bands_mhz: list of (lo, hi) tuples in MHz.
    """
    return [
        # ── Broadcast ─────────────────────────────────────────────────────────
        dict(name="AM Broadcast", category="Broadcast",
             freq_bands_mhz=[(0.52, 1.71)],
             modulations=["AM_DSB_LC"], bw_min_hz=9_000, bw_max_hz=10_000),
        dict(name="FM Broadcast", category="Broadcast",
             freq_bands_mhz=[(87.5, 108.0)],
             modulations=["FM_WB"], bw_min_hz=130_000, bw_max_hz=300_000),
        dict(name="HD Radio", category="Broadcast",
             freq_bands_mhz=[(87.5, 108.0)],
             modulations=["OFDM"], bw_min_hz=300_000, bw_max_hz=500_000, ofdm=True),
        dict(name="DAB/DAB+", category="Broadcast",
             freq_bands_mhz=[(174.0, 240.0), (1452.0, 1492.0)],
             modulations=["OFDM"], bw_min_hz=1_400_000, bw_max_hz=1_600_000, ofdm=True),
        dict(name="DRM", category="Broadcast",
             freq_bands_mhz=[(3.0, 30.0)],
             modulations=["OFDM"], bw_min_hz=9_000, bw_max_hz=18_000, ofdm=True),

        # ── Aviation ──────────────────────────────────────────────────────────
        dict(name="VHF Airband AM", category="Aviation",
             freq_bands_mhz=[(118.0, 137.0)],
             modulations=["AM_DSB_LC"], bw_min_hz=6_000, bw_max_hz=30_000),
        dict(name="ADS-B", category="Aviation",
             freq_bands_mhz=[(1089.0, 1091.0)],
             bw_min_hz=1_500_000, bw_max_hz=4_000_000, burst=True),
        dict(name="ACARS", category="Aviation",
             freq_bands_mhz=[(129.0, 137.0)],
             modulations=["AM_DSB_LC"], bw_min_hz=1_500, bw_max_hz=3_000,
             sr_min_sps=1_200, sr_max_sps=2_400, burst=True),
        dict(name="VOR/ILS", category="Aviation",
             freq_bands_mhz=[(108.0, 118.0)],
             modulations=["AM_DSB_LC"]),

        # ── Marine ────────────────────────────────────────────────────────────
        dict(name="AIS", category="Marine",
             freq_bands_mhz=[(161.9, 162.1)],
             modulations=["MSK/GMSK", "FSK"],
             bw_min_hz=20_000, bw_max_hz=30_000,
             sr_min_sps=8_000, sr_max_sps=12_000, burst=True),
        dict(name="Marine VHF FM", category="Marine",
             freq_bands_mhz=[(156.0, 174.0)],
             modulations=["FM_NB"], bw_min_hz=10_000, bw_max_hz=25_000),
        dict(name="DSC", category="Marine",
             freq_bands_mhz=[(156.5, 156.55)],
             modulations=["FSK"], bw_min_hz=800, bw_max_hz=1_500,
             sr_min_sps=600, sr_max_sps=2_400),

        # ── Amateur radio ──────────────────────────────────────────────────────
        dict(name="APRS", category="Amateur",
             freq_bands_mhz=[(144.3, 144.5), (144.7, 144.9)],
             modulations=["FSK", "MSK/GMSK"],
             bw_min_hz=10_000, bw_max_hz=20_000,
             sr_min_sps=800, sr_max_sps=1_600, burst=True),
        dict(name="FT8", category="Amateur",
             freq_bands_mhz=[(3.57, 3.58), (7.07, 7.08), (14.07, 14.08),
                              (21.07, 21.08), (28.07, 28.08)],
             modulations=["FSK"], bw_min_hz=40, bw_max_hz=80,
             sr_min_sps=3, sr_max_sps=10, burst=True),
        dict(name="WSPR", category="Amateur",
             freq_bands_mhz=[(14.09, 14.10), (10.14, 10.15)],
             modulations=["FSK"], bw_min_hz=3, bw_max_hz=10),
        dict(name="PSK31", category="Amateur",
             freq_bands_mhz=[(14.07, 14.08)],
             modulations=["BPSK"], bw_min_hz=20, bw_max_hz=60,
             sr_min_sps=20, sr_max_sps=45),
        dict(name="RTTY", category="Amateur",
             freq_bands_mhz=[(3.0, 30.0)],
             modulations=["FSK"],
             sr_min_sps=40, sr_max_sps=150),
        dict(name="D-STAR", category="Amateur",
             freq_bands_mhz=[(144.0, 148.0), (430.0, 440.0)],
             modulations=["MSK/GMSK", "FSK"],
             bw_min_hz=5_000, bw_max_hz=8_000,
             sr_min_sps=3_500, sr_max_sps=5_000),

        # ── Public Safety ─────────────────────────────────────────────────────
        dict(name="P25 Phase I", category="Public Safety",
             freq_bands_mhz=[(136.0, 174.0), (380.0, 512.0), (763.0, 870.0)],
             modulations=["QPSK", "FSK"],
             bw_min_hz=10_000, bw_max_hz=15_000,
             sr_min_sps=4_000, sr_max_sps=5_000),
        dict(name="TETRA", category="Public Safety",
             freq_bands_mhz=[(380.0, 400.0), (410.0, 430.0), (450.0, 470.0)],
             modulations=["QPSK"],
             bw_min_hz=22_000, bw_max_hz=28_000,
             sr_min_sps=17_000, sr_max_sps=19_000, burst=True),
        dict(name="DMR", category="Public Safety",
             freq_bands_mhz=[(136.0, 174.0), (403.0, 527.0), (762.0, 870.0)],
             modulations=["FSK", "QPSK"],
             bw_min_hz=10_000, bw_max_hz=13_000,
             sr_min_sps=4_000, sr_max_sps=5_500, burst=True),
        dict(name="MPT1327", category="Public Safety",
             freq_bands_mhz=[(138.0, 174.0), (450.0, 470.0)],
             modulations=["FSK"],
             bw_min_hz=10_000, bw_max_hz=25_000,
             sr_min_sps=1_000, sr_max_sps=1_500),

        # ── Cellular ──────────────────────────────────────────────────────────
        dict(name="GSM", category="Cellular",
             freq_bands_mhz=[(850.0, 894.0), (880.0, 960.0),
                              (1710.0, 1880.0), (1850.0, 1990.0)],
             modulations=["MSK/GMSK", "FSK"],
             bw_min_hz=180_000, bw_max_hz=220_000,
             sr_min_sps=250_000, sr_max_sps=290_000, burst=True),
        dict(name="UMTS/WCDMA", category="Cellular",
             freq_bands_mhz=[(850.0, 900.0), (1700.0, 2200.0)],
             modulations=["QPSK"],
             bw_min_hz=4_000_000, bw_max_hz=6_000_000, dsss=True),
        dict(name="LTE", category="Cellular",
             freq_bands_mhz=[(700.0, 900.0), (1700.0, 2200.0), (2500.0, 2700.0)],
             modulations=["OFDM"],
             bw_min_hz=1_400_000, bw_max_hz=20_000_000, ofdm=True),
        dict(name="5G NR", category="Cellular",
             freq_bands_mhz=[(600.0, 900.0), (2400.0, 2700.0),
                              (3300.0, 3800.0), (24000.0, 29500.0)],
             modulations=["OFDM"], ofdm=True),

        # ── ISM / Consumer / IoT ──────────────────────────────────────────────
        dict(name="Wi-Fi 2.4 GHz", category="ISM/IoT",
             freq_bands_mhz=[(2400.0, 2484.0)],
             modulations=["OFDM"],
             bw_min_hz=18_000_000, bw_max_hz=22_000_000, ofdm=True, burst=True),
        dict(name="Wi-Fi 5 GHz", category="ISM/IoT",
             freq_bands_mhz=[(5150.0, 5850.0)],
             modulations=["OFDM"],
             bw_min_hz=18_000_000, bw_max_hz=82_000_000, ofdm=True, burst=True),
        dict(name="Bluetooth Classic", category="ISM/IoT",
             freq_bands_mhz=[(2400.0, 2484.0)],
             modulations=["FSK", "QPSK"],
             bw_min_hz=700_000, bw_max_hz=1_200_000, fhss=True),
        dict(name="Bluetooth LE", category="ISM/IoT",
             freq_bands_mhz=[(2400.0, 2484.0)],
             modulations=["FSK", "MSK/GMSK"],
             bw_min_hz=1_500_000, bw_max_hz=2_500_000, burst=True),
        dict(name="Zigbee (802.15.4)", category="ISM/IoT",
             freq_bands_mhz=[(2400.0, 2484.0), (868.0, 869.0), (902.0, 928.0)],
             bw_min_hz=1_500_000, bw_max_hz=2_500_000,
             sr_min_sps=200_000, sr_max_sps=300_000, burst=True),
        dict(name="LoRa", category="ISM/IoT",
             freq_bands_mhz=[(433.0, 435.0), (868.0, 870.0), (915.0, 916.0)],
             modulations=["CSS"],
             bw_min_hz=100_000, bw_max_hz=550_000, burst=True),
        dict(name="Z-Wave", category="ISM/IoT",
             freq_bands_mhz=[(868.3, 868.5), (908.4, 908.5)],
             modulations=["FSK"],
             bw_min_hz=150_000, bw_max_hz=250_000,
             sr_min_sps=10_000, sr_max_sps=100_000),
        dict(name="TPMS", category="ISM/IoT",
             freq_bands_mhz=[(315.0, 316.0), (433.8, 434.0)],
             modulations=["FSK", "M-ASK/OOK"],
             bw_min_hz=10_000, bw_max_hz=80_000,
             sr_min_sps=15_000, sr_max_sps=40_000, burst=True),
        dict(name="Key Fob / Remote", category="ISM/IoT",
             freq_bands_mhz=[(315.0, 316.0), (433.8, 434.0),
                              (868.0, 870.0), (915.0, 916.0)],
             modulations=["M-ASK/OOK", "FSK"],
             bw_min_hz=5_000, bw_max_hz=50_000, burst=True),
        dict(name="ANT+", category="ISM/IoT",
             freq_bands_mhz=[(2450.0, 2460.0)],
             modulations=["FSK"], bw_min_hz=500_000, bw_max_hz=1_500_000),

        # ── Satellite / Space ─────────────────────────────────────────────────
        dict(name="GPS L1 C/A", category="Satellite",
             freq_bands_mhz=[(1575.3, 1575.5)],
             modulations=["BPSK"], bw_min_hz=1_000_000, bw_max_hz=3_000_000),
        dict(name="NOAA APT", category="Satellite",
             freq_bands_mhz=[(137.0, 138.0)],
             modulations=["FM_NB", "FM_WB"],
             bw_min_hz=30_000, bw_max_hz=38_000,
             sr_min_sps=2_000, sr_max_sps=2_800),
        dict(name="Meteor LRPT", category="Satellite",
             freq_bands_mhz=[(137.0, 137.5)],
             modulations=["QPSK"],
             bw_min_hz=100_000, bw_max_hz=150_000,
             sr_min_sps=70_000, sr_max_sps=90_000),
        dict(name="Iridium", category="Satellite",
             freq_bands_mhz=[(1616.0, 1626.5)],
             modulations=["QPSK"],
             bw_min_hz=25_000, bw_max_hz=40_000, burst=True),
        dict(name="Inmarsat AERO", category="Satellite",
             freq_bands_mhz=[(1525.0, 1559.0)],
             modulations=["BPSK", "QPSK"],
             bw_min_hz=8_000, bw_max_hz=12_000),

        # ── Radar / Misc ──────────────────────────────────────────────────────
        dict(name="FMCW Radar", category="Radar",
             modulations=["CSS"], chirp=True),
        dict(name="Pulse Radar", category="Radar",
             burst=True, bw_min_hz=500_000, bw_max_hz=50_000_000),
        dict(name="Weather Radar (WSR-88D)", category="Radar",
             freq_bands_mhz=[(2700.0, 2900.0)], burst=True),
    ]
