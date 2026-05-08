"""
Pure-Python / NumPy mirror of AnalysisApp's C++ FeatureExtractor.

Produces the same SignalFeatures struct fields so the Python test harness
can validate the feature extraction without requiring the C++ build.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.fft import fft, fftshift, fftfreq
from scipy.signal import welch, find_peaks


# ── Feature vector (mirrors include/SignalFeatures.hpp) ───────────────────────

@dataclass
class SignalFeatures:
    center_freq_hz:          float = 0.0
    bandwidth_hz:            float = 0.0
    snr_db:                  float = 0.0
    spectral_flatness:       float = 0.0
    spectral_symmetry:       float = 0.0

    envelope_mean:           float = 0.0
    envelope_std:            float = 0.0
    envelope_variance_norm:  float = 0.0
    inst_freq_mean_hz:       float = 0.0
    inst_freq_std_hz:        float = 0.0
    inst_phase_std:          float = 0.0
    am_index:                float = 0.0
    fm_deviation_hz:         float = 0.0

    c40_real:                float = 0.0
    c40_imag:                float = 0.0
    c42:                     float = 0.0
    c41_real:                float = 0.0

    symbol_rate_sps:         float = 0.0
    bit_rate_bps:            float = 0.0

    is_burst:                bool  = False
    burst_duty_cycle:        float = 1.0
    burst_period_ms:         float = 0.0
    ofdm_detected:           bool  = False
    ofdm_fft_size_est:       int   = 0
    ofdm_cp_ratio:           float = 0.0
    fhss_detected:           bool  = False
    fhss_hop_rate_hz:        float = 0.0
    dsss_detected:           bool  = False
    chirp_detected:          bool  = False
    chirp_rate_hz_s:         float = 0.0

    sample_count:            int   = 0
    sample_rate_sps:         float = 0.0


# ── Extractor ─────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extract signal features from complex IQ samples.

    Parameters
    ----------
    fft_size : int
        FFT size for PSD computation (must be power-of-two).
    """

    def __init__(self, fft_size: int = 4096):
        self.fft_size = fft_size

    def extract(self,
                iq:             np.ndarray,
                sample_rate:    float,
                center_freq:    float = 0.0) -> SignalFeatures:
        """
        Main entry point.

        Parameters
        ----------
        iq          : complex64 array, shape (N,)
        sample_rate : samples per second
        center_freq : nominal centre frequency in Hz (for metadata only)
        """
        iq = iq.astype(np.complex64)
        f  = SignalFeatures()
        f.sample_count    = len(iq)
        f.sample_rate_sps = sample_rate
        f.center_freq_hz  = center_freq

        if len(iq) < 256:
            return f

        # ── 1. PSD ────────────────────────────────────────────────────────────
        freqs, psd = welch(iq, fs=sample_rate, nperseg=min(self.fft_size, len(iq)//4),
                           return_onesided=False)
        psd  = np.abs(psd)
        psd  = np.fft.fftshift(psd)
        freqs = np.fft.fftshift(freqs)

        # ── 2. Bandwidth (-10 dB) ─────────────────────────────────────────────
        peak_idx   = int(np.argmax(psd))
        peak_power = psd[peak_idx]
        threshold  = peak_power / 10.0       # -10 dB
        above = psd >= threshold
        if above.any():
            lo  = freqs[np.where(above)[0][0]]
            hi  = freqs[np.where(above)[0][-1]]
            f.bandwidth_hz = float(hi - lo)

        # ── 3. SNR ────────────────────────────────────────────────────────────
        sorted_psd = np.sort(psd)
        noise_floor = float(np.mean(sorted_psd[:len(sorted_psd)//5]))
        signal_power = float(np.mean(psd[above]) if above.any() else peak_power)
        if noise_floor > 0:
            f.snr_db = float(10 * np.log10(signal_power / noise_floor))

        # ── 4. Spectral flatness ──────────────────────────────────────────────
        eps = 1e-30
        geo_mean = float(np.exp(np.mean(np.log(psd + eps))))
        ari_mean = float(np.mean(psd) + eps)
        f.spectral_flatness = float(np.clip(geo_mean / ari_mean, 0, 1))

        # ── 5. Spectral symmetry ──────────────────────────────────────────────
        n = len(psd)
        left  = psd[:n//2]
        right = psd[n//2:][::-1][:len(left)]
        denom = np.mean(left + right) + eps
        f.spectral_symmetry = float(1.0 - np.mean(np.abs(left - right)) / denom)

        # ── 6. Envelope statistics ────────────────────────────────────────────
        env  = np.abs(iq)
        emean = float(np.mean(env))
        estd  = float(np.std(env))
        f.envelope_mean = emean
        f.envelope_std  = estd
        if emean > 0:
            f.envelope_variance_norm = float(np.var(env) / (emean ** 2))
        emax, emin = float(env.max()), float(env.min())
        if emax + emin > 0:
            f.am_index = float((emax - emin) / (emax + emin))

        # ── 7. Instantaneous frequency ────────────────────────────────────────
        phase_diff = np.angle(iq[1:] * np.conj(iq[:-1]))
        inst_freq  = phase_diff * sample_rate / (2 * np.pi)
        f.inst_freq_mean_hz = float(np.mean(inst_freq))
        f.inst_freq_std_hz  = float(np.std(inst_freq))
        f.fm_deviation_hz   = f.inst_freq_std_hz

        # Instantaneous phase std (unwrapped)
        inst_phase = np.unwrap(np.angle(iq))
        f.inst_phase_std = float(np.std(np.diff(inst_phase)))

        # ── 8. Higher-order cumulants ─────────────────────────────────────────
        x    = iq.astype(np.complex128)
        n_s  = len(x)
        M20  = np.mean(x * x)
        M21  = float(np.mean(np.abs(x) ** 2))
        M40  = np.mean(x * x * x * x)
        M42  = float(np.mean(np.abs(x) ** 4))

        C40  = M40 - 3 * M20 ** 2
        C42  = M42 - abs(M20) ** 2 - 2 * M21 ** 2

        if M21 ** 2 > 1e-30:
            f.c40_real = float(np.real(C40) / M21 ** 2)
            f.c40_imag = float(np.imag(C40) / M21 ** 2)
            f.c42      = float(np.real(C42) / M21 ** 2)

        # ── 9. Symbol rate estimation ─────────────────────────────────────────
        f.symbol_rate_sps = self._estimate_symbol_rate(iq, sample_rate)

        # ── 10. OFDM detection ────────────────────────────────────────────────
        f.ofdm_detected, f.ofdm_fft_size_est, f.ofdm_cp_ratio = \
            self._detect_ofdm(iq, sample_rate)

        # ── 11. FHSS detection ────────────────────────────────────────────────
        f.fhss_detected, f.fhss_hop_rate_hz = \
            self._detect_fhss(iq, sample_rate)

        # ── 12. Chirp detection ───────────────────────────────────────────────
        f.chirp_detected, f.chirp_rate_hz_s = \
            self._detect_chirp(iq, sample_rate)

        # ── 13. Burst detection ───────────────────────────────────────────────
        f.is_burst, f.burst_duty_cycle, f.burst_period_ms = \
            self._detect_burst(iq, sample_rate)

        # ── 14. Bit rate estimate ─────────────────────────────────────────────
        # Will be filled by classifier once m_ary is known; set 0 for now
        f.bit_rate_bps = 0.0

        return f

    # ── Private helpers ───────────────────────────────────────────────────────

    def _estimate_symbol_rate(self, iq: np.ndarray, sr: float) -> float:
        """
        Estimate symbol rate via cyclostationary spectral lines.

        Two passes:
          1. Squared envelope |x|^2 − mean → peaks at symbol rate (ASK/FSK)
          2. Fourth power x^4 → peaks at 2×symbol rate (BPSK) or 4×sr (QPSK)
        """
        n     = min(len(iq), 65536)
        x     = iq[:n]
        freqs = fftfreq(n, d=1.0/sr)

        # Pass 1: squared envelope
        y = np.abs(x) ** 2
        y -= np.mean(y)
        Y = np.abs(fft(y))
        Y[0] = 0
        # Only look in range [BW/100, sr/2]
        bw_lo = max(1, int(n * (self.fft_size // 100) / n))
        half  = n // 2
        Y_half = Y[bw_lo:half]
        f_half = freqs[bw_lo:half]
        pk1_idx = int(np.argmax(Y_half))
        sr1 = float(abs(f_half[pk1_idx]))
        pk1_amp = float(Y_half[pk1_idx])

        # Pass 2: fourth power
        z = x ** 4
        Z = np.abs(fft(z))
        Z[0] = 0
        Z_half = Z[bw_lo:half]
        pk2_idx = int(np.argmax(Z_half))
        sr2_raw = float(abs(f_half[pk2_idx]))
        pk2_amp = float(Z_half[pk2_idx])
        # Fourth power peak is at 4×symrate for QPSK or 2×symrate for BPSK
        sr2 = sr2_raw / 4.0

        # Noise threshold: median * 20
        threshold = np.median(Y_half) * 20

        if pk1_amp > threshold and pk1_amp >= pk2_amp:
            return sr1
        if pk2_amp > threshold:
            return sr2
        return 0.0

    def _detect_ofdm(self, iq: np.ndarray, sr: float) \
            -> tuple[bool, int, float]:
        """
        Detect OFDM via cyclic-prefix autocorrelation.

        For each candidate (Nfft, Ncp), correlate the CP samples of each
        OFDM symbol with the corresponding tail of the IFFT output.
        Only triggered when spectral_flatness > 0.3 (OFDM has many equal-power
        subcarriers; AM/FM signals are tonal → flatness near 0).
        """
        nfft_candidates = [64, 128, 256, 512, 1024, 2048]
        cp_ratios       = [1/4, 1/8, 1/16, 1/32]

        x = iq[:min(len(iq), 32768)]
        pwr = float(np.mean(np.abs(x) ** 2))
        if pwr < 1e-12:
            return False, 0, 0.0

        # Spectral flatness gate: OFDM has near-flat spectrum
        n_fft_check = min(len(x), 4096)
        P    = np.abs(np.fft.fft(x[:n_fft_check])) ** 2
        eps  = 1e-30
        geo  = float(np.exp(np.mean(np.log(P + eps))))
        ari  = float(np.mean(P) + eps)
        spec_flat = float(np.clip(geo / ari, 0, 1))
        if spec_flat < 0.30:          # tonal/AM/FM → not OFDM
            return False, 0, 0.0

        best_corr = 0.0
        best_nfft = 0
        best_cp   = 0.0

        for nfft in nfft_candidates:
            for cp_frac in cp_ratios:
                ncp    = max(1, int(nfft * cp_frac))
                stride = nfft + ncp
                n_sym  = len(x) // stride
                if n_sym < 2:
                    continue

                # Correlate CP (start of symbol) with corresponding tail of IFFT
                corr_acc = 0.0
                count    = 0
                for i in range(n_sym):
                    cp_start   = i * stride
                    data_start = cp_start + ncp
                    tail_start = data_start + nfft - ncp
                    cp_end     = cp_start  + ncp
                    tail_end   = tail_start + ncp
                    if tail_end > len(x):
                        break
                    cp_seg   = x[cp_start:cp_end]
                    tail_seg = x[tail_start:tail_end]
                    corr_acc += float(np.abs(np.mean(cp_seg * np.conj(tail_seg))))
                    count    += 1

                if count == 0:
                    continue
                corr = (corr_acc / count) / (pwr + 1e-30)
                if corr > best_corr:
                    best_corr = corr
                    best_nfft = nfft
                    best_cp   = cp_frac

        detected = best_corr > 0.60
        return detected, best_nfft if detected else 0, best_cp if detected else 0.0

    def _detect_fhss(self, iq: np.ndarray, sr: float) \
            -> tuple[bool, float]:
        """
        Detect FHSS by measuring spectral centroid variance over time.
        Divides the signal into 32 segments and computes centroid per segment.
        """
        n_seg    = 32
        seg_len  = len(iq) // n_seg
        if seg_len < 64:
            return False, 0.0

        freqs    = fftfreq(seg_len, d=1.0/sr)
        centroids = []

        for i in range(n_seg):
            seg = iq[i*seg_len:(i+1)*seg_len]
            P   = np.abs(fft(seg)) ** 2
            P[0] = 0
            pwr  = P.sum()
            if pwr > 1e-12:
                c = float(np.sum(freqs * P) / pwr)
                centroids.append(c)

        if len(centroids) < 4:
            return False, 0.0

        centroid_std = float(np.std(centroids))
        bw_est = sr * 0.1   # rough 10% BW estimate

        detected  = centroid_std > bw_est
        hop_rate  = 0.0
        if detected and len(centroids) > 2:
            # Hop rate estimate: count transitions > 0.5*centroid_std per second
            transitions = sum(
                1 for j in range(1, len(centroids))
                if abs(centroids[j] - centroids[j-1]) > 0.5 * centroid_std
            )
            duration_s = len(iq) / sr
            hop_rate   = float(transitions / duration_s)

        return detected, hop_rate

    def _detect_chirp(self, iq: np.ndarray, sr: float) \
            -> tuple[bool, float]:
        """
        Detect linear FM chirp via instantaneous frequency linear regression.
        Uses the first 2000 samples for speed.
        """
        n = min(len(iq), 2048)
        x = iq[:n]
        if n < 64:
            return False, 0.0

        phase_diff = np.angle(x[1:] * np.conj(x[:-1]))
        inst_freq  = phase_diff * sr / (2 * np.pi)
        t          = np.arange(len(inst_freq)) / sr

        # Linear regression
        coeffs     = np.polyfit(t, inst_freq, 1)
        slope_hz_s = float(coeffs[0])
        fitted     = np.polyval(coeffs, t)
        ss_res     = float(np.sum((inst_freq - fitted) ** 2))
        ss_tot     = float(np.sum((inst_freq - np.mean(inst_freq)) ** 2))

        r_sq = 1.0 - ss_res / (ss_tot + 1e-30)

        # Chirp: R² > 0.90 and slope significant (> sr/200 Hz/s)
        min_slope = sr / 200.0
        detected  = (r_sq > 0.90) and (abs(slope_hz_s) > min_slope)

        return detected, slope_hz_s if detected else 0.0

    def _detect_burst(self, iq: np.ndarray, sr: float) \
            -> tuple[bool, float, float]:
        """
        Detect burst (duty < 0.85) via 100-window power thresholding.
        Returns (is_burst, duty_cycle, burst_period_ms).
        """
        n_win   = 100
        win_len = max(1, len(iq) // n_win)
        powers  = np.array([
            float(np.mean(np.abs(iq[i*win_len:(i+1)*win_len]) ** 2))
            for i in range(n_win)
        ])

        peak_power  = float(powers.max())
        threshold   = peak_power * 0.10
        on_mask     = powers >= threshold
        duty_cycle  = float(on_mask.mean())

        burst_period_ms = 0.0
        if duty_cycle < 0.85 and duty_cycle > 0.0:
            # Autocorrelation to estimate burst period
            on_f  = (on_mask.astype(float) - duty_cycle)
            ac    = np.correlate(on_f, on_f, mode='full')[n_win-1:]
            ac    = ac / (ac[0] + 1e-30)
            peaks, props = find_peaks(ac, height=0.3, distance=2)
            if len(peaks) > 0:
                period_windows   = float(peaks[0])
                period_ms        = period_windows * win_len / sr * 1000.0
                burst_period_ms  = float(period_ms)

        return duty_cycle < 0.85, duty_cycle, burst_period_ms
