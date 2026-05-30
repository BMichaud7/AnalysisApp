#pragma once
/**
 * @file SignalFeatures.hpp
 * @note Frequency fields use au::QuantityD<au::Hertz>;
 *       burst_period_ms uses au::QuantityD<au::Seconds>.
 * @brief Feature vector extracted from a block of IQ samples.
 *
 * SignalFeatures is populated by FeatureExtractor::extract() and consumed
 * by ModulationClassifier and ProtocolMapper.  Every field has a zero/false
 * default; FeatureExtractor sets only the fields it can reliably estimate.
 */
#include <au/units/hertz.hh>
#include <au/units/seconds.hh>
#include <cstdint>
#include <complex>
#include <vector>

namespace analysis {

/**
 * @brief Complete feature vector for one signal block.
 *
 * Features are grouped into five categories:
 * - Spectral (PSD shape, flatness, symmetry)
 * - Envelope and instantaneous frequency/phase
 * - Higher-order cumulants (HOC) for modulation classification
 * - Symbol/bit rate estimates
 * - Structural features (burst, OFDM, FHSS, DSSS, chirp)
 */
struct SignalFeatures {
    // ── Spectral ─────────────────────────────────────────────────────────────
    au::QuantityD<au::Hertz> center_freq_hz{au::hertz(0.0)}; ///< Centre frequency passed in by the caller.
    au::QuantityD<au::Hertz> bandwidth_hz{au::hertz(0.0)};   ///< −10 dB occupied bandwidth.
    double snr_db            = 0;  ///< Signal-to-noise ratio (dB).
    double spectral_flatness = 0;  ///< Wiener entropy: 0 = tonal, 1 = flat/noise-like.
    double spectral_symmetry = 0;  ///< PSD mirror symmetry: 1 = symmetric, 0 = asymmetric.

    // ── Envelope / instantaneous ─────────────────────────────────────────────
    double envelope_mean          = 0; ///< Mean of the signal envelope |x(t)|.
    double envelope_std           = 0; ///< Std deviation of the envelope.
    double envelope_variance_norm = 0; ///< var(|x|) / mean(|x|)^2 — amplitude modulation depth.
    au::QuantityD<au::Hertz> inst_freq_mean_hz{au::hertz(0.0)}; ///< Mean instantaneous frequency.
    au::QuantityD<au::Hertz> inst_freq_std_hz{au::hertz(0.0)};  ///< Std deviation of instantaneous frequency.
    double inst_phase_std         = 0; ///< Std deviation of instantaneous phase (rad).
    double am_index               = 0; ///< Estimated AM modulation index.
    au::QuantityD<au::Hertz> fm_deviation_hz{au::hertz(0.0)};   ///< Estimated FM peak deviation.

    // ── Higher-order cumulants (normalised by power^2) ───────────────────────
    double c40_real = 0; ///< Re{C40/M21^2} — 4th-order cumulant (real part).
    double c40_imag = 0; ///< Im{C40/M21^2} — 4th-order cumulant (imaginary part).
    double c42      = 0; ///< C42/M21^2 — mixed 4th-order cumulant (real-valued).
    double c41_real = 0; ///< Re{C41/M21^2} — mixed cumulant (real part).

    // ── Symbol / bit rate ────────────────────────────────────────────────────
    au::QuantityD<au::Hertz> symbol_rate_sps{au::hertz(0.0)}; ///< Estimated symbol rate (symbols/s; 0 = not found).
    double bit_rate_bps    = 0; ///< Estimated bit rate (bits/s; 0 = not found).

    // ── Structural ───────────────────────────────────────────────────────────
    bool   is_burst        = false; ///< True if burst (non-continuous) transmission.
    double burst_duty_cycle= 1.0;   ///< Fraction of time the carrier is active (0–1).
    au::QuantityD<au::Seconds> burst_period_ms{au::seconds(0.0)}; ///< Burst repetition period (0 = aperiodic).
    bool   ofdm_detected   = false; ///< True if OFDM cyclic-prefix structure found.
    int    ofdm_fft_size_est = 0;   ///< Estimated OFDM FFT size (0 if not OFDM).
    double ofdm_cp_ratio   = 0;     ///< Cyclic prefix ratio (CP length / FFT size).
    bool   fhss_detected   = false; ///< True if frequency-hopping pattern detected.
    double fhss_hop_rate_hz= 0;     ///< Estimated FHSS hop rate (Hz; 0 if not FHSS).
    bool   dsss_detected   = false; ///< True if direct-sequence spreading detected.
    bool   chirp_detected  = false; ///< True if linear FM chirp detected.
    double chirp_rate_hz_s = 0;     ///< Chirp sweep rate (Hz/s; 0 if not chirp).

    // ── Metadata ─────────────────────────────────────────────────────────────
    int    sample_count    = 0;  ///< Number of IQ samples analysed.
    au::QuantityD<au::Hertz> sample_rate_sps{au::hertz(0.0)}; ///< Sample rate of the analysed block.
};

} // namespace analysis
