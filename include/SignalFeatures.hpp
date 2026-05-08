#pragma once
#include <cstdint>
#include <complex>
#include <vector>

namespace analysis {

struct SignalFeatures {
    // ── Spectral ─────────────────────────────────────────────────────────
    double center_freq_hz    = 0;
    double bandwidth_hz      = 0;   // -10 dB occupied bandwidth
    double snr_db            = 0;
    double spectral_flatness = 0;   // 0=tonal, 1=flat/noise-like
    double spectral_symmetry = 0;   // 1=symmetric, 0=asymmetric

    // ── Envelope / instantaneous ─────────────────────────────────────────
    double envelope_mean     = 0;
    double envelope_std      = 0;
    double envelope_variance_norm = 0;  // var(|x|) / mean(|x|)^2
    double inst_freq_mean_hz = 0;       // mean instantaneous frequency
    double inst_freq_std_hz  = 0;       // std of instantaneous frequency
    double inst_phase_std    = 0;       // std of instantaneous phase
    double am_index          = 0;       // estimated AM modulation index
    double fm_deviation_hz   = 0;       // estimated FM deviation

    // ── Higher-order cumulants (normalized by power^2) ───────────────────
    double c40_real          = 0;   // Re{C40/M21^2}
    double c40_imag          = 0;
    double c42               = 0;   // C42/M21^2  (real-valued)
    double c41_real          = 0;

    // ── Symbol / bit rate ────────────────────────────────────────────────
    double symbol_rate_sps   = 0;   // estimated symbol rate (0 = not found)
    double bit_rate_bps      = 0;   // estimated bit rate

    // ── Structural ───────────────────────────────────────────────────────
    bool   is_burst          = false;
    double burst_duty_cycle  = 1.0; // 0–1
    double burst_period_ms   = 0;
    bool   ofdm_detected     = false;
    int    ofdm_fft_size_est = 0;
    double ofdm_cp_ratio     = 0;
    bool   fhss_detected     = false;
    double fhss_hop_rate_hz  = 0;
    bool   dsss_detected     = false;
    bool   chirp_detected    = false;
    double chirp_rate_hz_s   = 0;   // Hz/s sweep rate

    // ── Metadata ─────────────────────────────────────────────────────────
    int     sample_count     = 0;
    double  sample_rate_sps  = 0;
};

} // namespace analysis
