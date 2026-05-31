#include "ModulationClassifier.hpp"
#include <au/units/hertz.hh>
#include <au/units/seconds.hh>
#include <cmath>
#include <spdlog/spdlog.h>

namespace analysis {

// ── Internal helpers ──────────────────────────────────────────────────────────

bool ModulationClassifier::isAnalog(const SignalFeatures& f) const
{
    // High envelope variance strongly suggests AM family
    if (f.envelope_variance_norm > 0.25 && f.fm_deviation_hz < f.bandwidth_hz / 4.0)
        return true;
    // Wide FM deviation and low envelope variance suggests FM
    if (f.fm_deviation_hz > f.bandwidth_hz * 0.1 && f.envelope_variance_norm < 0.1)
        return true;
    // PM: moderate phase std, very constant envelope, low freq std
    if (f.inst_phase_std > 0.3
        && f.envelope_variance_norm < 0.15
        && f.inst_freq_std_hz < f.bandwidth_hz * 0.1)
        return true;
    return false;
}

bool ModulationClassifier::isConstantEnvelope(const SignalFeatures& f) const
{
    return f.envelope_variance_norm < 0.05;
}

// ── Analog classification ─────────────────────────────────────────────────────

std::string ModulationClassifier::classifyAnalog(const SignalFeatures& f,
                                                   double& index_out) const
{
    // FM family: constant envelope + large deviation
    if (f.fm_deviation_hz > f.bandwidth_hz * 0.1 && f.envelope_variance_norm < 0.1) {
        index_out = f.fm_deviation_hz.in(au::hertz);
        if (f.fm_deviation_hz > au::hertz(50000.0)) return "FM_WB";
        return "FM_NB";
    }

    // PM family
    if (f.inst_phase_std > 0.3
        && f.envelope_variance_norm < 0.15
        && f.inst_freq_std_hz < f.bandwidth_hz * 0.1) {
        index_out = f.inst_phase_std;
        return "PM";
    }

    // AM family
    index_out = f.am_index;

    // SSB: asymmetric spectrum
    if (f.am_index > 0.1 && f.spectral_symmetry < 0.5) {
        // Determine USB vs LSB from inst frequency centroid direction
        if (f.inst_freq_mean_hz.in(au::hertz) > 0) return "SSB_USB";
        return "SSB_LSB";
    }

    // DSB-SC: suppressed carrier, very small am_index
    if (f.am_index < 0.05 && f.envelope_variance_norm > 0.25)
        return "AM_DSB_SC";

    // CW / OOK: high flatness or very burst-like
    if (f.spectral_flatness > 0.8 || (f.is_burst && f.burst_duty_cycle < 0.6)) {
        index_out = 0;
        return "CW";
    }

    // Default: DSB with carrier
    return "AM_DSB_LC";
}

// ── Constant-envelope digital ─────────────────────────────────────────────────

std::string ModulationClassifier::classifyConstantEnvelope(const SignalFeatures& f,
                                                             int& m_ary_out) const
{
    // GMSK / MSK: very low spectral flatness (tonal), constant envelope, narrow
    if (f.spectral_flatness < 0.2 && f.envelope_variance_norm < 0.05) {
        m_ary_out = 2;
        return "GMSK";
    }

    // FSK: c42 ≈ -2, small |c40|
    if (std::abs(f.c42 - (-2.0)) < 0.5 && std::abs(f.c40_real) < 0.3) {
        m_ary_out = 2;
        return "FSK";
    }

    // BPSK: c40 ≈ -2, c42 ≈ -2  (real signal s=±1; both cumulants equal -2)
    if (std::abs(f.c40_real - (-2.0)) < 0.4 && f.c42 < -1.5) {
        m_ary_out = 2;
        return "BPSK";
    }

    // QPSK: c40 ≈ -2, c42 ≈ -1  (complex signal; E[s²]=0 → C42=E[|s|⁴]-2=-1)
    if (std::abs(f.c40_real - (-2.0)) < 0.5 && std::abs(f.c42 + 1.0) < 0.6) {
        m_ary_out = 4;
        return "QPSK";
    }

    // 8PSK: c40 close to 0, constant envelope
    if (std::abs(f.c40_real) < 0.4 && std::abs(f.c42 + 1.0) < 0.4) {
        m_ary_out = 8;
        return "8PSK";
    }

    // Fall through — generic PSK
    m_ary_out = 2;
    return "PSK";
}

// ── Variable-amplitude digital ────────────────────────────────────────────────

std::string ModulationClassifier::classifyVariableAmplitude(const SignalFeatures& f,
                                                              int& m_ary_out) const
{
    // QAM16: c40 ≈ -0.68, c42 ≈ -0.68
    if (std::abs(f.c40_real - (-0.68)) < 0.2 && std::abs(f.c42 - (-0.68)) < 0.2) {
        m_ary_out = 16;
        return "QAM16";
    }

    // QAM64: c40 ≈ -0.62, c42 ≈ -0.62
    if (std::abs(f.c40_real - (-0.62)) < 0.15 && std::abs(f.c42 - (-0.62)) < 0.15) {
        m_ary_out = 64;
        return "QAM64";
    }

    // QAM32: c40 ≈ -0.5, c42 ≈ -0.5
    if (std::abs(f.c40_real - (-0.5)) < 0.2 && std::abs(f.c42 - (-0.5)) < 0.2) {
        m_ary_out = 32;
        return "QAM32";
    }

    // BPSK-like but with variable amplitude (DSB-SC corner case)
    if (std::abs(f.c40_real - (-2.0)) < 0.3 && std::abs(f.c42 - 2.0) < 0.3) {
        m_ary_out = 2;
        return "BPSK";
    }

    // QPSK with some amplitude variation
    if (std::abs(f.c40_real - (-2.0)) < 0.4 && std::abs(f.c42) < 0.2) {
        m_ary_out = 4;
        return "QPSK";
    }

    // Default variable amplitude → QAM16
    m_ary_out = 16;
    return "QAM16";
}

// ── Classify digital ──────────────────────────────────────────────────────────

std::string ModulationClassifier::classifyDigital(const SignalFeatures& f,
                                                    int& m_ary_out) const
{
    if (isConstantEnvelope(f))
        return classifyConstantEnvelope(f, m_ary_out);
    return classifyVariableAmplitude(f, m_ary_out);
}

// ── Main classify ─────────────────────────────────────────────────────────────

void ModulationClassifier::classify(const SignalFeatures& f,
                                     AnalysisResult& r) const
{
    // Copy spectral basics
    r.center_freq_hz  = f.center_freq_hz;
    r.bandwidth_hz    = f.bandwidth_hz;
    r.snr_db          = f.snr_db;
    r.symbol_rate_sps = f.symbol_rate_sps;

    // Layer 3: Channel structure (always available)
    r.is_burst           = f.is_burst;
    r.burst_duty_cycle   = f.burst_duty_cycle;
    r.is_fhss            = f.fhss_detected;
    r.is_dsss            = f.dsss_detected;
    r.is_tdma            = f.is_burst && (f.burst_period_ms.in(au::seconds) > 0);
    r.ofdm_subcarrier_spacing_hz = au::hertz(0.0);

    // Layer 4 defaults
    r.bit_rate_bps    = 0;
    r.line_code       = "";
    r.fec_detected    = false;
    r.has_sync_pattern= false;
    r.classified      = true;
    r.analog_index    = 0;
    r.m_ary           = 0;
    r.is_ofdm         = false;
    r.is_spread        = f.fhss_detected || f.dsss_detected;
    r.analog_modulation  = "";
    r.digital_modulation = "";

    // ── Step 1: SNR gate ──────────────────────────────────────────────────
    if (f.snr_db < 5.0) {
        r.classified    = false;
        r.reject_reason = "SNR too low";
        spdlog::debug("Classifier: SNR {:.1f} dB below threshold, unclassified", f.snr_db);
        return;
    }

    // ── Step 2: Chirp ─────────────────────────────────────────────────────
    if (f.chirp_detected) {
        r.digital_modulation = "CSS";
        r.m_ary = 2;
        spdlog::debug("Classifier: chirp detected (rate={:.0f} Hz/s) → CSS",
                      f.chirp_rate_hz_s);
        return;
    }

    // ── Step 3: OFDM ─────────────────────────────────────────────────────
    if (f.ofdm_detected) {
        r.digital_modulation = "OFDM";
        r.is_ofdm = true;
        if (f.ofdm_fft_size_est > 0)
            r.symbol_rate_sps = f.sample_rate_sps / f.ofdm_fft_size_est;
        r.ofdm_subcarrier_spacing_hz = (f.ofdm_fft_size_est > 0)
                                       ? f.sample_rate_sps / f.ofdm_fft_size_est
                                       : au::hertz(0.0);
        spdlog::debug("Classifier: OFDM Nfft={}", f.ofdm_fft_size_est);
        return;
    }

    // ── Step 4: FHSS ─────────────────────────────────────────────────────
    if (f.fhss_detected) {
        r.digital_modulation = "FHSS";
        r.is_spread = true;
        spdlog::debug("Classifier: FHSS detected hop_rate={:.0f} Hz",
                      f.fhss_hop_rate_hz);
        return;
    }

    // ── Step 5: DSSS ─────────────────────────────────────────────────────
    if (f.dsss_detected) {
        r.digital_modulation = "DSSS";
        r.is_spread = true;
        return;
    }

    // ── Step 6: Analog vs digital decision ───────────────────────────────
    if (isAnalog(f)) {
        double idx = 0;
        r.analog_modulation = classifyAnalog(f, idx);
        r.analog_index      = idx;
        spdlog::debug("Classifier: analog → {}, index={:.3f}",
                      r.analog_modulation, idx);
        return;
    }

    // ── Step 7: Digital modulation ────────────────────────────────────────
    int m = 1;
    r.digital_modulation = classifyDigital(f, m);
    r.m_ary = m;

    // Line code heuristic for FSK
    if (r.digital_modulation == "FSK" || r.digital_modulation == "GMSK") {
        // Manchester encoding doubles the bandwidth relative to symbol rate
        const double bw_hz = f.bandwidth_hz.in(au::hertz);
        const double sr_hz = f.symbol_rate_sps.in(au::hertz);
        if (bw_hz > 0 && sr_hz > 0) {
            double ratio = bw_hz / sr_hz;
            if (ratio > 1.8 && ratio < 2.5)
                r.line_code = "Manchester";
        }
        if (r.line_code.empty()) r.line_code = "NRZ";
    }

    // ── Step 8/9: Bitstream traits ────────────────────────────────────────
    if (r.m_ary > 1 && r.symbol_rate_sps.in(au::hertz) > 0) {
        double bits_per_sym = std::log2(r.m_ary);
        r.bit_rate_bps = r.symbol_rate_sps.in(au::hertz) * bits_per_sym;
    }

    spdlog::debug("Classifier: digital → {} m_ary={} sr={:.0f} sps bit_rate={:.0f} bps",
                  r.digital_modulation, r.m_ary,
                  r.symbol_rate_sps.in(au::hertz), r.bit_rate_bps);
}

} // namespace analysis
