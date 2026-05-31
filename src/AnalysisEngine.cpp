#include "AnalysisEngine.hpp"
#include <au/units/seconds.hh>
#include <au/prefix.hh>
#include <chrono>
#include <spdlog/spdlog.h>

namespace analysis {

AnalysisEngine::AnalysisEngine(const EngineConfig& cfg)
    : cfg_(cfg)
    , extractor_(static_cast<int>(cfg.fft_size))
    , onnx_(cfg.onnx)
    , onnx_low_snr_(cfg.onnx_low_snr)
{}

// ── Rule confidence heuristic ─────────────────────────────────────────────────
// Assigns 0–1 certainty to a rule-based result so the engine can decide
// whether to invoke the ONNX fallback.
//
// High confidence (≥ 0.85): structural detections that don't depend on
//   cumulants and are reliable down to low SNR.
// Medium confidence (0.60): cumulant-based PSK/QAM where impairments may
//   corrupt the discriminant (see windowed-cumulant limitations).
// Low confidence (0.0): UNKNOWN — rule-based gave up entirely.

float AnalysisEngine::ruleConfidence(const AnalysisResult& r,
                                     const SignalFeatures&  f)
{
    if (!r.classified) return 0.f;

    const std::string& mod = r.digital_modulation.empty()
                             ? r.analog_modulation
                             : r.digital_modulation;

    if (mod.empty() || mod == "UNKNOWN" || mod == "PSK") return 0.f;

    // Structural detections: the CP correlator / chirp detector / FHSS hop
    // detector fire independently of cumulants — high confidence.
    if (r.is_ofdm   || mod == "OFDM")  return 0.95f;
    if (mod == "CSS")                   return 0.95f;
    if (r.is_fhss   || mod == "FHSS")  return 0.90f;

    // FM/AM: envelope and instantaneous-frequency features are robust to
    // phase noise; only degrade under very low SNR.
    if (mod == "FM_WB" || mod == "FM_NB") return 0.90f;
    if (mod.substr(0,2) == "AM" || mod == "SSB_USB" || mod == "SSB_LSB"
        || mod == "CW") return 0.85f;

    // FSK/MSK: inst_freq bimodal detection is reliable above SNR threshold.
    if (mod == "FSK" || mod == "MSK/GMSK") return 0.80f;

    // PSK/QAM: cumulant-based; phase noise and multipath reduce certainty.
    // The windowed-cumulant fix helps up to RTL-SDR class hardware,
    // but OTA multipath still breaks it.
    if (mod == "BPSK" || mod == "QPSK" || mod == "8PSK") return 0.65f;
    if (mod.substr(0,3) == "QAM")                         return 0.60f;

    return 0.70f;  // generic fallback for other classified signals
}


// ── Main analysis pipeline ─────────────────────────────────────────────────

AnalysisResult AnalysisEngine::analyze(const std::vector<float>& iq_cf32,
                                        au::QuantityD<au::Hertz> sample_rate_sps,
                                        au::QuantityD<au::Hertz> center_freq_hz,
                                        const std::string& detection_id,
                                        const std::string& scanner_id) const
{
    auto t0 = std::chrono::steady_clock::now();

    AnalysisResult result{};
    result.detection_id   = detection_id;
    result.scanner_id     = scanner_id;
    result.center_freq_hz = center_freq_hz;
    result.timestamp_ms   = au::milli(au::seconds)(
                                static_cast<double>(
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count()));

    if (iq_cf32.empty()) {
        result.classified    = false;
        result.reject_reason = "empty IQ buffer";
        return result;
    }

    // ── Layer 0: Feature extraction ──────────────────────────────────────
    // SNR must be known before we can pick high-SNR vs low-SNR model.
    SignalFeatures features = extractor_.extract(iq_cf32, sample_rate_sps,
                                                  center_freq_hz);
    result.snr_db       = features.snr_db;
    result.bandwidth_hz = features.bandwidth_hz;

    // ── Layers 1–4: Rule-based modulation classification ──────────────────
    classifier_.classify(features, result);
    result.rule_confidence = ruleConfidence(result, features);

    // ── ONNX inference — pick model based on measured SNR ─────────────────
    const bool use_low_snr_model = (features.snr_db < cfg_.snr_model_split_db)
                                   && onnx_low_snr_.loaded();
    const OnnxClassifier& active_onnx = use_low_snr_model ? onnx_low_snr_ : onnx_;
    const OnnxConfig&     ocfg        = use_low_snr_model ? cfg_.onnx_low_snr : cfg_.onnx;

    if (active_onnx.loaded()) {
        auto onnx_res = active_onnx.classify(iq_cf32, sample_rate_sps);
        bool use_onnx = onnx_res.valid && (
            (ocfg.fallback_on_unknown && result.rule_confidence == 0.f) ||
            (result.rule_confidence   <  static_cast<float>(ocfg.fallback_confidence))
        );
        if (use_onnx) {
            spdlog::debug("AnalysisEngine: ONNX{} override rule={} ({:.0f}%) → {} ({:.0f}%)",
                          use_low_snr_model ? "[low-SNR]" : "",
                          result.digital_modulation.empty()
                              ? result.analog_modulation : result.digital_modulation,
                          result.rule_confidence * 100.f,
                          onnx_res.modulation,
                          onnx_res.confidence * 100.f);
            result.digital_modulation = onnx_res.modulation;
            result.analog_modulation  = "";
            result.classified         = true;
            result.onnx_used          = true;
            result.onnx_confidence    = onnx_res.confidence;
        }
    }

    // ── Layer 5: Protocol mapping ─────────────────────────────────────────
    if (result.classified) {
        mapper_.map(features, result);
    }

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();

    const std::string& mod = result.digital_modulation.empty()
                             ? result.analog_modulation
                             : result.digital_modulation;

    spdlog::info("AnalysisEngine: [{:.3f} MHz] SNR={:.1f}dB mod={} "
                 "rule={:.0f}%{} hyp='{}' [{} ms]",
                 center_freq_hz.in(au::mega(au::hertz)),
                 result.snr_db,
                 mod.empty() ? "UNCLASSIFIED" : mod,
                 result.rule_confidence * 100.f,
                 result.onnx_used ? " +ONNX" : "",
                 result.hypotheses.empty() ? "none" : result.hypotheses[0].system,
                 elapsed_ms);

    return result;
}

// ── Fast-path: ONNX on the embedded snapshot ──────────────────────────────────
// No feature extraction, no SDR, no task — just classify the 1 024 IQ samples
// that arrived inside the RF_DETECTION AMQP message.
AnalysisResult AnalysisEngine::analyzeSnapshot(
    const std::vector<float>& iq_snapshot,
    au::QuantityD<au::Hertz> sample_rate_sps,
    au::QuantityD<au::Hertz> center_freq_hz,
    const std::string& detection_id,
    const std::string& scanner_id) const
{
    AnalysisResult result{};
    result.detection_id   = detection_id;
    result.scanner_id     = scanner_id;
    result.center_freq_hz = center_freq_hz;
    result.timestamp_ms   = au::milli(au::seconds)(
                                static_cast<double>(
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count()));

    if (iq_snapshot.empty() || !onnx_.loaded()) {
        result.classified    = false;
        result.reject_reason = "no snapshot or ONNX not loaded";
        return result;
    }

    auto onnx_res = onnx_.classify(iq_snapshot, sample_rate_sps);
    if (onnx_res.valid) {
        result.digital_modulation = onnx_res.modulation;
        result.classified         = true;
        result.onnx_used          = true;
        result.onnx_confidence    = onnx_res.confidence;

        // Run protocol mapping so hypotheses are populated even on the fast path.
        SignalFeatures minimal{};
        minimal.center_freq_hz  = center_freq_hz;
        minimal.sample_rate_sps = sample_rate_sps;
        mapper_.map(minimal, result);

        spdlog::debug("AnalysisEngine::analyzeSnapshot: {:.3f} MHz → {} ({:.0f}%)",
                      center_freq_hz.in(au::mega(au::hertz)),
                      onnx_res.modulation, onnx_res.confidence * 100.f);
    } else {
        result.classified    = false;
        result.reject_reason = "ONNX inference failed";
    }

    return result;
}
