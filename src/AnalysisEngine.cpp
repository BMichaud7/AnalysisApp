#include "AnalysisEngine.hpp"
#include <chrono>
#include <spdlog/spdlog.h>

namespace analysis {

AnalysisEngine::AnalysisEngine(const EngineConfig& cfg)
    : cfg_(cfg)
    , extractor_(static_cast<int>(cfg.fft_size))
{}

AnalysisResult AnalysisEngine::analyze(const std::vector<float>& iq_cf32,
                                        double sample_rate_sps,
                                        double center_freq_hz,
                                        const std::string& detection_id,
                                        const std::string& scanner_id) const
{
    auto t0 = std::chrono::steady_clock::now();

    AnalysisResult result{};
    result.detection_id  = detection_id;
    result.scanner_id    = scanner_id;
    result.center_freq_hz= center_freq_hz;
    result.timestamp_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();

    if (iq_cf32.empty()) {
        result.classified   = false;
        result.reject_reason= "empty IQ buffer";
        return result;
    }

    // Layer 0: Feature extraction
    SignalFeatures features = extractor_.extract(iq_cf32, sample_rate_sps, center_freq_hz);

    result.snr_db       = features.snr_db;
    result.bandwidth_hz = features.bandwidth_hz;

    // Layers 1–4: Modulation classification
    classifier_.classify(features, result);

    // Layer 5: Protocol mapping (only if classified)
    if (result.classified) {
        mapper_.map(features, result);
    }

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();

    spdlog::info("AnalysisEngine: [{:.3f} MHz] SNR={:.1f}dB mod={} hyp='{}' [{} ms]",
                 center_freq_hz / 1e6,
                 result.snr_db,
                 result.digital_modulation.empty()
                     ? result.analog_modulation
                     : result.digital_modulation,
                 result.hypotheses.empty() ? "none" : result.hypotheses[0].system,
                 elapsed_ms);

    return result;
}

} // namespace analysis
