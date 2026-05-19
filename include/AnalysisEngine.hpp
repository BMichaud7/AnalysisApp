#pragma once
#include "Config.hpp"
#include "SignalFeatures.hpp"
#include "AnalysisResult.hpp"
#include "FeatureExtractor.hpp"
#include "ModulationClassifier.hpp"
#include "ProtocolMapper.hpp"
#include "OnnxClassifier.hpp"
#include <vector>
#include <string>
#include <future>

namespace analysis {

class AnalysisEngine {
public:
    explicit AnalysisEngine(const EngineConfig& cfg);

    // Full analysis: feature extraction + rule classifier + ONNX (parallel).
    AnalysisResult analyze(const std::vector<float>& iq_cf32,
                           double sample_rate_sps,
                           double center_freq_hz,
                           const std::string& detection_id,
                           const std::string& scanner_id) const;

    // Fast-path: ONNX only, no feature extraction, no SDR re-acquisition.
    // Returns a result with onnx_used=true and onnx_confidence set.
    // The caller checks onnx_confidence and falls back to analyze() if too low.
    AnalysisResult analyzeSnapshot(const std::vector<float>& iq_snapshot,
                                   double sample_rate_sps,
                                   double center_freq_hz,
                                   const std::string& detection_id,
                                   const std::string& scanner_id) const;

    bool onnxLoaded() const { return onnx_.loaded(); }

private:
    // Returns 0–1 certainty of the rule-based result.
    // Structural detections (OFDM CP, chirp, FHSS) → 1.0.
    // Cumulant-based PSK/QAM → moderate.
    // UNKNOWN → 0.0.
    static float ruleConfidence(const AnalysisResult& r,
                                const SignalFeatures&  f);

    EngineConfig         cfg_;
    FeatureExtractor     extractor_;
    ModulationClassifier classifier_;
    ProtocolMapper       mapper_;
    OnnxClassifier       onnx_;
};

} // namespace analysis
