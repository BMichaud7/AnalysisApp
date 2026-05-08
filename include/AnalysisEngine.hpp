#pragma once
#include "Config.hpp"
#include "SignalFeatures.hpp"
#include "AnalysisResult.hpp"
#include "FeatureExtractor.hpp"
#include "ModulationClassifier.hpp"
#include "ProtocolMapper.hpp"
#include <vector>
#include <string>

namespace analysis {

class AnalysisEngine {
public:
    explicit AnalysisEngine(const EngineConfig& cfg);

    // Full pipeline: raw IQ → AnalysisResult
    AnalysisResult analyze(const std::vector<float>& iq_cf32,
                           double sample_rate_sps,
                           double center_freq_hz,
                           const std::string& detection_id,
                           const std::string& scanner_id) const;

private:
    EngineConfig         cfg_;
    FeatureExtractor     extractor_;
    ModulationClassifier classifier_;
    ProtocolMapper       mapper_;
};

} // namespace analysis
