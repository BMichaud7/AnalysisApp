#pragma once
/**
 * @file AnalysisEngine.hpp
 * @brief Orchestrates the full signal analysis pipeline.
 *
 * AnalysisEngine combines FeatureExtractor, ModulationClassifier,
 * ProtocolMapper, and OnnxClassifier into two entry points:
 *
 * - **analyze()** — full pipeline: feature extraction + rule-based
 *   classification + optional ONNX override.  Requires a full IQ block
 *   (~65 536 samples at 2 MSPS ≈ 32 ms of RF time).
 *
 * - **analyzeSnapshot()** — fast path: ONNX-only inference on the 1 024-sample
 *   snapshot embedded in the RF_DETECTION message (~5 ms, no SDR re-acquisition).
 *   The caller should fall back to analyze() if onnx_confidence is below the
 *   configured fallback_confidence threshold.
 *
 * AnalysisEngine is stateless (const methods); multiple AnalysisService worker
 * threads may call it concurrently.
 */
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

/**
 * @brief Stateless analysis pipeline — thread-safe for concurrent calls.
 */
class AnalysisEngine {
public:
    /**
     * @brief Construct the engine and optionally load the ONNX model.
     * @param cfg Engine configuration (FFT size, SNR threshold, ONNX path, …).
     */
    explicit AnalysisEngine(const EngineConfig& cfg);

    /**
     * @brief Full analysis: feature extraction → rule classifier → ONNX (parallel).
     *
     * The rule-based classifier and the ONNX classifier run concurrently
     * (std::async).  If onnx_confidence > rule_confidence and onnx_confidence ≥
     * the fallback_confidence threshold, the ONNX result overrides the rules.
     *
     * @param iq_cf32       Interleaved float32 IQ samples (I,Q,I,Q,…).
     * @param sample_rate_sps Sample rate (samples/s).
     * @param center_freq_hz  Centre frequency (Hz); used by ProtocolMapper.
     * @param detection_id    UUID from the originating RF_DETECTION message.
     * @param scanner_id      Scanner identifier.
     * @return AnalysisResult with all five layers populated.
     */
    AnalysisResult analyze(const std::vector<float>& iq_cf32,
                           double sample_rate_sps,
                           double center_freq_hz,
                           const std::string& detection_id,
                           const std::string& scanner_id) const;

    /**
     * @brief Fast-path: ONNX-only inference on a short IQ snapshot.
     *
     * Skips feature extraction.  Returns a result with @p fast_path = true
     * and @p onnx_used = true.  Only Layers 1–2 (modulation type) are filled.
     *
     * @param iq_snapshot     1 024-sample CF32 snapshot from the RF_DETECTION message.
     * @param sample_rate_sps Sample rate of the snapshot (samples/s).
     * @param center_freq_hz  Centre frequency (Hz).
     * @param detection_id    UUID from the originating RF_DETECTION message.
     * @param scanner_id      Scanner identifier.
     * @return AnalysisResult with onnx_used = true and fast_path = true.
     */
    AnalysisResult analyzeSnapshot(const std::vector<float>& iq_snapshot,
                                   double sample_rate_sps,
                                   double center_freq_hz,
                                   const std::string& detection_id,
                                   const std::string& scanner_id) const;

    /// @brief True if the ONNX model was loaded successfully at construction.
    bool onnxLoaded() const { return onnx_.loaded(); }

private:
    /**
     * @brief Compute a certainty score for the rule-based result.
     *
     * Structural detections (OFDM CP, chirp, FHSS) return 1.0.
     * Cumulant-based PSK/QAM classifications return a moderate score.
     * UNKNOWN returns 0.0.
     *
     * @param r Result from the rule classifier.
     * @param f Feature vector used for classification.
     * @return Certainty in [0, 1].
     */
    static float ruleConfidence(const AnalysisResult& r, const SignalFeatures& f);

    EngineConfig         cfg_;
    FeatureExtractor     extractor_;
    ModulationClassifier classifier_;
    ProtocolMapper       mapper_;
    OnnxClassifier       onnx_;
};

} // namespace analysis
