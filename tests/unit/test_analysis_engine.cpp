/*
========================================================================
Project: OpenRFStack
Author:  Brendan Michaud
Year:    2026
Part of OpenRFStack (https://github.com/OpenRFStack)

Licensed under the Personal Use License.
Do not use for commercial, organizational, or military purposes.
Contact author for permission: https://github.com/OpenRFStack
========================================================================
*/
#include <au/units/seconds.hh>
#include <au/units/hertz.hh>
#include <gtest/gtest.h>
#include "AnalysisEngine.hpp"
#include "Config.hpp"
#include <vector>
#include <cmath>

using namespace analysis;

// ── Helpers ───────────────────────────────────────────────────────────────────

static EngineConfig makeEngineConfig() {
    EngineConfig cfg;
    cfg.fft_size         = 256;
    cfg.snr_threshold_db = 5.0;
    cfg.rank             = 1;
    // ONNX disabled — model_path empty
    cfg.onnx.enabled     = false;
    cfg.onnx.model_path  = "";
    return cfg;
}

// Synthesise 2048 floats of a complex tone (interleaved I,Q) at freq_frac.
static std::vector<float> makeToneIq(float freq_frac = 0.25f, float amp = 1.0f) {
    std::vector<float> iq(2048);
    for (int i = 0; i < 1024; ++i) {
        float phi = 2.f * static_cast<float>(M_PI) * freq_frac * (float)i;
        iq[2 * i]     = amp * std::cos(phi);
        iq[2 * i + 1] = amp * std::sin(phi);
    }
    return iq;
}

// ── onnxLoaded() ──────────────────────────────────────────────────────────────

TEST(AnalysisEngine, OnnxLoaded_FalseWhenNoModel) {
    AnalysisEngine engine(makeEngineConfig());
    EXPECT_FALSE(engine.onnxLoaded())
        << "onnxLoaded() must return false when no model path is configured";
}

// ── analyzeSnapshot() — no ONNX ───────────────────────────────────────────────

TEST(AnalysisEngine, AnalyzeSnapshot_EmptyIq_ReturnsUnclassified) {
    AnalysisEngine engine(makeEngineConfig());
    std::vector<float> empty;
    auto result = engine.analyzeSnapshot(empty, au::hertz(20e6), au::hertz(433e6), "test-id", "test-scanner");

    EXPECT_FALSE(result.classified)
        << "analyzeSnapshot with empty IQ must return classified=false";
    EXPECT_FALSE(result.onnx_used)
        << "onnx_used must be false when no inference ran";
    EXPECT_FALSE(result.reject_reason.empty())
        << "reject_reason should explain why classification was not attempted";
}

TEST(AnalysisEngine, AnalyzeSnapshot_NoOnnx_ReturnsUnclassified) {
    AnalysisEngine engine(makeEngineConfig());
    auto iq = makeToneIq();
    auto result = engine.analyzeSnapshot(iq, au::hertz(20e6), au::hertz(915e6), "test-id", "test-scanner");

    EXPECT_FALSE(result.classified)
        << "analyzeSnapshot without an ONNX model must return classified=false";
    EXPECT_FALSE(result.onnx_used);
    EXPECT_FALSE(result.reject_reason.empty());
}

TEST(AnalysisEngine, AnalyzeSnapshot_PopulatesMetadata) {
    AnalysisEngine engine(makeEngineConfig());
    auto iq = makeToneIq();
    const double cf = 915e6;
    auto result = engine.analyzeSnapshot(iq, au::hertz(20e6), au::hertz(cf), "req-abc", "scanner-0");

    EXPECT_EQ(result.detection_id, "req-abc");
    EXPECT_EQ(result.scanner_id,   "scanner-0");
    EXPECT_NEAR(result.center_freq_hz.in(au::hertz), cf, 1.0);
    EXPECT_GT(result.timestamp_ms.in(au::seconds), 0.0)
        << "timestamp_ms must be set even for unclassified results";
}

// ── analyzeSnapshot() + analyzeSnapshot() independence ───────────────────────

TEST(AnalysisEngine, AnalyzeSnapshot_TwiceIsIdempotent) {
    AnalysisEngine engine(makeEngineConfig());
    auto iq = makeToneIq();
    auto r1 = engine.analyzeSnapshot(iq, au::hertz(20e6), au::hertz(100e6), "id1", "s");
    auto r2 = engine.analyzeSnapshot(iq, au::hertz(20e6), au::hertz(100e6), "id2", "s");

    // With no ONNX both results must be consistently unclassified
    EXPECT_EQ(r1.classified, r2.classified);
    EXPECT_EQ(r1.onnx_used,  r2.onnx_used);
}

// ── analyze() still works alongside analyzeSnapshot() ────────────────────────

TEST(AnalysisEngine, FullAnalyze_ZeroInput_StillRuns) {
    AnalysisEngine engine(makeEngineConfig());
    // All-zero IQ → very low SNR → should not crash regardless of result
    std::vector<float> zeros(65536 * 2, 0.f);
    EXPECT_NO_THROW({
        engine.analyze(zeros, au::hertz(2e6), au::hertz(100e6), "id", "scanner");
    });
}

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
