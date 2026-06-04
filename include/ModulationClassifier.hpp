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
#pragma once
/**
 * @file ModulationClassifier.hpp
 * @brief Rule-based modulation classifier from signal features.
 *
 * ModulationClassifier fills Layers 1–4 of AnalysisResult from a SignalFeatures
 * vector using decision-tree rules derived from higher-order cumulants,
 * envelope statistics, and structural indicators.
 *
 * It does **not** fill Layer 5 (protocol hypotheses) — that is ProtocolMapper's job.
 *
 * Classification tree:
 * 1. Analog vs digital decision (FM deviation, AM index, envelope variance).
 * 2. Constant-envelope vs variable-amplitude digital.
 * 3. PSK/MPSK vs QAM via C42 cumulant.
 * 4. Order M via C40.
 */
#include "SignalFeatures.hpp"
#include "AnalysisResult.hpp"

namespace analysis {

/// @brief Stateless rule-based modulation classifier.
class ModulationClassifier {
public:
    /**
     * @brief Classify a signal and populate Layers 1–4 of @p r.
     *
     * Reads SignalFeatures and writes to the analog_modulation, digital_modulation,
     * symbol_rate_sps, m_ary, is_ofdm, is_spread, is_burst, is_tdma, is_fhss,
     * is_dsss, ofdm_subcarrier_spacing_hz, bit_rate_bps, line_code, fec_detected,
     * and has_sync_pattern fields of @p r.
     *
     * @param f Feature vector from FeatureExtractor::extract().
     * @param r Result struct to populate (Layers 1–4 only).
     */
    void classify(const SignalFeatures& f, AnalysisResult& r) const;

private:
    bool isAnalog(const SignalFeatures& f) const;
    bool isConstantEnvelope(const SignalFeatures& f) const;

    std::string classifyAnalog(const SignalFeatures& f, double& index_out) const;
    std::string classifyDigital(const SignalFeatures& f, int& m_ary_out) const;
    std::string classifyConstantEnvelope(const SignalFeatures& f, int& m_ary_out) const;
    std::string classifyVariableAmplitude(const SignalFeatures& f, int& m_ary_out) const;
};

} // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
