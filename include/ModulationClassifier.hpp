#pragma once
#include "SignalFeatures.hpp"
#include "AnalysisResult.hpp"

namespace analysis {

class ModulationClassifier {
public:
    // Fills Layer 1–4 fields of result from features.
    // Does not fill hypotheses (that's ProtocolMapper's job).
    void classify(const SignalFeatures& f, AnalysisResult& r) const;

private:
    // Analog/digital decision
    bool isAnalog(const SignalFeatures& f) const;
    bool isConstantEnvelope(const SignalFeatures& f) const;

    std::string classifyAnalog(const SignalFeatures& f,
                                double& index_out) const;
    std::string classifyDigital(const SignalFeatures& f,
                                 int& m_ary_out) const;
    std::string classifyConstantEnvelope(const SignalFeatures& f,
                                          int& m_ary_out) const;
    std::string classifyVariableAmplitude(const SignalFeatures& f,
                                           int& m_ary_out) const;
};

} // namespace analysis
