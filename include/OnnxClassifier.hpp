#pragma once
#include "AnalysisResult.hpp"
#include <string>
#include <vector>
#include <memory>

#ifdef ANALYSIS_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace analysis {

struct OnnxConfig {
    std::string model_path;                          // path to .onnx file
    std::string classes_path;                        // path to .classes.json
    bool        enabled                 = false;     // false if model_path empty
    bool        use_gpu                 = true;      // try CUDA, fall back to CPU
    double      fallback_confidence     = 0.60;      // run ONNX if rule confidence < this
    bool        fallback_on_unknown     = true;      // always run ONNX on UNKNOWN result
    int         input_len               = 1024;      // samples per inference window
};

struct OnnxResult {
    std::string modulation;     // e.g. "BPSK", "QPSK", "FM"
    float       confidence;     // softmax probability of top class
    bool        valid = false;  // false if model not loaded or inference failed
};

class OnnxClassifier {
public:
    explicit OnnxClassifier(const OnnxConfig& cfg);
    ~OnnxClassifier();

    bool loaded() const { return loaded_; }

    // Run inference on raw IQ (interleaved float32 I,Q,I,Q,...).
    // Internally crops / pads to input_len samples.
    OnnxResult classify(const std::vector<float>& iq_cf32,
                        double sample_rate_sps) const;

    const std::vector<std::string>& classNames() const { return class_names_; }

private:
    OnnxConfig  cfg_;
    bool        loaded_  = false;
    std::vector<std::string> class_names_;

#ifdef ANALYSIS_WITH_ONNX
    mutable Ort::Env            env_;
    mutable Ort::Session        session_{nullptr};
    mutable Ort::AllocatorWithDefaultOptions allocator_;
#endif
};

} // namespace analysis
