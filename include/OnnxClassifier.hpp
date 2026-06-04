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
 * @file OnnxClassifier.hpp
 * @brief ONNX Runtime-based modulation classifier.
 *
 * OnnxClassifier loads a pre-trained modulation classifier model
 * (`modulation_classifier.onnx`) at startup and exposes single-signal and
 * batch inference entry points.
 *
 * ## Execution providers
 * TensorRT is tried first (fp16, best throughput), then CUDA, then CPU.
 * The first available provider is used.
 *
 * ## Fast path
 * When an RF_DETECTION message includes a 1 024-sample IQ snapshot
 * (`iq_snapshot_b64`), AnalysisEngine::analyzeSnapshot() calls
 * OnnxClassifier::classify() directly — no SDR re-acquisition, no feature
 * extraction.  Classification latency: ~5 ms on GPU.
 *
 * ## Compilation
 * The class compiles to a no-op stub when @c ANALYSIS_WITH_ONNX is not defined.
 * loaded() returns false and classify() returns an invalid OnnxResult.
 */
#include "AnalysisResult.hpp"
#include <au/units/hertz.hh>
#include <string>
#include <vector>
#include <memory>

#ifdef ANALYSIS_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace analysis {

/**
 * @brief ONNX model and inference configuration.
 *
 * Loaded from the @c \<onnx\> or @c \<onnx_low_snr\> XML block in analysis.xml.
 *
 * @par Current model — 47-class RadioResNet (val_acc = 0.748, best epoch 38/40)
 * - Input:  @c (1, 2, 512) — 2-channel real (I,Q), 512 complex samples
 * - Output: @c (1, 47) — softmax probability per class
 * - Training: 1.06M samples, focal loss γ=2, GPU, 40 epochs
 *
 * @par Execution provider selection
 * TensorRT (if @c use_tensorrt=true) → CUDA (if @c use_gpu=true) → CPU
 */
struct OnnxConfig {
    std::string model_path;                  ///< Absolute or relative path to the @c .onnx model file. Empty = disabled.
    std::string classes_path;                ///< Path to the @c .classes.json label index file.
    bool        enabled          = false;    ///< Auto-set to true when model_path is non-empty. Do not set manually.
    bool        use_gpu          = true;     ///< Enable CUDA execution provider. Falls back to CPU if unavailable.
    bool        use_tensorrt     = false;    ///< Try TensorRT EP before CUDA (requires TensorRT ≥ 8.x installed).
    bool        tensorrt_fp16    = true;     ///< Use fp16 precision kernels on RTX Tensor Cores (faster, same accuracy).
    int         tensorrt_cache_mb= 128;      ///< TensorRT engine cache workspace (MB). Increase for large models.
    double      fallback_confidence = 0.60;  ///< Run ONNX when rule engine confidence is below this value (0–1).
    bool        fallback_on_unknown = true;  ///< Always run ONNX when the rule engine returns UNKNOWN modulation.
    int         input_len        = 1024;     ///< IQ samples per inference window (512 complex = 1024 real). Must match training.
    int         max_batch        = 8;        ///< Maximum signals batched per GPU inference call. Higher = better GPU utilisation.
};

/// @brief Result of one ONNX inference.
struct OnnxResult {
    std::string modulation;     ///< Top class label (e.g. "BPSK", "FM").
    float       confidence;     ///< Softmax probability of the top class (0–1).
    bool        valid = false;  ///< False when the model is not loaded or inference failed.
};

/**
 * @brief ONNX Runtime wrapper for modulation classification.
 *
 * Thread-safe: const inference methods may be called from multiple threads
 * simultaneously (ONNX Runtime sessions are thread-safe by design).
 */
class OnnxClassifier {
public:
    /**
     * @brief Load the model from the path in @p cfg.
     * @param cfg ONNX configuration.  If cfg.model_path is empty, the classifier
     *            initialises as unloaded (loaded() returns false).
     */
    explicit OnnxClassifier(const OnnxConfig& cfg);
    ~OnnxClassifier();

    /// @brief True if the model was loaded successfully.
    bool loaded() const { return loaded_; }

    /**
     * @brief Classify one signal.
     * @param iq_cf32         Interleaved float32 IQ samples (length = OnnxConfig::input_len).
     * @param sample_rate_sps Sample rate (samples/s); may be used for normalisation.
     * @return OnnxResult with valid = true on success.
     */
    OnnxResult classify(const std::vector<float>& iq_cf32,
                        au::QuantityD<au::Hertz> sample_rate_sps) const;

    /**
     * @brief Batch classify multiple signals in one GPU launch.
     *
     * Faster than calling classify() in a loop when multiple signals are ready.
     * At most OnnxConfig::max_batch signals are processed per call.
     *
     * @param signals         Per-signal IQ vectors (each length = OnnxConfig::input_len).
     * @param sample_rate_sps Common sample rate for all signals.
     * @return One OnnxResult per input signal (same order).
     */
    std::vector<OnnxResult> classifyBatch(
        const std::vector<std::vector<float>>& signals,
        au::QuantityD<au::Hertz> sample_rate_sps) const;

    /// @brief Class label names in softmax output order.
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
