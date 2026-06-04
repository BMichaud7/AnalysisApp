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
#include "OnnxClassifier.hpp"

#include <au/units/hertz.hh>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace analysis {

using json = nlohmann::json;

// ── Helpers ────────────────────────────────────────────────────────────────

static std::vector<std::string> loadClasses(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open classes file: " + path);
    auto j = json::parse(f);
    return j.get<std::vector<std::string>>();
}

static void softmax(std::vector<float>& v)
{
    float mx = *std::max_element(v.begin(), v.end());
    float sum = 0.f;
    for (auto& x : v) { x = std::exp(x - mx); sum += x; }
    for (auto& x : v) x /= sum;
}

// ── OnnxClassifier ─────────────────────────────────────────────────────────

OnnxClassifier::OnnxClassifier(const OnnxConfig& cfg)
    : cfg_(cfg)
#ifdef ANALYSIS_WITH_ONNX
    , env_(ORT_LOGGING_LEVEL_WARNING, "sdr_analysis")
#endif
{
    if (!cfg_.enabled || cfg_.model_path.empty()) {
        spdlog::info("OnnxClassifier: disabled (no model path configured)");
        return;
    }

#ifdef ANALYSIS_WITH_ONNX
    try {
        class_names_ = loadClasses(cfg_.classes_path);

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        bool gpu_ok = false;

        // ── TensorRT EP (fastest on RTX — uses FP16 Tensor Cores) ────────────
        if (cfg_.use_gpu && cfg_.use_tensorrt) {
            try {
                OrtTensorRTProviderOptions trt{};
                trt.device_id                  = 0;
                trt.trt_fp16_enable            = cfg_.tensorrt_fp16 ? 1 : 0;
                trt.trt_int8_enable            = 0;
                trt.trt_max_workspace_size     =
                    (size_t)cfg_.tensorrt_cache_mb * 1024 * 1024;
                // Cache compiled TRT engines so the first-run compilation
                // (~5–30 s) is not repeated on every restart.
                trt.trt_engine_cache_enable    = 1;
                trt.trt_engine_cache_path      = "/var/cache/sdr-analysis";
                trt.trt_max_partition_iterations = 1000;
                trt.trt_min_subgraph_size      = 1;
                opts.AppendExecutionProvider_TensorRT(trt);
                spdlog::info("OnnxClassifier: TensorRT EP enabled (fp16={})",
                             cfg_.tensorrt_fp16);
                gpu_ok = true;
            } catch (const Ort::Exception& e) {
                spdlog::warn("OnnxClassifier: TensorRT unavailable ({})", e.what());
            }
        }

        // ── CUDA EP (fallback when TRT not available) ─────────────────────────
        if (cfg_.use_gpu && !gpu_ok) {
            try {
                OrtCUDAProviderOptions cuda{};
                cuda.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cuda);
                spdlog::info("OnnxClassifier: CUDA EP enabled");
                gpu_ok = true;
            } catch (const Ort::Exception& e) {
                spdlog::warn("OnnxClassifier: CUDA unavailable ({}), using CPU",
                             e.what());
            }
        }

        if (!gpu_ok)
            spdlog::info("OnnxClassifier: running on CPU");

        session_ = Ort::Session(env_, cfg_.model_path.c_str(), opts);
        loaded_  = true;

        spdlog::info("OnnxClassifier: loaded '{}' — {} classes, input_len={}, max_batch={}",
                     cfg_.model_path, class_names_.size(),
                     cfg_.input_len, cfg_.max_batch);
    } catch (const std::exception& e) {
        spdlog::error("OnnxClassifier: failed to load model: {}", e.what());
    }
#else
    spdlog::warn("OnnxClassifier: built without ONNX Runtime (-DWITH_ONNX=OFF)");
#endif
}

OnnxClassifier::~OnnxClassifier() = default;

OnnxResult OnnxClassifier::classify(const std::vector<float>& iq_cf32,
                                     au::QuantityD<au::Hertz> /*sample_rate_sps*/) const
{
    OnnxResult out;
    if (!loaded_) return out;

#ifdef ANALYSIS_WITH_ONNX
    try {
        const int L      = cfg_.input_len;
        const int n_samp = static_cast<int>(iq_cf32.size()) / 2;

        // Build (1, 2, L) tensor: channel 0 = I, channel 1 = Q
        std::vector<float> input(2 * L, 0.f);
        int copy_len = std::min(n_samp, L);
        for (int i = 0; i < copy_len; ++i) {
            input[i]     = iq_cf32[i * 2];
            input[L + i] = iq_cf32[i * 2 + 1];
        }

        // Unit-power normalise (matches training preprocessing)
        float pwr = 0.f;
        for (float v : input) pwr += v * v;
        pwr /= static_cast<float>(2 * L);
        if (pwr > 0.f) {
            float scale = 1.f / std::sqrt(pwr);
            for (float& v : input) v *= scale;
        }

        std::array<int64_t, 3> shape{1, 2, L};
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor   = Ort::Value::CreateTensor<float>(
            mem_info, input.data(), input.size(), shape.data(), shape.size());

        const char* in_names[]  = {"iq_input"};
        const char* out_names[] = {"logits"};
        auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                    in_names, &tensor, 1,
                                    out_names, 1);

        float*  logits = outputs[0].GetTensorMutableData<float>();
        int64_t n_cls  = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];

        std::vector<float> probs(logits, logits + n_cls);
        softmax(probs);

        int top_idx = static_cast<int>(
            std::max_element(probs.begin(), probs.end()) - probs.begin());

        out.valid      = true;
        out.confidence = probs[top_idx];
        out.modulation = (top_idx < static_cast<int>(class_names_.size()))
                         ? class_names_[top_idx] : std::to_string(top_idx);

    } catch (const std::exception& e) {
        spdlog::warn("OnnxClassifier: inference error: {}", e.what());
    }
#endif

    return out;
}

// Batch variant: classify multiple signals in one forward pass.
// More efficient than calling classify() N times when the backlog is large.
std::vector<OnnxResult> OnnxClassifier::classifyBatch(
    const std::vector<std::vector<float>>& signals,
    au::QuantityD<au::Hertz> /*sample_rate_sps*/) const
{
    std::vector<OnnxResult> results(signals.size());
    if (!loaded_ || signals.empty()) return results;

#ifdef ANALYSIS_WITH_ONNX
    try {
        const int L   = cfg_.input_len;
        const int B   = static_cast<int>(signals.size());

        std::vector<float> input(B * 2 * L, 0.f);
        for (int b = 0; b < B; ++b) {
            const auto& iq = signals[b];
            int n_samp = static_cast<int>(iq.size()) / 2;
            int copy   = std::min(n_samp, L);
            float* dst_i = input.data() + b * 2 * L;
            float* dst_q = dst_i + L;
            for (int i = 0; i < copy; ++i) {
                dst_i[i] = iq[i * 2];
                dst_q[i] = iq[i * 2 + 1];
            }
            // Normalise per sample
            float pwr = 0.f;
            for (int i = 0; i < 2 * L; ++i) pwr += dst_i[i] * dst_i[i];
            pwr /= static_cast<float>(2 * L);
            if (pwr > 0.f) {
                float s = 1.f / std::sqrt(pwr);
                for (int i = 0; i < 2 * L; ++i) dst_i[i] *= s;
            }
        }

        std::array<int64_t, 3> shape{B, 2, L};
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor   = Ort::Value::CreateTensor<float>(
            mem_info, input.data(), input.size(), shape.data(), shape.size());

        const char* in_names[]  = {"iq_input"};
        const char* out_names[] = {"logits"};
        auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                    in_names, &tensor, 1, out_names, 1);

        float*  logits = outputs[0].GetTensorMutableData<float>();
        int64_t n_cls  = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];

        for (int b = 0; b < B; ++b) {
            std::vector<float> probs(logits + b * n_cls, logits + (b + 1) * n_cls);
            softmax(probs);
            int top = static_cast<int>(
                std::max_element(probs.begin(), probs.end()) - probs.begin());
            results[b].valid      = true;
            results[b].confidence = probs[top];
            results[b].modulation = (top < static_cast<int>(class_names_.size()))
                                    ? class_names_[top] : std::to_string(top);
        }
    } catch (const std::exception& e) {
        spdlog::warn("OnnxClassifier: batch inference error: {}", e.what());
    }
#endif

    return results;
}

} // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
