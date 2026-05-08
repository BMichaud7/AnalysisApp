#include "OnnxClassifier.hpp"

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

// Softmax in-place
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
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (cfg_.use_gpu) {
            try {
                OrtCUDAProviderOptions cuda{};
                opts.AppendExecutionProvider_CUDA(cuda);
                spdlog::info("OnnxClassifier: CUDA execution provider enabled");
            } catch (const Ort::Exception& e) {
                spdlog::warn("OnnxClassifier: CUDA unavailable ({}), using CPU", e.what());
            }
        }

        session_ = Ort::Session(env_, cfg_.model_path.c_str(), opts);
        loaded_  = true;

        spdlog::info("OnnxClassifier: loaded '{}' — {} classes, input_len={}",
                     cfg_.model_path, class_names_.size(), cfg_.input_len);
    } catch (const std::exception& e) {
        spdlog::error("OnnxClassifier: failed to load model: {}", e.what());
    }
#else
    spdlog::warn("OnnxClassifier: built without ONNX Runtime (-DWITH_ONNX=OFF)");
#endif
}

OnnxClassifier::~OnnxClassifier() = default;

OnnxResult OnnxClassifier::classify(const std::vector<float>& iq_cf32,
                                     double /*sample_rate_sps*/) const
{
    OnnxResult out;
    if (!loaded_) return out;

#ifdef ANALYSIS_WITH_ONNX
    try {
        const int L = cfg_.input_len;
        const int n_samp = static_cast<int>(iq_cf32.size()) / 2;

        // Build (1, 2, L) tensor: channel 0 = I, channel 1 = Q
        std::vector<float> input(2 * L, 0.f);
        int copy_len = std::min(n_samp, L);
        for (int i = 0; i < copy_len; ++i) {
            input[i]     = iq_cf32[i * 2];       // I
            input[L + i] = iq_cf32[i * 2 + 1];   // Q
        }

        // Unit-power normalise
        float pwr = 0.f;
        for (float v : input) pwr += v * v;
        pwr /= static_cast<float>(2 * L);
        if (pwr > 0.f) {
            float scale = 1.f / std::sqrt(pwr);
            for (float& v : input) v *= scale;
        }

        std::array<int64_t, 3> shape{1, 2, L};
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input.data(), input.size(),
            shape.data(), shape.size());

        const char* in_names[]  = {"iq_input"};
        const char* out_names[] = {"logits"};
        auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                    in_names, &input_tensor, 1,
                                    out_names, 1);

        float* logits = outputs[0].GetTensorMutableData<float>();
        int64_t n_cls = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];

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

} // namespace analysis
