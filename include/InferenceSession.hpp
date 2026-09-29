#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "DSPPipeline.hpp"

namespace speech {

class InferenceSession final {
public:
    explicit InferenceSession(const std::string& model_path);
    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;

    void reset() noexcept;
    void infer(const DSPPipeline::Features& feature,
               DSPPipeline::Mask& mask_real,
               DSPPipeline::Mask& mask_imaginary);
    static void mask_provider(const DSPPipeline::Features& feature,
                              DSPPipeline::Mask& mask_real,
                              DSPPipeline::Mask& mask_imaginary,
                              void* context);
    const std::string& model_path() const noexcept { return model_path_; }

private:
    void validate_graph();
    static Ort::Value tensor(float* data, std::size_t length,
                             const int64_t* shape, std::size_t rank,
                             const Ort::MemoryInfo& memory);

    std::string model_path_;
    Ort::Env env_;
    Ort::SessionOptions options_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_{nullptr};
    Ort::RunOptions run_options_;
    std::array<float, DSPPipeline::bins> feature_{};
    std::array<std::array<float, DSPPipeline::bins>, 2> mask_storage_{};
    std::array<std::array<float, 256>, 2> hidden_{};
    std::array<std::array<Ort::Value, 2>, 2> inputs_;
    std::array<std::array<Ort::Value, 3>, 2> outputs_;
    std::size_t current_hidden_ = 0;
};

}  // namespace speech
