#include "InferenceSession.hpp"

#include <filesystem>
#include <stdexcept>

#include <algorithm>
#include <utility>
namespace speech {
namespace {
constexpr int64_t feature_shape[] = {1, 1, 257};
constexpr int64_t state_shape[] = {2, 1, 128};
constexpr std::array<const char*, 2> input_names{"input_mag", "h_in"};
constexpr std::array<const char*, 3> output_names{"mask_r", "mask_i", "h_out"};
constexpr std::array<std::array<int64_t, 3>, 5> expected_shapes{{
    {{1, 1, 257}}, {{2, 1, 128}}, {{1, 1, 257}}, {{1, 1, 257}}, {{2, 1, 128}}
}};
constexpr std::array<const char*, 5> all_names{
    "input_mag", "h_in", "mask_r", "mask_i", "h_out"
};

void validate_value(const Ort::Session& session, bool input, std::size_t index,
                    const char* expected_name, const int64_t* expected_shape,
                    std::size_t expected_rank) {
    auto allocated_name = input ? session.GetInputNameAllocated(index, Ort::AllocatorWithDefaultOptions{})
                                : session.GetOutputNameAllocated(index, Ort::AllocatorWithDefaultOptions{});
    if (std::string(allocated_name.get()) != expected_name) {
        throw std::runtime_error(std::string("Unexpected ONNX graph value name: ") + allocated_name.get());
    }
    auto info = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    if (info.GetONNXType() != ONNX_TYPE_TENSOR) {
        throw std::runtime_error(std::string(expected_name) + " must be a tensor");
    }
    auto tensor_info = info.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(std::string(expected_name) + " must be float32");
    }
    const auto shape = tensor_info.GetShape();
    if (shape.size() != expected_rank) {
        throw std::runtime_error(std::string(expected_name) + " has an unexpected rank");
    }
    for (std::size_t dim = 0; dim < expected_rank; ++dim) {
        if (shape[dim] != expected_shape[dim]) {
            throw std::runtime_error(std::string(expected_name) + " has an unexpected shape");
        }
    }
}
}  // namespace

Ort::Value InferenceSession::tensor(float* data, std::size_t length,
                                    const int64_t* shape, std::size_t rank,
                                    const Ort::MemoryInfo& memory) {
    return Ort::Value::CreateTensor<float>(memory, data, length, shape, rank);
}

InferenceSession::InferenceSession(const std::string& model_path)
    : model_path_(model_path), env_(ORT_LOGGING_LEVEL_WARNING, "SpeechIsolation") {
    if (!std::filesystem::is_regular_file(model_path_)) {
        throw std::runtime_error("Model file is missing or not a regular file: " + model_path_);
    }
    options_.SetIntraOpNumThreads(1);
    options_.SetInterOpNumThreads(1);
    options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, std::filesystem::path(model_path_).c_str(), options_);
    memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    validate_graph();

    for (std::size_t state = 0; state < 2; ++state) {
        inputs_[state][0] = tensor(feature_.data(), feature_.size(), feature_shape, 3, memory_);
        inputs_[state][1] = tensor(hidden_[state].data(), hidden_[state].size(), state_shape, 3, memory_);
        outputs_[state][0] = tensor(mask_storage_[0].data(), mask_storage_[0].size(), feature_shape, 3, memory_);
        outputs_[state][1] = tensor(mask_storage_[1].data(), mask_storage_[1].size(), feature_shape, 3, memory_);
        outputs_[state][2] = tensor(hidden_[state].data(), hidden_[state].size(), state_shape, 3, memory_);
    }
    run_options_.SetRunLogVerbosityLevel(0);
}

void InferenceSession::validate_graph() {
    if (session_.GetInputCount() != 2 || session_.GetOutputCount() != 3) {
        throw std::runtime_error("ONNX model must have two inputs and three outputs");
    }
    validate_value(session_, true, 0, all_names[0], expected_shapes[0].data(), 3);
    validate_value(session_, true, 1, all_names[1], expected_shapes[1].data(), 3);
    validate_value(session_, false, 0, all_names[2], expected_shapes[2].data(), 3);
    validate_value(session_, false, 1, all_names[3], expected_shapes[3].data(), 3);
    validate_value(session_, false, 2, all_names[4], expected_shapes[4].data(), 3);
}

void InferenceSession::reset() noexcept {
    hidden_[0].fill(0.f);
    hidden_[1].fill(0.f);
    current_hidden_ = 0;
}

void InferenceSession::infer(const DSPPipeline::Features& feature,
                             DSPPipeline::Mask& mask_real,
                             DSPPipeline::Mask& mask_imaginary) {
    std::copy(feature.begin(), feature.end(), feature_.begin());
    const std::size_t next = 1 - current_hidden_;
    session_.Run(run_options_, input_names.data(), inputs_[current_hidden_].data(), 2,
                 output_names.data(), outputs_[next].data(), 3);
    std::copy(mask_storage_[0].begin(), mask_storage_[0].end(), mask_real.begin());
    std::copy(mask_storage_[1].begin(), mask_storage_[1].end(), mask_imaginary.begin());
    current_hidden_ = next;
}

void InferenceSession::mask_provider(const DSPPipeline::Features& feature,
                                     DSPPipeline::Mask& mask_real,
                                     DSPPipeline::Mask& mask_imaginary,
                                     void* context) {
    if (context == nullptr) throw std::invalid_argument("Inference context is null");
    static_cast<InferenceSession*>(context)->infer(feature, mask_real, mask_imaginary);
}

}  // namespace speech
