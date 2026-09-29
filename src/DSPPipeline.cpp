#include "DSPPipeline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace speech {
namespace {
constexpr double pi = 3.14159265358979323846264338327950288;
}

DSPPipeline::DSPPipeline() {
    forward_ = kiss_fftr_alloc(static_cast<int>(window_size), 0, nullptr, nullptr);
    inverse_ = kiss_fftr_alloc(static_cast<int>(window_size), 1, nullptr, nullptr);
    if (!forward_ || !inverse_) {
        kiss_fftr_free(forward_);
        kiss_fftr_free(inverse_);
        throw std::runtime_error("Cannot allocate KissFFT forward/inverse configurations");
    }
    std::array<double, hop_size> phase{};
    for (std::size_t n = 0; n < window_size; ++n) {
        const double w = std::sin(pi * (static_cast<double>(n) + .5) / window_size);
        analysis_window_[n] = static_cast<float>(w);
        phase[n % hop_size] += w * w;
    }
    for (std::size_t n = 0; n < window_size; ++n) {
        synthesis_window_[n] = static_cast<float>(analysis_window_[n] / phase[n % hop_size]);
    }
}

DSPPipeline::~DSPPipeline() {
    kiss_fftr_free(forward_);
    kiss_fftr_free(inverse_);
}

void DSPPipeline::reset() noexcept {
    history_.fill(0.f);
    ola_.fill(0.f);
    features_.fill(0.f);
    frames_ = 0;
    awaiting_synthesis_ = false;
}

const DSPPipeline::Features& DSPPipeline::analyze(const Hop& input) {
    if (awaiting_synthesis_) {
        throw std::logic_error("Previous DSP frame requires synthesis or reset");
    }
    std::move(history_.begin() + hop_size, history_.end(), history_.begin());
    std::copy(input.begin(), input.end(), history_.end() - hop_size);
    for (std::size_t n = 0; n < window_size; ++n) {
        analysis_time_[n] = history_[n] * analysis_window_[n];
    }
    kiss_fftr(forward_, analysis_time_.data(), frequency_.data());
    for (std::size_t k = 0; k < bins; ++k) {
        const auto& bin = frequency_[k];
        features_[k] = std::log10(bin.r * bin.r + bin.i * bin.i + 1e-7f);
    }
    awaiting_synthesis_ = true;
    return features_;
}

void DSPPipeline::synthesize(const Mask& real, const Mask& imaginary, Hop& output) {
    if (!awaiting_synthesis_) {
        throw std::logic_error("Analyze a DSP frame before synthesis");
    }
    for (std::size_t k = 0; k < bins; ++k) {
        const auto& y = frequency_[k];
        inverse_frequency_[k] = {y.r * real[k] - y.i * imaginary[k],
                                 y.r * imaginary[k] + y.i * real[k]};
    }
    inverse_frequency_[0].i = 0.f;
    inverse_frequency_[bins - 1].i = 0.f;
    kiss_fftri(inverse_, inverse_frequency_.data(), inverse_time_.data());
    for (std::size_t n = 0; n < window_size; ++n) {
        ola_[n] += inverse_time_[n] * (1.f / window_size) * synthesis_window_[n];
    }
    std::copy_n(ola_.begin(), hop_size, output.begin());
    // Samples before index zero are padding, not audible output.
    const std::size_t negative = frames_ * hop_size < delay ?
        std::min(hop_size, delay - frames_ * hop_size) : 0;
    std::fill_n(output.begin(), negative, 0.f);
    std::move(ola_.begin() + hop_size, ola_.end(), ola_.begin());
    std::fill(ola_.end() - hop_size, ola_.end(), 0.f);
    ++frames_;
    awaiting_synthesis_ = false;
}

std::vector<float> DSPPipeline::process_file(const std::vector<float>& input,
                                             MaskProvider provider, void* context) {
    if (input.empty() || !provider) {
        throw std::invalid_argument("Finite input and mask provider are required");
    }
    reset();
    const std::size_t input_hops = input.size() / hop_size + (input.size() % hop_size != 0);
    constexpr std::size_t flush_hops = (delay + hop_size - 1) / hop_size;
    std::vector<float> result(input.size());
    Hop hop{}, rendered{};
    Mask real{}, imaginary{};
    for (std::size_t frame = 0; frame < input_hops + flush_hops; ++frame) {
        hop.fill(0.f);
        if (frame < input_hops) {
            const std::size_t begin = frame * hop_size;
            std::copy_n(input.begin() + begin, std::min(hop_size, input.size() - begin), hop.begin());
        }
        const auto& feature = analyze(hop);
        provider(feature, real, imaginary, context);
        synthesize(real, imaginary, rendered);
        const std::size_t begin = frame * hop_size;
        for (std::size_t i = 0; i < hop_size; ++i) {
            const std::size_t absolute = begin + i;
            if (absolute >= delay && absolute - delay < result.size()) {
                result[absolute - delay] = rendered[i];
            }
        }
    }
    return result;
}

}  // namespace speech
