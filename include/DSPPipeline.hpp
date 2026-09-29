#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <kiss_fftr.h>

namespace speech {

class DSPPipeline final {
public:
    static constexpr std::size_t window_size = 512;
    static constexpr std::size_t hop_size = 160;
    static constexpr std::size_t bins = 257;
    static constexpr std::size_t delay = window_size - hop_size;
    using Hop = std::array<float, hop_size>;
    using Features = std::array<float, bins>;
    using Mask = std::array<float, bins>;
    using MaskProvider = void (*)(const Features&, Mask&, Mask&, void*);

    DSPPipeline();
    ~DSPPipeline();
    DSPPipeline(const DSPPipeline&) = delete;
    DSPPipeline& operator=(const DSPPipeline&) = delete;

    void reset() noexcept;
    // Exactly one synthesize call must follow each analyze call.
    const Features& analyze(const Hop& input);
    void synthesize(const Mask& real, const Mask& imaginary, Hop& output);
    // Clears state on entry; zero-pads final hop, flushes three hops, trims delay.
    // Provider is mandatory: no synthetic/fallback neural mask in production.
    std::vector<float> process_file(const std::vector<float>& input,
                                    MaskProvider provider, void* context);

private:
    kiss_fftr_cfg forward_ = nullptr;
    kiss_fftr_cfg inverse_ = nullptr;
    std::array<float, window_size> history_{};
    std::array<float, window_size> analysis_time_{};
    std::array<kiss_fft_cpx, bins> frequency_{};
    std::array<kiss_fft_cpx, bins> inverse_frequency_{};
    std::array<float, window_size> inverse_time_{};
    std::array<float, window_size> ola_{};
    std::array<float, window_size> analysis_window_{};
    std::array<float, window_size> synthesis_window_{};
    Features features_{};
    std::size_t frames_ = 0;
    bool awaiting_synthesis_ = false;
};

}  // namespace speech
