#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <optional>

#include <miniaudio.h>

#include "DSPPipeline.hpp"
#include "InferenceSession.hpp"
#include "RingBuffer.hpp"

namespace speech {

class AudioEngine final {
public:
    explicit AudioEngine(const std::string& model_path);
    ~AudioEngine();
    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    std::vector<float> process_file(const std::vector<float>& input);
    static std::string list_devices();
    void start_live(std::optional<std::uint32_t> capture_device_index = std::nullopt,
                    std::optional<std::uint32_t> playback_device_index = std::nullopt);
    void stop_live() noexcept;
    bool faulted() const noexcept { return error_.load(std::memory_order_acquire); }
    std::size_t capture_overruns() const noexcept { return input_overruns_.load(); }
    std::size_t render_underruns() const noexcept { return output_underruns_.load(); }
    const std::string& error_message() const noexcept { return error_message_; }

    std::string device_description() const;
private:
    enum class Phase : uint8_t { active, reset_requested, worker_ready, priming };
    static void audio_callback(ma_device*, void*, const void*, ma_uint32);
    void on_audio(float* output, const float* input, std::size_t frames) noexcept;
    void worker_loop() noexcept;
    void request_reset(Phase expected, bool input_overrun) noexcept;
    void set_error(const char* message) noexcept;

    InferenceSession inference_;
    DSPPipeline dsp_;
    SPSCRingBuffer<float> input_ring_{1024};
    SPSCRingBuffer<float> output_ring_{1024};
    std::atomic<Phase> phase_{Phase::priming};
    std::atomic<bool> stop_worker_{false};
    std::atomic<bool> error_{false};
    std::atomic<std::size_t> input_overruns_{0};
    std::atomic<std::size_t> output_underruns_{0};
    std::thread worker_;
    std::mutex error_mutex_;
    std::string error_message_;
    ma_context context_{};
    ma_device device_{};
    bool context_initialized_ = false;
    bool device_initialized_ = false;
    bool device_started_ = false;
};

}  // namespace speech
