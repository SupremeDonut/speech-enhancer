#include "AudioEngine.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace speech {
namespace {
class WasapiContext final {
public:
    WasapiContext() {
        const ma_backend backend = ma_backend_wasapi;
        if (ma_context_init(&backend, 1, nullptr, &context_) != MA_SUCCESS)
            throw std::runtime_error("Cannot initialize WASAPI context");
    }

    ~WasapiContext() { ma_context_uninit(&context_); }

    ma_context* get() { return &context_; }

private:
    ma_context context_{};
};
}

std::string AudioEngine::list_devices() {
    WasapiContext context;
    ma_device_info* playback_devices = nullptr;
    ma_device_info* capture_devices = nullptr;
    ma_uint32 playback_count = 0;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(context.get(), &playback_devices, &playback_count,
                               &capture_devices, &capture_count) != MA_SUCCESS)
        throw std::runtime_error("Cannot enumerate WASAPI devices");

    std::ostringstream output;
    output << "Capture devices:\n";
    for (ma_uint32 i = 0; i < capture_count; ++i)
        output << "  [" << i << "] " << capture_devices[i].name
               << (capture_devices[i].isDefault ? " (default)" : "") << '\n';
    if (capture_count == 0) output << "  (none)\n";
    output << "Playback devices:\n";
    for (ma_uint32 i = 0; i < playback_count; ++i)
        output << "  [" << i << "] " << playback_devices[i].name
               << (playback_devices[i].isDefault ? " (default)" : "") << '\n';
    if (playback_count == 0) output << "  (none)\n";
    return output.str();
}


AudioEngine::AudioEngine(const std::string& model_path) : inference_(model_path) {}
AudioEngine::~AudioEngine() { stop_live(); }

std::vector<float> AudioEngine::process_file(const std::vector<float>& input) {
    inference_.reset();
    return dsp_.process_file(input, &InferenceSession::mask_provider, &inference_);
}

std::string AudioEngine::device_description() const {
    if (!device_initialized_) return "WASAPI device not initialized";
    return std::string("capture=") + device_.capture.name +
        " capture_hardware_rate=" + std::to_string(device_.capture.internalSampleRate) +
        " capture_period_frames=" + std::to_string(device_.capture.internalPeriodSizeInFrames) +
        " playback=" + device_.playback.name +
        " playback_hardware_rate=" + std::to_string(device_.playback.internalSampleRate) +
        " playback_period_frames=" + std::to_string(device_.playback.internalPeriodSizeInFrames);
}

void AudioEngine::start_live(std::optional<std::uint32_t> capture_device_index,
                             std::optional<std::uint32_t> playback_device_index) {
    if (capture_device_index.has_value() != playback_device_index.has_value())
        throw std::invalid_argument("Capture and playback device selections must be provided together");
    if (context_initialized_ || worker_.joinable()) throw std::logic_error("Live engine already started");
    const ma_backend backend = ma_backend_wasapi;
    ma_result result = ma_context_init(&backend, 1, nullptr, &context_);
    if (result != MA_SUCCESS) throw std::runtime_error("Cannot initialize WASAPI context");
    context_initialized_ = true;

    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    config.sampleRate = 16000;
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.dataCallback = &AudioEngine::audio_callback;
    config.pUserData = this;

    if (capture_device_index) {
        ma_device_info* playback_devices = nullptr;
        ma_device_info* capture_devices = nullptr;
        ma_uint32 playback_count = 0;
        ma_uint32 capture_count = 0;
        result = ma_context_get_devices(&context_, &playback_devices, &playback_count,
                                        &capture_devices, &capture_count);
        if (result != MA_SUCCESS) {
            stop_live();
            throw std::runtime_error("Cannot enumerate WASAPI devices");
        }
        if (*capture_device_index >= capture_count) {
            stop_live();
            throw std::out_of_range("Capture device index is not listed by --list-devices");
        }
        if (*playback_device_index >= playback_count) {
            stop_live();
            throw std::out_of_range("Playback device index is not listed by --list-devices");
        }
        config.capture.pDeviceID = &capture_devices[*capture_device_index].id;
        config.playback.pDeviceID = &playback_devices[*playback_device_index].id;
    }

    result = ma_device_init(&context_, &config, &device_);
    if (result != MA_SUCCESS) {
        stop_live();
        throw std::runtime_error("Cannot initialize selected mono 16 kHz float32 WASAPI duplex device");
    }
    device_initialized_ = true;
    stop_worker_.store(false, std::memory_order_release);
    error_.store(false, std::memory_order_release);
    phase_.store(Phase::priming, std::memory_order_release);
    try {
        worker_ = std::thread(&AudioEngine::worker_loop, this);
    } catch (...) {
        stop_live();
        throw;
    }
    result = ma_device_start(&device_);
    if (result != MA_SUCCESS) {
        stop_live();
        throw std::runtime_error("Cannot start selected WASAPI duplex device");
    }
    device_started_ = true;
}

void AudioEngine::stop_live() noexcept {
    if (device_started_) {
        ma_device_stop(&device_);
        device_started_ = false;
    }
    if (device_initialized_) {
        ma_device_uninit(&device_);
        device_initialized_ = false;
    }
    stop_worker_.store(true, std::memory_order_release);
    if (worker_.joinable()) worker_.join();
    if (context_initialized_) {
        ma_context_uninit(&context_);
        context_initialized_ = false;
    }
}

void AudioEngine::audio_callback(ma_device* device, void* output,
                                 const void* input, ma_uint32 frames) {
    auto* self = static_cast<AudioEngine*>(device->pUserData);
    self->on_audio(static_cast<float*>(output), static_cast<const float*>(input), frames);
}

void AudioEngine::request_reset(Phase expected, bool input_overrun) noexcept {
    if (phase_.compare_exchange_strong(expected, Phase::reset_requested,
                                       std::memory_order_acq_rel)) {
        if (input_overrun) input_overruns_.fetch_add(1, std::memory_order_relaxed);
        else output_underruns_.fetch_add(1, std::memory_order_relaxed);
    }
}

void AudioEngine::on_audio(float* output, const float* input, std::size_t frames) noexcept {
    std::fill_n(output, frames, 0.f);
    if (error_.load(std::memory_order_acquire)) return;

    Phase current = phase_.load(std::memory_order_acquire);
    if (current == Phase::reset_requested) return;
    if (current == Phase::worker_ready) {
        output_ring_.discard_all();
        phase_.store(Phase::priming, std::memory_order_release);
        current = Phase::priming;
    }
    if (input == nullptr) {
        request_reset(current, true);
        return;
    }
    if (input_ring_.write(input, frames) != frames) {
        request_reset(current, true);
        return;
    }
    if (current == Phase::priming) {
        if (output_ring_.readable() < frames) return;
        Phase expected = Phase::priming;
        if (!phase_.compare_exchange_strong(expected, Phase::active,
                                            std::memory_order_acq_rel)) return;
    }
    const std::size_t received = output_ring_.read(output, frames);
    if (received < frames) {
        std::fill(output + received, output + frames, 0.f);
        request_reset(Phase::active, false);
    }
}

void AudioEngine::set_error(const char* message) noexcept {
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_message_ = message;
    }
    error_.store(true, std::memory_order_release);
}

void AudioEngine::worker_loop() noexcept {
    DSPPipeline::Hop input_hop{}, output_hop{};
    DSPPipeline::Mask mask_real{}, mask_imaginary{};
    try {
        while (!stop_worker_.load(std::memory_order_acquire)) {
            const Phase current = phase_.load(std::memory_order_acquire);
            if (current == Phase::reset_requested) {
                input_ring_.discard_all();
                dsp_.reset();
                inference_.reset();
                phase_.store(Phase::worker_ready, std::memory_order_release);
                continue;
            }
            if (current == Phase::worker_ready || input_ring_.readable() < DSPPipeline::hop_size ||
                output_ring_.writable() < DSPPipeline::hop_size) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (input_ring_.read(input_hop.data(), input_hop.size()) != input_hop.size()) continue;
            const auto& feature = dsp_.analyze(input_hop);
            inference_.infer(feature, mask_real, mask_imaginary);
            dsp_.synthesize(mask_real, mask_imaginary, output_hop);
            if (phase_.load(std::memory_order_acquire) == Phase::reset_requested) continue;
            if (output_ring_.write(output_hop.data(), output_hop.size()) != output_hop.size()) {
                set_error("Output ring invariant failed in worker");
                return;
            }
        }
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown worker failure");
    }
}

}  // namespace speech
