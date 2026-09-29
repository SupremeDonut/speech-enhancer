#include "AudioEngine.hpp"
#include "WavFile.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <charconv>
#include <optional>

#include <array>
#include <cstring>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif
namespace {
std::string cpu_name() {
#if defined(_MSC_VER) && defined(_M_X64)
    int registers[4]{};
    __cpuid(registers, static_cast<int>(0x80000000));
    if (static_cast<unsigned int>(registers[0]) >= 0x80000004u) {
        std::array<char,49> brand{};
        for (int leaf=0; leaf<3; ++leaf) {
            __cpuid(registers, static_cast<int>(0x80000002u + leaf));
            std::memcpy(brand.data()+leaf*16, registers, 16);
        }
        return std::string(brand.data());
    }
#endif
    return "unknown";
}
std::string compiler_name() {
#if defined(_MSC_FULL_VER)
    return "MSVC-" + std::to_string(_MSC_FULL_VER);
#else
    return "unknown";
#endif
}
volatile std::sig_atomic_t keep_running = 1;
void on_sigint(int) { keep_running = 0; }

struct Arguments {
    std::string model;
    std::string input;
    std::string output;
    std::optional<std::uint32_t> capture_device;
    std::optional<std::uint32_t> playback_device;
    bool live = false;
    bool system_audio = false;
    bool list_devices = false;
};

std::uint32_t device_index(const std::string& option, const std::string& value) {
    std::uint32_t index = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), index);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
        throw std::invalid_argument(option + " requires a nonnegative device index");
    return index;
}

Arguments parse(int argc, char** argv) {
    Arguments args;
    bool have_model = false, have_input = false, have_output = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        auto value = [&]() -> std::string {
            if (i+1 >= argc) throw std::invalid_argument("Missing value after " + option);
            return argv[++i];
        };
        if (option == "--model") { args.model = value(); have_model = true; }
        else if (option == "--input") { args.input = value(); have_input = true; }
        else if (option == "--output") { args.output = value(); have_output = true; }
        else if (option == "--capture-device")
            args.capture_device = device_index(option, value());
        else if (option == "--playback-device")
            args.playback_device = device_index(option, value());
        else if (option == "--live") args.live = true;
        else if (option == "--system-audio") args.system_audio = true;
        else if (option == "--list-devices") args.list_devices = true;
        else throw std::invalid_argument("Unknown option: " + option);
    }
    if (args.list_devices) {
        if (argc != 2) throw std::invalid_argument("--list-devices must be used alone");
        return args;
    }
    if (!have_model) throw std::invalid_argument("--model PATH is required");
    if (args.live && args.system_audio)
        throw std::invalid_argument("--live and --system-audio are mutually exclusive");
    if (args.live) {
        if (have_input || have_output || args.capture_device || args.playback_device)
            throw std::invalid_argument("--live cannot be combined with file or device-selection options");
    } else if (args.system_audio) {
        if (have_input || have_output)
            throw std::invalid_argument("--system-audio cannot be combined with --input/--output");
        if (!args.capture_device || !args.playback_device)
            throw std::invalid_argument(
                "--system-audio requires both --capture-device and --playback-device");
    } else if (args.capture_device || args.playback_device) {
        throw std::invalid_argument("--capture-device/--playback-device require --system-audio");
    } else if (!have_input || !have_output) {
        throw std::invalid_argument(
            "Use --model PATH --live, --model PATH --system-audio --capture-device INDEX "
            "--playback-device INDEX, or --model PATH --input INPUT --output OUTPUT.wav");
    }
    return args;
}

void usage() {
    std::cerr << "Usage: speech_isolation --list-devices\n"
                 "   or: speech_isolation --model PATH --live\n"
                 "   or: speech_isolation --model PATH --system-audio "
                 "--capture-device INDEX --playback-device INDEX\n"
                 "   or: speech_isolation --model PATH --input INPUT --output OUTPUT.wav\n";
}

}

int main(int argc, char** argv) {
    try {
        const Arguments args = parse(argc, argv);
        if (args.list_devices) {
            std::cout << speech::AudioEngine::list_devices();
            return 0;
        }
        if (args.live || args.system_audio) {
            speech::AudioEngine engine(args.model);
            std::signal(SIGINT, on_sigint);
            engine.start_live(args.capture_device, args.playback_device);
            std::cout << (args.system_audio ? "mode=system-audio " : "mode=live ")
                      << "model=" << args.model << " cpu=\"" << cpu_name()
                      << "\" compiler=" << compiler_name() << " onnxruntime=" << Ort::GetVersionString()
                      << " backend=WASAPI app_format=mono-float32 app_rate=16000 "
                      << engine.device_description() << '\n';
            while (keep_running && !engine.faulted())
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            engine.stop_live();
            std::cout << "capture_overruns=" << engine.capture_overruns()
                      << " render_underruns=" << engine.render_underruns() << '\n';
            if (engine.faulted()) throw std::runtime_error("Live worker failed: " + engine.error_message());
        } else {
            // Validate/read all user input before creating runtime or audio state.
            const auto samples = speech::read_input_audio(args.input);
            speech::AudioEngine engine(args.model);
            const auto output = engine.process_file(samples);
            speech::write_float_wav(args.output, output);
            std::cout << "mode=offline model=" << args.model << " cpu=\"" << cpu_name()
                      << "\" compiler=" << compiler_name() << " onnxruntime=" << Ort::GetVersionString()
                      << " sample_rate=16000 input_samples=" << samples.size()
                      << " output_samples=" << output.size() << " output=" << args.output << '\n';
        }
        return 0;
    } catch (const std::invalid_argument& error) {
        usage();
        std::cerr << "Argument error: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
