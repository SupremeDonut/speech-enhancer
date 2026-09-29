#include "WavFile.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <system_error>
#include <vector>

namespace speech {
namespace {
uint16_t u16(const unsigned char* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}
uint32_t u32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
void put16(std::ostream& out, uint16_t value) {
    const char bytes[] = {static_cast<char>(value), static_cast<char>(value >> 8)};
    out.write(bytes, sizeof(bytes));
}
void put32(std::ostream& out, uint32_t value) {
    const char bytes[] = {static_cast<char>(value), static_cast<char>(value >> 8),
                          static_cast<char>(value >> 16), static_cast<char>(value >> 24)};
    out.write(bytes, sizeof(bytes));
}
void require_read(std::istream& in, char* buffer, std::streamsize count, const char* what) {
    in.read(buffer, count);
    if (in.gcount() != count) throw std::runtime_error(std::string("Truncated WAV ") + what);
}
std::wstring quote_argument(const std::wstring& argument) {
    std::wstring result(1, L'"');
    std::size_t backslashes = 0;
    for (const wchar_t ch : argument) {
        if (ch == L'\\') {
            ++backslashes;
        } else if (ch == L'"') {
            result.append(backslashes * 2 + 1, L'\\');
            result.push_back(ch);
            backslashes = 0;
        } else {
            result.append(backslashes, L'\\');
            result.push_back(ch);
            backslashes = 0;
        }
    }
    result.append(backslashes * 2, L'\\');
    result.push_back(L'"');
    return result;
}

class TemporaryFile {
public:
    TemporaryFile() {
        std::array<wchar_t, MAX_PATH> directory{};
        const DWORD directory_size = GetTempPathW(
            static_cast<DWORD>(directory.size()), directory.data());
        if (directory_size == 0 || directory_size >= directory.size())
            throw std::runtime_error("Cannot locate the Windows temporary directory");
        std::array<wchar_t, MAX_PATH> filename{};
        if (GetTempFileNameW(directory.data(), L"spc", 0, filename.data()) == 0)
            throw std::runtime_error("Cannot create a temporary WAV path");
        path_ = filename.data();
    }

    ~TemporaryFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

void convert_with_ffmpeg(const std::filesystem::path& input,
                         const std::filesystem::path& output) {
    std::wstring command = L"ffmpeg.exe -hide_banner -nostdin -loglevel error -y -i ";
    command += quote_argument(input.wstring());
    command += L" -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le -f wav ";
    command += quote_argument(output.wstring());
    std::vector<wchar_t> writable_command(command.begin(), command.end());
    writable_command.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(nullptr, writable_command.data(), nullptr, nullptr, FALSE,
                        0, nullptr, nullptr, &startup, &process)) {
        throw std::runtime_error(
            "Cannot start ffmpeg.exe; install FFmpeg and ensure it is on PATH "
            "(Windows error " + std::to_string(GetLastError()) + ")");
    }
    CloseHandle(process.hThread);
    const DWORD wait_result = WaitForSingleObject(process.hProcess, INFINITE);
    if (wait_result != WAIT_OBJECT_0) {
        const DWORD error = GetLastError();
        TerminateProcess(process.hProcess, 1);
        CloseHandle(process.hProcess);
        throw std::runtime_error("Waiting for FFmpeg failed (Windows error " +
                                 std::to_string(error) + ")");
    }
    DWORD exit_code = 0;
    const BOOL got_exit_code = GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hProcess);
    if (!got_exit_code)
        throw std::runtime_error("Cannot read FFmpeg's exit code");
    if (exit_code != 0)
        throw std::runtime_error("FFmpeg could not decode/convert the input (exit code " +
                                 std::to_string(exit_code) + ")");
}

}

std::vector<float> read_input_wav(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input WAV: " + path.string());
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    if (end < 12) throw std::runtime_error("WAV is empty or truncated: " + path.string());
    const uint64_t file_size = static_cast<uint64_t>(end);
    in.seekg(0);
    std::array<unsigned char, 12> header{};
    require_read(in, reinterpret_cast<char*>(header.data()), header.size(), "header");
    if (std::memcmp(header.data(), "RIFF", 4) || std::memcmp(header.data()+8, "WAVE", 4))
        throw std::runtime_error("Input is not a RIFF/WAVE file: " + path.string());
    const uint64_t riff_end = static_cast<uint64_t>(u32(header.data()+4)) + 8;
    if (riff_end > file_size || riff_end < 12) throw std::runtime_error("Invalid RIFF size");

    bool have_format = false, have_data = false;
    uint16_t format = 0, channels = 0, bits = 0, block_align = 0;
    uint32_t sample_rate = 0, data_size = 0;
    uint64_t data_offset = 0;
    uint64_t offset = 12;
    while (offset + 8 <= riff_end) {
        in.seekg(static_cast<std::streamoff>(offset));
        std::array<unsigned char, 8> chunk{};
        require_read(in, reinterpret_cast<char*>(chunk.data()), chunk.size(), "chunk header");
        const uint32_t chunk_size = u32(chunk.data()+4);
        const uint64_t payload = offset + 8;
        if (payload + chunk_size > riff_end) throw std::runtime_error("WAV chunk exceeds RIFF boundary");
        if (!std::memcmp(chunk.data(), "fmt ", 4)) {
            if (chunk_size < 16) throw std::runtime_error("WAV fmt chunk is too short");
            std::array<unsigned char, 16> fmt{};
            require_read(in, reinterpret_cast<char*>(fmt.data()), fmt.size(), "fmt chunk");
            format = u16(fmt.data()); channels = u16(fmt.data()+2);
            sample_rate = u32(fmt.data()+4); block_align = u16(fmt.data()+12); bits = u16(fmt.data()+14);
            have_format = true;
        } else if (!std::memcmp(chunk.data(), "data", 4) && !have_data) {
            data_offset = payload;
            data_size = chunk_size;
            have_data = true;
        }
        offset = payload + chunk_size + (chunk_size & 1u);
    }
    if (!have_format || !have_data) throw std::runtime_error("WAV must contain fmt and data chunks");
    if (channels != 1 || sample_rate != 16000)
        throw std::runtime_error("Input WAV must be mono at 16000 Hz");
    const bool pcm16 = format == 1 && bits == 16 && block_align == 2;
    const bool float32 = format == 3 && bits == 32 && block_align == 4;
    if (!pcm16 && !float32) throw std::runtime_error("Input WAV must be PCM16 or IEEE float32");
    if (data_size == 0 || data_size % block_align != 0)
        throw std::runtime_error("Input WAV has empty or misaligned sample data");
    const std::size_t count = data_size / block_align;
    std::vector<float> samples(count);
    in.seekg(static_cast<std::streamoff>(data_offset));
    if (pcm16) {
        std::array<unsigned char, 2> bytes{};
        for (float& sample : samples) {
            require_read(in, reinterpret_cast<char*>(bytes.data()), bytes.size(), "PCM16 sample");
            const auto value = static_cast<int16_t>(u16(bytes.data()));
            sample = static_cast<float>(value) / 32768.f;
        }
    } else {
        static_assert(sizeof(float) == sizeof(uint32_t));
        for (float& sample : samples) {
            std::array<unsigned char, 4> bytes{};
            require_read(in, reinterpret_cast<char*>(bytes.data()), bytes.size(), "float32 sample");
            const uint32_t bits_value = u32(bytes.data());
            std::memcpy(&sample, &bits_value, sizeof(sample));
            if (!std::isfinite(sample)) throw std::runtime_error("Input WAV contains non-finite samples");
        }
    }
    return samples;
}

std::vector<float> read_input_audio(const std::filesystem::path& path) {
    try {
        return read_input_wav(path);
    } catch (const std::runtime_error& wav_error) {
        std::error_code file_error;
        if (!std::filesystem::is_regular_file(path, file_error)) throw;
        try {
            TemporaryFile converted;
            convert_with_ffmpeg(path, converted.path());
            return read_input_wav(converted.path());
        } catch (const std::exception& ffmpeg_error) {
            throw std::runtime_error(
                "Input is not in the supported WAV format (" + std::string(wav_error.what()) +
                "); FFmpeg conversion failed: " + ffmpeg_error.what());
        }
    }
}

void write_float_wav(const std::filesystem::path& path, const std::vector<float>& samples) {
    if (samples.empty()) throw std::invalid_argument("Cannot write an empty output WAV");
    constexpr uint64_t header_size = 44;
    const uint64_t data_size = samples.size() * sizeof(float);
    if (data_size > std::numeric_limits<uint32_t>::max() - header_size)
        throw std::runtime_error("Output exceeds RIFF/WAV size limit");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot create output WAV: " + path.string());
    out.write("RIFF", 4); put32(out, static_cast<uint32_t>(36 + data_size)); out.write("WAVE", 4);
    out.write("fmt ", 4); put32(out, 16); put16(out, 3); put16(out, 1); put32(out, 16000);
    put32(out, 16000 * sizeof(float)); put16(out, sizeof(float)); put16(out, 32);
    out.write("data", 4); put32(out, static_cast<uint32_t>(data_size));
    for (float sample : samples) {
        uint32_t bits_value;
        std::memcpy(&bits_value, &sample, sizeof(bits_value));
        put32(out, bits_value);
    }
    if (!out) throw std::runtime_error("Failed writing output WAV: " + path.string());
}

}  // namespace speech
