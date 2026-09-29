#pragma once

#include <filesystem>
#include <vector>

namespace speech {

std::vector<float> read_input_wav(const std::filesystem::path& path);
std::vector<float> read_input_audio(const std::filesystem::path& path);
void write_float_wav(const std::filesystem::path& path, const std::vector<float>& samples);

}  // namespace speech
