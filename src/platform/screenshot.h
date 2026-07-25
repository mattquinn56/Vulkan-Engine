#pragma once

#include <cstdint>
#include <string>

namespace screenshot {

// Encodes 8-bit RGBA pixels as a PNG. Rows are top-to-bottom, tightly packed.
// Returns false if the file could not be written.
bool write_png(const std::string& path, uint32_t width, uint32_t height, const uint8_t* rgba);

} // namespace screenshot
