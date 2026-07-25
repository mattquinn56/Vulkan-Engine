#pragma once

#include <cstdint>
#include <string>

namespace screenshot {

// Encodes 8-bit RGBA pixels as a PNG. Rows are top-to-bottom, tightly packed.
// Returns false if the file could not be written.
bool write_png(const std::string& path, uint32_t width, uint32_t height, const uint8_t* rgba);

struct Comparison
{
    bool referenceLoaded{false};
    bool sizeMatches{false};
    uint32_t referenceWidth{0};
    uint32_t referenceHeight{0};
    double meanChannelDifference{0.0}; // average |a-b| over RGB channels, 0-255
    uint32_t maxChannelDifference{0};
    double differingPixelFraction{0.0}; // pixels where any channel differs by > tolerance
};

// Compares RGBA pixels against a reference PNG on disk. A pixel counts as
// differing when any channel is more than perChannelTolerance apart.
Comparison compare_png(const std::string& referencePath, uint32_t width, uint32_t height, const uint8_t* rgba,
                       uint32_t perChannelTolerance);

} // namespace screenshot
