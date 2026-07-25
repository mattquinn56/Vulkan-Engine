#include "platform/screenshot.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "stb_image.h"

#include <algorithm>
#include <cstdlib>

bool screenshot::write_png(const std::string& path, uint32_t width, uint32_t height, const uint8_t* rgba) {
    if (width == 0 || height == 0 || rgba == nullptr) {
        return false;
    }

    const int stride = int(width) * 4;
    return stbi_write_png(path.c_str(), int(width), int(height), 4, rgba, stride) != 0;
}

screenshot::Comparison screenshot::compare_png(const std::string& referencePath, uint32_t width, uint32_t height,
                                               const uint8_t* rgba, uint32_t perChannelTolerance) {
    Comparison result;

    int refWidth = 0, refHeight = 0, refChannels = 0;
    stbi_uc* reference = stbi_load(referencePath.c_str(), &refWidth, &refHeight, &refChannels, 4);
    if (reference == nullptr) {
        return result;
    }
    result.referenceLoaded = true;
    result.referenceWidth = uint32_t(refWidth);
    result.referenceHeight = uint32_t(refHeight);

    if (uint32_t(refWidth) != width || uint32_t(refHeight) != height) {
        stbi_image_free(reference);
        return result;
    }
    result.sizeMatches = true;

    const size_t pixels = size_t(width) * height;
    uint64_t channelSum = 0;
    uint64_t differing = 0;
    for (size_t i = 0; i < pixels; i++) {
        uint32_t worst = 0;
        for (int c = 0; c < 3; c++) {
            const int a = rgba[i * 4 + c];
            const int b = reference[i * 4 + c];
            const uint32_t delta = uint32_t(std::abs(a - b));
            channelSum += delta;
            worst = std::max(worst, delta);
        }
        result.maxChannelDifference = std::max(result.maxChannelDifference, worst);
        if (worst > perChannelTolerance) {
            differing++;
        }
    }

    result.meanChannelDifference = double(channelSum) / double(pixels * 3);
    result.differingPixelFraction = double(differing) / double(pixels);

    stbi_image_free(reference);
    return result;
}
