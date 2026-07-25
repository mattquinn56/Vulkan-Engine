#include "screenshot.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool screenshot::write_png(const std::string& path, uint32_t width, uint32_t height, const uint8_t* rgba) {
    if (width == 0 || height == 0 || rgba == nullptr) {
        return false;
    }

    const int stride = int(width) * 4;
    return stbi_write_png(path.c_str(), int(width), int(height), 4, rgba, stride) != 0;
}
