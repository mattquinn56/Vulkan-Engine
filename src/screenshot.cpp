#include "screenshot.h"

#include <cstdio>
#include <vector>

// A minimal PNG encoder. PNG's only defined compression is DEFLATE, but DEFLATE
// permits "stored" (uncompressed) blocks, so a valid file needs no compressor
// and no third-party dependency. Diagnostic captures are written once and read
// by tooling, so file size does not matter here.

namespace {

uint32_t crc32_of(const uint8_t* data, size_t length, uint32_t crc = 0xFFFFFFFFu) {
    static uint32_t table[256];
    static bool built = false;
    if (!built) {
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[n] = c;
        }
        built = true;
    }
    for (size_t i = 0; i < length; i++) {
        crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc;
}

void push_be32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(uint8_t(value >> 24));
    out.push_back(uint8_t(value >> 16));
    out.push_back(uint8_t(value >> 8));
    out.push_back(uint8_t(value));
}

// A PNG chunk is length, type, payload, then CRC over type+payload.
void push_chunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& payload) {
    push_be32(out, uint32_t(payload.size()));
    const size_t crcStart = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), payload.begin(), payload.end());
    push_be32(out, crc32_of(out.data() + crcStart, out.size() - crcStart) ^ 0xFFFFFFFFu);
}

// zlib stream wrapping DEFLATE stored blocks, each capped at 65535 bytes.
std::vector<uint8_t> zlib_stored(const std::vector<uint8_t>& raw) {
    std::vector<uint8_t> out;
    out.push_back(0x78); // CMF: deflate, 32K window
    out.push_back(0x01); // FLG: no dictionary, check bits

    size_t offset = 0;
    while (offset < raw.size() || raw.empty()) {
        const size_t remaining = raw.size() - offset;
        const uint16_t blockSize = uint16_t(remaining > 65535 ? 65535 : remaining);
        const bool isFinal = (size_t(blockSize) == remaining);

        out.push_back(isFinal ? 1 : 0);
        out.push_back(uint8_t(blockSize & 0xFF));
        out.push_back(uint8_t(blockSize >> 8));
        out.push_back(uint8_t(~blockSize & 0xFF));
        out.push_back(uint8_t((~blockSize >> 8) & 0xFF));
        out.insert(out.end(), raw.begin() + offset, raw.begin() + offset + blockSize);

        offset += blockSize;
        if (isFinal) {
            break;
        }
    }

    uint32_t a = 1, b = 0;
    for (uint8_t byte : raw) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    push_be32(out, (b << 16) | a);
    return out;
}

} // namespace

bool screenshot::write_png(const std::string& path, uint32_t width, uint32_t height, const uint8_t* rgba) {
    if (width == 0 || height == 0 || rgba == nullptr) {
        return false;
    }

    // Each scanline is prefixed with its filter type; 0 means no filtering.
    std::vector<uint8_t> raw;
    raw.reserve(size_t(height) * (size_t(width) * 4 + 1));
    for (uint32_t y = 0; y < height; y++) {
        raw.push_back(0);
        const uint8_t* row = rgba + size_t(y) * size_t(width) * 4;
        raw.insert(raw.end(), row, row + size_t(width) * 4);
    }

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};

    std::vector<uint8_t> ihdr;
    push_be32(ihdr, width);
    push_be32(ihdr, height);
    ihdr.push_back(8); // bit depth
    ihdr.push_back(6); // colour type: truecolour with alpha
    ihdr.push_back(0); // deflate
    ihdr.push_back(0); // adaptive filtering
    ihdr.push_back(0); // no interlace
    push_chunk(png, "IHDR", ihdr);
    push_chunk(png, "IDAT", zlib_stored(raw));
    push_chunk(png, "IEND", {});

    FILE* file = nullptr;
    if (fopen_s(&file, path.c_str(), "wb") != 0 || file == nullptr) {
        return false;
    }
    const size_t written = fwrite(png.data(), 1, png.size(), file);
    fclose(file);
    return written == png.size();
}
