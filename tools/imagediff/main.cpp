// Compares two PNGs and optionally writes a visualization of the difference.
// Kept out of the engine on purpose: this is image processing, not rendering,
// and staying a separate executable means it can be run and tested on its own.
//
//   imagediff <reference.png> <actual.png> [options]
//     --diff=<path>          write a difference visualization
//     --tolerance=<n>        per-channel difference ignored, 0-255 (default 4)
//     --max-differing=<f>    fraction of pixels allowed to differ (default 0.002)
//
// Exit codes: 0 match, 1 mismatch, 2 could not read or sizes differ.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options
{
    std::string referencePath;
    std::string actualPath;
    std::string diffPath;
    int tolerance = 4;
    double maxDifferingFraction = 0.002;
};

void print_usage() {
    std::fprintf(stderr, "Usage: imagediff <reference.png> <actual.png> [options]\n"
                         "  --diff=<path>         write a difference visualization\n"
                         "  --tolerance=<n>       per-channel difference ignored, 0-255 (default 4)\n"
                         "  --max-differing=<f>   fraction of pixels allowed to differ (default 0.002)\n");
}

bool parse_args(int argc, char** argv, Options& out) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg.starts_with("--diff=")) {
            out.diffPath = std::string(arg.substr(7));
        } else if (arg.starts_with("--tolerance=")) {
            out.tolerance = std::atoi(argv[i] + 12);
        } else if (arg.starts_with("--max-differing=")) {
            out.maxDifferingFraction = std::atof(argv[i] + 16);
        } else if (arg.starts_with("--")) {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        } else if (out.referencePath.empty()) {
            out.referencePath = std::string(arg);
        } else if (out.actualPath.empty()) {
            out.actualPath = std::string(arg);
        } else {
            std::fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
            return false;
        }
    }
    return !out.referencePath.empty() && !out.actualPath.empty();
}

// Blue for a barely-there difference through to red for a large one, which
// reads more clearly than a plain magnitude ramp when differences are subtle.
void heat_color(double t, uint8_t& r, uint8_t& g, uint8_t& b) {
    t = std::clamp(t, 0.0, 1.0);
    double rf, gf, bf;
    if (t < 0.5) {
        const double k = t / 0.5; // blue -> green
        rf = 0.0;
        gf = k;
        bf = 1.0 - k;
    } else {
        const double k = (t - 0.5) / 0.5; // green -> red
        rf = k;
        gf = 1.0 - k;
        bf = 0.0;
    }
    r = uint8_t(rf * 255.0);
    g = uint8_t(gf * 255.0);
    b = uint8_t(bf * 255.0);
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_args(argc, argv, options)) {
        print_usage();
        return 2;
    }

    int refW = 0, refH = 0, refC = 0;
    stbi_uc* reference = stbi_load(options.referencePath.c_str(), &refW, &refH, &refC, 4);
    if (reference == nullptr) {
        std::fprintf(stderr, "Could not read reference: %s\n", options.referencePath.c_str());
        return 2;
    }

    int actW = 0, actH = 0, actC = 0;
    stbi_uc* actual = stbi_load(options.actualPath.c_str(), &actW, &actH, &actC, 4);
    if (actual == nullptr) {
        std::fprintf(stderr, "Could not read actual: %s\n", options.actualPath.c_str());
        stbi_image_free(reference);
        return 2;
    }

    if (refW != actW || refH != actH) {
        std::fprintf(stderr, "Size mismatch: reference is %dx%d, actual is %dx%d\n", refW, refH, actW, actH);
        stbi_image_free(reference);
        stbi_image_free(actual);
        return 2;
    }

    const size_t pixels = size_t(refW) * size_t(refH);
    std::vector<uint8_t> diffImage;
    if (!options.diffPath.empty()) {
        diffImage.resize(pixels * 4);
    }

    uint64_t channelSum = 0;
    uint64_t differing = 0;
    int worstOverall = 0;

    for (size_t i = 0; i < pixels; i++) {
        int worst = 0;
        for (int c = 0; c < 3; c++) {
            const int delta = std::abs(int(actual[i * 4 + c]) - int(reference[i * 4 + c]));
            channelSum += uint64_t(delta);
            worst = std::max(worst, delta);
        }
        worstOverall = std::max(worstOverall, worst);
        const bool differs = worst > options.tolerance;
        if (differs) {
            differing++;
        }

        if (!diffImage.empty()) {
            if (differs) {
                // Scale against 64 rather than 255: most real regressions are
                // subtle, and a full-range ramp would leave them all dark blue.
                uint8_t r, g, b;
                heat_color(double(worst) / 64.0, r, g, b);
                diffImage[i * 4 + 0] = r;
                diffImage[i * 4 + 1] = g;
                diffImage[i * 4 + 2] = b;
            } else {
                // Dimmed grayscale of the reference, so differences are located
                // against the scene instead of floating in a void.
                const int luma = (int(reference[i * 4 + 0]) * 299 + int(reference[i * 4 + 1]) * 587 +
                                  int(reference[i * 4 + 2]) * 114) /
                                 1000;
                const uint8_t dim = uint8_t(luma / 4);
                diffImage[i * 4 + 0] = dim;
                diffImage[i * 4 + 1] = dim;
                diffImage[i * 4 + 2] = dim;
            }
            diffImage[i * 4 + 3] = 255;
        }
    }

    const double meanChannel = double(channelSum) / double(pixels * 3);
    const double differingFraction = double(differing) / double(pixels);
    const bool pass = differingFraction <= options.maxDifferingFraction;

    if (!diffImage.empty()) {
        if (stbi_write_png(options.diffPath.c_str(), refW, refH, 4, diffImage.data(), refW * 4) == 0) {
            std::fprintf(stderr, "Could not write diff image: %s\n", options.diffPath.c_str());
        } else {
            std::printf("Diff image: %s\n", options.diffPath.c_str());
        }
    }

    std::printf("%s: %.4f%% of pixels differ (limit %.4f%%), mean %.3f, max %d\n", pass ? "PASS" : "FAIL",
                differingFraction * 100.0, options.maxDifferingFraction * 100.0, meanChannel, worstOverall);

    stbi_image_free(reference);
    stbi_image_free(actual);
    return pass ? 0 : 1;
}
