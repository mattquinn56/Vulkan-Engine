#include "core/rt_engine.h"

#include <charconv>
#include <string_view>

namespace {

bool parse_int(std::string_view text, int& out) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

void print_usage() {
    fmt::print(stderr, "Options:\n"
                       "  --aa=taa|adaptive     antialiasing mode\n"
                       "  --tonemap=on|off      ACES + sRGB tonemapping\n"
                       "  --screenshot=<path>   write a PNG of the presented frame, then exit\n"
                       "  --frames=<n>          frame to capture on (default 30)\n");
}

bool configure_engine(RtEngine& engine, int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];

        if (argument == "--aa=taa") {
            engine._aaMode = RtEngine::AAMode::TAA;
        } else if (argument == "--aa=adaptive") {
            engine._aaMode = RtEngine::AAMode::AdaptiveMSAA;
        } else if (argument == "--tonemap=on") {
            engine._enableTonemap = true;
        } else if (argument == "--tonemap=off") {
            engine._enableTonemap = false;
        } else if (argument.starts_with("--screenshot=")) {
            engine._screenshotPath = std::string(argument.substr(13));
            if (engine._screenshotPath.empty()) {
                fmt::print(stderr, "--screenshot requires a path\n");
                return false;
            }
        } else if (argument.starts_with("--frames=")) {
            if (!parse_int(argument.substr(9), engine._screenshotFrame) || engine._screenshotFrame < 0) {
                fmt::print(stderr, "--frames requires a non-negative integer\n");
                return false;
            }
        } else {
            fmt::print(stderr, "Unknown startup option: {}\n", argument);
            print_usage();
            return false;
        }
    }

    fmt::println("Startup configuration: aa={} tonemap={}",
                 engine._aaMode == RtEngine::AAMode::TAA ? "taa" : "adaptive", engine._enableTonemap ? "on" : "off");
    if (!engine._screenshotPath.empty()) {
        fmt::println("Screenshot: {} on frame {}", engine._screenshotPath, engine._screenshotFrame);
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    RtEngine engine;

    if (!configure_engine(engine, argc, argv)) {
        return 2;
    }

    engine.init();

    engine.run();

    engine.cleanup();

    return 0;
}
