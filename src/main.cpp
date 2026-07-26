#include "core/rt_engine.h"

#include <charconv>
#include <string_view>

namespace {

bool parse_int(std::string_view text, int& out) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool parse_float(std::string_view text, float& out) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

void print_usage() {
    fmt::print(stderr, "Options:\n"
                       "  --tonemap=on|off      ACES + sRGB tonemapping\n"
                       "  --screenshot=<path>   write a PNG of the presented frame, then exit\n"
                       "  --frames=<n>          frame to capture on (default 30)\n"
                       "  --no-ui               render without the ImGui overlay\n"
                       "  --debug-view=<n>      0 shaded, 1 normal, 2 hit distance, 3 motion, 4 instance,\n"
                       "                        5 history rejection\n"
                       "  --orbit=<deg>         yaw the camera this many degrees every frame\n"
                       "  --orbit-frames=<n>    stop orbiting after n frames\n");
}

bool configure_engine(RtEngine& engine, int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];

        if (argument == "--tonemap=on") {
            engine._enableTonemap = true;
        } else if (argument == "--tonemap=off") {
            engine._enableTonemap = false;
        } else if (argument.starts_with("--screenshot=")) {
            engine._screenshotPath = std::string(argument.substr(13));
            if (engine._screenshotPath.empty()) {
                fmt::print(stderr, "--screenshot requires a path\n");
                return false;
            }
        } else if (argument == "--no-ui") {
            engine._showUi = false;
        } else if (argument.starts_with("--debug-view=")) {
            if (!parse_int(argument.substr(13), engine._debugView) || engine._debugView < 0 ||
                engine._debugView >= RtEngine::kDebugViewCount) {
                fmt::print(stderr, "--debug-view requires an integer in [0, {})\n", RtEngine::kDebugViewCount);
                return false;
            }
        } else if (argument.starts_with("--orbit-frames=")) {
            if (!parse_int(argument.substr(15), engine._orbitFrames) || engine._orbitFrames < 0) {
                fmt::print(stderr, "--orbit-frames requires a non-negative integer\n");
                return false;
            }
        } else if (argument.starts_with("--orbit=")) {
            if (!parse_float(argument.substr(8), engine._orbitDegreesPerFrame)) {
                fmt::print(stderr, "--orbit requires a number of degrees per frame\n");
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

    fmt::println("Startup configuration: tonemap={}", engine._enableTonemap ? "on" : "off");
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
