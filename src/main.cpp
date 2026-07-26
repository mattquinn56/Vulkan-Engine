#include "core/rt_engine.h"

#include <CLI/CLI.hpp>

#include <map>
#include <string>
#include <utility>

namespace {

void describe_options(CLI::App& app, RtEngine& engine, bool& noUi) {
    static const std::map<std::string, bool> onOff{{"on", true}, {"off", false}};

    app.add_option("--tonemap", engine._enableTonemap, "ACES + sRGB tonemapping")
        ->transform(CLI::CheckedTransformer(onOff, CLI::ignore_case));
    app.add_option("--screenshot", engine._screenshotPath, "write a PNG of the presented frame, then exit")
        ->check(CLI::Validator([](const std::string& path) { return path.empty() ? "path is empty" : ""; }, "PATH"));
    app.add_option("--frames", engine._screenshotFrame, "frame to capture on")->check(CLI::NonNegativeNumber);
    app.add_flag("--no-ui", noUi, "render without the ImGui overlay");
    app.add_option("--debug-view", engine._debugView,
                   "0 shaded, 1 normal, 2 hit distance, 3 motion, 4 instance, 5 history reuse")
        ->check(CLI::Range(0, RtEngine::kDebugViewCount - 1));
    app.add_option("--orbit", engine._orbitDegreesPerFrame, "yaw the camera this many degrees every frame");
    app.add_option("--orbit-frames", engine._orbitFrames, "stop orbiting after n frames")
        ->check(CLI::NonNegativeNumber);
    app.add_option_function<std::pair<uint32_t, uint32_t>>(
           "--resolution",
           [&engine](const std::pair<uint32_t, uint32_t>& size) {
               engine._windowExtent = {size.first, size.second};
               engine._resolutionPinned = true;
           },
           "window size as WIDTH HEIGHT, overriding the desktop-relative default")
        ->type_name("WIDTH HEIGHT");
}

} // namespace

int main(int argc, char* argv[]) {
    RtEngine engine;

    CLI::App app{"Vulkan hardware ray tracing engine"};
    bool noUi = false;
    describe_options(app, engine, noUi);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        // --help is a successful exit; anything else is a bad invocation, which
        // callers distinguish from a render that started and then failed.
        return app.exit(e) == 0 ? 0 : 2;
    }
    engine._showUi = !noUi;

    fmt::println("Startup configuration: tonemap={}", engine._enableTonemap ? "on" : "off");
    if (!engine._screenshotPath.empty()) {
        fmt::println("Screenshot: {} on frame {}", engine._screenshotPath, engine._screenshotFrame);
    }

    engine.init();

    engine.run();

    engine.cleanup();

    return 0;
}
