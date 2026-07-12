#include <vk_engine.h>

#include <string_view>

namespace {
bool configure_engine(VulkanEngine& engine, int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];

        if (argument == "--render-path=raytrace") {
            engine._useRayTracing = true;
        } else if (argument == "--render-path=raster") {
            engine._useRayTracing = false;
        } else if (argument == "--aa=taa") {
            engine._aaMode = VulkanEngine::AAMode::TAA;
        } else if (argument == "--aa=none") {
            engine._aaMode = VulkanEngine::AAMode::None;
        } else if (argument == "--tonemap=on") {
            engine._enableTonemap = true;
        } else if (argument == "--tonemap=off") {
            engine._enableTonemap = false;
        } else {
            fmt::print(stderr, "Unknown startup option: {}\n", argument);
            return false;
        }
    }

    fmt::println(
        "Startup configuration: render_path={} aa={} tonemap={}", engine._useRayTracing ? "raytrace" : "raster",
        engine._aaMode == VulkanEngine::AAMode::TAA ? "taa" : "none", engine._enableTonemap ? "on" : "off");
    return true;
}
} // namespace

int main(int argc, char* argv[])
{
    VulkanEngine engine;

    if (!configure_engine(engine, argc, argv)) {
        return 2;
    }

    engine.init();

    engine.run();

    engine.cleanup();

    return 0;
}
