#include <vk_engine.h>

#include <iostream>
#include <string_view>

namespace {
bool configure_engine(VulkanEngine& engine, int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];

        if (argument == "--render-path=raytrace") {
            engine.useRaytracer = true;
        } else if (argument == "--render-path=raster") {
            engine.useRaytracer = false;
        } else if (argument == "--aa=taa") {
            engine.aaMode = VulkanEngine::AAMode::TAA;
        } else if (argument == "--aa=adaptive") {
            engine.aaMode = VulkanEngine::AAMode::AdaptiveMSAA;
        } else if (argument == "--tonemap=on") {
            engine._enableTonemap = true;
        } else if (argument == "--tonemap=off") {
            engine._enableTonemap = false;
        } else {
            std::cerr << "Unknown startup option: " << argument << '\n';
            return false;
        }
    }

    std::cout << "Startup configuration: render_path=" << (engine.useRaytracer ? "raytrace" : "raster")
              << " aa=" << (engine.aaMode == VulkanEngine::AAMode::TAA ? "taa" : "adaptive")
              << " tonemap=" << (engine._enableTonemap ? "on" : "off") << '\n';
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
