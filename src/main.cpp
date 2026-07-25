#include <vk_engine.h>

#include <string_view>

namespace {
bool configure_engine(VulkanEngine& engine, int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];

        if (argument == "--aa=taa") {
            engine._aaMode = VulkanEngine::AAMode::TAA;
        } else if (argument == "--aa=adaptive") {
            engine._aaMode = VulkanEngine::AAMode::AdaptiveMSAA;
        } else if (argument == "--tonemap=on") {
            engine._enableTonemap = true;
        } else if (argument == "--tonemap=off") {
            engine._enableTonemap = false;
        } else {
            fmt::print(stderr, "Unknown startup option: {}\n", argument);
            return false;
        }
    }

    fmt::println("Startup configuration: aa={} tonemap={}",
                 engine._aaMode == VulkanEngine::AAMode::TAA ? "taa" : "adaptive",
                 engine._enableTonemap ? "on" : "off");
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
