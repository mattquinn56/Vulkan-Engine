#include "core/rt_engine.h"

#include "gpu/descriptor_alloc.h"
#include "scene/gltf_import.h"
#include "gpu/image_utils.h"
#include "passes/ray_tracing_pipeline.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include "core/gpu_types.h"
#include "gpu/vk_init.h"

#include "VkBootstrap.h"

#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

RtEngine* loadedEngine = nullptr;

RtEngine& RtEngine::get() {
    return *loadedEngine;
}

void RtEngine::init() {
    // Backs the RtEngine::get() singleton, only one instance is supported.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    SDL_Init(SDL_INIT_VIDEO);

    // Sized against the desktop unless --resolution pinned it, so the window is
    // proportionate on any display. Capture runs must pin it: the extent decides
    // the rendered image, and a reference is only comparable at a fixed size.
    if (!_resolutionPinned) {
        SDL_DisplayMode desktop{};
        if (SDL_GetDesktopDisplayMode(0, &desktop) == 0) {
            _windowExtent.width = uint32_t(desktop.w * 0.85f);
            _windowExtent.height = uint32_t(desktop.h * 0.85f);
        }
    }

    // A capture run still needs a surface to present through, but the window
    // stays hidden and does not take the mouse.
    const bool capturing = !_screenshotPath.empty();
    _cursorLocked = !capturing;
    Uint32 window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
    if (capturing) {
        window_flags |= SDL_WINDOW_HIDDEN;
    }

    _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                               _windowExtent.height, (SDL_WindowFlags)window_flags);

    if (!capturing) {
        SDL_SetWindowGrab(_window, SDL_TRUE);
        SDL_SetRelativeMouseMode(SDL_TRUE);
    }

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    create_gbuffer_images();

    init_descriptors();

    init_pipelines();

    create_postprocess_resources();

    init_default_data();

    init_raytracing();

    init_renderables();

    init_lights();

    init_imgui();

    _isInitialized = true;

    _mainCamera.velocity = glm::vec3(0.f);
    _mainCamera.position = glm::vec3(.406f, 2.346f, 5.630f);

    _mainCamera.pitch = -.349f;
    _mainCamera.yaw = .005f;
}

void RtEngine::init_default_data() {
    // 1x1 fallback textures for materials with a missing or failed image.
    uint32_t white = 0xFFFFFFFF;
    _whiteImage =
        create_image((void*)&white, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    // The dark square of the error checkerboard below.
    uint32_t black = 0x000000FF;

    // Magenta checkerboard, the conventional "texture is wrong" marker.
    uint32_t magenta = 0xFF00FFFF;
    std::array<uint32_t, 16 * 16> pixels;
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    _errorCheckerboardImage =
        create_image(pixels.data(), VkExtent3D{16, 16, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo sampl = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

    sampl.magFilter = VK_FILTER_NEAREST;
    sampl.minFilter = VK_FILTER_NEAREST;

    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerNearest);

    sampl.magFilter = VK_FILTER_LINEAR;
    sampl.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerLinear);

    const AllocatedImage whiteImage = _whiteImage;
    const AllocatedImage checkerboardImage = _errorCheckerboardImage;
    const VkSampler nearestSampler = _defaultSamplerNearest;
    const VkSampler linearSampler = _defaultSamplerLinear;
    _mainDeletionQueue.push_function([this, whiteImage, checkerboardImage, nearestSampler, linearSampler]() {
        vkDestroySampler(_device, nearestSampler, nullptr);
        vkDestroySampler(_device, linearSampler, nullptr);
        destroy_image(whiteImage);
        destroy_image(checkerboardImage);
    });
}

void RtEngine::cleanup() {
    if (_isInitialized) {

        // Nothing below may run while the GPU still references these resources.
        vkDeviceWaitIdle(_device);

        _loadedScenes.clear();

        for (auto& frame : _frames) {
            frame._deletionQueue.flush();
        }

        destroy_taa_history_images();
        destroy_monte_carlo_images();
        destroy_gbuffer_images();
        destroy_render_targets();
        destroy_swapchain();

        _mainDeletionQueue.flush();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);

        vmaDestroyAllocator(_allocator);

        vkDestroyDevice(_device, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debugMessenger);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }
}

void RtEngine::init_pipelines() {
    create_monte_carlo_pipeline_resources();

    create_taa_pipeline_resources();
}

void RtEngine::run() {
    SDL_Event e;
    bool bQuit = false;
    _renderingFrozen = false;

    while (!bQuit) {
        auto start = std::chrono::system_clock::now();

        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {

                if (e.window.event == SDL_WINDOWEVENT_RESIZED || e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                    _resizeRequested = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    _renderingFrozen = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    _renderingFrozen = false;
                }
            }

            if (_showUi && e.type == SDL_KEYDOWN && e.key.repeat == 0 && e.key.keysym.scancode == SDL_SCANCODE_TAB) {
                set_settings_open(!_settingsOpen);
            }

            if (_cursorLocked) {
                _mainCamera.process_sdl_event(e);
            }

            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        if (_renderingFrozen)
            continue;

        if (_resizeRequested) {
            if (!resize_swapchain()) {
                continue;
            }
            if (_accelerationStructuresCreated) {
                _rayTracer->update_output_descriptor();
            }
        }

        draw_ui();

        update_scene();

        if (!_accelerationStructuresCreated) {
            _rayTracer->create_bottom_level_acceleration_structures();
            _rayTracer->create_top_level_acceleration_structure();
            _rayTracer->create_descriptor_set();
            _rayTracer->create_material_descriptor_set();
            _rayTracer->create_pipeline();
            _rayTracer->create_shader_binding_table();
            _accelerationStructuresCreated = true;
        }

        draw();

        // A capture run is non-interactive: exit as soon as the shot is written.
        if (_screenshotDone) {
            bQuit = true;
        }

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        _stats.frameTime = elapsed.count() / 1000.f;
    }
}

void RtEngine::set_settings_open(bool open) {
    _settingsOpen = open;
    _cursorLocked = !open;

    SDL_SetWindowGrab(_window, _cursorLocked ? SDL_TRUE : SDL_FALSE);
    SDL_SetRelativeMouseMode(_cursorLocked ? SDL_TRUE : SDL_FALSE);
    if (_cursorLocked) {
        SDL_WarpMouseInWindow(_window, _windowExtent.width / 2, _windowExtent.height / 2);
    }
}
