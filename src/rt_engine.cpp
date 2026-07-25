#include "rt_engine.h"

#include "descriptor_alloc.h"
#include "gltf_import.h"
#include "image_utils.h"
#include "ray_tracing_pipeline.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <gpu_types.h>
#include <vk_init.h>

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

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                               _windowExtent.height, window_flags);

    SDL_SetWindowGrab(_window, SDL_TRUE);
    SDL_SetRelativeMouseMode(SDL_TRUE);

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

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
    std::array<Vertex, 4> rect_vertices;

    rect_vertices[0].position = {0.5, -0.5, 0};
    rect_vertices[1].position = {0.5, 0.5, 0};
    rect_vertices[2].position = {-0.5, -0.5, 0};
    rect_vertices[3].position = {-0.5, 0.5, 0};

    rect_vertices[0].color = {0, 0, 0, 1};
    rect_vertices[1].color = {0.5, 0.5, 0.5, 1};
    rect_vertices[2].color = {1, 0, 0, 1};
    rect_vertices[3].color = {0, 1, 0, 1};

    rect_vertices[0].uvX = 1;
    rect_vertices[0].uvY = 0;
    rect_vertices[1].uvX = 0;
    rect_vertices[1].uvY = 0;
    rect_vertices[2].uvX = 1;
    rect_vertices[2].uvY = 1;
    rect_vertices[3].uvX = 0;
    rect_vertices[3].uvY = 1;

    std::array<uint32_t, 6> rect_indices;

    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;

    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;

    _defaultRectangle = upload_mesh(rect_indices, rect_vertices);

    // 1x1 fallback textures for materials with a missing or failed image.
    uint32_t white = 0xFFFFFFFF;
    _whiteImage =
        create_image((void*)&white, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = 0xAAAAAAFF;
    _greyImage = create_image((void*)&grey, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = 0x000000FF;
    _blackImage =
        create_image((void*)&black, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

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

    const GPUMeshBuffers defaultRectangle = _defaultRectangle;
    const AllocatedImage whiteImage = _whiteImage;
    const AllocatedImage greyImage = _greyImage;
    const AllocatedImage blackImage = _blackImage;
    const AllocatedImage checkerboardImage = _errorCheckerboardImage;
    const VkSampler nearestSampler = _defaultSamplerNearest;
    const VkSampler linearSampler = _defaultSamplerLinear;
    _mainDeletionQueue.push_function([this, defaultRectangle, whiteImage, greyImage, blackImage, checkerboardImage,
                                      nearestSampler, linearSampler]() {
        vkDestroySampler(_device, nearestSampler, nullptr);
        vkDestroySampler(_device, linearSampler, nullptr);
        destroy_image(whiteImage);
        destroy_image(greyImage);
        destroy_image(blackImage);
        destroy_image(checkerboardImage);
        destroy_buffer(defaultRectangle.vertexBuffer);
        destroy_buffer(defaultRectangle.indexBuffer);
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
    bool cursorLocked = true;
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

            if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_LALT) {
                SDL_SetRelativeMouseMode(cursorLocked ? SDL_FALSE : SDL_TRUE);
                SDL_WarpMouseInWindow(_window, _windowExtent.width / 2, _windowExtent.height / 2);
                cursorLocked = !cursorLocked;
            }

            if (cursorLocked) {
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

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        _stats.frameTime = elapsed.count() / 1000.f;
    }
}
