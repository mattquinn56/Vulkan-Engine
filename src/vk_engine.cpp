
#include "vk_engine.h"

#include "vk_images.h"
#include "vk_loader.h"
#include "vk_descriptors.h"
#include "vk_raytracer.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

#include "VkBootstrap.h"

#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <stb_image.h>
#include <fastgltf/parser.hpp>

constexpr bool bUseValidationLayers = true;

using namespace std;

#define CHAPTER_STAGE 1

VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::get()
{
    return *loadedEngine;
}

void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
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

    // everything went fine
    _isInitialized = true;

    _mainCamera.velocity = glm::vec3(0.f);
    _mainCamera.position = glm::vec3(.406f, 2.346f, 5.630f);

    _mainCamera.pitch = -.349f;
    _mainCamera.yaw = .005f;
}

void VulkanEngine::init_default_data()
{
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

    //3 default textures, white, grey, black. 1 pixel each
    uint32_t white = 0xFFFFFFFF;
    _whiteImage =
        create_image((void*)&white, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = 0xAAAAAAFF;
    _greyImage = create_image((void*)&grey, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = 0x000000FF;
    _blackImage =
        create_image((void*)&black, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    //checkerboard image
    uint32_t magenta = 0xFF00FFFF;
    std::array<uint32_t, 16 * 16> pixels; //for 16x16 checkerboard texture
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

void VulkanEngine::cleanup()
{
    if (_isInitialized) {

        // make sure the gpu has stopped doing its things
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

void VulkanEngine::init_background_pipelines()
{
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;

    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_gradientPipelineLayout));

    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module("../../shaders/gradient_color.comp.spv", _device, &gradientShader)) {
        fmt::print(stderr, "Failed to build gradient compute shader\n");
    }

    VkShaderModule skyShader;
    if (!vkutil::load_shader_module("../../shaders/sky.comp.spv", _device, &skyShader)) {
        fmt::print(stderr, "Failed to build sky compute shader\n");
    }

    VkPipelineShaderStageCreateInfo stageinfo{};
    stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageinfo.pNext = nullptr;
    stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageinfo.module = gradientShader;
    stageinfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = _gradientPipelineLayout;
    computePipelineCreateInfo.stage = stageinfo;

    ComputeEffect gradient{};
    gradient.layout = _gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};
    gradient.pipeline = VK_NULL_HANDLE;

    //default colors
    gradient.data.data1 = glm::vec4(1, 0, 0, 1);
    gradient.data.data2 = glm::vec4(0, 0, 1, 1);

    VK_CHECK(
        vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

    //change the shader module only to create the sky shader
    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky;
    sky.layout = _gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};
    //default sky parameters
    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

    //add the 2 background effects into the array
    _backgroundEffects.push_back(sky);
    _backgroundEffects.push_back(gradient);

    //destroy structures properly
    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);
    const VkPipelineLayout backgroundLayout = _gradientPipelineLayout;
    const VkPipeline skyPipeline = sky.pipeline;
    const VkPipeline gradientPipeline = gradient.pipeline;
    _mainDeletionQueue.push_function([this, backgroundLayout, skyPipeline, gradientPipeline]() {
        vkDestroyPipelineLayout(_device, backgroundLayout, nullptr);
        vkDestroyPipeline(_device, skyPipeline, nullptr);
        vkDestroyPipeline(_device, gradientPipeline, nullptr);
    });
}

void VulkanEngine::draw_main(VkCommandBuffer cmd)
{
    ComputeEffect& effect = _backgroundEffects[_currentBackgroundEffect];

    // bind the background compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    // bind the descriptor set containing the draw image for the compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipelineLayout, 0, 1, &_drawImageDescriptors,
                            0, nullptr);

    vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants),
                       &effect.data);
    // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
    vkCmdDispatch(cmd, (_windowExtent.width + 15) / 16, (_windowExtent.height + 15) / 16, 1);

    //draw the triangle

    VkRenderingAttachmentInfo colorAttachment =
        vkinit::attachment_info(_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_GENERAL);
    VkRenderingAttachmentInfo depthAttachment =
        vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_windowExtent, &colorAttachment, &depthAttachment);

    vkCmdBeginRendering(cmd, &renderInfo);
    auto start = std::chrono::system_clock::now();
    draw_geometry(cmd);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    _stats.meshDrawTime = elapsed.count() / 1000.f;

    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment =
        vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_GENERAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_windowExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void seed_taa_history(VulkanEngine* e, VkCommandBuffer cmd)
{
    for (int i = 0; i < 2; ++i) {
        vkutil::transition_image(cmd, e->_taaHistory[i].image, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkutil::transition_image(cmd, e->_drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        vkutil::copy_image_to_image(cmd, e->_drawImage.image, e->_taaHistory[i].image, e->_windowExtent,
                                    e->_windowExtent);
        vkutil::transition_image(cmd, e->_taaHistory[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_GENERAL);
        vkutil::transition_image(cmd, e->_drawImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 VK_IMAGE_LAYOUT_GENERAL);
    }
    e->_taaIndex = 0;
}

void VulkanEngine::draw()
{
    // Wait for the previous frame using this FrameData to finish
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, VK_TRUE, UINT64_MAX));

    // Per-frame cleanup
    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);

    // Acquire next swapchain image, signaling this frame's _swapchainSemaphore when ready
    uint32_t imageIndex = 0;
    VkResult acquire = vkAcquireNextImageKHR(_device, _swapchain, UINT64_MAX, get_current_frame()._swapchainSemaphore,
                                             VK_NULL_HANDLE, &imageIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
        _resizeRequested = true;
        return;
    }
    if (acquire == VK_SUBOPTIMAL_KHR) {
        _resizeRequested = true;
    } // continue; we'll still draw this frame

    // We are going to submit new work for this frame
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    // Reset and begin command buffer recording
    VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));
    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Make draw/depth images writable for compute/graphics
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    // Ray tracing path writes into _drawImage (GENERAL) or raster path draws and then composites
    if (_useRayTracing) {
        _rayTracer->raytrace(cmd);
    } else {
        draw_main(cmd);
    }

    // If UI changed something that affects appearance, clear histories now
    if (_resetAccumNextFrame) {
        // Reset MC accumulators (count=0, copy current _drawImage into accumColor)
        reset_monte_carlo_history(cmd);

        // Reset/seed TAA history from current frame so blending starts clean
        if (_aaMode == AAMode::TAA) {
            seed_taa_history(this, cmd);
            _taaInitialized = true;
        }
        _resetAccumNextFrame = false;
    }

    bool doProgressive = _progressiveMonteCarlo && (_aaMode == AAMode::TAA ? !_cameraMoving : true);

    // Bind descriptors for MC accumulation
    {
        DescriptorWriter w;
        w.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.write_image(1, _mcAccumColor.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.write_image(2, _mcAccumCount.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.write_image(3, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.update_set(_device, _mcSet);
    }

    if (doProgressive) {
        // Optional: delay reset for a couple frames after movement ends
        static int resetCooldown = 0;
        if (_cameraMoving)
            resetCooldown = _monteCarloResetFrames;
        if (resetCooldown > 0) {
            reset_monte_carlo_history(cmd);
            resetCooldown--;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _mcPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _mcPipeLayout, 0, 1, &_mcSet, 0, nullptr);

        struct
        {
            float perFrameSpp;
            float movingFlag;
        } pc{float(_monteCarloSamplesPerFrame), _cameraMoving ? 1.f : 0.f};
        vkCmdPushConstants(cmd, _mcPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t gx = (_windowExtent.width + 7) / 8;
        uint32_t gy = (_windowExtent.height + 7) / 8;
        vkCmdDispatch(cmd, gx, gy, 1);

        // Note: mc resolve writes back into _drawImage (binding 3), so TAA will consume it next
    } else {
        reset_monte_carlo_history(cmd);
    }

    if (_aaMode == AAMode::TAA) {
        // Make RT writes visible to compute reads
        VkImageMemoryBarrier2 imgBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        imgBarrier.srcStageMask =
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR; // or COLOR_ATTACHMENT_OUTPUT if raster
        imgBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        imgBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        imgBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imgBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imgBarrier.image = _drawImage.image;
        imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &imgBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);

        if (!_taaInitialized) {
            seed_taa_history(this, cmd);
            _taaInitialized = true;
        }
        int prev = _taaIndex;
        int next = 1 - _taaIndex;

        // seed history when we first switch to TAA or on strong movement with zero alpha
        if (_cameraMoving && _taaMovingAlpha == 0.0f) {
            seed_taa_history(this, cmd);
            prev = _taaIndex;
            next = 1 - _taaIndex;
        }

        {
            DescriptorWriter w;
            w.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.write_image(1, _taaHistory[prev].imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.write_image(2, _taaHistory[next].imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.update_set(_device, _taaSet[next]);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _taaPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _taaPipelineLayout, 0, 1, &_taaSet[next], 0,
                                nullptr);

        struct
        {
            float alpha;
            float clampK;
        } pc{_cameraMoving ? _taaMovingAlpha : _taaAlpha, _taaClamp};
        vkCmdPushConstants(cmd, _taaPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t gx = (_windowExtent.width + 7) / 8;
        uint32_t gy = (_windowExtent.height + 7) / 8;
        vkCmdDispatch(cmd, gx, gy, 1);

        VkImageMemoryBarrier2 histBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        histBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        histBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        histBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        histBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        histBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        histBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        histBarrier.image = _taaHistory[next].image;
        histBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        dep = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &histBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);

        vkutil::transition_image(cmd, _taaHistory[next].image, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkutil::copy_image_to_image(cmd, _taaHistory[next].image, _drawImage.image, _windowExtent, _windowExtent);
        vkutil::transition_image(cmd, _taaHistory[next].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 VK_IMAGE_LAYOUT_GENERAL);
        vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

        _taaIndex = next;
    }

    // --- POST: ACES + sRGB (optional) ---
    if (_enableTonemap) {
        // Make sure _drawImage writes are visible to compute
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.image = _drawImage.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // Transition _ldrImage to GENERAL on first use (or from last frame's TRANSFER_SRC)
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
            b.srcAccessMask = 0;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            b.oldLayout = _ldrNeedsInit ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.image = _ldrImage.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
            _ldrNeedsInit = false;
        }

        // Bind descriptors and dispatch post compute
        {
            DescriptorWriter w;
            w.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.write_image(1, _ldrImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.update_set(_device, _postSet);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _postPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _postPipeLayout, 0, 1, &_postSet, 0, nullptr);

        vkCmdPushConstants(cmd, _postPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &_exposure);

        uint32_t gx = (_windowExtent.width + 7) / 8;
        uint32_t gy = (_windowExtent.height + 7) / 8;
        vkCmdDispatch(cmd, gx, gy, 1);

        // Make post-process writes visible before copying from the LDR image.
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.image = _ldrImage.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // Transition the swapchain image before copying the LDR result into it.
        vkutil::transition_image(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkExtent2D extent{_windowExtent.width, _windowExtent.height};
        vkutil::copy_image_to_image(cmd, _ldrImage.image, _swapchainImages[imageIndex], extent, extent);
    } else {
        // No tonemap: copy HDR directly (debug)
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.image = _drawImage.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        vkutil::transition_image(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkExtent2D extent{_windowExtent.width, _windowExtent.height};
        vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[imageIndex], extent, extent);
    }

    // Draw ImGui on the swapchain image
    vkutil::transition_image(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    draw_imgui(cmd, _swapchainImageViews[imageIndex]);

    // Prepare for present
    vkutil::transition_image(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit: wait on image-available for this frame, signal render-finished for this frame
    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);
    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
                                                                   get_current_frame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame()._renderSemaphore);
    VkSubmitInfo2 submitInfo = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submitInfo, get_current_frame()._renderFence));

    // Present: wait on this frame's render-finished semaphore
    VkPresentInfoKHR presentInfo = vkinit::present_info();
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;

    VkResult present = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (present == VK_ERROR_OUT_OF_DATE_KHR || present == VK_SUBOPTIMAL_KHR) {
        _resizeRequested = true;
    }

    _frameNumber++;
}
bool is_visible(const RenderObject& obj, const glm::mat4& viewproj)
{
    std::array<glm::vec3, 8> corners{
        glm::vec3{1, 1, 1},  glm::vec3{1, 1, -1},  glm::vec3{1, -1, 1},  glm::vec3{1, -1, -1},
        glm::vec3{-1, 1, 1}, glm::vec3{-1, 1, -1}, glm::vec3{-1, -1, 1}, glm::vec3{-1, -1, -1},
    };

    glm::mat4 matrix = viewproj * obj.transform;

    glm::vec3 min = {1.5, 1.5, 1.5};
    glm::vec3 max = {-1.5, -1.5, -1.5};

    for (int c = 0; c < 8; c++) {
        // project each corner into clip space
        glm::vec4 v = matrix * glm::vec4(obj.bounds.origin + (corners[c] * obj.bounds.extents), 1.f);

        // perspective correction
        v.x = v.x / v.w;
        v.y = v.y / v.w;
        v.z = v.z / v.w;

        min = glm::min(glm::vec3{v.x, v.y, v.z}, min);
        max = glm::max(glm::vec3{v.x, v.y, v.z}, max);
    }

    // check the clip space box is within the view
    if (min.z > 1.f || max.z < 0.f || min.x > 1.f || max.x < -1.f || min.y > 1.f || max.y < -1.f) {
        return false;
    } else {
        return true;
    }
}

void VulkanEngine::update_global_descriptor()
{

    //allocate a new uniform buffer for the scene data
    AllocatedBuffer gpuSceneDataBuffer =
        create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    //add it to the deletion queue of this frame so it gets deleted once its been used
    get_current_frame()._deletionQueue.push_function([=, this]() { destroy_buffer(gpuSceneDataBuffer); });

    //write the buffer
    GPUSceneData* sceneUniformData = (GPUSceneData*)gpuSceneDataBuffer.allocation->GetMappedData();
    *sceneUniformData = _sceneData;

    //create a descriptor set that binds that buffer and update it
    _globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);

    {
        DescriptorWriter writer;
        writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(_device, _globalDescriptor);
    }

    // do the same for objDesc set
    _objectDescriptionBuffer = create_buffer_data(sizeof(ObjDesc) * _drawContext.objectDescriptions.size(),
                                                  _drawContext.objectDescriptions.data(),
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Delete it only after this frame's fence signals and the descriptor is no longer in use.
    const AllocatedBuffer objectDescriptionBuffer = _objectDescriptionBuffer;
    get_current_frame()._deletionQueue.push_function(
        [this, objectDescriptionBuffer]() { destroy_buffer(objectDescriptionBuffer); });

    _objDescSet = get_current_frame()._frameDescriptors.allocate(_device, _objDescLayout);

    {
        DescriptorWriter writer;
        writer.write_buffer(0, _objectDescriptionBuffer.buffer, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(_device, _objDescSet);
    }
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    std::vector<uint32_t> opaque_draws;
    opaque_draws.reserve(_drawContext.opaqueSurfaces.size());

    for (int i = 0; i < _drawContext.opaqueSurfaces.size(); i++) {
        if (is_visible(_drawContext.opaqueSurfaces[i], _sceneData.viewproj)) {
            opaque_draws.push_back(i);
        }
    }

    // sort the opaque surfaces by material and mesh
    std::sort(opaque_draws.begin(), opaque_draws.end(), [&](const auto& iA, const auto& iB) {
        const RenderObject& A = _drawContext.opaqueSurfaces[iA];
        const RenderObject& B = _drawContext.opaqueSurfaces[iB];
        if (A.material == B.material) {
            return A.indexBuffer < B.indexBuffer;
        } else {
            return A.material < B.material;
        }
    });

    update_global_descriptor();

    MaterialPipeline* lastPipeline = nullptr;
    MaterialInstance* lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    auto draw = [&](const RenderObject& r) {
        if (r.material != lastMaterial) {
            lastMaterial = r.material;
            if (r.material->pipeline != lastPipeline) {

                lastPipeline = r.material->pipeline;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 0, 1,
                                        &_globalDescriptor, 0, nullptr);

                VkViewport viewport = {};
                viewport.x = 0;
                viewport.y = 0;
                viewport.width = (float)_windowExtent.width;
                viewport.height = (float)_windowExtent.height;
                viewport.minDepth = 0.f;
                viewport.maxDepth = 1.f;

                vkCmdSetViewport(cmd, 0, 1, &viewport);

                VkRect2D scissor = {};
                scissor.offset.x = 0;
                scissor.offset.y = 0;
                scissor.extent.width = _windowExtent.width;
                scissor.extent.height = _windowExtent.height;

                vkCmdSetScissor(cmd, 0, 1, &scissor);
            }

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 1, 1,
                                    &r.material->materialSet, 0, nullptr);
        }
        if (r.indexBuffer != lastIndexBuffer) {
            lastIndexBuffer = r.indexBuffer;
            vkCmdBindIndexBuffer(cmd, r.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }
        // calculate final mesh matrix
        GPUDrawPushConstants push_constants;
        push_constants.worldMatrix = r.transform;
        push_constants.vertexBuffer = r.vertexBufferAddress;
        push_constants.lightBuffer = get_buffer_device_address(_device, _lightBuffer.buffer);
        ;
        push_constants.numLights = _lightCount;

        vkCmdPushConstants(cmd, r.material->pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(GPUDrawPushConstants), &push_constants);

        _stats.drawCallCount++;
        _stats.triangleCount += r.indexCount / 3;
        vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, 0);
    };

    _stats.drawCallCount = 0;
    _stats.triangleCount = 0;

    for (auto& r : opaque_draws) {
        draw(_drawContext.opaqueSurfaces[r]);
    }
}

void VulkanEngine::run()
{
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

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame(_window);

        ImGui::NewFrame();

        bool reset_accum = false;

        ImGui::Begin("Main Control");

        reset_accum |= ImGui::Checkbox("Ray Tracer mode", &_useRayTracing);
        reset_accum |= ImGui::Checkbox("Debug setting", &_debugEnabled);
        reset_accum |= ImGui::Checkbox("Use Microfacet BRDF (GGX/Smith/Schlick)", &_useMicrofacetBrdf);
        ImGui::Text("frameTime %f ms", _stats.frameTime);
        glm::vec3 viewDir = _mainCamera.get_view_direction();
        ImGui::Text("position: %f %f %f", _mainCamera.position.x, _mainCamera.position.y, _mainCamera.position.z);
        ImGui::Text("view direction: %f %f %f", viewDir.x, viewDir.y, viewDir.z);
        ImGui::Text("pitch and yaw: %f %f", _mainCamera.pitch, _mainCamera.yaw);
        if (ImGui::CollapsingHeader("Color & Tone", ImGuiTreeNodeFlags_DefaultOpen)) {
            reset_accum |= ImGui::Checkbox("ACES + sRGB tonemap", &_enableTonemap);
            reset_accum |= ImGui::SliderFloat("Exposure", &_exposure, 0.1f, 4.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::TextUnformatted(_enableTonemap ? "Output: tonemapped sRGB" : "Output: raw linear HDR (debug)");
        }
        ImGui::End();

        ImGui::Begin("Antialiasing");
        int aa = (_aaMode == AAMode::TAA) ? 1 : 0;

        bool rb0 = ImGui::RadioButton("None", aa == 0);
        if (rb0) {
            aa = 0;
        }
        ImGui::SameLine();
        bool rb1 = ImGui::RadioButton("TAA", aa == 1);
        if (rb1) {
            aa = 1;
        }

        AAMode newMode = (aa == 1) ? AAMode::TAA : AAMode::None;
        if (newMode != _aaMode) {
            _aaMode = newMode;
            reset_accum = true;
        }

        ImGui::BeginDisabled();
        reset_accum |= ImGui::Checkbox("Progressive Monte Carlo", &_progressiveMonteCarlo);
        ImGui::EndDisabled();
        if (_aaMode == AAMode::TAA) {
            reset_accum |= ImGui::SliderInt("MC per-frame spp", &_monteCarloSamplesPerFrame, 0, 20);
            reset_accum |= ImGui::SliderInt("MC reset frames", &_monteCarloResetFrames, 0, 8);
            reset_accum |= ImGui::SliderFloat("TAA alpha (still)", &_taaAlpha, 0.0f, 0.99f);
            ImGui::Text("Camera moving: %s", _cameraMoving ? "yes" : "no");
            _progressiveMonteCarlo = true;
        } else {
            _progressiveMonteCarlo = false;
        }
        ImGui::End();

        ImGui::Begin("Medium");
        {
            auto* mp = (GPUMediumParams*)_volume.mediumParams.allocation->GetMappedData();

            // unpack helpers for readability
            auto& sigma_a = mp->sigma_a_step;     // xyz used, w = step
            auto& sigma_s = mp->sigma_s_maxT;     // xyz used, w = maxT
            auto& g_e_d = mp->g_emis_density_pad; // x=g, y=emission, z=densityScale

            ImGui::Text("Homogeneous Medium");
            reset_accum |= ImGui::DragFloat3("sigma_a (absorption)", &sigma_a.x, 0.001f, 0.0f, 5.0f);
            reset_accum |= ImGui::DragFloat3("sigma_s (scattering)", &sigma_s.x, 0.001f, 0.0f, 5.0f);
            reset_accum |= ImGui::DragFloat("stepSize", &sigma_a.w, 0.001f, 0.001f, 1.0f);
            reset_accum |= ImGui::DragFloat("maxT", &sigma_s.w, 1.0f, 0.0f, 20000.0f);
            reset_accum |= ImGui::DragFloat("g (anisotropy)", &g_e_d.x, 0.001f, -0.99f, 0.99f);
            reset_accum |= ImGui::DragFloat("emission", &g_e_d.y, 0.001f, 0.0f, 10.0f);
            reset_accum |= ImGui::DragFloat("densityScale", &g_e_d.z, 0.001f, 0.0f, 10.0f);

            {
                auto* mp = (GPUMediumParams*)_volume.mediumParams.allocation->GetMappedData();
                bool fogEnv = mp->g_emis_density_pad.w > 0.5f;
                if (ImGui::Checkbox("Fog affects environment", &fogEnv)) {
                    mp->g_emis_density_pad.w = fogEnv ? 1.0f : 0.0f;
                }
            }
        }
        ImGui::End();

        bool collapsedDataWindow = ImGui::Begin("Background (Raster)");
        if (collapsedDataWindow) {

            ComputeEffect& selected = _backgroundEffects[_currentBackgroundEffect];

            ImGui::Text("Selected background: ", selected.name);

            ImGui::SliderInt("Effect", &_currentBackgroundEffect, 0, static_cast<int>(_backgroundEffects.size()) - 1);

            ImGui::InputFloat4("Top Gradient", (float*)&selected.data.data1);
            ImGui::InputFloat4("Bottom Gradient", (float*)&selected.data.data2);
        }
        ImGui::End();

        ImGui::Begin("Hierarchy");

        // For all pairs in _loadedScenes, draw an expandable tree node
        for (auto& [name, scene] : _loadedScenes) {

            if (ImGui::TreeNode(name.c_str())) {
                render_loaded_gltf(scene);
                ImGui::TreePop();
            }
        }
        ImGui::End();

        ImGui::Render();

        if (reset_accum) {
            request_accum_reset();
        }

        // imgui commands

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

void VulkanEngine::render_loaded_gltf(std::shared_ptr<LoadedGLTF> gltf)
{
    auto topLevelNodes = gltf->topNodes;

    for (auto& node : topLevelNodes) {
        recursively_render_node(gltf, node);
    }
}

void VulkanEngine::recursively_render_node(std::shared_ptr<LoadedGLTF> gltf, std::shared_ptr<Node> node)
{
    if (node->children.size() > 0) {
        if (ImGui::TreeNode(gltf->nodeNames[node].c_str())) {
            for (auto& child : node->children) {
                recursively_render_node(gltf, child);
            }
            ImGui::TreePop();
        }
    } else {
        ImGui::Text(gltf->nodeNames[node].c_str());
    }
}

void VulkanEngine::update_scene()
{
    _mainCamera.update();

    glm::mat4 view = _mainCamera.get_view_matrix();
    glm::mat4 projection =
        glm::perspective(glm::radians(70.f), (float)_windowExtent.width / (float)_windowExtent.height, 10000.f, 0.1f);
    projection[1][1] *= -1;

    // motion detection
    glm::vec3 camPos = _mainCamera.position;
    glm::vec3 viewDir = _mainCamera.get_view_direction();

    float linDelta = _hasPrevCamera ? glm::length(camPos - _prevCamPos) : 0.f;
    float angDelta =
        _hasPrevCamera
            ? glm::degrees(acos(glm::clamp(glm::dot(glm::normalize(viewDir), glm::normalize(_prevViewDir)), -1.f, 1.f)))
            : 0.f;
    _cameraMoving = _hasPrevCamera && (linDelta > _taaVelocityThreshold || angDelta > _taaRotationThreshold);
    _prevCamPos = camPos;
    _prevViewDir = viewDir;
    _hasPrevCamera = true;

    _sceneData.view = view;
    _sceneData.proj = projection;
    _sceneData.viewproj = projection * view;
    const float perFrameSpp = _progressiveMonteCarlo ? float(_monteCarloSamplesPerFrame) : float(_monteCarloSamples);
    _sceneData.data = glm::vec4(_frameNumber, perFrameSpp, _msaaSamples, (_aaMode == AAMode::TAA) ? 1.f : 0.f);

    _drawContext.opaqueSurfaces.clear();
    _drawContext.objectDescriptions.clear();
    _drawContext.TransparentSurfaces.clear();
    _loadedScenes["structure"]->draw(glm::mat4{1.f}, _drawContext);
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    // allocate buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = memoryUsage;
    vmaallocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    // allocate the buffer
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation,
                             &newBuffer.info));

    return newBuffer;
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);

    // treat 3D extents as 3D textures
    if (size.depth > 1) {
        img_info.imageType = VK_IMAGE_TYPE_3D;
        img_info.arrayLayers = 1;
    }

    if (mipmapped) {
        const uint32_t mx = std::max({size.width, size.height, size.depth});
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(mx))) + 1;
    }

    VmaAllocationCreateInfo allocinfo{};
    allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vmaCreateImage(_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr));

    VkImageAspectFlags aspectFlag =
        (format == VK_FORMAT_D32_SFLOAT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);

    // 3D view when needed
    if (size.depth > 1) {
        view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
        view_info.subresourceRange.layerCount = 1;
    }

    view_info.subresourceRange.levelCount = img_info.mipLevels;
    VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));
    return newImage;
}
AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
                                          bool mipmapped)
{
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadbuffer =
        create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(
        size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &copyRegion);

        if (mipmapped) {
            vkutil::generate_mipmaps(cmd, new_image.image,
                                     VkExtent2D{new_image.imageExtent.width, new_image.imageExtent.height});
        } else {
            vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    });
    destroy_buffer(uploadbuffer);
    return new_image;
}

GPUMeshBuffers VulkanEngine::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    newSurface.vertexCount = static_cast<int>(vertices.size());

    newSurface.vertexBuffer = create_buffer(vertexBufferSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                            VMA_MEMORY_USAGE_GPU_ONLY);

    VkBufferDeviceAddressInfo deviceAdressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                               .buffer = newSurface.vertexBuffer.buffer};
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAdressInfo);

    newSurface.indexBuffer =
        create_buffer(indexBufferSize,
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                      VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging =
        create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.allocation->GetMappedData();

    // copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{0};
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{0};
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    destroy_buffer(staging);

    return newSurface;
}

FrameData& VulkanEngine::get_current_frame()
{
    return _frames[_frameNumber % FRAME_OVERLAP];
}

FrameData& VulkanEngine::get_last_frame()
{
    return _frames[(_frameNumber - 1) % FRAME_OVERLAP];
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;
    // begin the command buffer recording. We will use this command buffer exactly
    // once, so we want to let vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, std::numeric_limits<uint64_t>::max()));
}

void VulkanEngine::destroy_image(const AllocatedImage& img)
{
    if (img.imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(_device, img.imageView, nullptr);
    }
    if (img.image != VK_NULL_HANDLE) {
        vmaDestroyImage(_allocator, img.image, img.allocation);
    }
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
    }
}

void VulkanEngine::check_extensions()
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(_chosenGPU, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(_chosenGPU, nullptr, &extensionCount, availableExtensions.data());

    std::vector<bool> hasExtension(availableExtensions.size(), false);
    for (const auto& extension : availableExtensions) {
        for (int i = 0; i < _deviceExtensions.size(); i++) {
            if (strcmp(extension.extensionName, _deviceExtensions[i]) == 0) {
                hasExtension[i] = true;
            }
        }
    }

    for (int i = 0; i < _deviceExtensions.size(); i++) {
        if (!hasExtension[i]) {
            throw std::runtime_error("Missing device extension: " + std::string(_deviceExtensions[i]));
        }
    }

    fmt::println("All required device extensions found");
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;

    // make the vulkan instance, with basic debug features
    auto inst_ret = builder.set_app_name("Vulkan Engine")
                        .request_validation_layers(bUseValidationLayers)
                        .use_default_debug_messenger()
                        .require_api_version(1, 3, 0)
                        .build();

    vkb::Instance vkb_inst = inst_ret.value();

    // grab the instance
    _instance = vkb_inst.instance;
    _debugMessenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    VkPhysicalDeviceVulkan13Features features13{};
    features13.dynamicRendering = true;
    features13.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;

    features12.scalarBlockLayout = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingVariableDescriptorCount = VK_TRUE;

    VkPhysicalDeviceFeatures features{};
    features.shaderInt64 = VK_TRUE; // Enable 64-bit integers in shaders

    // use vkbootstrap to select a gpu.
    // We want a gpu that can write to the SDL surface and supports vulkan 1.2
    vkb::PhysicalDeviceSelector selector{vkb_inst};
    for (const auto& extension : _deviceExtensions) {
        selector.add_required_extension(extension);
    }
    vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(1, 3)
                                             .set_required_features_13(features13)
                                             .set_required_features_12(features12)
                                             .set_required_features(features)
                                             .set_surface(_surface)
                                             .select()
                                             .value();

    // physicalDevice.features.
    // create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};

    // enable acceleration structure and RT extension
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
    accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelerationStructureFeatures.accelerationStructure = VK_TRUE;
    deviceBuilder.add_pNext(&accelerationStructureFeatures);

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{};
    rayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    deviceBuilder.add_pNext(&rayTracingPipelineFeatures);

    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a vulkan application
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    // DEBUG: print GPU name
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(_chosenGPU, &deviceProperties);
    fmt::println("GPU: {}", deviceProperties.deviceName);

    // use vkbootstrap to get a Graphics queue
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();

    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);
}

void VulkanEngine::init_raytracing()
{
    // Create the raytracing support structure
    _rayTracer = new VulkanRayTracer(this);
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);
    create_render_targets();
}

void VulkanEngine::create_render_targets()
{

    //depth image size will match the window
    VkExtent3D drawImageExtent = {_windowExtent.width, _windowExtent.height, 1};

    //hardcoding the draw format to 32 bit float
    _drawImage.imageFormat = VK_FORMAT_R32G32B32A32_SFLOAT;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    //for the draw image, we want to allocate it from gpu local memory
    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    //allocate and create the image
    vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    //build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo rview_info =
        vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

    //create a depth image too
    //hardcoding the draw format to 32 bit float
    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;

    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    //allocate and create the image
    vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &_depthImage.image, &_depthImage.allocation, nullptr);

    //build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo dview_info =
        vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));

    // LDR target for the tonemapped sRGB result.
    _ldrImage =
        create_image(VkExtent3D{_windowExtent.width, _windowExtent.height, 1}, VK_FORMAT_R16G16B16A16_SFLOAT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    _ldrNeedsInit = true;
}

void VulkanEngine::destroy_render_targets()
{
    destroy_image(_drawImage);
    destroy_image(_depthImage);
    destroy_image(_ldrImage);
    _drawImage = {};
    _depthImage = {};
    _ldrImage = {};
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};

    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain =
        swapchainBuilder
            //.use_default_format_selection()
            .set_desired_format(
                VkSurfaceFormatKHR{.format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            //use vsync present mode
            .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_extent(width, height)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .build()
            .value();

    //store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
    _windowExtent = vkbSwapchain.extent;
}
void VulkanEngine::destroy_swapchain()
{
    for (VkImageView imageView : _swapchainImageViews) {
        vkDestroyImageView(_device, imageView, nullptr);
    }
    _swapchainImageViews.clear();
    _swapchainImages.clear();

    if (_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(_device, _swapchain, nullptr);
        _swapchain = VK_NULL_HANDLE;
    }
}

bool VulkanEngine::resize_swapchain()
{
    int w, h;
    SDL_Vulkan_GetDrawableSize(_window, &w, &h);
    if (w <= 0 || h <= 0) {
        return false;
    }

    VK_CHECK(vkDeviceWaitIdle(_device));

    destroy_taa_history_images();
    destroy_monte_carlo_images();
    destroy_render_targets();
    destroy_swapchain();

    create_swapchain(static_cast<uint32_t>(w), static_cast<uint32_t>(h));
    create_render_targets();
    create_monte_carlo_images();
    create_taa_history_images();

    DescriptorWriter drawImageWriter;
    drawImageWriter.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    drawImageWriter.update_set(_device, _drawImageDescriptors);

    _taaIndex = 0;
    _taaInitialized = false;
    _resetAccumNextFrame = true;
    _lastMonteCarlo = (_monteCarloSamples == 0);
    _lastMsaaSamples = -1;
    _resizeRequested = false;
    return true;
}

void VulkanEngine::init_commands()
{
    // create a command pool for commands submitted to the graphics queue.
    // we also want the pool to allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo =
        vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (int i = 0; i < FRAME_OVERLAP; i++) {

        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

        // allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));

        _mainDeletionQueue.push_function([=]() { vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr); });
    }

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    // allocate the default command buffer that we will use for rendering
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

    _mainDeletionQueue.push_function([=]() { vkDestroyCommandPool(_device, _immCommandPool, nullptr); });
}

void VulkanEngine::init_sync_structures()
{
    // create syncronization structures
    // one fence to control when the gpu has finished rendering the frame,
    // and 2 semaphores to syncronize rendering with swapchain
    // we want the fence to start signalled so we can wait on it on the first
    // frame
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));

    _mainDeletionQueue.push_function([=]() { vkDestroyFence(_device, _immFence, nullptr); });

    for (int i = 0; i < FRAME_OVERLAP; i++) {

        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));

        _mainDeletionQueue.push_function([=]() {
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);
            vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
        });
    }
}

AllocatedImage VulkanEngine::load_image_from_file(std::string path)
{
    // Load image file
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    VkDeviceSize imageSize = texWidth * texHeight * 4; // 4 for RGBA

    // Create a Vulkan image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = static_cast<uint32_t>(texWidth);
    imageInfo.extent.height = static_cast<uint32_t>(texHeight);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkImage image;
    VmaAllocation allocation;
    vmaCreateImage(_allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr);

    // Upload pixels to the Vulkan image
    VkBuffer stagingBuffer;
    VmaAllocation stagingBufferAllocation;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo bufferAllocInfo = {};
    bufferAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    vmaCreateBuffer(_allocator, &bufferInfo, &bufferAllocInfo, &stagingBuffer, &stagingBufferAllocation, nullptr);

    void* data;
    vmaMapMemory(_allocator, stagingBufferAllocation, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vmaUnmapMemory(_allocator, stagingBufferAllocation);

    // Clean up pixel array
    stbi_image_free(pixels);

    // Transfer data to the image
    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkutil::copy_buffer_to_image(cmd, stagingBuffer, image, static_cast<uint32_t>(texWidth),
                                     static_cast<uint32_t>(texHeight));
        vkutil::transition_image(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    // Clean up staging buffer
    vmaDestroyBuffer(_allocator, stagingBuffer, stagingBufferAllocation);

    // Create image view
    VkImageView imageView;
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(_device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    return AllocatedImage{image,
                          imageView,
                          allocation,
                          {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1},
                          VK_FORMAT_R8G8B8A8_SRGB};
}

void VulkanEngine::init_renderables()
{
    _structurePath = {"..\\..\\assets\\livingroom_vkr.glb"};
    _lightPath = {"..\\..\\assets\\livingroom.json"};
    auto structureFile = load_gltf(this, _structurePath);

    assert(structureFile.has_value());

    _loadedScenes["structure"] = *structureFile;

    // load environment map .png
    _environmentMapPath = {"..\\..\\assets\\142_hdrmaps_com_free_10K.png"};
    _environmentMap = load_image_from_file(_environmentMapPath);
    const AllocatedImage loadedEnvironmentMap = _environmentMap;
    _mainDeletionQueue.push_function([this, loadedEnvironmentMap]() { destroy_image(loadedEnvironmentMap); });
}

void VulkanEngine::init_lights()
{
    std::vector<RenderLight> parsedLights = load_lights(_lightPath);
    _lightCount = static_cast<int>(parsedLights.size());
    fmt::println("Loaded {} lights", _lightCount);

    // create a buffer for the lights
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    _lightBuffer =
        create_buffer_data(sizeof(RenderLight) * _lightCount, parsedLights.data(), usage, VMA_MEMORY_USAGE_CPU_TO_GPU);
    const AllocatedBuffer lightBuffer = _lightBuffer;
    _mainDeletionQueue.push_function([this, lightBuffer]() { destroy_buffer(lightBuffer); });
}

void VulkanEngine::init_imgui()
{
    // 1: create descriptor pool for IMGUI
    //  the size of the pool is very oversize, but it's copied from imgui demo
    //  itself.
    VkDescriptorPoolSize pool_sizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                                         {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                                         {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                                         {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

    // 2: initialize imgui library

    // this initializes the core structures of imgui
    ImGui::CreateContext();

    // this initializes imgui for SDL
    ImGui_ImplSDL2_InitForVulkan(_window);

    // this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _instance;
    init_info.PhysicalDevice = _chosenGPU;
    init_info.Device = _device;
    init_info.Queue = _graphicsQueue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;
    init_info.ColorAttachmentFormat = _swapchainImageFormat;

    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info, VK_NULL_HANDLE);

    // execute a gpu command to upload imgui font textures
    immediate_submit([&](VkCommandBuffer cmd) { ImGui_ImplVulkan_CreateFontsTexture(cmd); });

    // clear font textures from cpu data
    ImGui_ImplVulkan_DestroyFontUploadObjects();

    // add the destroy the imgui created structures
    _mainDeletionQueue.push_function([=]() {
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
        ImGui_ImplVulkan_Shutdown();
    });
}

void VulkanEngine::init_pipelines()
{
    // COMPUTE PIPELINES
    init_background_pipelines();

    _metalRoughMaterial.build_pipelines(this);

    create_monte_carlo_pipeline_resources();

    create_taa_pipeline_resources();
}

void VulkanEngine::init_descriptors()
{
    // create a descriptor pool
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4},  // add
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4}, // add
        {VK_DESCRIPTOR_TYPE_SAMPLER, 4},        // add
    };

    _globalDescriptorAllocator.init_pool(_device, 10, sizes);
    const VkDescriptorPool globalDescriptorPool = _globalDescriptorAllocator.pool;
    _mainDeletionQueue.push_function(
        [this, globalDescriptorPool]() { vkDestroyDescriptorPool(_device, globalDescriptorPool, nullptr); });

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout =
            builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    }
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        _objDescLayout = builder.build(_device, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    }

    const VkDescriptorSetLayout drawImageLayout = _drawImageDescriptorLayout;
    const VkDescriptorSetLayout sceneDataLayout = _gpuSceneDataDescriptorLayout;
    const VkDescriptorSetLayout objectLayout = _objDescLayout;
    _mainDeletionQueue.push_function([this, drawImageLayout, sceneDataLayout, objectLayout]() {
        vkDestroyDescriptorSetLayout(_device, drawImageLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, sceneDataLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, objectLayout, nullptr);
    });

    _drawImageDescriptors = _globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.update_set(_device, _drawImageDescriptors);
    }
    for (int i = 0; i < FRAME_OVERLAP; i++) {
        // create a descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init_pools(_device, 1000, frame_sizes);
        _mainDeletionQueue.push_function([&, i]() { _frames[i]._frameDescriptors.destroy_pools(_device); });
    }

    // Volume set: binding 0 = medium params (uniform/storage buffer)
    //             binding 1 = 3D density image (sampled image)
    //             binding 2 = sampler
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        builder.add_binding(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
        builder.add_binding(2, VK_DESCRIPTOR_TYPE_SAMPLER);
        _volumeSetLayout = builder.build(_device, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                                      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    }

    const VkDescriptorSetLayout volumeLayout = _volumeSetLayout;
    _mainDeletionQueue.push_function(
        [this, volumeLayout]() { vkDestroyDescriptorSetLayout(_device, volumeLayout, nullptr); });

    // Allocate once and update the buffer contents when parameters change.
    _volumeSet = _globalDescriptorAllocator.allocate(_device, _volumeSetLayout);

    create_volume_resources();
    initialize_default_medium();
}

void GLTFMetallic_Roughness::build_pipelines(VulkanEngine* engine)
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.frag.spv", engine->_device, &meshFragShader)) {
        fmt::print(stderr, "Failed to build triangle fragment shader module\n");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.vert.spv", engine->_device, &meshVertexShader)) {
        fmt::print(stderr, "Failed to build triangle vertex shader module\n");
    }

    VkPushConstantRange matrixRange{};
    matrixRange.offset = 0;
    matrixRange.size = sizeof(GPUDrawPushConstants);
    matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    materialLayout = layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = {engine->_gpuSceneDataDescriptorLayout, materialLayout};

    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 2;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pPushConstantRanges = &matrixRange;
    mesh_layout_info.pushConstantRangeCount = 1;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &mesh_layout_info, nullptr, &newLayout));

    opaquePipeline.layout = newLayout;
    transparentPipeline.layout = newLayout;

    // build the stage-create-info for both vertex and fragment stages. This lets
    // the pipeline know the shader modules per stage
    PipelineBuilder pipelineBuilder;

    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);

    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);

    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);

    pipelineBuilder.set_multisampling_none();

    pipelineBuilder.disable_blending();

    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    //render format
    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

    // use the triangle layout we created
    pipelineBuilder._pipelineLayout = newLayout;

    // finally build the pipeline
    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    // create the transparent variant
    pipelineBuilder.enable_blending_additive();

    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    vkDestroyShaderModule(engine->_device, meshFragShader, nullptr);
    vkDestroyShaderModule(engine->_device, meshVertexShader, nullptr);

    const VkDescriptorSetLayout layoutToDestroy = materialLayout;
    const VkPipelineLayout pipelineLayoutToDestroy = newLayout;
    const VkPipeline opaquePipelineToDestroy = opaquePipeline.pipeline;
    const VkPipeline transparentPipelineToDestroy = transparentPipeline.pipeline;
    engine->_mainDeletionQueue.push_function(
        [engine, layoutToDestroy, pipelineLayoutToDestroy, opaquePipelineToDestroy, transparentPipelineToDestroy]() {
            vkDestroyPipeline(engine->_device, opaquePipelineToDestroy, nullptr);
            vkDestroyPipeline(engine->_device, transparentPipelineToDestroy, nullptr);
            vkDestroyPipelineLayout(engine->_device, pipelineLayoutToDestroy, nullptr);
            vkDestroyDescriptorSetLayout(engine->_device, layoutToDestroy, nullptr);
        });
}

MaterialInstance GLTFMetallic_Roughness::write_material(VkDevice device, MaterialPass pass,
                                                        const MaterialResources& resources,
                                                        DescriptorAllocatorGrowable& descriptorAllocator)
{
    MaterialInstance matData;
    matData.passType = pass;
    if (pass == MaterialPass::Transparent) {
        matData.pipeline = &transparentPipeline;
    } else {
        matData.pipeline = &opaquePipeline;
    }

    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);

    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset,
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    writer.update_set(device, matData.materialSet);

    return matData;
}

// function to make buffers available on device
AllocatedBuffer VulkanEngine::allocate_and_bind_buffer(VkBuffer buffer, VmaMemoryUsage memoryUsage)
{
    if (_allocator == VK_NULL_HANDLE || buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Invalid allocator or buffer handle");
    }

    // Get the buffer memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(_device, buffer, &memRequirements);

    // Create the allocation information structure
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memoryUsage; // Use CPU to GPU memory

    // Allocate memory for the buffer
    VmaAllocation allocation;
    VmaAllocationInfo allocationInfo;
    if (vmaAllocateMemoryForBuffer(_allocator, buffer, &allocInfo, &allocation, &allocationInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate memory for buffer");
    }

    // Bind the buffer memory
    if (vmaBindBufferMemory(_allocator, allocation, buffer) != VK_SUCCESS) {
        vmaFreeMemory(_allocator, allocation);
        throw std::runtime_error("Failed to bind buffer memory");
    }

    // Create an AllocatedBuffer struct to hold the buffer and its memory allocation
    AllocatedBuffer allocatedBuffer;
    allocatedBuffer.buffer = buffer;
    allocatedBuffer.allocation = allocation;
    allocatedBuffer.info = allocationInfo;

    return allocatedBuffer;
}

void MeshNode::draw(const glm::mat4& topMatrix, DrawContext& ctx)
{
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto& s : mesh->surfaces) {
        RenderObject def;
        def.indexCount = s.count;
        def.firstIndex = s.startIndex;
        def.indexBuffer = mesh->meshBuffers.indexBuffer.buffer;
        def.material = &s.material->data;
        def.bounds = s.bounds;
        def.transform = nodeMatrix;
        def.vertexBuffer = mesh->meshBuffers.vertexBuffer.buffer;
        def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;
        def.vertexCount = mesh->meshBuffers.vertexCount;

        ObjDesc od;
        od.vertexAddress = mesh->meshBuffers.vertexBufferAddress;
        od.indexAddress = engine->get_buffer_device_address(engine->_device, mesh->meshBuffers.indexBuffer.buffer);
        od.materialAddress = s.material->materialAddressRT;

        if (s.material->data.passType == MaterialPass::Transparent) {
            ctx.TransparentSurfaces.push_back(def);
        } else {
            ctx.opaqueSurfaces.push_back(def);
            ctx.objectDescriptions.push_back(od);
        }
    }

    // recurse down
    Node::draw(topMatrix, ctx);
}

VkDeviceAddress VulkanEngine::get_buffer_device_address(VkDevice device, VkBuffer buffer)
{
    if (buffer == VK_NULL_HANDLE)
        return 0ULL;

    VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &info);
}
// Creates a buffer with data mapped in
AllocatedBuffer VulkanEngine::create_buffer_data(VkDeviceSize size, const void* data, VkBufferUsageFlags usage,
                                                 const VmaMemoryUsage memUsage)
{

    AllocatedBuffer resultBuffer = create_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, memUsage);

    // Create a staging buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAllocation;
    vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo, &stagingBuffer, &stagingAllocation, nullptr);

    // Map the buffer and copy the data
    void* mappedData;
    vmaMapMemory(_allocator, stagingAllocation, &mappedData);
    memcpy(mappedData, data, size);
    vmaUnmapMemory(_allocator, stagingAllocation);

    // Record command to transfer data from staging buffer to the destination buffer
    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0; // Assuming data starts at the beginning of the staging buffer
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    immediate_submit(
        [&](VkCommandBuffer cmd) { vkCmdCopyBuffer(cmd, stagingBuffer, resultBuffer.buffer, 1, &copyRegion); });
    vmaDestroyBuffer(_allocator, stagingBuffer, stagingAllocation);

    return resultBuffer;
}

void VulkanEngine::create_taa_pipeline_resources()
{
    create_taa_history_images();

    // Descriptor set layout: curr, prev, out = 3 storage images
    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    _taaSetLayout = b.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    const VkDescriptorSetLayout taaSetLayout = _taaSetLayout;
    _mainDeletionQueue.push_function(
        [this, taaSetLayout]() { vkDestroyDescriptorSetLayout(_device, taaSetLayout, nullptr); });

    // Allocate two descriptor sets (we'll ping-pong prev/out between histories)
    _taaSet[0] = _globalDescriptorAllocator.allocate(_device, _taaSetLayout);
    _taaSet[1] = _globalDescriptorAllocator.allocate(_device, _taaSetLayout);

    // Compute pipeline
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = sizeof(float) * 2; // alpha, clampK
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pc;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &_taaSetLayout;
    VK_CHECK(vkCreatePipelineLayout(_device, &pli, nullptr, &_taaPipelineLayout));
    const VkPipelineLayout taaPipelineLayout = _taaPipelineLayout;
    _mainDeletionQueue.push_function(
        [this, taaPipelineLayout]() { vkDestroyPipelineLayout(_device, taaPipelineLayout, nullptr); });

    VkShaderModule taaCS;
    if (!vkutil::load_shader_module("../../shaders/temporal_resolve.comp.spv", _device, &taaCS)) {
        throw std::runtime_error("failed to load temporal_resolve.comp.spv");
    }
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = taaCS;
    ss.pName = "main";
    ci.stage = ss;
    ci.layout = _taaPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &ci, nullptr, &_taaPipeline));
    vkDestroyShaderModule(_device, taaCS, nullptr);
    const VkPipeline taaPipeline = _taaPipeline;
    _mainDeletionQueue.push_function([this, taaPipeline]() { vkDestroyPipeline(_device, taaPipeline, nullptr); });
}

void VulkanEngine::create_taa_history_images()
{
    VkExtent3D ext{_windowExtent.width, _windowExtent.height, 1};
    auto make_history = [&](AllocatedImage& img) {
        img = create_image(ext, VK_FORMAT_R16G16B16A16_SFLOAT,
                           VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        immediate_submit([&](VkCommandBuffer cmd) {
            vkutil::transition_image(cmd, img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        });
    };
    make_history(_taaHistory[0]);
    make_history(_taaHistory[1]);
}

void VulkanEngine::destroy_taa_history_images()
{
    destroy_image(_taaHistory[0]);
    destroy_image(_taaHistory[1]);
    _taaHistory[0] = {};
    _taaHistory[1] = {};
}

void VulkanEngine::create_monte_carlo_pipeline_resources()
{
    create_monte_carlo_images();

    // Descriptor set layout: currColor, accumColor, accumCount, outColor
    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); // curr (from raygen)
    b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); // accum color (avg)
    b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); // accum count (r32ui)
    b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); // out (resolved)
    _mcSetLayout = b.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    const VkDescriptorSetLayout mcSetLayout = _mcSetLayout;
    _mainDeletionQueue.push_function(
        [this, mcSetLayout]() { vkDestroyDescriptorSetLayout(_device, mcSetLayout, nullptr); });

    _mcSet = _globalDescriptorAllocator.allocate(_device, _mcSetLayout);

    // Pipeline + layout (push: resetFrames, movingFlag)
    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 2};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pc;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &_mcSetLayout;
    VK_CHECK(vkCreatePipelineLayout(_device, &pli, nullptr, &_mcPipeLayout));
    const VkPipelineLayout mcPipelineLayout = _mcPipeLayout;
    _mainDeletionQueue.push_function(
        [this, mcPipelineLayout]() { vkDestroyPipelineLayout(_device, mcPipelineLayout, nullptr); });

    VkShaderModule mcCS;
    if (!vkutil::load_shader_module("../../shaders/mc_accum.comp.spv", _device, &mcCS)) {
        throw std::runtime_error("failed to load mc_accum.comp.spv");
    }
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = mcCS;
    ss.pName = "main";
    ci.stage = ss;
    ci.layout = _mcPipeLayout;
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &ci, nullptr, &_mcPipeline));
    vkDestroyShaderModule(_device, mcCS, nullptr);
    const VkPipeline mcPipeline = _mcPipeline;
    _mainDeletionQueue.push_function([this, mcPipeline]() { vkDestroyPipeline(_device, mcPipeline, nullptr); });
}

void VulkanEngine::create_monte_carlo_images()
{

    VkExtent3D ext{_windowExtent.width, _windowExtent.height, 1};

    // Accum color: running average
    _mcAccumColor =
        create_image(ext, VK_FORMAT_R16G16B16A16_SFLOAT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, _mcAccumColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    });

    // Accum count: number of accumulated samples per pixel
    _mcAccumCount =
        create_image(ext, VK_FORMAT_R32_UINT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, _mcAccumCount.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        // zero it
        vkutil::transition_image(cmd, _mcAccumCount.image, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkutil::clear_color_image_uint(cmd, _mcAccumCount.image, 0, 0, 0, 0); // helper: vkCmdClearColorImage for UINT
        vkutil::transition_image(cmd, _mcAccumCount.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_GENERAL);
    });
}

void VulkanEngine::destroy_monte_carlo_images()
{
    destroy_image(_mcAccumColor);
    destroy_image(_mcAccumCount);
    _mcAccumColor = {};
    _mcAccumCount = {};
}

void VulkanEngine::reset_monte_carlo_history(VkCommandBuffer cmd)
{
    // Clear count=0 and copy current draw into accumColor so the first blend is stable
    vkutil::transition_image(cmd, _mcAccumCount.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkutil::clear_color_image_uint(cmd, _mcAccumCount.image, 0, 0, 0, 0);
    vkutil::transition_image(cmd, _mcAccumCount.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

    vkutil::transition_image(cmd, _mcAccumColor.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::copy_image_to_image(cmd, _drawImage.image, _mcAccumColor.image, _windowExtent, _windowExtent);
    vkutil::transition_image(cmd, _mcAccumColor.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
}

void VulkanEngine::create_postprocess_resources()
{
    // descriptor layout: hdrIn (0), ldrOut (1)
    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    _postSetLayout = b.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    const VkDescriptorSetLayout postSetLayout = _postSetLayout;
    _mainDeletionQueue.push_function(
        [this, postSetLayout]() { vkDestroyDescriptorSetLayout(_device, postSetLayout, nullptr); });

    _postSet = _globalDescriptorAllocator.allocate(_device, _postSetLayout);

    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = sizeof(float);

        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &_postSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pc;

        VK_CHECK(vkCreatePipelineLayout(_device, &pli, nullptr, &_postPipeLayout));
        const VkPipelineLayout postPipelineLayout = _postPipeLayout;
        _mainDeletionQueue.push_function(
            [this, postPipelineLayout]() { vkDestroyPipelineLayout(_device, postPipelineLayout, nullptr); });
    }

    VkShaderModule cs;
    if (!vkutil::load_shader_module("../../shaders/post_tonemap.comp.spv", _device, &cs)) {
        throw std::runtime_error("failed to load post_tonemap.comp.spv");
    }
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = cs;
    ss.pName = "main";
    ci.stage = ss;
    ci.layout = _postPipeLayout;
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &ci, nullptr, &_postPipeline));
    vkDestroyShaderModule(_device, cs, nullptr);
    const VkPipeline postPipeline = _postPipeline;
    _mainDeletionQueue.push_function([this, postPipeline]() { vkDestroyPipeline(_device, postPipeline, nullptr); });
}

void VulkanEngine::request_accum_reset()
{
    _resetAccumNextFrame = true;
}

void VulkanEngine::create_volume_resources()
{
    // Create std140 medium UBO (persistently mapped CPU->GPU)
    if (_volume.mediumParams.buffer == VK_NULL_HANDLE) {
        _volume.mediumParams =
            create_buffer(sizeof(GPUMediumParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        const AllocatedBuffer mediumParams = _volume.mediumParams;
        _mainDeletionQueue.push_function([this, mediumParams]() { destroy_buffer(mediumParams); });
    }

    // Create a dedicated 3D sampler (linear, clamp)
    if (_volume.densitySampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(_device, &sci, nullptr, &_volume.densitySampler));
        const VkSampler densitySampler = _volume.densitySampler;
        _mainDeletionQueue.push_function(
            [this, densitySampler]() { vkDestroySampler(_device, densitySampler, nullptr); });
    }

    // Initial descriptor write: UBO + sampler (no density image yet)
    {
        DescriptorWriter w;
        w.write_buffer(0, _volume.mediumParams.buffer, sizeof(GPUMediumParams), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        w.write_image(2, VK_NULL_HANDLE, _volume.densitySampler, VK_IMAGE_LAYOUT_UNDEFINED, VK_DESCRIPTOR_TYPE_SAMPLER);
        w.update_set(_device, _volumeSet);
    }
}

void VulkanEngine::initialize_default_medium()
{
    // No 3D density bound initially; homogeneous only.
    _volume.hasDensity = false;

    // One time defaults
    GPUMediumParams p{};
    p.sigma_a_step = {0.02f, 0.02f, 0.02f, 0.02f}; // stepSize as .w
    p.sigma_s_maxT = {0.00f, 0.00f, 0.00f, 200.0f};
    p.g_emis_density_pad = {0.0f, 0.0f, 1.0f, 0.0f}; // ... , fogEnvFlag=0 (skip fog on env)
    set_medium_params(p);
}

void VulkanEngine::upload_volume_density(const void* voxels, VkExtent3D extent, VkFormat fmt)
{
    // Create a 3D image (R16_SFLOAT or R8_UNORM or R32_SFLOAT)
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    _volume.densityTex3D = create_image(extent, fmt, usage, /*mipmapped=*/false);

    // Upload via staging (reusing create_image(void*,...) path is 2D-only; do a custom upload)
    size_t pixelSize = (fmt == VK_FORMAT_R32_SFLOAT) ? 4 : (fmt == VK_FORMAT_R16_SFLOAT) ? 2 : 1; // R8_UNORM
    size_t total = size_t(extent.width) * extent.height * extent.depth * pixelSize;

    AllocatedBuffer staging = create_buffer(total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    memcpy(staging.allocation->GetMappedData(), voxels, total);

    immediate_submit([&](VkCommandBuffer cmd) {
        // Transition
        vkutil::transition_image(cmd, _volume.densityTex3D.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel = 0;
        copy.imageSubresource.baseArrayLayer = 0;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = extent;

        vkCmdCopyBufferToImage(cmd, staging.buffer, _volume.densityTex3D.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &copy);

        vkutil::transition_image(cmd, _volume.densityTex3D.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    destroy_buffer(staging);

    _volume.hasDensity = true;

    // Update descriptor set with the image at binding 1
    {
        DescriptorWriter w;
        w.write_buffer(0, _volume.mediumParams.buffer, sizeof(GPUMediumParams), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        w.write_image(1, _volume.densityTex3D.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
        w.write_image(2, VK_NULL_HANDLE, _volume.densitySampler, VK_IMAGE_LAYOUT_UNDEFINED, VK_DESCRIPTOR_TYPE_SAMPLER);
        w.update_set(_device, _volumeSet);
    }

    // Cleanup hook
    const AllocatedImage densityImage = _volume.densityTex3D;
    _mainDeletionQueue.push_function([this, densityImage]() { destroy_image(densityImage); });
}

void VulkanEngine::set_medium_params(const GPUMediumParams& p)
{
    GPUMediumParams* dst = (GPUMediumParams*)_volume.mediumParams.allocation->GetMappedData();
    *dst = p;
}
