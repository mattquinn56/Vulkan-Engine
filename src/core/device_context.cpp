#include "core/rt_engine.h"

#include "gpu/descriptor_alloc.h"
#include "scene/gltf_import.h"
#include "gpu/image_utils.h"
#include "passes/ray_tracing_pipeline.h"
#include "gpu/shader_module.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include "core/gpu_types.h"
#include "gpu/vk_init.h"

#include "VkBootstrap.h"

#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

#include <stb_image.h>

void RtEngine::init_vulkan() {
    vkb::InstanceBuilder builder;

    auto inst_ret = builder.set_app_name("Vulkan Engine")
                        .request_validation_layers(bUseValidationLayers)
                        .use_default_debug_messenger()
                        .require_api_version(1, 3, 0)
                        .build();

    vkb::Instance vkb_inst = inst_ret.value();

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
    features.shaderInt64 = VK_TRUE; // buffer device addresses are 64-bit in shaders

    // Requires a GPU that can present to the SDL surface and supports Vulkan 1.3
    // plus the ray tracing extensions in _deviceExtensions.
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

    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    // DEBUG: print GPU name
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(_chosenGPU, &deviceProperties);
    fmt::println("GPU: {}", deviceProperties.deviceName);

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

void RtEngine::init_raytracing() {
    _rayTracer = new VulkanRayTracer(this);
}

void RtEngine::init_swapchain() {
    create_swapchain(_windowExtent.width, _windowExtent.height);
    create_render_targets();
}

void RtEngine::create_render_targets() {

    VkExtent3D drawImageExtent = {_windowExtent.width, _windowExtent.height, 1};

    // Full 32-bit float so the HDR path has headroom before tonemapping.
    _drawImage.imageFormat = VK_FORMAT_R32G32B32A32_SFLOAT;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vk_init::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    VkImageViewCreateInfo rview_info =
        vk_init::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

    // LDR target for the tonemapped sRGB result.
    _ldrImage =
        create_image(VkExtent3D{_windowExtent.width, _windowExtent.height, 1}, VK_FORMAT_R16G16B16A16_SFLOAT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    _ldrNeedsInit = true;
}

void RtEngine::destroy_render_targets() {
    destroy_image(_drawImage);
    destroy_image(_ldrImage);
    _drawImage = {};
    _ldrImage = {};
}

void RtEngine::create_swapchain(uint32_t width, uint32_t height) {
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
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
            .build()
            .value();

    //store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
    _windowExtent = vkbSwapchain.extent;
}

void RtEngine::destroy_swapchain() {
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

bool RtEngine::resize_swapchain() {
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

    _taaIndex = 0;
    _taaInitialized = false;
    _resetAccumNextFrame = true;
    _resizeRequested = false;
    return true;
}
