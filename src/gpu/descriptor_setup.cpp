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

void RtEngine::init_descriptors() {
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
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout =
            builder.build(_device, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    }
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        _objDescLayout = builder.build(_device, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    }

    const VkDescriptorSetLayout sceneDataLayout = _gpuSceneDataDescriptorLayout;
    const VkDescriptorSetLayout objectLayout = _objDescLayout;
    _mainDeletionQueue.push_function([this, sceneDataLayout, objectLayout]() {
        vkDestroyDescriptorSetLayout(_device, sceneDataLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, objectLayout, nullptr);
    });
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

// Backs an already-created VkBuffer with memory and binds the two together.
