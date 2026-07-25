#include "rt_engine.h"

#include "descriptor_alloc.h"
#include "gltf_import.h"
#include "image_utils.h"
#include "ray_tracing_pipeline.h"
#include "shader_module.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <gpu_types.h>
#include <vk_init.h>

#include "VkBootstrap.h"

#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

#include <stb_image.h>

void RtEngine::create_volume_resources() {
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

void RtEngine::initialize_default_medium() {
    // No 3D density bound initially; homogeneous only.
    _volume.hasDensity = false;

    // One time defaults
    GPUMediumParams p{};
    p.sigma_a_step = {0.02f, 0.02f, 0.02f, 0.02f}; // stepSize as .w
    p.sigma_s_maxT = {0.00f, 0.00f, 0.00f, 200.0f};
    p.g_emis_density_pad = {0.0f, 0.0f, 1.0f, 0.0f}; // ... , fogEnvFlag=0 (skip fog on env)
    set_medium_params(p);
}

void RtEngine::upload_volume_density(const void* voxels, VkExtent3D extent, VkFormat fmt) {
    // Create a 3D image (R16_SFLOAT or R8_UNORM or R32_SFLOAT)
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    _volume.densityTex3D = create_image(extent, fmt, usage, /*mipmapped=*/false);

    // Upload via staging (reusing create_image(void*,...) path is 2D-only; do a custom upload)
    size_t pixelSize = (fmt == VK_FORMAT_R32_SFLOAT) ? 4 : (fmt == VK_FORMAT_R16_SFLOAT) ? 2 : 1; // R8_UNORM
    size_t total = size_t(extent.width) * extent.height * extent.depth * pixelSize;

    AllocatedBuffer staging = create_buffer(total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    memcpy(staging.info.pMappedData, voxels, total);

    immediate_submit([&](VkCommandBuffer cmd) {
        // Transition
        vk_img::transition_image(cmd, _volume.densityTex3D.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel = 0;
        copy.imageSubresource.baseArrayLayer = 0;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = extent;

        vkCmdCopyBufferToImage(cmd, staging.buffer, _volume.densityTex3D.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &copy);

        vk_img::transition_image(cmd, _volume.densityTex3D.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
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

void RtEngine::set_medium_params(const GPUMediumParams& p) {
    GPUMediumParams* dst = (GPUMediumParams*)_volume.mediumParams.info.pMappedData;
    *dst = p;
}
