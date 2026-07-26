#include "core/rt_engine.h"
#include "platform/resource_path.h"

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

void RtEngine::create_monte_carlo_pipeline_resources() {
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
    if (!vk_shader::load_shader_module(resource::shader("mc_accum.comp.spv").c_str(), _device, &mcCS)) {
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

    update_monte_carlo_descriptors();
}

// The set names only render targets, so it stays valid until one is recreated.
// Rewriting it per frame would touch a set the previous frame still has pending.
void RtEngine::update_monte_carlo_descriptors() {
    DescriptorWriter w;
    w.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    w.write_image(1, _mcAccumColor.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    w.write_image(2, _mcAccumCount.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    w.write_image(3, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    w.update_set(_device, _mcSet);
}

void RtEngine::create_monte_carlo_images() {

    VkExtent3D ext{_windowExtent.width, _windowExtent.height, 1};

    // Accum color: running average
    _mcAccumColor =
        create_image(ext, VK_FORMAT_R16G16B16A16_SFLOAT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    immediate_submit([&](VkCommandBuffer cmd) {
        vk_img::transition_image(cmd, _mcAccumColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    });

    // Accum count: number of accumulated samples per pixel
    _mcAccumCount =
        create_image(ext, VK_FORMAT_R32_UINT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    immediate_submit([&](VkCommandBuffer cmd) {
        vk_img::transition_image(cmd, _mcAccumCount.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        // zero it
        vk_img::transition_image(cmd, _mcAccumCount.image, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vk_img::clear_color_image_uint(cmd, _mcAccumCount.image, 0, 0, 0, 0); // helper: vkCmdClearColorImage for UINT
        vk_img::transition_image(cmd, _mcAccumCount.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_GENERAL);
    });
}

void RtEngine::destroy_monte_carlo_images() {
    destroy_image(_mcAccumColor);
    destroy_image(_mcAccumCount);
    _mcAccumColor = {};
    _mcAccumCount = {};
}

void RtEngine::reset_monte_carlo_history(VkCommandBuffer cmd) {
    // Clear count=0 and copy current draw into accumColor so the first blend is stable
    vk_img::transition_image(cmd, _mcAccumCount.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vk_img::clear_color_image_uint(cmd, _mcAccumCount.image, 0, 0, 0, 0);
    vk_img::transition_image(cmd, _mcAccumCount.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

    vk_img::transition_image(cmd, _mcAccumColor.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vk_img::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vk_img::copy_image_to_image(cmd, _drawImage.image, _mcAccumColor.image, _windowExtent, _windowExtent);
    vk_img::transition_image(cmd, _mcAccumColor.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    vk_img::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
}

void RtEngine::request_accum_reset() {
    _resetAccumNextFrame = true;
}
