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

void seed_taa_history(RtEngine* e, VkCommandBuffer cmd) {
    for (int i = 0; i < 2; ++i) {
        vk_img::transition_image(cmd, e->_taaHistory[i].image, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vk_img::transition_image(cmd, e->_drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        vk_img::copy_image_to_image(cmd, e->_drawImage.image, e->_taaHistory[i].image, e->_windowExtent,
                                    e->_windowExtent);
        vk_img::transition_image(cmd, e->_taaHistory[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_GENERAL);
        vk_img::transition_image(cmd, e->_drawImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 VK_IMAGE_LAYOUT_GENERAL);
    }
    e->_taaIndex = 0;
}

void RtEngine::create_taa_pipeline_resources() {
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
    if (!vk_shader::load_shader_module("../../shaders/temporal_resolve.comp.spv", _device, &taaCS)) {
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

void RtEngine::create_taa_history_images() {
    VkExtent3D ext{_windowExtent.width, _windowExtent.height, 1};
    auto make_history = [&](AllocatedImage& img) {
        img = create_image(ext, VK_FORMAT_R16G16B16A16_SFLOAT,
                           VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        immediate_submit([&](VkCommandBuffer cmd) {
            vk_img::transition_image(cmd, img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        });
    };
    make_history(_taaHistory[0]);
    make_history(_taaHistory[1]);
}

void RtEngine::destroy_taa_history_images() {
    destroy_image(_taaHistory[0]);
    destroy_image(_taaHistory[1]);
    _taaHistory[0] = {};
    _taaHistory[1] = {};
}
