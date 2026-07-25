#include "rt_engine.h"
#include "resource_path.h"

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

void RtEngine::create_postprocess_resources() {
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
    if (!vk_shader::load_shader_module(resource::shader("post_tonemap.comp.spv").c_str(), _device, &cs)) {
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
