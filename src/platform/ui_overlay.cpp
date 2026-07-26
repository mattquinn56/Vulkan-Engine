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

void RtEngine::init_imgui() {
    // Generously sized, per the ImGui demo. ImGui does not report its own needs.
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

    ImGui::CreateContext();

    ImGui_ImplSDL2_InitForVulkan(_window);

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

    immediate_submit([&](VkCommandBuffer cmd) { ImGui_ImplVulkan_CreateFontsTexture(cmd); });

    ImGui_ImplVulkan_DestroyFontUploadObjects();

    _mainDeletionQueue.push_function([=]() {
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
        ImGui_ImplVulkan_Shutdown();
    });
}

void RtEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView) {
    VkRenderingAttachmentInfo colorAttachment =
        vk_init::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_GENERAL);
    VkRenderingInfo renderInfo = vk_init::rendering_info(_windowExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void RtEngine::draw_ui() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame(_window);

    ImGui::NewFrame();

    bool reset_accum = false;

    ImGui::Begin("Main Control");

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

    ImGui::Begin("G-buffer");
    {
        static const char* const kViewNames[] = {"Shaded", "Normal", "Hit distance", "Motion vectors", "Instance ID"};
        static_assert(IM_ARRAYSIZE(kViewNames) == RtEngine::kDebugViewCount);
        reset_accum |= ImGui::Combo("View", &_debugView, kViewNames, IM_ARRAYSIZE(kViewNames));
    }
    ImGui::End();

    ImGui::Begin("Antialiasing");
    reset_accum |= ImGui::Checkbox("Progressive Monte Carlo", &_progressiveMonteCarlo);
    reset_accum |= ImGui::SliderInt("MC per-frame spp", &_monteCarloSamplesPerFrame, 1, 20);
    reset_accum |= ImGui::SliderInt("MC reset frames", &_monteCarloResetFrames, 0, 8);
    reset_accum |= ImGui::SliderFloat("TAA alpha (still)", &_taaAlpha, 0.0f, 0.99f);
    ImGui::Text("Camera moving: %s", _cameraMoving ? "yes" : "no");
    ImGui::End();

    ImGui::Begin("Medium");
    {
        auto* mp = (GPUMediumParams*)_volume.mediumParams.info.pMappedData;

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
            auto* mp = (GPUMediumParams*)_volume.mediumParams.info.pMappedData;
            bool fogEnv = mp->g_emis_density_pad.w > 0.5f;
            if (ImGui::Checkbox("Fog affects environment", &fogEnv)) {
                mp->g_emis_density_pad.w = fogEnv ? 1.0f : 0.0f;
            }
        }
    }
    ImGui::End();

    ImGui::Begin("Hierarchy");

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
}
