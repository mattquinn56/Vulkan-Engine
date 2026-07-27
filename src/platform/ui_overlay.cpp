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
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.WindowPadding = ImVec2(16.0f, 16.0f);
    style.ItemSpacing = ImVec2(8.0f, 8.0f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(0.25f, 0.72f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.25f, 0.72f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.45f, 0.82f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.12f, 0.35f, 0.52f, 1.0f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.16f, 0.45f, 0.66f, 1.0f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.55f, 0.78f, 1.0f);

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

    static const char* const kViewNames[] = {"Beauty",         "World normals", "Hit distance",
                                             "Motion vectors", "Instance ID",   "History reuse"};
    static_assert(IM_ARRAYSIZE(kViewNames) == RtEngine::kDebugViewCount);

    bool resetAccum = false;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();

    if (_settingsOpen) {
        const float panelWidth = viewport->WorkSize.x < 410.0f ? viewport->WorkSize.x : 410.0f;
        ImGui::SetNextWindowPos(viewport->WorkPos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(panelWidth, viewport->WorkSize.y), ImGuiCond_Always);
        constexpr ImGuiWindowFlags panelFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                                                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                                ImGuiWindowFlags_NoSavedSettings;

        ImGui::Begin("Renderer Settings", nullptr, panelFlags);
        ImGui::TextUnformatted("RENDERER SETTINGS");
        ImGui::SameLine();
        ImGui::TextDisabled("Tab to close");
        ImGui::Separator();

        if (ImGui::BeginTabBar("Settings pages")) {
            if (ImGui::BeginTabItem("Essentials")) {
                ImGui::Spacing();
                ImGui::TextUnformatted("Output");
                ImGui::TextDisabled("View");
                ImGui::SetNextItemWidth(-1.0f);
                resetAccum |= ImGui::Combo("##View", &_debugView, kViewNames, IM_ARRAYSIZE(kViewNames));
                resetAccum |= ImGui::Checkbox("ACES tone mapping", &_enableTonemap);
                ImGui::BeginDisabled(!_enableTonemap);
                ImGui::TextDisabled("Exposure");
                ImGui::SetNextItemWidth(-1.0f);
                resetAccum |=
                    ImGui::SliderFloat("##Exposure", &_exposure, 0.1f, 4.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::EndDisabled();

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                ImGui::TextUnformatted("Image quality");
                resetAccum |= ImGui::Checkbox("Progressive accumulation", &_progressiveMonteCarlo);
                ImGui::TextDisabled("Area-light samples");
                ImGui::SetNextItemWidth(-1.0f);
                resetAccum |= ImGui::SliderInt("##Area-light samples", &_monteCarloSamplesPerFrame, 1, 20);
                ImGui::TextDisabled("More samples reduce shadow noise but cost GPU time.");

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                ImGui::TextUnformatted("Controls");
                ImGui::BulletText("W A S D    Move");
                ImGui::BulletText("Mouse      Look");
                ImGui::BulletText("Q / E      Down / up");
                ImGui::BulletText("Shift/Ctrl Fast / slow");
                ImGui::BulletText("Tab        Close settings");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Advanced")) {
                if (ImGui::CollapsingHeader("Temporal filtering", ImGuiTreeNodeFlags_DefaultOpen)) {
                    resetAccum |= ImGui::SliderInt("Camera settle frames", &_cameraSettleFrames, 0, 16);
                    resetAccum |= ImGui::SliderFloat("History weight (still)", &_taaAlpha, 0.0f, 0.99f);
                    resetAccum |= ImGui::SliderFloat("History weight (moving)", &_taaMovingAlpha, 0.0f, 0.99f);
                    resetAccum |= ImGui::SliderFloat("Depth tolerance", &_taaDepthTolerance, 0.0f, 0.5f);
                    resetAccum |= ImGui::SliderFloat("Normal tolerance", &_taaNormalTolerance, 0.0f, 1.0f);
                    ImGui::TextDisabled("Camera state: %s", _cameraMoving ? "moving" : "settled");
                }

                if (ImGui::CollapsingHeader("Homogeneous medium")) {
                    auto* medium = static_cast<GPUMediumParams*>(_volume.mediumParams.info.pMappedData);
                    resetAccum |= ImGui::DragFloat3("Absorption", &medium->sigma_a_step.x, 0.001f, 0.0f, 5.0f, "%.3f");
                    resetAccum |=
                        ImGui::DragFloat3("Out-scattering", &medium->sigma_s_maxT.x, 0.001f, 0.0f, 5.0f, "%.3f");
                    resetAccum |=
                        ImGui::DragFloat("Maximum distance", &medium->sigma_s_maxT.w, 1.0f, 0.0f, 20000.0f, "%.0f");
                    resetAccum |= ImGui::DragFloat("Emission", &medium->g_emis_density_pad.y, 0.001f, 0.0f, 10.0f);

                    bool affectsEnvironment = medium->g_emis_density_pad.w > 0.5f;
                    if (ImGui::Checkbox("Affect environment", &affectsEnvironment)) {
                        medium->g_emis_density_pad.w = affectsEnvironment ? 1.0f : 0.0f;
                        resetAccum = true;
                    }
                }

                if (ImGui::CollapsingHeader("Diagnostics")) {
                    ImGui::Text("Frame time: %.2f ms", _stats.frameTime);
                    ImGui::Text("Camera: %.2f, %.2f, %.2f", _mainCamera.position.x, _mainCamera.position.y,
                                _mainCamera.position.z);
                    for (auto& [name, scene] : _loadedScenes) {
                        if (ImGui::TreeNode(name.c_str())) {
                            render_loaded_gltf(scene);
                            ImGui::TreePop();
                        }
                    }
                }
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 42.0f);
        if (ImGui::Button("Close settings  [Tab]", ImVec2(-1.0f, 0.0f))) {
            set_settings_open(false);
        }
        ImGui::End();
    } else {
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 12.0f, viewport->WorkPos.y + 12.0f), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.72f);
        constexpr ImGuiWindowFlags hintFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                               ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                               ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove |
                                               ImGuiWindowFlags_NoMouseInputs;
        ImGui::Begin("Controls hint", nullptr, hintFlags);
        ImGui::TextUnformatted("[Tab] Settings");
        ImGui::SameLine();
        ImGui::TextDisabled("  WASD move  |  Mouse look  |  Q/E vertical");
        ImGui::End();
    }

    ImGui::Render();

    if (resetAccum) {
        request_accum_reset();
    }
}
