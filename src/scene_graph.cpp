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

void RtEngine::init_renderables() {
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

void RtEngine::init_lights() {
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

void RtEngine::render_loaded_gltf(std::shared_ptr<GltfScene> gltf) {
    auto topLevelNodes = gltf->topNodes;

    for (auto& node : topLevelNodes) {
        recursively_render_node(gltf, node);
    }
}

void RtEngine::recursively_render_node(std::shared_ptr<GltfScene> gltf, std::shared_ptr<Node> node) {
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

void RtEngine::update_scene() {
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
    _loadedScenes["structure"]->draw(glm::mat4{1.f}, _drawContext);
}

void MeshNode::draw(const glm::mat4& topMatrix, SceneDrawList& ctx) {
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto& s : mesh->surfaces) {
        // Blended materials are not represented in the acceleration structure.
        if (s.material->passType == SurfaceAlphaMode::Transparent) {
            continue;
        }

        GeometryInstance def;
        def.indexCount = s.count;
        def.firstIndex = s.startIndex;
        def.indexBuffer = mesh->meshBuffers.indexBuffer.buffer;
        def.bounds = s.bounds;
        def.transform = nodeMatrix;
        def.vertexBuffer = mesh->meshBuffers.vertexBuffer.buffer;
        def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;
        def.vertexCount = mesh->meshBuffers.vertexCount;

        ObjDesc od;
        od.vertexAddress = mesh->meshBuffers.vertexBufferAddress;
        od.indexAddress = engine->get_buffer_device_address(engine->_device, mesh->meshBuffers.indexBuffer.buffer);
        od.materialAddress = s.material->materialAddressRT;

        ctx.opaqueSurfaces.push_back(def);
        ctx.objectDescriptions.push_back(od);
    }

    Node::draw(topMatrix, ctx);
}
