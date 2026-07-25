
#pragma once

#include "core/gpu_types.h"

#include "gpu/descriptor_alloc.h"
#include <unordered_map>
#include <filesystem>

class RtEngine;

struct Bounds
{
    glm::vec3 origin{};
    float sphereRadius{0.0f};
    glm::vec3 extents{};
};

struct SceneMaterial
{
    SurfaceAlphaMode passType{SurfaceAlphaMode::Other};
    VkDeviceAddress materialAddressRT{0};
};

struct MeshPrimitive
{
    uint32_t startIndex{0};
    uint32_t count{0};
    Bounds bounds;
    std::shared_ptr<SceneMaterial> material;
};

struct MeshResource
{
    std::string name;

    std::vector<MeshPrimitive> surfaces;
    GPUMeshBuffers meshBuffers;
};

struct GltfScene : public IRenderable
{

    // storage for all the data on a given gltf file
    std::unordered_map<std::string, std::shared_ptr<MeshResource>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::shared_ptr<Node>, std::string> nodeNames;
    std::unordered_map<std::string, AllocatedImage> images;
    std::unordered_map<std::string, std::shared_ptr<SceneMaterial>> materials;
    std::vector<RenderLight> lights;

    // Root nodes used to traverse the scene in tree order.
    std::vector<std::shared_ptr<Node>> topNodes;

    std::vector<VkSampler> samplers;

    RtEngine* creator{nullptr};

    ~GltfScene() {
        destroy_owned_resources();
    };

    virtual void draw(const glm::mat4& topMatrix, SceneDrawList& ctx);

  private:
    void destroy_owned_resources();
};

std::vector<RenderLight> load_lights(std::string filePath);
std::optional<std::shared_ptr<GltfScene>> load_gltf(RtEngine* engine, std::string_view filePath);
