// or project specific include files.

#pragma once

#include <vk_types.h>

#include "vk_descriptors.h"
#include <unordered_map>
#include <filesystem>

class VulkanEngine;

struct Bounds
{
    glm::vec3 origin{};
    float sphereRadius{0.0f};
    glm::vec3 extents{};
};

struct GLTFMaterial
{
    MaterialInstance data;
    VkDeviceAddress materialAddressRT{0};
};

struct GeoSurface
{
    uint32_t startIndex{0};
    uint32_t count{0};
    Bounds bounds;
    std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset
{
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

struct LoadedGLTF : public IRenderable
{

    // storage for all the data on a given gltf file
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::shared_ptr<Node>, std::string> nodeNames;
    std::unordered_map<std::string, AllocatedImage> images;
    std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;
    std::vector<RenderLight> lights;

    // Root nodes used to traverse the scene in tree order.
    std::vector<std::shared_ptr<Node>> topNodes;

    std::vector<VkSampler> samplers;

    DescriptorAllocatorGrowable descriptorPool;

    AllocatedBuffer materialDataBuffer;

    VulkanEngine* creator{nullptr};

    ~LoadedGLTF()
    {
        destroy_owned_resources();
    };

    virtual void draw(const glm::mat4& topMatrix, DrawContext& ctx);

  private:
    void destroy_owned_resources();
};

std::vector<RenderLight> load_lights(std::string filePath);
std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(VulkanEngine* engine, std::string_view filePath);
