// or project specific include files.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <functional>
#include <deque>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

// we will add our main reusable types here
struct AllocatedImage
{
    VkImage image{VK_NULL_HANDLE};
    VkImageView imageView{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    VkExtent3D imageExtent{};
    VkFormat imageFormat{VK_FORMAT_UNDEFINED};
};

struct AllocatedBuffer
{
    VkBuffer buffer{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    VmaAllocationInfo info{};
};

struct GPUGLTFMaterial
{
    glm::vec4 colorFactors{};
    glm::vec4 metalRoughFactors{};
    glm::vec4 extra[14]{};
};

static_assert(sizeof(GPUGLTFMaterial) == 256);

struct RenderLight
{
    glm::vec4 position{};
    glm::vec4 color{};
    glm::vec4 v0{};
    glm::vec4 v1{};
};

struct GPUSceneData
{
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::mat4 viewproj{1.0f};
    glm::vec4 data{};
};
enum class MaterialPass : uint8_t
{
    MainColor,
    Transparent,
    Other
};
struct MaterialPipeline
{
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkPipelineLayout layout{VK_NULL_HANDLE};
};

struct MaterialInstance
{
    MaterialPipeline* pipeline{nullptr};
    VkDescriptorSet materialSet{VK_NULL_HANDLE};
    MaterialPass passType{MaterialPass::Other};
};
struct Vertex
{

    glm::vec3 position{};
    float uvX{0.0f};
    glm::vec3 normal{};
    float uvY{0.0f};
    glm::vec4 color{};
};

// holds the resources needed for a mesh
struct GPUMeshBuffers
{

    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress{0};
    int vertexCount{0};
};

// push constants for our mesh object draws
struct GPUDrawPushConstants
{
    glm::mat4 worldMatrix{1.0f};
    VkDeviceAddress vertexBuffer{0};
    VkDeviceAddress lightBuffer{0};
    int numLights{0};
};
struct DrawContext;

// base class for a renderable dynamic object
class IRenderable
{

    virtual void draw(const glm::mat4& topMatrix, DrawContext& ctx) = 0;
};

// to declare that the engine class will exist
class VulkanEngine;

// implementation of a drawable scene node.
// the scene node can hold children and will also keep a transform to propagate
// to them
struct Node : public IRenderable
{

    // pointer to main engine
    VulkanEngine* engine{nullptr};

    // parent pointer must be a weak pointer to avoid circular dependencies
    std::weak_ptr<Node> parent;
    std::vector<std::shared_ptr<Node>> children;

    glm::mat4 localTransform{1.0f};
    glm::mat4 worldTransform{1.0f};

    void refresh_transform(const glm::mat4& parentMatrix)
    {
        worldTransform = parentMatrix * localTransform;
        for (auto c : children) {
            c->refresh_transform(worldTransform);
        }
    }

    virtual void draw(const glm::mat4& topMatrix, DrawContext& ctx)
    {
        // draw children
        for (auto& c : children) {
            c->draw(topMatrix, ctx);
        }
    }
};
#define VK_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        VkResult err = x;                                                                                              \
        if (err) {                                                                                                     \
            fmt::print("Detected Vulkan error: {}", string_VkResult(err));                                             \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
