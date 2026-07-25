#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <deque>
#include <source_location>
#include <string_view>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

// Core types shared across the renderer.
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

// Mirrors RenderLight in shaders/raycommon.glsl. The w channels are packed
// parameters, not padding.
struct RenderLight
{
    glm::vec4 position{}; // xyz direction (directional) or third corner (area), w intensity
    glm::vec4 color{};    // w type: 0 point, 1 ambient, 2 directional, 3 area
    glm::vec4 v0{};       // remaining area-light corners, unused by other types
    glm::vec4 v1{};
};

// Mirrors SceneData in shaders/raycommon.glsl.
struct GPUFrameConstants
{
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::mat4 viewproj{1.0f};
    glm::vec4 data{}; // x accumulated frame count, y sampling enable flag
};
// Alpha-blended surfaces are excluded from the traced scene, so this only
// distinguishes "blended" from everything else.
enum class SurfaceAlphaMode : uint8_t
{
    MainColor,
    Transparent,
    Other
};
// UVs are split around the vec3s so each pair fills one 16-byte std430 slot
// with no padding. Mirrored in shaders/mesh.vert and shaders/raycommon.glsl.
struct Vertex
{
    glm::vec3 position{};
    float uvX{0.0f};
    glm::vec3 normal{};
    float uvY{0.0f};
    glm::vec4 color{};
};

// GPU-side buffers backing a single mesh.
struct GPUMeshBuffers
{
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress{0};
    int vertexCount{0};
};

struct SceneDrawList;

// Base class for anything that can submit draws.
class IRenderable
{

    virtual void draw(const glm::mat4& topMatrix, SceneDrawList& ctx) = 0;
};

class RtEngine;

// Scene graph node. Holds a local transform that is composed with its parent's
// and propagated down to children.
struct Node : public IRenderable
{
    RtEngine* engine{nullptr};

    // Weak, so a parent holding its children shared does not form a cycle.
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

    virtual void draw(const glm::mat4& topMatrix, SceneDrawList& ctx)
    {
        for (auto& c : children) {
            c->draw(topMatrix, ctx);
        }
    }
};
inline void check_vk_result(VkResult result, std::string_view expression,
                            std::source_location location = std::source_location::current())
{
    if (result == VK_SUCCESS) {
        return;
    }

    fmt::print(stderr, "Vulkan call failed: {} returned {} at {}:{}\n", expression, string_VkResult(result),
               location.file_name(), location.line());
    std::abort();
}

#define VK_CHECK(expression) check_vk_result((expression), #expression)
