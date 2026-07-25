
#pragma once

#include "core/gpu_types.h"

#include <deque>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <vk_mem_alloc.h>

#include "scene/camera.h"
#include "gpu/descriptor_alloc.h"
#include "scene/gltf_import.h"
#include "gpu/shader_module.h"

struct MeshResource;

class VulkanRayTracer;

namespace fastgltf {
struct Mesh;
}

struct CleanupQueue
{
    std::deque<std::function<void()>> callbacks;

    void push_function(std::function<void()>&& function) {
        callbacks.push_back(function);
    }

    void flush() {
        // Reverse order, so resources are destroyed before what they depend on.
        for (auto it = callbacks.rbegin(); it != callbacks.rend(); it++) {
            (*it)();
        }

        callbacks.clear();
    }
};

struct GeometryInstance
{
    uint32_t indexCount{0};
    uint32_t firstIndex{0};
    VkBuffer indexBuffer{VK_NULL_HANDLE};

    Bounds bounds{};
    glm::mat4 transform{1.0f};
    VkBuffer vertexBuffer{VK_NULL_HANDLE};
    VkDeviceAddress vertexBufferAddress{0};
    int vertexCount{0};

    uint32_t blasIndex{0};
};

struct FrameContext
{
    VkSemaphore _swapchainSemaphore{VK_NULL_HANDLE};
    VkSemaphore _renderSemaphore{VK_NULL_HANDLE};
    VkFence _renderFence{VK_NULL_HANDLE};

    DescriptorAllocatorGrowable _frameDescriptors;
    CleanupQueue _deletionQueue;

    VkCommandPool _commandPool{VK_NULL_HANDLE};
    VkCommandBuffer _mainCommandBuffer{VK_NULL_HANDLE};
};

constexpr unsigned int FRAME_OVERLAP = 2;

// Per-object buffer addresses, uploaded as _objectDescriptionBuffer so the
// closest-hit shader can reach geometry the ray hit.
struct ObjDesc
{
    uint64_t vertexAddress;
    uint64_t indexAddress;
    uint64_t materialAddress;
};

// Alpha-blended surfaces are excluded entirely; the ray tracer traces only
// what lands in opaqueSurfaces, and objectDescriptions runs parallel to it.
struct SceneDrawList
{
    std::vector<GeometryInstance> opaqueSurfaces;
    std::vector<ObjDesc> objectDescriptions;
};

struct EngineStats
{
    float frameTime{0.0f};
};

struct MeshNode : public Node
{

    std::shared_ptr<MeshResource> mesh;

    virtual void draw(const glm::mat4& topMatrix, SceneDrawList& ctx) override;
};

// Homogeneous medium coefficients plus ray-march settings. Mirrors the Medium
// uniform block in shaders/raycommon.glsl.
struct GPUMediumParams
{
    glm::vec4 sigma_a_step{};       // xyz = sigma_a, w = stepSize
    glm::vec4 sigma_s_maxT{};       // xyz = sigma_s, w = maxT
    glm::vec4 g_emis_density_pad{}; // x = g, y = emission, z = densityScale, w = fogEnvFlag (1=affect env, 0=skip)
};

// Optional 3D density texture and its sampler, plus the medium params buffer.
struct VolumeResources
{
    AllocatedImage densityTex3D; // R16F or R8_UNORM or R32F depending on memory
    VkSampler densitySampler{VK_NULL_HANDLE};
    AllocatedBuffer mediumParams; // sizeof(GPUMediumParams)
    bool hasDensity{false};
};

// Release runs without validation: it costs frame time and would require the
// Vulkan SDK layers on the target machine.
#ifdef NDEBUG
inline constexpr bool bUseValidationLayers = false;
#else
inline constexpr bool bUseValidationLayers = true;
#endif

class RtEngine
{
  public:
    bool _isInitialized{false};
    std::vector<const char*> _deviceExtensions{VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                               VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                                               VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    bool _accelerationStructuresCreated{false};
    int _frameNumber{0};
    int _monteCarloSamples{0};
    int _msaaSamples{1};
    bool _debugEnabled{false};

    // Previous-frame values, used to detect setting changes. Not UI-controlled.
    int _lastMonteCarlo{-1};
    int _lastMsaaSamples{-1};

    VkExtent2D _windowExtent{1250, 800};

    std::string _structurePath;
    std::string _lightPath;
    std::string _environmentMapPath;

    struct SDL_Window* _window{nullptr};

    VkInstance _instance{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT _debugMessenger{VK_NULL_HANDLE};
    VkPhysicalDevice _chosenGPU{VK_NULL_HANDLE};
    VkDevice _device{VK_NULL_HANDLE};

    VkQueue _graphicsQueue{VK_NULL_HANDLE};
    uint32_t _graphicsQueueFamily{0};

    AllocatedBuffer _objectDescriptionBuffer;
    AllocatedBuffer _lightBuffer;
    int _lightCount{0};

    FrameContext _frames[FRAME_OVERLAP];

    VkSurfaceKHR _surface{VK_NULL_HANDLE};
    VkSwapchainKHR _swapchain{VK_NULL_HANDLE};
    VkFormat _swapchainImageFormat{VK_FORMAT_UNDEFINED};

    DescriptorAllocator _globalDescriptorAllocator;

    VulkanRayTracer* _rayTracer{nullptr};

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;

    CleanupQueue _mainDeletionQueue;

    VmaAllocator _allocator{VK_NULL_HANDLE};

    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout{VK_NULL_HANDLE};
    VkDescriptorSet _globalDescriptor{VK_NULL_HANDLE};

    VkDescriptorSetLayout _objDescLayout{VK_NULL_HANDLE};
    VkDescriptorSet _objDescSet{VK_NULL_HANDLE};

    // draw resources
    AllocatedImage _drawImage;

    // immediate submit structures
    VkFence _immFence{VK_NULL_HANDLE};
    VkCommandBuffer _immCommandBuffer{VK_NULL_HANDLE};
    VkCommandPool _immCommandPool{VK_NULL_HANDLE};

    AllocatedImage _whiteImage;
    AllocatedImage _errorCheckerboardImage;
    AllocatedImage _environmentMap;

    VkSampler _defaultSamplerLinear{VK_NULL_HANDLE};
    VkSampler _defaultSamplerNearest{VK_NULL_HANDLE};

    SceneDrawList _drawContext;

    GPUFrameConstants _sceneData;

    Camera _mainCamera;

    EngineStats _stats;

    // Volumetrics
    VkDescriptorSetLayout _volumeSetLayout{VK_NULL_HANDLE};
    VkDescriptorSet _volumeSet{VK_NULL_HANDLE};
    VolumeResources _volume{};

    // Singleton accessor; multiple engine instances are not supported.
    static RtEngine& get();

    void init();

    void cleanup();

    void draw();
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
    void draw_ui();

    // Diagnostic capture. When _screenshotPath is set, the frame numbered
    // _screenshotFrame is written to disk and the engine then exits.
    std::string _screenshotPath;
    int _screenshotFrame{30};
    bool _screenshotDone{false};
    bool _showUi{true};
    void capture_swapchain(VkCommandBuffer cmd, uint32_t imageIndex, AllocatedBuffer& dst);
    void write_capture(const AllocatedBuffer& src);

    void update_global_descriptor();

    void run();

    void update_scene();

    // Uploads a mesh into a device-local index/vertex buffer pair.
    GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    FrameContext& get_current_frame();

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
                                bool mipmapped = false);

    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

    std::unordered_map<std::string, std::shared_ptr<GltfScene>> _loadedScenes;

    void destroy_image(const AllocatedImage& img);
    void destroy_buffer(const AllocatedBuffer& buffer);

    bool _resizeRequested{false};
    bool _renderingFrozen{false};

    VkDeviceAddress get_buffer_device_address(VkDevice device, VkBuffer buffer);

    AllocatedBuffer create_buffer_data(VkDeviceSize size, const void* data, VkBufferUsageFlags usage,
                                       const VmaMemoryUsage memUsage);

    AllocatedImage load_image_from_file(std::string path);

    // antialiasing
    enum class AAMode : int
    {
        AdaptiveMSAA = 0,
        TAA = 1
    };
    AAMode _aaMode{AAMode::TAA};
    float _taaAlpha{0.99f}; // history weight
    float _taaClamp{0.10f}; // neighborhood clamps

    float _taaMovingAlpha{0.0f};          // alpha when moving (0 = full reset behavior)
    float _taaVelocityThreshold{0.0001f}; // world units / frame
    float _taaRotationThreshold{0.1f};    // degrees / frame
    bool _taaInitialized{false};

    bool _cameraMoving{false};
    glm::vec3 _prevCamPos{};
    glm::vec3 _prevViewDir{};
    bool _hasPrevCamera{false};

    // TAA GPU resources
    AllocatedImage _taaHistory[2];
    int _taaIndex{0};
    VkDescriptorSetLayout _taaSetLayout{VK_NULL_HANDLE};
    VkPipelineLayout _taaPipelineLayout{VK_NULL_HANDLE};
    VkPipeline _taaPipeline{VK_NULL_HANDLE};
    VkDescriptorSet _taaSet[2]{}; // 2 sets for ping-pong

    // helpers
    void create_taa_pipeline_resources();
    void create_taa_history_images();
    void destroy_taa_history_images();

    // Progressive Monte Carlo accumulation
    bool _progressiveMonteCarlo{true};
    int _monteCarloSamplesPerFrame{5};
    int _monteCarloResetFrames{2}; // frames of motion before history is cleared

    AllocatedImage _mcAccumColor; // rgba16f, running average
    AllocatedImage _mcAccumCount; // r32ui, sample counts

    VkDescriptorSetLayout _mcSetLayout{VK_NULL_HANDLE};
    VkPipelineLayout _mcPipeLayout{VK_NULL_HANDLE};
    VkPipeline _mcPipeline{VK_NULL_HANDLE};
    VkDescriptorSet _mcSet{VK_NULL_HANDLE};

    // helpers
    void create_monte_carlo_pipeline_resources();
    void create_monte_carlo_images();
    void destroy_monte_carlo_images();
    void reset_monte_carlo_history(VkCommandBuffer cmd);

    // Post-tonemap (ACES + sRGB) pass
    VkDescriptorSetLayout _postSetLayout{VK_NULL_HANDLE};
    VkPipelineLayout _postPipeLayout{VK_NULL_HANDLE};
    VkPipeline _postPipeline{VK_NULL_HANDLE};
    VkDescriptorSet _postSet{VK_NULL_HANDLE};
    bool _enableTonemap{true};
    bool _ldrNeedsInit{true}; // _ldrImage still needs its first layout transition
    float _exposure{1.0f};

    // LDR target copied to the swapchain after tonemapping.
    AllocatedImage _ldrImage;
    void create_postprocess_resources();

    // False falls back to the legacy BRDF; see useMicrofacet in the RT push constants.
    bool _useMicrofacetBrdf{true};

    // Forces MC and TAA history to be discarded on the next draw().
    void request_accum_reset();
    bool _resetAccumNextFrame{false};

    // Volumetrics
    void set_medium_params(const GPUMediumParams& p);

  private:
    void init_vulkan();

    void init_raytracing();

    void init_swapchain();

    void create_render_targets();

    void destroy_render_targets();

    void create_swapchain(uint32_t width, uint32_t height);

    bool resize_swapchain();

    void destroy_swapchain();

    void init_commands();

    void init_pipelines();

    void init_descriptors();

    void init_sync_structures();

    void init_renderables();

    void init_lights();

    void init_imgui();

    void init_default_data();

    void render_loaded_gltf(std::shared_ptr<GltfScene> scene);

    void recursively_render_node(std::shared_ptr<GltfScene> scene, std::shared_ptr<Node> node);

    // volumetric additions
    void create_volume_resources();
    void initialize_default_medium();
    void upload_volume_density(const void* voxels, VkExtent3D extent, VkFormat fmt);
};

// Seeds the TAA history from the current frame so blending starts clean.
void seed_taa_history(RtEngine* e, VkCommandBuffer cmd);
