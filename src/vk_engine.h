// or project specific include files.

#pragma once

#include <vk_types.h>

#include <deque>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <vk_mem_alloc.h>

#include <camera.h>
#include <vk_descriptors.h>
#include <vk_loader.h>
#include <vk_pipelines.h>

struct MeshAsset;

class VulkanRayTracer;

namespace fastgltf {
struct Mesh;
}

struct DeletionQueue
{
    std::deque<std::function<void()>> callbacks;

    void push_function(std::function<void()>&& function)
    {
        callbacks.push_back(function);
    }

    void flush()
    {
        // reverse iterate the deletion queue to execute all the functions
        for (auto it = callbacks.rbegin(); it != callbacks.rend(); it++) {
            (*it)(); // call functors
        }

        callbacks.clear();
    }
};

struct ComputePushConstants
{
    glm::vec4 data1{};
    glm::vec4 data2{};
    glm::vec4 data3{};
    glm::vec4 data4{};
};

struct ComputeEffect
{
    const char* name{nullptr};

    VkPipeline pipeline{VK_NULL_HANDLE};
    VkPipelineLayout layout{VK_NULL_HANDLE};

    ComputePushConstants data;
};

struct RenderObject
{
    uint32_t indexCount{0};
    uint32_t firstIndex{0};
    VkBuffer indexBuffer{VK_NULL_HANDLE};

    MaterialInstance* material{nullptr};
    Bounds bounds{};
    glm::mat4 transform{1.0f};
    VkBuffer vertexBuffer{VK_NULL_HANDLE};
    VkDeviceAddress vertexBufferAddress{0};
    int vertexCount{0};

    uint32_t blasIndex{0};
};

struct FrameData
{
    VkSemaphore _swapchainSemaphore{VK_NULL_HANDLE};
    VkSemaphore _renderSemaphore{VK_NULL_HANDLE};
    VkFence _renderFence{VK_NULL_HANDLE};

    DescriptorAllocatorGrowable _frameDescriptors;
    DeletionQueue _deletionQueue;

    VkCommandPool _commandPool{VK_NULL_HANDLE};
    VkCommandBuffer _mainCommandBuffer{VK_NULL_HANDLE};
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct ObjDesc
{
    // a buffer with a vector of these objects `_objectDescriptionBuffer` will be passed to the ray tracing closest hit shader
    uint64_t vertexAddress;   // Address of the vertex buffer
    uint64_t indexAddress;    // Address of the index buffer
    uint64_t materialAddress; // Address of the material buffer
};

struct DrawContext
{
    // Only drawing + using RT for opaque surfaces
    std::vector<RenderObject> opaqueSurfaces;
    std::vector<ObjDesc> objectDescriptions; // Model descriptions for device access by opaque surfaces.
    std::vector<RenderObject> TransparentSurfaces;
};

struct EngineStats
{
    float frameTime{0.0f};
    int triangleCount{0};
    int drawCallCount{0};
    float meshDrawTime{0.0f};
};

struct GLTFMetallic_Roughness
{
    MaterialPipeline opaquePipeline;
    MaterialPipeline transparentPipeline;

    VkDescriptorSetLayout materialLayout{VK_NULL_HANDLE};

    struct MaterialConstants
    {
        glm::vec4 colorFactors{};
        glm::vec4 metalRoughFactors{};
        //padding, we need it anyway for uniform buffers
        glm::vec4 extra[14]{};
    };

    struct MaterialResources
    {
        AllocatedImage colorImage;
        VkSampler colorSampler{VK_NULL_HANDLE};
        AllocatedImage metalRoughImage;
        VkSampler metalRoughSampler{VK_NULL_HANDLE};
        VkBuffer dataBuffer{VK_NULL_HANDLE};
        uint32_t dataBufferOffset{0};
    };

    DescriptorWriter writer;

    void build_pipelines(VulkanEngine* engine);
    MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources,
                                    DescriptorAllocatorGrowable& descriptorAllocator);
};

struct MeshNode : public Node
{

    std::shared_ptr<MeshAsset> mesh;

    virtual void draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

// volumetric additions
// Medium parameters for a homogeneous base + settings controlling ray marching
struct GPUMediumParams
{
    glm::vec4 sigma_a_step{};       // xyz = sigma_a, w = stepSize
    glm::vec4 sigma_s_maxT{};       // xyz = sigma_s, w = maxT
    glm::vec4 g_emis_density_pad{}; // x = g, y = emission, z = densityScale, w = fogEnvFlag (1=affect env, 0=skip)
};

// Volume resources: optional 3D density + sampler + params buffer
struct VolumeResources
{
    AllocatedImage densityTex3D; // R16F or R8_UNORM or R32F depending on memory
    VkSampler densitySampler{VK_NULL_HANDLE};
    AllocatedBuffer mediumParams; // sizeof(GPUMediumParams)
    bool hasDensity{false};
};

class VulkanEngine
{
  public:
    bool _isInitialized{false};
    std::vector<const char*> _deviceExtensions{VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                               VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                                               VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    bool _accelerationStructuresCreated{false};
    int _frameNumber{0};
    bool _useRayTracing{true};
    int _monteCarloSamples{0};
    int _msaaSamples{1};
    bool _debugEnabled{false};

    int _lastMonteCarlo{-1};  // Not controlled by the UI.
    int _lastMsaaSamples{-1}; // not controlled by UI

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

    AllocatedBuffer _defaultGLTFMaterialData;
    AllocatedBuffer _objectDescriptionBuffer;
    AllocatedBuffer _lightBuffer;
    int _lightCount{0};

    FrameData _frames[FRAME_OVERLAP];

    VkSurfaceKHR _surface{VK_NULL_HANDLE};
    VkSwapchainKHR _swapchain{VK_NULL_HANDLE};
    VkFormat _swapchainImageFormat{VK_FORMAT_UNDEFINED};

    VkDescriptorPool _descriptorPool{VK_NULL_HANDLE};

    DescriptorAllocator _globalDescriptorAllocator;

    VulkanRayTracer* _rayTracer{nullptr};

    VkPipeline _gradientPipeline{VK_NULL_HANDLE};
    VkPipelineLayout _gradientPipelineLayout{VK_NULL_HANDLE};

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;

    std::vector<VkSemaphore> _imageAcquireSems;
    std::vector<VkSemaphore> _imageRenderSems;

    VkDescriptorSet _drawImageDescriptors{VK_NULL_HANDLE};
    VkDescriptorSetLayout _drawImageDescriptorLayout{VK_NULL_HANDLE};

    DeletionQueue _mainDeletionQueue;

    VmaAllocator _allocator{VK_NULL_HANDLE}; // vma lib allocator

    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout{VK_NULL_HANDLE};
    VkDescriptorSet _globalDescriptor{VK_NULL_HANDLE};

    VkDescriptorSetLayout _objDescLayout{VK_NULL_HANDLE};
    VkDescriptorSet _objDescSet{VK_NULL_HANDLE};

    GLTFMetallic_Roughness _metalRoughMaterial;

    // draw resources
    AllocatedImage _drawImage;
    AllocatedImage _depthImage;

    // immediate submit structures
    VkFence _immFence{VK_NULL_HANDLE};
    VkCommandBuffer _immCommandBuffer{VK_NULL_HANDLE};
    VkCommandPool _immCommandPool{VK_NULL_HANDLE};

    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;
    AllocatedImage _environmentMap;

    VkSampler _defaultSamplerLinear{VK_NULL_HANDLE};
    VkSampler _defaultSamplerNearest{VK_NULL_HANDLE};

    GPUMeshBuffers _defaultRectangle;
    DrawContext _drawContext;

    GPUSceneData _sceneData;

    Camera _mainCamera;

    EngineStats _stats;

    // some volumetric additions
    VkDescriptorSetLayout _volumeSetLayout{VK_NULL_HANDLE};
    VkDescriptorSet _volumeSet{VK_NULL_HANDLE};
    VolumeResources _volume{};

    std::vector<ComputeEffect> _backgroundEffects;
    int _currentBackgroundEffect{0};

    // singleton style getter.multiple engines is not supported
    static VulkanEngine& get();

    // initializes everything in the engine
    void init();

    // checks that the needed extensions are available (currently unused)
    void check_extensions();

    // shuts down the engine
    void cleanup();

    // draw loop
    void draw();
    void draw_main(VkCommandBuffer cmd);
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);

    void render_nodes();

    void update_global_descriptor();
    void draw_geometry(VkCommandBuffer cmd);

    // run main loop
    void run();

    void update_scene();

    // upload a mesh into a pair of gpu buffers. If descriptor allocator is not
    // null, it will also create a descriptor that points to the vertex buffer
    GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    FrameData& get_current_frame();
    FrameData& get_last_frame();

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
                                bool mipmapped = false);

    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

    std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> _loadedScenes;
    std::vector<std::shared_ptr<LoadedGLTF>> _brickadiaScene;

    void destroy_image(const AllocatedImage& img);
    void destroy_buffer(const AllocatedBuffer& buffer);

    bool _resizeRequested{false};
    bool _renderingFrozen{false};

    VkDeviceAddress get_buffer_device_address(VkDevice device, VkBuffer buffer);

    AllocatedBuffer create_buffer_data(VkDeviceSize size, const void* data, VkBufferUsageFlags usage,
                                       const VmaMemoryUsage memUsage);

    AllocatedBuffer allocate_and_bind_buffer(VkBuffer buffer, VmaMemoryUsage memoryUsage);

    AllocatedImage load_image_from_file(std::string path);

    // antialiasing
    enum class AAMode : int
    {
        None = 0,
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

    // progressive mc things
    bool _progressiveMonteCarlo{true}; // enable progressive MC accumulation
    int _monteCarloSamplesPerFrame{5}; // Samples per pixel per frame.
    int _monteCarloResetFrames{2};     // clear history this many frames after motion

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
    bool _enableTonemap{true}; // default ON
    bool _ldrNeedsInit{true};  // first-use transition
    float _exposure{1.0f};

    // LDR target copied to the swapchain after tonemapping.
    AllocatedImage _ldrImage;
    void create_postprocess_resources();

    // Microfacet addition
    bool _useMicrofacetBrdf{true};

    // Force reset of MC/TAA history on the next draw()
    void request_accum_reset();
    bool _resetAccumNextFrame{false};

    // volumetric additions
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
    void init_background_pipelines();

    void init_descriptors();

    void init_sync_structures();

    void init_renderables();

    void init_lights();

    void init_imgui();

    void init_default_data();

    void render_loaded_gltf(std::shared_ptr<LoadedGLTF> scene);

    void recursively_render_node(std::shared_ptr<LoadedGLTF> scene, std::shared_ptr<Node> node);

    // volumetric additions
    void create_volume_resources();
    void initialize_default_medium();
    void upload_volume_density(const void* voxels, VkExtent3D extent, VkFormat fmt);
};
