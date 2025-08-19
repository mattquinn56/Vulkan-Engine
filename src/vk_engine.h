// vulkan_guide.h : Include file for standard system include files,
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

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()>&& function)
    {
        deletors.push_back(function);
    }

    void flush()
    {
        // reverse iterate the deletion queue to execute all the functions
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)(); // call functors
        }

        deletors.clear();
    }
};

struct ComputePushConstants {
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeEffect {
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	ComputePushConstants data;
};

struct RenderObject {
    uint32_t indexCount;
    uint32_t firstIndex;
    VkBuffer indexBuffer;
    
    MaterialInstance* material;
    Bounds bounds;
    glm::mat4 transform;
    VkBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress;
    int vertexCount;

    uint32_t blasIndex{ 0 };
};

struct FrameData {
	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;

    DescriptorAllocatorGrowable _frameDescriptors;
    DeletionQueue _deletionQueue;

    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct ObjDesc {
    // a buffer with a vector of these objects `m_bObjDesc` will be passed to the ray tracing closest hit shader
    uint64_t vertexAddress; // Address of the vertex buffer
    uint64_t indexAddress; // Address of the index buffer
    uint64_t materialAddress; // Address of the material buffer
};

struct DrawContext {
    // Only drawing + using RT for opaque surfaces
    std::vector<RenderObject> OpaqueSurfaces;
    std::vector<ObjDesc> m_objDesc; // Model description for device access, opaque surfaces
    std::vector<RenderObject> TransparentSurfaces;
};

struct EngineStats {
    float frametime;
    int triangle_count;
    int drawcall_count;
    float mesh_draw_time;
};

struct GLTFMetallic_Roughness {
    MaterialPipeline opaquePipeline;
    MaterialPipeline transparentPipeline;

    VkDescriptorSetLayout materialLayout;

    struct MaterialConstants {
		glm::vec4 colorFactors;
		glm::vec4 metal_rough_factors;
        //padding, we need it anyway for uniform buffers
		glm::vec4 extra[14];
    };

    struct MaterialResources {
        AllocatedImage colorImage; 
        VkSampler colorSampler;
        AllocatedImage metalRoughImage;
        VkSampler metalRoughSampler;
        VkBuffer dataBuffer; 
        uint32_t dataBufferOffset;
    };

    DescriptorWriter writer;

    void build_pipelines(VulkanEngine* engine);
    void clear_resources(VkDevice device);

    MaterialInstance write_material(VkDevice device,MaterialPass pass,const MaterialResources& resources , DescriptorAllocatorGrowable& descriptorAllocator);
};

struct MeshNode : public Node {

	std::shared_ptr<MeshAsset> mesh;

	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

// volumetric additions
// Medium parameters for a homogeneous base + settings controlling ray marching
struct GPUMediumParams {
    glm::vec3 sigma_a;   float stepSize;   // absorption, world units step
    glm::vec3 sigma_s;   float maxT;      // scattering, world-space march cap
    float g;             float emission;  // Henyey–Greenstein anisotropy, simple emission
    float densityScale;  float padding0;  // scales texture density into sigma_t
    glm::vec2 padding1;
};

// Volume resources: optional 3D density + sampler + params buffer
struct VolumeResources {
    AllocatedImage densityTex3D;  // R16F or R8_UNORM or R32F depending on memory
    VkSampler      densitySampler;
    AllocatedBuffer mediumParams; // sizeof(GPUMediumParams)
    bool hasDensity = false;
};

class VulkanEngine {
public:
    bool _isInitialized { false };
    std::vector<const char*> _deviceExtensions{ 
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, 
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, 
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME 
    };
    bool createdAS{ false };
    int _frameNumber { 0 };
    bool useRaytracer = true;
    int computeMonteCarlo = 0;
    int msaaSetting = 1;
    bool debugSetting = false;

    bool lastMonteCarlo = -1; // not controlled by UI
    int lastMSAA = -1; // not controlled by UI

    VkExtent2D _windowExtent { 1000, 600 };

    std::string structurePath;
    std::string lightPath;
    std::string envMapPath;

    struct SDL_Window* _window { nullptr };

    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debug_messenger;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    AllocatedBuffer _defaultGLTFMaterialData;
    AllocatedBuffer m_bObjDesc;
    AllocatedBuffer m_lightBuffer;
    int m_numLights;

    FrameData _frames[FRAME_OVERLAP];

    VkSurfaceKHR _surface;
    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;

    VkDescriptorPool _descriptorPool;

    DescriptorAllocator globalDescriptorAllocator;

    VulkanRayTracer* rayTracer;

    VkPipeline _gradientPipeline;
    VkPipelineLayout _gradientPipelineLayout;
    
    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;

    std::vector<VkSemaphore> _imageAcquireSems;
    std::vector<VkSemaphore> _imageRenderSems;

	VkDescriptorSet _drawImageDescriptors;
	VkDescriptorSetLayout _drawImageDescriptorLayout;

    DeletionQueue _mainDeletionQueue;

    VmaAllocator _allocator; // vma lib allocator

	VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;
    VkDescriptorSet _globalDescriptor;

    VkDescriptorSetLayout _objDescLayout;
    VkDescriptorSet _objDescSet;

    GLTFMetallic_Roughness metalRoughMaterial;

    // draw resources
    AllocatedImage _drawImage;
    AllocatedImage _depthImage;

    // immediate submit structures
    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

	AllocatedImage _whiteImage;
	AllocatedImage _blackImage;
	AllocatedImage _greyImage;
	AllocatedImage _errorCheckerboardImage;
    AllocatedImage environmentMap;

	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;
	
    GPUMeshBuffers rectangle;
    DrawContext drawCommands;

    GPUSceneData sceneData;

    Camera mainCamera;

    EngineStats stats;

    // some volumetric additions
    VkDescriptorSetLayout _volumeSetLayout = {VK_NULL_HANDLE};
    VkDescriptorSet       _volumeSet = {VK_NULL_HANDLE};
    VolumeResources       _volume{};

	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{ 0 };

    // singleton style getter.multiple engines is not supported
    static VulkanEngine& Get();

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
	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    FrameData& get_current_frame();
    FrameData& get_last_frame();

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage , bool mipmapped = false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);

    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

    std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;
    std::vector<std::shared_ptr<LoadedGLTF>> brickadiaScene;

    void destroy_image(const AllocatedImage& img);
    void destroy_buffer(const AllocatedBuffer& buffer);

    bool resize_requested;
    bool freeze_rendering;

    VkDeviceAddress getBufferDeviceAddress(VkDevice device, VkBuffer buffer);

    AllocatedBuffer create_buffer_data(VkDeviceSize size, const void* data, VkBufferUsageFlags usage, const VmaMemoryUsage memUsage);

    AllocatedBuffer allocateAndBindBuffer(VkBuffer buffer, VmaMemoryUsage memoryUsage);

    AllocatedImage loadImageFromFile(std::string path);

    // volumetric additions
    void setMediumParams(const GPUMediumParams& p);

private:
    void init_vulkan();

    void init_raytracing();

    void init_swapchain();

    void create_swapchain(uint32_t width, uint32_t height);

	void resize_swapchain();

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
    void init_volume_descriptors();
    void create_default_volume();                         // start with homogeneous only
    void upload_volume_3d(const void* voxels, VkExtent3D extent, VkFormat fmt); // later for grids
};
