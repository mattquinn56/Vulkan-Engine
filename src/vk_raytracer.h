#pragma once

#include "vk_engine.h"

struct BlasInput
{
    // Data used to build acceleration structure geometry
    std::vector<VkAccelerationStructureGeometryKHR> asGeometry;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
    VkBuildAccelerationStructureFlagsKHR flags{0};
};

struct AccelKHR
{
    VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
    AllocatedBuffer buffer;
};

struct BuildAccelerationStructure
{
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo{nullptr};
    AccelKHR as; // result acceleration structure
    AccelKHR cleanupAS;
};

class VulkanRayTracer
{
  public:
    // pointer to main engine
    VulkanEngine* _engine{nullptr};

    // pointers to extension functions
    PFN_vkGetAccelerationStructureBuildSizesKHR _vkGetAccelerationStructureBuildSizes{nullptr};
    PFN_vkCmdBuildAccelerationStructuresKHR _vkCmdBuildAccelerationStructures{nullptr};
    PFN_vkCmdCopyAccelerationStructureKHR _vkCmdCopyAccelerationStructure{nullptr};
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR _vkCmdWriteAccelerationStructuresProperties{nullptr};
    PFN_vkCreateAccelerationStructureKHR _vkCreateAccelerationStructure{nullptr};
    PFN_vkDestroyAccelerationStructureKHR _vkDestroyAccelerationStructure{nullptr};
    PFN_vkGetAccelerationStructureDeviceAddressKHR _vkGetAccelerationStructureDeviceAddress{nullptr};
    PFN_vkCreateRayTracingPipelinesKHR _vkCreateRayTracingPipelines{nullptr};
    PFN_vkGetRayTracingShaderGroupHandlesKHR _vkGetRayTracingShaderGroupHandles{nullptr};
    PFN_vkCmdTraceRaysKHR _vkCmdTraceRays{nullptr};

    VkPhysicalDeviceAccelerationStructurePropertiesKHR _accelerationStructureProperties{};

    std::vector<AccelKHR> _bottomLevelStructures; // Bottom-level acceleration structure
    AccelKHR _topLevelStructure;                  // Top-level acceleration structure

    VulkanRayTracer(VulkanEngine* owner);
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR _rayTracingProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

    BlasInput object_to_vk_geometry(const RenderObject object);

    void create_bottom_level_acceleration_structures();

    AccelKHR create_acceleration_structure(VkAccelerationStructureCreateInfoKHR& accel_);

    void record_blas_build(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices,
                           std::vector<BuildAccelerationStructure>& buildAs, VkDeviceAddress scratchAddress,
                           VkQueryPool queryPool);

    void record_blas_compaction(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices,
                                std::vector<BuildAccelerationStructure>& buildAs, VkQueryPool queryPool);

    void destroy_non_compacted_structures(std::vector<uint32_t> indices,
                                          std::vector<BuildAccelerationStructure>& buildAs);

    void build_bottom_level_structures(const std::vector<BlasInput>& input, VkBuildAccelerationStructureFlagsKHR flags);

    bool has_flag(VkFlags item, VkFlags flag)
    {
        return (item & flag) == flag;
    }

    void create_top_level_acceleration_structure();

    VkTransformMatrixKHR to_transform_matrix(glm::mat4 matrix);

    VkDeviceAddress get_blas_device_address(uint32_t blasId);

    void build_top_level_structure(const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                                   VkBuildAccelerationStructureFlagsKHR flags, bool update, bool motion);

    void record_top_level_structure_build(VkCommandBuffer cmdBuf, uint32_t countInstance,
                                          VkDeviceAddress instBufferAddr, AllocatedBuffer& scratchBuffer,
                                          VkBuildAccelerationStructureFlagsKHR flags, bool update, bool motion);

    void create_descriptor_set();

    DescriptorAllocator _descriptorAllocator;
    DescriptorWriter _descriptorWriter;
    VkDescriptorPool _descriptorPool{VK_NULL_HANDLE};
    VkDescriptorSetLayout _descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorSet _descriptorSet{VK_NULL_HANDLE};

    void update_output_descriptor();

    void create_pipeline();

    const int MAX_RAY_RECURSION_DEPTH = 4;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> _shaderGroups;
    VkPipelineLayout _pipelineLayout{VK_NULL_HANDLE};
    VkPipeline _pipeline{VK_NULL_HANDLE};

    // Push constant structure for the ray tracer
    struct PushConstantRay
    {
        glm::vec4 clearColor{};
        uint64_t lightAddress{0};
        int numLights{0};
        int useMicrofacet{0}; // 0 = legacy, 1 = GGX+Smith+Schlick
    };

    // Push constant for ray tracer
    PushConstantRay _pushConstants{};

    void create_shader_binding_table();

    AllocatedBuffer _shaderBindingTableBuffer{};
    VkStridedDeviceAddressRegionKHR _rayGenerationRegion{};
    VkStridedDeviceAddressRegionKHR _missRegion{};
    VkStridedDeviceAddressRegionKHR _hitRegion{};
    VkStridedDeviceAddressRegionKHR _callableRegion{};

    struct MaterialRT
    {
        glm::vec4 colorFactors{};
        glm::vec4 metalRoughFactors{};
        int textureID{0};
    };

    std::vector<VkImageView> _colorTextures;
    std::vector<VkImageView> _metalRoughTextures;
    std::vector<VkSampler> _colorSamplers;
    std::vector<VkSampler> _metalRoughSamplers;

    DescriptorAllocatorGrowable _materialDescriptorAllocator;
    DescriptorWriter _materialDescriptorWriter;
    VkDescriptorPool _materialDescriptorPool{VK_NULL_HANDLE};
    VkDescriptorSetLayout _materialDescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorSet _materialDescriptorSet{VK_NULL_HANDLE};

    VkDeviceAddress upload_material(MaterialRT mat);

    void create_material_descriptor_set();

    void raytrace(const VkCommandBuffer& cmdBuf);
};
