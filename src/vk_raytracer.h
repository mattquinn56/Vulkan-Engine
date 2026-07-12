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
    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo;
    AccelKHR as; // result acceleration structure
    AccelKHR cleanupAS;
};

class VulkanRayTracer
{
  public:
    // pointer to main engine
    VulkanEngine* _engine;

    // pointers to extension functions
    PFN_vkGetAccelerationStructureBuildSizesKHR _vkGetAccelerationStructureBuildSizes;
    PFN_vkCmdBuildAccelerationStructuresKHR _vkCmdBuildAccelerationStructures;
    PFN_vkCmdCopyAccelerationStructureKHR _vkCmdCopyAccelerationStructure;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR _vkCmdWriteAccelerationStructuresProperties;
    PFN_vkCreateAccelerationStructureKHR _vkCreateAccelerationStructure;
    PFN_vkDestroyAccelerationStructureKHR _vkDestroyAccelerationStructure;
    PFN_vkGetAccelerationStructureDeviceAddressKHR _vkGetAccelerationStructureDeviceAddress;
    PFN_vkCreateRayTracingPipelinesKHR _vkCreateRayTracingPipelines;
    PFN_vkGetRayTracingShaderGroupHandlesKHR _vkGetRayTracingShaderGroupHandles;
    PFN_vkCmdTraceRaysKHR _vkCmdTraceRays;

    VkPhysicalDeviceAccelerationStructurePropertiesKHR _accelerationStructureProperties;

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
    VkDescriptorPool _descriptorPool;
    VkDescriptorSetLayout _descriptorSetLayout;
    VkDescriptorSet _descriptorSet;

    void update_output_descriptor();

    void create_pipeline();

    const int MAX_RAY_RECURSION_DEPTH = 4;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> _shaderGroups;
    VkPipelineLayout _pipelineLayout;
    VkPipeline _pipeline;

    // Push constant structure for the ray tracer
    struct PushConstantRay
    {
        glm::vec4 clearColor;
        uint64_t lightAddress;
        int numLights;
        int useMicrofacet; // 0 = legacy, 1 = GGX+Smith+Schlick
    };

    // Push constant for ray tracer
    PushConstantRay _pushConstants{};

    void create_shader_binding_table();

    AllocatedBuffer _shaderBindingTableBuffer;
    VkStridedDeviceAddressRegionKHR _rayGenerationRegion{};
    VkStridedDeviceAddressRegionKHR _missRegion{};
    VkStridedDeviceAddressRegionKHR _hitRegion{};
    VkStridedDeviceAddressRegionKHR _callableRegion{};

    struct MaterialRT
    {
        glm::vec4 colorFactors;
        glm::vec4 metalRoughFactors;
        int textureID;
    };

    std::vector<VkImageView> _colorTextures;
    std::vector<VkImageView> _metalRoughTextures;
    std::vector<VkSampler> _colorSamplers;
    std::vector<VkSampler> _metalRoughSamplers;

    DescriptorAllocatorGrowable _materialDescriptorAllocator;
    DescriptorWriter _materialDescriptorWriter;
    VkDescriptorPool _materialDescriptorPool;
    VkDescriptorSetLayout _materialDescriptorSetLayout;
    VkDescriptorSet _materialDescriptorSet;

    VkDeviceAddress upload_material(MaterialRT mat);

    void create_material_descriptor_set();

    void raytrace(const VkCommandBuffer& cmdBuf);
};