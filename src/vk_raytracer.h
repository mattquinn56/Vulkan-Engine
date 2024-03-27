#pragma once

#include "vk_engine.h"

struct BlasInput
{
    // Data used to build acceleration structure geometry
    std::vector<VkAccelerationStructureGeometryKHR>       asGeometry;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
    VkBuildAccelerationStructureFlagsKHR                  flags{ 0 };
};

struct AccelKHR
{
    VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
    AllocatedBuffer            buffer;
};

struct PaddedIndex {
    glm::ivec3 index; // Original index data
    int pad;   // Padding to ensure 16-byte alignment
};

struct BuildAccelerationStructure
{
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo;
    AccelKHR                                  as;  // result acceleration structure
    AccelKHR                                  cleanupAS;
};

class VulkanRayTracer {
public:

    // pointer to main engine
    VulkanEngine* engine;

    // pointers to extension functions
    PFN_vkGetAccelerationStructureBuildSizesKHR pfnGetAccelerationStructureBuildSizesKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR pfnCmdBuildAccelerationStructuresKHR;
    PFN_vkCmdCopyAccelerationStructureKHR pfnCmdCopyAccelerationStructureKHR;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR pfnCmdWriteAccelerationStructuresPropertiesKHR;
    PFN_vkCreateAccelerationStructureKHR pfnCreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR pfnDestroyAccelerationStructureKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfnGetAccelerationStructureDeviceAddressKHR;
    PFN_vkCreateRayTracingPipelinesKHR pfnCreateRayTracingPipelinesKHR;
    PFN_vkGetRayTracingShaderGroupHandlesKHR pfnGetRayTracingShaderGroupHandlesKHR;
    PFN_vkCmdTraceRaysKHR pfnCmdTraceRaysKHR;

    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties;

    std::vector<AccelKHR> m_blas;  // Bottom-level acceleration structure
    AccelKHR              m_tlas;  // Top-level acceleration structure

    VulkanRayTracer(VulkanEngine* engine);
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };

    //-------------------- BLAS Creation --------------------//

    BlasInput objectToVkGeometryKHR(const RenderObject object);

    void createBottomLevelAS();

    AccelKHR createAcceleration(VkAccelerationStructureCreateInfoKHR& accel_);

    void cmdCreateBlas(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs,
        VkDeviceAddress scratchAddress, VkQueryPool queryPool);

    void cmdCompactBlas(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs,
        VkQueryPool queryPool);

    void destroyNonCompacted(std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs);

    void buildBlas(const std::vector<BlasInput>& input, VkBuildAccelerationStructureFlagsKHR flags);

    bool hasFlag(VkFlags item, VkFlags flag) { return (item & flag) == flag; }

    //-------------------- TLAS Creation --------------------//

    void createTopLevelAS();

    VkTransformMatrixKHR toTransformMatrixKHR(glm::mat4 matrix);

    VkDeviceAddress getBlasDeviceAddress(uint32_t blasId);

    void buildTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances,
        VkBuildAccelerationStructureFlagsKHR flags, bool update, bool motion);

    void cmdCreateTlas(VkCommandBuffer cmdBuf, uint32_t countInstance, VkDeviceAddress instBufferAddr,
        AllocatedBuffer& scratchBuffer, VkBuildAccelerationStructureFlagsKHR flags, bool update, bool motion);

    //-------------------- Ray Tracing Descriptor Set Creation --------------------//

    void createRtDescriptorSet();
    DescriptorAllocator         m_rtDescAllocator;
    DescriptorWriter            m_rtDescWriter;
    VkDescriptorPool            m_rtDescPool;
    VkDescriptorSetLayout       m_rtDescSetLayout;
    VkDescriptorSet             m_rtDescSet;

    void updateRtDescriptorSet();

    //-------------------- Ray Tracing Pipeline Creation --------------------//

    void createRtPipeline();

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
    VkPipelineLayout m_rtPipelineLayout;
    VkPipeline m_rtPipeline;

    // Push constant structure for the ray tracer
    struct PushConstantRay
    {
        glm::vec4 clearColor;
    };

    // Push constant for ray tracer
    PushConstantRay m_pcRay{};

    //-------------------- Binding Table Creation --------------------//

    void createRtShaderBindingTable();

    AllocatedBuffer m_rtSBTBuffer;
    VkStridedDeviceAddressRegionKHR m_rgenRegion{};
    VkStridedDeviceAddressRegionKHR m_missRegion{};
    VkStridedDeviceAddressRegionKHR m_hitRegion{};
    VkStridedDeviceAddressRegionKHR m_callRegion{};

    //-------------------- Ray Tracing Computation --------------------//

    void raytrace(const VkCommandBuffer& cmdBuf);
};