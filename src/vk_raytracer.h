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

    std::vector<AccelKHR> m_blas;  // Bottom-level acceleration structure
    AccelKHR              m_tlas;  // Top-level acceleration structure

    VulkanRayTracer(VulkanEngine* engine);

    // BLAS Creation

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

    // TLAS Creation

    void createTopLevelAS();

    VkTransformMatrixKHR toTransformMatrixKHR(glm::mat4 matrix);

    VkDeviceAddress getBlasDeviceAddress(uint32_t blasId);

    void buildTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances,
        VkBuildAccelerationStructureFlagsKHR flags, bool update, bool motion);
};