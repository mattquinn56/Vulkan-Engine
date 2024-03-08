#pragma once

#include "vk_engine.h"

struct BlasInput
{
    // Data used to build acceleration structure geometry
    std::vector<VkAccelerationStructureGeometryKHR>       asGeometry;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
    VkBuildAccelerationStructureFlagsKHR                  flags{ 0 };
};

class VulkanRayTracer {
public:

    VulkanEngine* engine;

    VulkanRayTracer(VulkanEngine* engine);

    void init();

    BlasInput objectToVkGeometryKHR(const RenderObject object);

    void createBottomLevelAS();

    void buildBlas(const std::vector<BlasInput>& input, VkBuildAccelerationStructureFlagsKHR flags);
};