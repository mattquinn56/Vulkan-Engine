#include "vk_raytracer.h"

VulkanRayTracer::VulkanRayTracer(VulkanEngine* engine)
{
	VulkanRayTracer::engine = engine;
}

void VulkanRayTracer::init()
{
    // Looking at the ray tracing extensions
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(engine->_chosenGPU, &prop2);
}

//--------------------------------------------------------------------------------------------------
// Convert an OBJ model into the ray tracing geometry used to build the BLAS
//
BlasInput VulkanRayTracer::objectToVkGeometryKHR(const RenderObject object)
{
    // BLAS builder requires raw device addresses.
    VkDeviceAddress vertexAddress = object.vertexBufferAddress;
    VkDeviceAddress indexAddress = engine->getBufferDeviceAddress(engine->_device, object.indexBuffer);

    uint32_t maxPrimitiveCount = object.indexCount / 3;

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride = sizeof(Vertex);
    // Describe index data (32-bit unsigned int)
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexAddress;
    // Indicate identity transform by setting transformData to null device pointer.
    //triangles.transformData = {};
    triangles.maxVertex = object.vertexCount - 1;

    // Identify the above data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR asGeom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    asGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    asGeom.geometry.triangles = triangles;

    // The entire array will be used to build the BLAS.
    VkAccelerationStructureBuildRangeInfoKHR offset;
    offset.firstVertex = 0;
    offset.primitiveCount = maxPrimitiveCount;
    offset.primitiveOffset = 0;
    offset.transformOffset = 0;

    // Our blas is made from only one geometry, but could be made of many geometries
    BlasInput input;
    input.asGeometry.emplace_back(asGeom);
    input.asBuildOffsetInfo.emplace_back(offset);

    return input;
}

void VulkanRayTracer::createBottomLevelAS()
{
    // BLAS - Storing each primitive in a geometry
    std::vector<BlasInput> allBlas;
    allBlas.reserve(engine->drawCommands.OpaqueSurfaces.size());
    for (const auto& obj : engine->drawCommands.OpaqueSurfaces)
    {
        BlasInput blas = objectToVkGeometryKHR(obj);

        // We could add more geometry in each BLAS, but we add only one for now
        allBlas.emplace_back(blas);
    }

    buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

void VulkanRayTracer::buildBlas(const std::vector<BlasInput>& input, VkBuildAccelerationStructureFlagsKHR flags) {
    // TODO
    return;
}