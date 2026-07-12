#include "vk_raytracer.h"

VulkanRayTracer::VulkanRayTracer(VulkanEngine* owner)
{
    _engine = owner;

    // Load extension functions
    _vkGetAccelerationStructureBuildSizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        vkGetDeviceProcAddr(_engine->_device, "vkGetAccelerationStructureBuildSizesKHR"));
    _vkCmdBuildAccelerationStructures = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(_engine->_device, "vkCmdBuildAccelerationStructuresKHR"));
    _vkCmdCopyAccelerationStructure = reinterpret_cast<PFN_vkCmdCopyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(_engine->_device, "vkCmdCopyAccelerationStructureKHR"));
    _vkCmdWriteAccelerationStructuresProperties = reinterpret_cast<PFN_vkCmdWriteAccelerationStructuresPropertiesKHR>(
        vkGetDeviceProcAddr(_engine->_device, "vkCmdWriteAccelerationStructuresPropertiesKHR"));
    _vkCreateAccelerationStructure = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        vkGetDeviceProcAddr(_engine->_device, "vkCreateAccelerationStructureKHR"));
    _vkDestroyAccelerationStructure = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(_engine->_device, "vkDestroyAccelerationStructureKHR"));
    _vkGetAccelerationStructureDeviceAddress = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        vkGetDeviceProcAddr(_engine->_device, "vkGetAccelerationStructureDeviceAddressKHR"));
    _vkCreateRayTracingPipelines = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
        vkGetDeviceProcAddr(_engine->_device, "vkCreateRayTracingPipelinesKHR"));
    _vkGetRayTracingShaderGroupHandles = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
        vkGetDeviceProcAddr(_engine->_device, "vkGetRayTracingShaderGroupHandlesKHR"));
    _vkCmdTraceRays =
        reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(_engine->_device, "vkCmdTraceRaysKHR"));

    // Query ray tracing and acceleration structure limits through separate property chains.
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &_rayTracingProperties;
    vkGetPhysicalDeviceProperties2(_engine->_chosenGPU, &prop2);

    // The shaders require more recursion than Vulkan's minimum guarantee.
    if (_rayTracingProperties.maxRayRecursionDepth <= 1) {
        throw std::runtime_error(
            "Device fails to support ray recursion (_rayTracingProperties.maxRayRecursionDepth <= 1)");
    }

    _accelerationStructureProperties = {};
    _accelerationStructureProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2 deviceProperties2 = {};
    deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProperties2.pNext = &_accelerationStructureProperties;

    vkGetPhysicalDeviceProperties2(_engine->_chosenGPU, &deviceProperties2);
}

// Converts a render object into the geometry used to build its BLAS.
BlasInput VulkanRayTracer::object_to_vk_geometry(const RenderObject object)
{
    // BLAS builder requires raw device addresses.
    VkDeviceAddress vertexAddress = object.vertexBufferAddress;

    VkDeviceAddress indexAddress = _engine->get_buffer_device_address(_engine->_device, object.indexBuffer);

    uint32_t maxPrimitiveCount = object.indexCount / 3;

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride = sizeof(Vertex);
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexAddress;
    triangles.maxVertex = object.vertexCount - 1;

    VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    asGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    asGeom.geometry.triangles = triangles;

    VkAccelerationStructureBuildRangeInfoKHR offset;
    offset.firstVertex = 0;
    offset.primitiveCount = maxPrimitiveCount;
    offset.primitiveOffset = object.firstIndex * sizeof(uint32_t);
    offset.transformOffset = 0;

    BlasInput input;
    input.asGeometry.emplace_back(asGeom);
    input.asBuildOffsetInfo.emplace_back(offset);

    return input;
}

void VulkanRayTracer::create_bottom_level_acceleration_structures()
{
    // BLAS - Storing each primitive in a geometry
    std::vector<BlasInput> allBlas;
    allBlas.reserve(_engine->_drawContext.opaqueSurfaces.size());
    for (RenderObject& obj : _engine->_drawContext.opaqueSurfaces) {
        BlasInput blas = object_to_vk_geometry(obj);
        obj.blasIndex = static_cast<uint32_t>(allBlas.size());

        // We could add more geometry in each BLAS, but we add only one for now
        allBlas.emplace_back(blas);
    }

    build_bottom_level_structures(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    const auto blasToDestroy = _bottomLevelStructures;
    const auto destroyAccelerationStructure = _vkDestroyAccelerationStructure;
    _engine->_mainDeletionQueue.push_function([this, blasToDestroy, destroyAccelerationStructure]() {
        for (const auto& blas : blasToDestroy) {
            destroyAccelerationStructure(_engine->_device, blas.accel, nullptr);
            _engine->destroy_buffer(blas.buffer);
        }
    });
}

AccelKHR VulkanRayTracer::create_acceleration_structure(VkAccelerationStructureCreateInfoKHR& accel_)
{
    AccelKHR resultAccel;
    resultAccel.buffer = _engine->create_buffer(
        accel_.size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    accel_.buffer = resultAccel.buffer.buffer;
    VK_CHECK(_vkCreateAccelerationStructure(_engine->_device, &accel_, nullptr, &resultAccel.accel));

    return resultAccel;
}
// Creating the bottom level acceleration structure for all indices of `buildAs` vector.
// The array of BuildAccelerationStructure was created in build_bottom_level_structures and the vector of
// indices limits the number of BLAS to create at once. This limits the amount of
// memory needed when compacting the BLAS.
void VulkanRayTracer::record_blas_build(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices,
                                        std::vector<BuildAccelerationStructure>& buildAs,
                                        VkDeviceAddress scratchAddress, VkQueryPool queryPool)
{

    if (queryPool) // For querying the compaction size
        vkResetQueryPool(_engine->_device, queryPool, 0, static_cast<uint32_t>(indices.size()));
    uint32_t queryCnt{0};

    for (const auto& idx : indices) {
        // Actual allocation of buffer and acceleration structure.
        VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        createInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize; // Will be used to allocate memory.
        buildAs[idx].as = create_acceleration_structure(createInfo);

        // BuildInfo #2 part
        buildAs[idx].buildInfo.dstAccelerationStructure = buildAs[idx].as.accel; // Setting where the build lands
        buildAs[idx].buildInfo.scratchData.deviceAddress =
            scratchAddress; // All build are using the same scratch buffer

        // Building the bottom-level-acceleration-structure
        _vkCmdBuildAccelerationStructures(cmdBuf, 1, &buildAs[idx].buildInfo, &buildAs[idx].rangeInfo);

        // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
        // is finished before starting the next one.
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0,
                             nullptr);

        if (queryPool) {
            // Add a query to find the 'real' amount of memory needed, use for compaction
            _vkCmdWriteAccelerationStructuresProperties(cmdBuf, 1, &buildAs[idx].buildInfo.dstAccelerationStructure,
                                                        VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                                                        queryPool, queryCnt++);
        }
    }
}
// Create and replace a new acceleration structure and buffer based on the size retrieved by the
// Query.
void VulkanRayTracer::record_blas_compaction(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices,
                                             std::vector<BuildAccelerationStructure>& buildAs, VkQueryPool queryPool)
{
    uint32_t queryCtn{0};

    // Get the compacted size result back
    std::vector<VkDeviceSize> compactSizes(static_cast<uint32_t>(indices.size()));
    vkGetQueryPoolResults(_engine->_device, queryPool, 0, (uint32_t)compactSizes.size(),
                          compactSizes.size() * sizeof(VkDeviceSize), compactSizes.data(), sizeof(VkDeviceSize),
                          VK_QUERY_RESULT_WAIT_BIT);

    for (auto idx : indices) {
        buildAs[idx].cleanupAS = buildAs[idx].as;                                   // previous AS to destroy
        buildAs[idx].sizeInfo.accelerationStructureSize = compactSizes[queryCtn++]; // new reduced size

        // Creating a compact version of the AS
        VkAccelerationStructureCreateInfoKHR asCreateInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        asCreateInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize;
        asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildAs[idx].as = create_acceleration_structure(asCreateInfo);

        // Copy the original BLAS to a compact version
        VkCopyAccelerationStructureInfoKHR copyInfo{VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
        copyInfo.src = buildAs[idx].buildInfo.dstAccelerationStructure;
        copyInfo.dst = buildAs[idx].as.accel;
        copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
        _vkCmdCopyAccelerationStructure(cmdBuf, &copyInfo);
    }
}
// Destroys the original structures after their compacted replacements are ready.
void VulkanRayTracer::destroy_non_compacted_structures(std::vector<uint32_t> indices,
                                                       std::vector<BuildAccelerationStructure>& buildAs)
{
    for (auto& i : indices) {
        AccelKHR& a_ = buildAs[i].cleanupAS;
        _vkDestroyAccelerationStructure(_engine->_device, a_.accel, nullptr);
        _engine->destroy_buffer(a_.buffer);
        a_.buffer = AllocatedBuffer();
        a_ = AccelKHR();
    }
}

void VulkanRayTracer::build_bottom_level_structures(const std::vector<BlasInput>& input,
                                                    VkBuildAccelerationStructureFlagsKHR flags)
{
    uint32_t nbBlas = static_cast<uint32_t>(input.size());
    VkDeviceSize asTotalSize{0};    // Memory size of all allocated BLAS
    uint32_t nbCompactions{0};      // Nb of BLAS requesting compaction
    VkDeviceSize maxScratchSize{0}; // Largest scratch size

    // Preparing the information for the acceleration build commands.
    std::vector<BuildAccelerationStructure> buildAs(nbBlas);
    for (uint32_t idx = 0; idx < nbBlas; idx++) {
        // Filling partially the VkAccelerationStructureBuildGeometryInfoKHR for querying the build sizes.
        // Other information will be filled in the createBlas (see #2)
        buildAs[idx].buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildAs[idx].buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildAs[idx].buildInfo.flags = input[idx].flags | flags;
        buildAs[idx].buildInfo.geometryCount = static_cast<uint32_t>(input[idx].asGeometry.size());
        buildAs[idx].buildInfo.pGeometries = input[idx].asGeometry.data();

        // Build range information
        buildAs[idx].rangeInfo = input[idx].asBuildOffsetInfo.data();

        // Finding sizes to create acceleration structures and scratch
        std::vector<uint32_t> maxPrimCount(input[idx].asBuildOffsetInfo.size());
        for (auto tt = 0; tt < input[idx].asBuildOffsetInfo.size(); tt++)
            maxPrimCount[tt] = input[idx].asBuildOffsetInfo[tt].primitiveCount; // Number of primitives/triangles
        _vkGetAccelerationStructureBuildSizes(_engine->_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                              &buildAs[idx].buildInfo, maxPrimCount.data(), &buildAs[idx].sizeInfo);

        // Extra info
        asTotalSize += buildAs[idx].sizeInfo.accelerationStructureSize;
        maxScratchSize = std::max(maxScratchSize, buildAs[idx].sizeInfo.buildScratchSize);
        nbCompactions +=
            has_flag(buildAs[idx].buildInfo.flags, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);
    }

    uint64_t align = _accelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment;

    // Allocate the scratch buffers holding the temporary data of the acceleration structure builder
    AllocatedBuffer scratchBuffer = _engine->create_buffer(
        maxScratchSize + align, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, scratchBuffer.buffer};
    VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(_engine->_device, &bufferInfo);
    VkDeviceAddress alignedAddress = (scratchAddress + align - 1) & ~(align - 1);

    // Allocate a query pool for storing the needed size for every BLAS compaction.
    VkQueryPool queryPool{VK_NULL_HANDLE};
    if (nbCompactions > 0) // Is compaction requested?
    {
        assert(nbCompactions == nbBlas); // Don't allow mix of on/off compaction
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryCount = nbBlas;
        qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        VK_CHECK(vkCreateQueryPool(_engine->_device, &qpci, nullptr, &queryPool));
    }

    // Batching creation/compaction of BLAS to allow staying in restricted amount of memory
    std::vector<uint32_t> indices; // Indices of the BLAS to create
    VkDeviceSize batchSize{0};
    VkDeviceSize batchLimit{256'000'000}; // 256 MB
    for (uint32_t idx = 0; idx < nbBlas; idx++) {
        indices.push_back(idx);
        batchSize += buildAs[idx].sizeInfo.accelerationStructureSize;
        // Over the limit or last BLAS element
        if (batchSize >= batchLimit || idx == nbBlas - 1) {
            _engine->immediate_submit(
                [&](VkCommandBuffer cmd) { record_blas_build(cmd, indices, buildAs, alignedAddress, queryPool); });

            if (queryPool) {
                _engine->immediate_submit(
                    [&](VkCommandBuffer cmd) { record_blas_compaction(cmd, indices, buildAs, queryPool); });

                // Destroy the non-compacted version
                destroy_non_compacted_structures(indices, buildAs);
            }
            // Reset
            batchSize = 0;
            indices.clear();
        }
    }

    // Keeping all the created acceleration structures
    for (auto& b : buildAs) {
        _bottomLevelStructures.emplace_back(b.as);
    }

    // Clean up
    vkDestroyQueryPool(_engine->_device, queryPool, nullptr);
    _engine->destroy_buffer(scratchBuffer);

    return;
}

void VulkanRayTracer::create_top_level_acceleration_structure()
{
    std::vector<VkAccelerationStructureInstanceKHR> tlas;
    tlas.reserve(_engine->_drawContext.opaqueSurfaces.size());
    for (const RenderObject& inst : _engine->_drawContext.opaqueSurfaces) {
        VkAccelerationStructureInstanceKHR rayInst{};
        rayInst.transform = to_transform_matrix(inst.transform); // Position of the instance
        rayInst.instanceCustomIndex = inst.blasIndex;            // gl_InstanceCustomIndexEXT
        rayInst.accelerationStructureReference = get_blas_device_address(inst.blasIndex);
        rayInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        rayInst.mask = 0xFF;                                //  Only be hit if rayMask & instance.mask != 0
        rayInst.instanceShaderBindingTableRecordOffset = 0; // We will use the same hit group for all objects
        tlas.emplace_back(rayInst);
    }
    build_top_level_structure(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, false, false);

    // cleanup
    const AccelKHR tlasToDestroy = _topLevelStructure;
    const auto destroyAccelerationStructure = _vkDestroyAccelerationStructure;
    _engine->_mainDeletionQueue.push_function([this, tlasToDestroy, destroyAccelerationStructure]() {
        destroyAccelerationStructure(_engine->_device, tlasToDestroy.accel, nullptr);
        _engine->destroy_buffer(tlasToDestroy.buffer);
    });
}

// Convert a Mat4x4 to the matrix required by acceleration structures
VkTransformMatrixKHR VulkanRayTracer::to_transform_matrix(glm::mat4 matrix)
{
    // VkTransformMatrixKHR uses a row-major memory layout, while glm::mat4
    // uses a column-major memory layout. We transpose the matrix so we can
    // memcpy the matrix's data directly.
    glm::mat4 temp = glm::transpose(matrix);
    VkTransformMatrixKHR out_matrix;
    memcpy(&out_matrix, &temp, sizeof(VkTransformMatrixKHR));
    return out_matrix;
}
// Return the device address of a Blas previously created.
VkDeviceAddress VulkanRayTracer::get_blas_device_address(uint32_t blasId)
{
    assert(size_t(blasId) < _bottomLevelStructures.size());
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = _bottomLevelStructures[blasId].accel;
    return _vkGetAccelerationStructureDeviceAddress(_engine->_device, &addressInfo);
}

void VulkanRayTracer::build_top_level_structure(const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                                                VkBuildAccelerationStructureFlagsKHR flags, bool update, bool motion)
{
    // Cannot call build_top_level_structure twice except to update.
    assert(_topLevelStructure.accel == VK_NULL_HANDLE || update);
    uint32_t countInstance = static_cast<uint32_t>(instances.size());

    // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
    AllocatedBuffer instancesBuffer; // Buffer of instances containing the matrices and BLAS ids
    VkDeviceSize size = sizeof(VkAccelerationStructureInstanceKHR) * instances.size();
    instancesBuffer =
        _engine->create_buffer_data(size, instances.data(),
                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                    VMA_MEMORY_USAGE_GPU_ONLY);
    VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, instancesBuffer.buffer};
    VkDeviceAddress instBufferAddr = vkGetBufferDeviceAddress(_engine->_device, &bufferInfo);

    // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    // Creating the TLAS
    AllocatedBuffer scratchBuffer;

    // Executing and destroying temporary data
    _engine->immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0,
                             nullptr);
    });

    _engine->immediate_submit([&](VkCommandBuffer cmd) {
        record_top_level_structure_build(cmd, countInstance, instBufferAddr, scratchBuffer, flags, update, motion);
    });

    _engine->destroy_buffer(scratchBuffer);
    _engine->destroy_buffer(instancesBuffer);
}
// Low level of Tlas creation - see build_top_level_structure
void VulkanRayTracer::record_top_level_structure_build(VkCommandBuffer cmdBuf, uint32_t countInstance,
                                                       VkDeviceAddress instBufferAddr, AllocatedBuffer& scratchBuffer,
                                                       VkBuildAccelerationStructureFlagsKHR flags, bool update,
                                                       bool motion)
{
    // Wraps a device pointer to the above uploaded instances.
    VkAccelerationStructureGeometryInstancesDataKHR instancesVk{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    instancesVk.data.deviceAddress = instBufferAddr;

    // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
    VkAccelerationStructureGeometryKHR topASGeometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    topASGeometry.geometry.instances = instancesVk;

    // Find sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.flags = flags;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &topASGeometry;
    buildInfo.ppGeometries = nullptr;
    buildInfo.mode =
        update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    _vkGetAccelerationStructureBuildSizes(_engine->_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
                                          &countInstance, &sizeInfo);

    VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    createInfo.size = sizeInfo.accelerationStructureSize;

    _topLevelStructure = create_acceleration_structure(createInfo);

    // Allocate the scratch memory
    uint64_t align = _accelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment;
    scratchBuffer = _engine->create_buffer(
        sizeInfo.buildScratchSize + align,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, scratchBuffer.buffer};
    VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(_engine->_device, &bufferInfo);
    VkDeviceAddress alignedAddress = (scratchAddress + align - 1) & ~(align - 1);

    // Update build information
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = _topLevelStructure.accel;
    buildInfo.scratchData.deviceAddress = alignedAddress;

    // Build Offsets info: n instances
    VkAccelerationStructureBuildRangeInfoKHR buildOffsetInfo{countInstance, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

    // Build the TLAS
    _vkCmdBuildAccelerationStructures(cmdBuf, 1, &buildInfo, &pBuildOffsetInfo);
}

void VulkanRayTracer::create_descriptor_set()
{
    // Descriptor set #0 holds the Acceleration structure and the output image.
    DescriptorLayoutBuilder descriptorLayoutBuilder;
    descriptorLayoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR); // TLAS
    descriptorLayoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);              // Output image
    descriptorLayoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);     // Environment map

    std::vector<DescriptorAllocator::PoolSizeRatio> rt_pool_sizes = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    };
    _descriptorAllocator.init_pool(_engine->_device, 1, rt_pool_sizes);
    _descriptorPool = _descriptorAllocator.pool;
    VkShaderStageFlags flags =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    _descriptorSetLayout = descriptorLayoutBuilder.build(_engine->_device, flags);

    VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool = _descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &_descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(_engine->_device, &allocateInfo, &_descriptorSet));

    VkAccelerationStructureKHR tlas = _topLevelStructure.accel;
    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    descASInfo.accelerationStructureCount = 1;
    descASInfo.pAccelerationStructures = &tlas;

    _descriptorWriter.clear();
    _descriptorWriter.write_buffer(0, 0, 0, 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    _descriptorWriter.writes[0].pNext = &descASInfo;
    _descriptorWriter.writes[0].pBufferInfo = nullptr;
    _descriptorWriter.write_image(1, _engine->_drawImage.imageView, {}, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    _descriptorWriter.write_image(2, _engine->_environmentMap.imageView, _engine->_defaultSamplerLinear,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    _descriptorWriter.update_set(_engine->_device, _descriptorSet);

    // add all to deletion queue
    const VkDescriptorPool descriptorPool = _descriptorPool;
    const VkDescriptorSetLayout descriptorLayout = _descriptorSetLayout;
    _engine->_mainDeletionQueue.push_function([this, descriptorPool, descriptorLayout]() {
        vkDestroyDescriptorPool(_engine->_device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(_engine->_device, descriptorLayout, nullptr);
    });
}

void VulkanRayTracer::create_material_descriptor_set()
{
    const uint32_t TEX_MAX = 256;

    DescriptorLayoutBuilder materialDescriptorLayoutBuilder;
    materialDescriptorLayoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, TEX_MAX);
    materialDescriptorLayoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, TEX_MAX);

    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> rt_mat_pool_sizes = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, float(TEX_MAX) * 2.0f}};
    _materialDescriptorAllocator.init_pools(_engine->_device, 1, rt_mat_pool_sizes);
    _materialDescriptorSetLayout =
        materialDescriptorLayoutBuilder.build(_engine->_device, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    _materialDescriptorSet = _materialDescriptorAllocator.allocate(_engine->_device, _materialDescriptorSetLayout);

    // resize current arrays to TEX_MAX
    _colorTextures.resize(TEX_MAX, _engine->_whiteImage.imageView);
    _colorSamplers.resize(TEX_MAX, _engine->_defaultSamplerLinear);
    _metalRoughTextures.resize(TEX_MAX, _engine->_whiteImage.imageView);
    _metalRoughSamplers.resize(TEX_MAX, _engine->_defaultSamplerLinear);

    _materialDescriptorWriter.clear();
    // arrays with only one element
    _materialDescriptorWriter.write_image_array(0, _colorTextures, _colorSamplers,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    _materialDescriptorWriter.write_image_array(1, _metalRoughTextures, _metalRoughSamplers,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    _materialDescriptorWriter.update_set(_engine->_device, _materialDescriptorSet);

    // add all to deletion queue
    const VkDescriptorSetLayout materialDescriptorLayout = _materialDescriptorSetLayout;
    _engine->_mainDeletionQueue.push_function([this, materialDescriptorLayout]() {
        _materialDescriptorAllocator.destroy_pools(_engine->_device);
        vkDestroyDescriptorSetLayout(_engine->_device, materialDescriptorLayout, nullptr);
    });
}
// Writes the output image to the descriptor set
// - Required when changing resolution
void VulkanRayTracer::update_output_descriptor()
{
    // update drawimage manually (without use of descriptor writer) to avoid updating the acceleration structure
    VkDescriptorImageInfo info{{}, _engine->_drawImage.imageView, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet = _descriptorSet;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wds.pImageInfo = &info;
    wds.dstBinding = 1;

    vkUpdateDescriptorSets(_engine->_device, 1, &wds, 0, nullptr);
}
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
void VulkanRayTracer::create_pipeline()
{
    enum StageIdx
    {
        eRaygen,
        eMiss,
        eMissShadow,
        eClosestHit,
        eStageCount
    };

    // Clear in case this is called more than once
    _shaderGroups.clear();

    // Load shaders or die early (don't proceed with junk)
    auto load_or_bail = [&](const char* path, VkShaderStageFlagBits stg) {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage = stg;
        s.pName = "main";
        if (!vkutil::load_shader_module(path, _engine->_device, &s.module)) {
            throw std::runtime_error(std::string("Failed to load shader: ") + path);
        }
        return s;
    };

    std::array<VkPipelineShaderStageCreateInfo, eStageCount> stages;
    stages[eRaygen] = load_or_bail("../../shaders/raytrace.rgen.spv", VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    stages[eMiss] = load_or_bail("../../shaders/raytrace.rmiss.spv", VK_SHADER_STAGE_MISS_BIT_KHR);
    stages[eMissShadow] = load_or_bail("../../shaders/raytraceShadow.rmiss.spv", VK_SHADER_STAGE_MISS_BIT_KHR);
    stages[eClosestHit] = load_or_bail("../../shaders/raytrace.rchit.spv", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

    // Sanity: every set layout must exist
    auto must = [](VkDescriptorSetLayout l, const char* name) {
        if (l == VK_NULL_HANDLE)
            throw std::runtime_error(std::string("Null set layout: ") + name);
    };
    must(_descriptorSetLayout, "rtDesc");
    must(_engine->_gpuSceneDataDescriptorLayout, "scene");
    must(_engine->_objDescLayout, "objDesc");
    must(_materialDescriptorSetLayout, "materials");
    must(_engine->_volumeSetLayout, "volume");

    // Push constants
    VkPushConstantRange pcr{};
    pcr.stageFlags =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    pcr.offset = 0;
    pcr.size = sizeof(PushConstantRay);

    // Pipeline layout
    _pipelineLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> layouts = {_descriptorSetLayout, _engine->_gpuSceneDataDescriptorLayout,
                                                  _engine->_objDescLayout, _materialDescriptorSetLayout,
                                                  _engine->_volumeSetLayout};

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = (uint32_t)layouts.size();
    plci.pSetLayouts = layouts.data();
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;

    VkResult plRes = vkCreatePipelineLayout(_engine->_device, &plci, nullptr, &_pipelineLayout);
    if (plRes != VK_SUCCESS || _pipelineLayout == VK_NULL_HANDLE) {
        for (auto& s : stages)
            if (s.module)
                vkDestroyShaderModule(_engine->_device, s.module, nullptr);
        throw std::runtime_error("vkCreatePipelineLayout failed, code " + std::to_string(plRes));
    }

    // Shader groups
    _shaderGroups.reserve(4);
    VkRayTracingShaderGroupCreateInfoKHR g{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    g.anyHitShader = VK_SHADER_UNUSED_KHR;
    g.closestHitShader = VK_SHADER_UNUSED_KHR;
    g.generalShader = VK_SHADER_UNUSED_KHR;
    g.intersectionShader = VK_SHADER_UNUSED_KHR;

    g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    g.generalShader = eRaygen;
    _shaderGroups.push_back(g);
    g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    g.generalShader = eMiss;
    _shaderGroups.push_back(g);
    g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    g.generalShader = eMissShadow;
    _shaderGroups.push_back(g);
    g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    g.generalShader = VK_SHADER_UNUSED_KHR;
    g.closestHitShader = eClosestHit;
    _shaderGroups.push_back(g);

    // Create pipeline
    VkRayTracingPipelineCreateInfoKHR rtp{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    rtp.stageCount = (uint32_t)stages.size();
    rtp.pStages = stages.data();
    rtp.groupCount = (uint32_t)_shaderGroups.size();
    rtp.pGroups = _shaderGroups.data();
    rtp.maxPipelineRayRecursionDepth = MAX_RAY_RECURSION_DEPTH;
    rtp.layout = _pipelineLayout;

    // Verify function pointer
    if (!_vkCreateRayTracingPipelines) {
        for (auto& s : stages)
            if (s.module)
                vkDestroyShaderModule(_engine->_device, s.module, nullptr);
        throw std::runtime_error("_vkCreateRayTracingPipelines is NULL (extension not enabled / bad load)");
    }

    VkDeferredOperationKHR deferred{VK_NULL_HANDLE};
    VkPipelineCache cache{VK_NULL_HANDLE};
    _pipeline = VK_NULL_HANDLE;

    VkResult pipeRes = _vkCreateRayTracingPipelines(_engine->_device, deferred, cache, 1, &rtp, nullptr, &_pipeline);

    // Cleanup shader modules regardless
    for (auto& s : stages)
        if (s.module)
            vkDestroyShaderModule(_engine->_device, s.module, nullptr);

    if (pipeRes != VK_SUCCESS || _pipeline == VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(_engine->_device, _pipelineLayout, nullptr);
        _pipelineLayout = VK_NULL_HANDLE;
        throw std::runtime_error("vkCreateRayTracingPipelinesKHR failed, code " + std::to_string(pipeRes));
    }

    // Deletion hooks
    const VkPipeline pipeline = _pipeline;
    const VkPipelineLayout pipelineLayout = _pipelineLayout;
    _engine->_mainDeletionQueue.push_function([this, pipeline, pipelineLayout]() {
        vkDestroyPipeline(_engine->_device, pipeline, nullptr);
        vkDestroyPipelineLayout(_engine->_device, pipelineLayout, nullptr);
    });
}

template <class integral> constexpr integral align_up(integral x, size_t a) noexcept
{
    return integral((x + (integral(a) - 1)) & ~integral(a - 1));
}
// The Shader Binding Table (SBT)
// - getting all shader handles and write them in a SBT buffer
// - Besides exception, this could be always done like this
void VulkanRayTracer::create_shader_binding_table()
{
    uint32_t missCount{2};
    uint32_t hitCount{1};
    auto handleCount = 1 + missCount + hitCount;
    uint32_t handleSize = _rayTracingProperties.shaderGroupHandleSize;

    // The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
    uint32_t handleSizeAligned = align_up(handleSize, _rayTracingProperties.shaderGroupHandleAlignment);

    _rayGenerationRegion.stride = align_up(handleSizeAligned, _rayTracingProperties.shaderGroupBaseAlignment);
    _rayGenerationRegion.size =
        _rayGenerationRegion.stride; // The size member of pRayGenShaderBindingTable must be equal to its stride member
    _missRegion.stride = handleSizeAligned;
    _missRegion.size = align_up(missCount * handleSizeAligned, _rayTracingProperties.shaderGroupBaseAlignment);
    _hitRegion.stride = handleSizeAligned;
    _hitRegion.size = align_up(hitCount * handleSizeAligned, _rayTracingProperties.shaderGroupBaseAlignment);

    // Get the shader group handles
    uint32_t dataSize = handleCount * handleSize;
    std::vector<uint8_t> handles(dataSize);
    VK_CHECK(_vkGetRayTracingShaderGroupHandles(_engine->_device, _pipeline, 0, handleCount, dataSize, handles.data()));

    // Allocate a buffer for storing the SBT.
    VkDeviceSize sbtSize = _rayGenerationRegion.size + _missRegion.size + _hitRegion.size + _callableRegion.size;
    _shaderBindingTableBuffer =
        _engine->create_buffer(sbtSize,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                   VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                               VMA_MEMORY_USAGE_CPU_ONLY);

    // Find the SBT addresses of each group
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr,
                                   _shaderBindingTableBuffer.buffer};
    VkDeviceAddress sbtAddress = vkGetBufferDeviceAddress(_engine->_device, &info);
    _rayGenerationRegion.deviceAddress = sbtAddress;
    _missRegion.deviceAddress = sbtAddress + _rayGenerationRegion.size;
    _hitRegion.deviceAddress = sbtAddress + _rayGenerationRegion.size + _missRegion.size;

    // Helper to retrieve the handle data
    auto getHandle = [&](int i) { return handles.data() + i * handleSize; };

    // Map the SBT buffer and write in the handles.
    uint8_t* pSBTBuffer;
    vmaMapMemory(_engine->_allocator, _shaderBindingTableBuffer.allocation, (void**)&pSBTBuffer);
    uint8_t* pData{nullptr};
    uint32_t handleIdx{0};

    // Raygen
    pData = pSBTBuffer;
    memcpy(pData, getHandle(handleIdx++), handleSize);

    // Miss
    pData = pSBTBuffer + _rayGenerationRegion.size;
    for (uint32_t c = 0; c < missCount; c++) {
        memcpy(pData, getHandle(handleIdx++), handleSize);
        pData += _missRegion.stride;
    }

    // Hit
    pData = pSBTBuffer + _rayGenerationRegion.size + _missRegion.size;
    for (uint32_t c = 0; c < hitCount; c++) {
        memcpy(pData, getHandle(handleIdx++), handleSize);
        pData += _hitRegion.stride;
    }

    // Cleanup
    const AllocatedBuffer sbtBuffer = _shaderBindingTableBuffer;
    _engine->_mainDeletionQueue.push_function([this, sbtBuffer]() {
        vmaUnmapMemory(_engine->_allocator, sbtBuffer.allocation);
        _engine->destroy_buffer(sbtBuffer);
    });
}
// Upload custom material structure to be referred to by the ray tracer
VkDeviceAddress VulkanRayTracer::upload_material(MaterialRT mat)
{
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    AllocatedBuffer materialBuffer =
        _engine->create_buffer_data(sizeof(MaterialRT), &mat, usage, VMA_MEMORY_USAGE_CPU_TO_GPU);

    _engine->_mainDeletionQueue.push_function([this, materialBuffer]() { _engine->destroy_buffer(materialBuffer); });

    return _engine->get_buffer_device_address(_engine->_device, materialBuffer.buffer);
}
// Ray Tracing the scene
void VulkanRayTracer::raytrace(const VkCommandBuffer& cmdBuf)
{
    // Initializing push constant values
    _pushConstants.clearColor = glm::vec4(.6, .6, .6, 1.00f);
    _pushConstants.lightAddress = _engine->get_buffer_device_address(_engine->_device, _engine->_lightBuffer.buffer);
    _pushConstants.numLights = _engine->_lightCount;
    _pushConstants.useMicrofacet = _engine->_useMicrofacetBrdf ? 1 : 0;
    _engine->update_global_descriptor();

    std::vector<VkDescriptorSet> descSets{
        _descriptorSet, _engine->_globalDescriptor, _engine->_objDescSet, _materialDescriptorSet,
        _engine->_volumeSet // volumetric addition
    };
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _pipelineLayout, 0,
                            (uint32_t)descSets.size(), descSets.data(), 0, nullptr);

    vkCmdPushConstants(cmdBuf, _pipelineLayout,
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                           VK_SHADER_STAGE_MISS_BIT_KHR,
                       0, sizeof(PushConstantRay), &_pushConstants);

    // don't run shader multiple times if computing monte carlo
    if (_engine->_monteCarloSamples == 0 || _engine->_lastMonteCarlo != _engine->_monteCarloSamples ||
        _engine->_lastMsaaSamples != _engine->_msaaSamples) {
        _vkCmdTraceRays(cmdBuf, &_rayGenerationRegion, &_missRegion, &_hitRegion, &_callableRegion,
                        _engine->_windowExtent.width, _engine->_windowExtent.height, 1);
    }
    _engine->_lastMonteCarlo = _engine->_monteCarloSamples;
    _engine->_lastMsaaSamples = _engine->_msaaSamples;
}
