#include "vk_raytracer.h"

VulkanRayTracer::VulkanRayTracer(VulkanEngine* engine)
{
	VulkanRayTracer::engine = engine;

    // Load extension functions
    pfnGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(engine->_device, "vkGetAccelerationStructureBuildSizesKHR"));
    pfnCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(engine->_device, "vkCmdBuildAccelerationStructuresKHR"));
    pfnCmdCopyAccelerationStructureKHR = reinterpret_cast<PFN_vkCmdCopyAccelerationStructureKHR>(vkGetDeviceProcAddr(engine->_device, "vkCmdCopyAccelerationStructureKHR"));
    pfnCmdWriteAccelerationStructuresPropertiesKHR = reinterpret_cast<PFN_vkCmdWriteAccelerationStructuresPropertiesKHR>(vkGetDeviceProcAddr(engine->_device, "vkCmdWriteAccelerationStructuresPropertiesKHR"));
    pfnCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(engine->_device, "vkCreateAccelerationStructureKHR"));
    pfnDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(engine->_device, "vkDestroyAccelerationStructureKHR"));
    pfnGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(engine->_device, "vkGetAccelerationStructureDeviceAddressKHR"));
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
    for (RenderObject& obj : engine->drawCommands.OpaqueSurfaces)
    {
        BlasInput blas = objectToVkGeometryKHR(obj);
        obj.blasIndex = static_cast<uint32_t>(allBlas.size());

        // We could add more geometry in each BLAS, but we add only one for now
        allBlas.emplace_back(blas);
    }

    buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);


    // cleanup
    engine->_mainDeletionQueue.push_function([=]() {
        for (auto& blas : m_blas)
		{
            pfnDestroyAccelerationStructureKHR(engine->_device, blas.accel, nullptr);
			engine->destroy_buffer(blas.buffer);
		}
	});
}


AccelKHR VulkanRayTracer::createAcceleration(VkAccelerationStructureCreateInfoKHR& accel_)
{
    AccelKHR resultAccel;
    // Allocating the buffer to hold the acceleration structure
    resultAccel.buffer = engine->create_buffer(accel_.size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    // Setting the buffer
    accel_.buffer = resultAccel.buffer.buffer;
    // Create the acceleration structure
    pfnCreateAccelerationStructureKHR(engine->_device, &accel_, nullptr, &resultAccel.accel);

    return resultAccel;
}

//--------------------------------------------------------------------------------------------------
// Creating the bottom level acceleration structure for all indices of `buildAs` vector.
// The array of BuildAccelerationStructure was created in buildBlas and the vector of
// indices limits the number of BLAS to create at once. This limits the amount of
// memory needed when compacting the BLAS.
void VulkanRayTracer::cmdCreateBlas(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs, VkDeviceAddress scratchAddress, VkQueryPool queryPool)
{

    if (queryPool)  // For querying the compaction size
        vkResetQueryPool(engine->_device, queryPool, 0, static_cast<uint32_t>(indices.size()));
    uint32_t queryCnt{ 0 };

    for (const auto& idx : indices)
    {
        // Actual allocation of buffer and acceleration structure.
        VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        createInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize;  // Will be used to allocate memory.
        buildAs[idx].as = createAcceleration(createInfo);

        // BuildInfo #2 part
        buildAs[idx].buildInfo.dstAccelerationStructure = buildAs[idx].as.accel;  // Setting where the build lands
        buildAs[idx].buildInfo.scratchData.deviceAddress = scratchAddress;  // All build are using the same scratch buffer

        // Building the bottom-level-acceleration-structure
        pfnCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildAs[idx].buildInfo, &buildAs[idx].rangeInfo);

        // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
        // is finished before starting the next one.
        VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        if (queryPool)
        {
            // Add a query to find the 'real' amount of memory needed, use for compaction
            pfnCmdWriteAccelerationStructuresPropertiesKHR(cmdBuf, 1, &buildAs[idx].buildInfo.dstAccelerationStructure,
                VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, queryCnt++);
        }
    }
}

//--------------------------------------------------------------------------------------------------
// Create and replace a new acceleration structure and buffer based on the size retrieved by the
// Query.
void VulkanRayTracer::cmdCompactBlas(VkCommandBuffer cmdBuf, std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs, VkQueryPool queryPool)
{
    uint32_t queryCtn{ 0 };

    // Get the compacted size result back
    std::vector<VkDeviceSize> compactSizes(static_cast<uint32_t>(indices.size()));
    vkGetQueryPoolResults(engine->_device, queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
        compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);

    for (auto idx : indices)
    {
        buildAs[idx].cleanupAS = buildAs[idx].as;           // previous AS to destroy
        buildAs[idx].sizeInfo.accelerationStructureSize = compactSizes[queryCtn++];  // new reduced size

        // Creating a compact version of the AS
        VkAccelerationStructureCreateInfoKHR asCreateInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        asCreateInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize;
        asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildAs[idx].as = createAcceleration(asCreateInfo);

        // Copy the original BLAS to a compact version
        VkCopyAccelerationStructureInfoKHR copyInfo{ VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR };
        copyInfo.src = buildAs[idx].buildInfo.dstAccelerationStructure;
        copyInfo.dst = buildAs[idx].as.accel;
        copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
        pfnCmdCopyAccelerationStructureKHR(cmdBuf, &copyInfo);
    }
}

//--------------------------------------------------------------------------------------------------
// Destroy all the non-compacted acceleration structures
//
void VulkanRayTracer::destroyNonCompacted(std::vector<uint32_t> indices, std::vector<BuildAccelerationStructure>& buildAs)
{
    for (auto& i : indices)
    {
        AccelKHR& a_ = buildAs[i].cleanupAS;
        pfnDestroyAccelerationStructureKHR(engine->_device, a_.accel, nullptr);
        engine->destroy_buffer(a_.buffer);
        //m_memAlloc->freeMemory(a_.buffer.memHandle);

        a_.buffer = AllocatedBuffer();
        a_ = AccelKHR();
    }
}

void VulkanRayTracer::buildBlas(const std::vector<BlasInput>& input, VkBuildAccelerationStructureFlagsKHR flags) {
    uint32_t     nbBlas = static_cast<uint32_t>(input.size());
    VkDeviceSize asTotalSize{ 0 };     // Memory size of all allocated BLAS
    uint32_t     nbCompactions{ 0 };   // Nb of BLAS requesting compaction
    VkDeviceSize maxScratchSize{ 0 };  // Largest scratch size

    // Preparing the information for the acceleration build commands.
    std::vector<BuildAccelerationStructure> buildAs(nbBlas);
    for (uint32_t idx = 0; idx < nbBlas; idx++)
    {
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
            maxPrimCount[tt] = input[idx].asBuildOffsetInfo[tt].primitiveCount;  // Number of primitives/triangles
        pfnGetAccelerationStructureBuildSizesKHR(engine->_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildAs[idx].buildInfo, maxPrimCount.data(), &buildAs[idx].sizeInfo);

        // Extra info
        asTotalSize += buildAs[idx].sizeInfo.accelerationStructureSize;
        maxScratchSize = std::max(maxScratchSize, buildAs[idx].sizeInfo.buildScratchSize);
        nbCompactions += hasFlag(buildAs[idx].buildInfo.flags, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);
    }

    // Allocate the scratch buffers holding the temporary data of the acceleration structure builder
    AllocatedBuffer scratchBuffer = engine->create_buffer(maxScratchSize, 
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    VkBufferDeviceAddressInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, scratchBuffer.buffer };
    VkDeviceAddress           scratchAddress = vkGetBufferDeviceAddress(engine->_device, &bufferInfo);

    // Allocate a query pool for storing the needed size for every BLAS compaction.
    VkQueryPool queryPool{ VK_NULL_HANDLE };
    if (nbCompactions > 0)  // Is compaction requested?
    {
        assert(nbCompactions == nbBlas);  // Don't allow mix of on/off compaction
        VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qpci.queryCount = nbBlas;
        qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        vkCreateQueryPool(engine->_device, &qpci, nullptr, &queryPool);
    }

    // Batching creation/compaction of BLAS to allow staying in restricted amount of memory
    std::vector<uint32_t> indices;  // Indices of the BLAS to create
    VkDeviceSize          batchSize{ 0 };
    VkDeviceSize          batchLimit{ 256'000'000 };  // 256 MB
    for (uint32_t idx = 0; idx < nbBlas; idx++)
    {
        indices.push_back(idx);
        batchSize += buildAs[idx].sizeInfo.accelerationStructureSize;
        // Over the limit or last BLAS element
        if (batchSize >= batchLimit || idx == nbBlas - 1)
        {
            engine->immediate_submit([&](VkCommandBuffer cmd) { cmdCreateBlas(cmd, indices, buildAs, scratchAddress, queryPool); });

            if (queryPool)
            {
                engine->immediate_submit([&](VkCommandBuffer cmd) { cmdCompactBlas(cmd, indices, buildAs, queryPool); });

                // Destroy the non-compacted version
                destroyNonCompacted(indices, buildAs);
            }
            // Reset
            batchSize = 0;
            indices.clear();
        }
    }

    // Keeping all the created acceleration structures
    for (auto& b : buildAs)
    {
        m_blas.emplace_back(b.as);
    }

    // Clean up
    vkDestroyQueryPool(engine->_device, queryPool, nullptr);
    engine->destroy_buffer(scratchBuffer);

    return;
}

void VulkanRayTracer::createTopLevelAS()
{
    std::vector<VkAccelerationStructureInstanceKHR> tlas;
    tlas.reserve(engine->drawCommands.OpaqueSurfaces.size());
    for (const RenderObject& inst : engine->drawCommands.OpaqueSurfaces)
    {
        VkAccelerationStructureInstanceKHR rayInst{};
        rayInst.transform = toTransformMatrixKHR(inst.transform);  // Position of the instance
        rayInst.instanceCustomIndex = inst.blasIndex;                               // gl_InstanceCustomIndexEXT
        rayInst.accelerationStructureReference = getBlasDeviceAddress(inst.blasIndex);
        rayInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        rayInst.mask = 0xFF;       //  Only be hit if rayMask & instance.mask != 0
        rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
        tlas.emplace_back(rayInst);
    }
    buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, false, false);

    // cleanup
    engine->_mainDeletionQueue.push_function([=]() {
        pfnDestroyAccelerationStructureKHR(engine->_device, m_tlas.accel, nullptr);
		engine->destroy_buffer(m_tlas.buffer);
    });
}


// Convert a Mat4x4 to the matrix required by acceleration structures
VkTransformMatrixKHR VulkanRayTracer::toTransformMatrixKHR(glm::mat4 matrix)
{
    // VkTransformMatrixKHR uses a row-major memory layout, while glm::mat4
    // uses a column-major memory layout. We transpose the matrix so we can
    // memcpy the matrix's data directly.
    glm::mat4            temp = glm::transpose(matrix);
    VkTransformMatrixKHR out_matrix;
    memcpy(&out_matrix, &temp, sizeof(VkTransformMatrixKHR));
    return out_matrix;
}

//--------------------------------------------------------------------------------------------------
// Return the device address of a Blas previously created.
//
VkDeviceAddress VulkanRayTracer::getBlasDeviceAddress(uint32_t blasId)
{
    assert(size_t(blasId) < m_blas.size());
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
    addressInfo.accelerationStructure = m_blas[blasId].accel;
    return pfnGetAccelerationStructureDeviceAddressKHR(engine->_device, &addressInfo);
}

void VulkanRayTracer::buildTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances,
    VkBuildAccelerationStructureFlagsKHR flags, bool update, bool motion)
{
    // Cannot call buildTlas twice except to update.
    assert(m_tlas.accel == VK_NULL_HANDLE || update);
    uint32_t countInstance = static_cast<uint32_t>(instances.size());

    // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
    AllocatedBuffer instancesBuffer;  // Buffer of instances containing the matrices and BLAS ids
    VkDeviceSize size = sizeof(VkAccelerationStructureInstanceKHR) * instances.size();
    instancesBuffer = engine->create_buffer(size, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | 
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_GPU_ONLY);
    VkBufferDeviceAddressInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, instancesBuffer.buffer };
    VkDeviceAddress           instBufferAddr = vkGetBufferDeviceAddress(engine->_device, &bufferInfo);

    // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    // Creating the TLAS
    AllocatedBuffer scratchBuffer;

    // Executing and destroying temporary data
    engine->immediate_submit([&](VkCommandBuffer cmd) {
        // should become necessary if we move buffer creation to the GPU
        //vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        cmdCreateTlas(cmd, countInstance, instBufferAddr, scratchBuffer, flags, update, motion);
    });
    engine->destroy_buffer(scratchBuffer);
    engine->destroy_buffer(instancesBuffer);
}

//--------------------------------------------------------------------------------------------------
// Low level of Tlas creation - see buildTlas
//
void VulkanRayTracer::cmdCreateTlas(VkCommandBuffer cmdBuf, uint32_t countInstance, VkDeviceAddress instBufferAddr,
    AllocatedBuffer& scratchBuffer, VkBuildAccelerationStructureFlagsKHR flags, bool update, bool motion)
{
    // Wraps a device pointer to the above uploaded instances.
    VkAccelerationStructureGeometryInstancesDataKHR instancesVk{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
    instancesVk.data.deviceAddress = instBufferAddr;

    // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
    VkAccelerationStructureGeometryKHR topASGeometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    topASGeometry.geometry.instances = instancesVk;

    // Find sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    buildInfo.flags = flags;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &topASGeometry;
    buildInfo.ppGeometries = nullptr;
    buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    pfnGetAccelerationStructureBuildSizesKHR(engine->_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
        &countInstance, &sizeInfo);

    VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    createInfo.size = sizeInfo.accelerationStructureSize;

    m_tlas = createAcceleration(createInfo);

    // Allocate the scratch memory
    scratchBuffer = engine->create_buffer(sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    VkBufferDeviceAddressInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, scratchBuffer.buffer };
    VkDeviceAddress           scratchAddress = vkGetBufferDeviceAddress(engine->_device, &bufferInfo);

    // Update build information
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = m_tlas.accel;
    buildInfo.scratchData.deviceAddress = scratchAddress;

    // Build Offsets info: n instances
    VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo{ countInstance, 0, 0, 0 };
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

    // Build the TLAS
    pfnCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildInfo, &pBuildOffsetInfo);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void VulkanRayTracer::createRtDescriptorSet()
{
    DescriptorLayoutBuilder m_rtDescLayoutBuilder;
    m_rtDescLayoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);  // TLAS
    m_rtDescLayoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // Output image

    std::vector<DescriptorAllocator::PoolSizeRatio> rt_pool_sizes = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
    };
    m_rtDescAllocator.init_pool(engine->_device, 1, rt_pool_sizes);
    m_rtDescPool = m_rtDescAllocator.pool;
    m_rtDescSetLayout = m_rtDescLayoutBuilder.build(engine->_device, VK_SHADER_STAGE_RAYGEN_BIT_KHR);

    VkDescriptorSetAllocateInfo allocateInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocateInfo.descriptorPool = m_rtDescPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &m_rtDescSetLayout;
    vkAllocateDescriptorSets(engine->_device, &allocateInfo, &m_rtDescSet);


    VkAccelerationStructureKHR tlas = m_tlas.accel;
    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
    descASInfo.accelerationStructureCount = 1;
    descASInfo.pAccelerationStructures = &tlas;

    m_rtDescWriter.clear();
    m_rtDescWriter.write_buffer(0, 0, 0, 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    m_rtDescWriter.writes[0].pNext = &descASInfo;
    m_rtDescWriter.write_image(1, engine->_drawImage.imageView, {}, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    m_rtDescWriter.update_set(engine->_device, m_rtDescSet);

    // add all to deletion queue
    engine->_mainDeletionQueue.push_function([=]() {
        vkDestroyDescriptorPool(engine->_device, m_rtDescPool, nullptr);
        vkDestroyDescriptorSetLayout(engine->_device, m_rtDescSetLayout, nullptr);
        m_rtDescAllocator.destroy_pool(engine->_device);
	});
}

//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void VulkanRayTracer::updateRtDescriptorSet()
{
    // update drawimage manually (without use of descriptor writer) to avoid updating the acceleration structure
    VkDescriptorImageInfo info{ {}, engine->_drawImage.imageView, VK_IMAGE_LAYOUT_GENERAL };

    VkWriteDescriptorSet wds{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    wds.dstSet = m_rtDescSet;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wds.pImageInfo = &info;
    wds.dstBinding = 1;

    vkUpdateDescriptorSets(engine->_device, 1, &wds, 0, nullptr);
}