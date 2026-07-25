#include "rt_engine.h"

#include "descriptor_alloc.h"
#include "gltf_import.h"
#include "image_utils.h"
#include "ray_tracing_pipeline.h"
#include "shader_module.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <gpu_types.h>
#include <vk_init.h>

#include "VkBootstrap.h"

#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

#include <stb_image.h>

AllocatedBuffer RtEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = memoryUsage;
    vmaallocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation,
                             &newBuffer.info));

    return newBuffer;
}

AllocatedBuffer RtEngine::create_buffer_data(VkDeviceSize size, const void* data, VkBufferUsageFlags usage,
                                             const VmaMemoryUsage memUsage)
{

    AllocatedBuffer resultBuffer = create_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, memUsage);

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAllocation;
    vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo, &stagingBuffer, &stagingAllocation, nullptr);

    void* mappedData;
    vmaMapMemory(_allocator, stagingAllocation, &mappedData);
    memcpy(mappedData, data, size);
    vmaUnmapMemory(_allocator, stagingAllocation);

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    immediate_submit(
        [&](VkCommandBuffer cmd) { vkCmdCopyBuffer(cmd, stagingBuffer, resultBuffer.buffer, 1, &copyRegion); });
    vmaDestroyBuffer(_allocator, stagingBuffer, stagingAllocation);

    return resultBuffer;
}

AllocatedBuffer RtEngine::allocate_and_bind_buffer(VkBuffer buffer, VmaMemoryUsage memoryUsage)
{
    if (_allocator == VK_NULL_HANDLE || buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Invalid allocator or buffer handle");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(_device, buffer, &memRequirements);

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memoryUsage;

    VmaAllocation allocation;
    VmaAllocationInfo allocationInfo;
    if (vmaAllocateMemoryForBuffer(_allocator, buffer, &allocInfo, &allocation, &allocationInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate memory for buffer");
    }

    if (vmaBindBufferMemory(_allocator, allocation, buffer) != VK_SUCCESS) {
        vmaFreeMemory(_allocator, allocation);
        throw std::runtime_error("Failed to bind buffer memory");
    }

    AllocatedBuffer allocatedBuffer;
    allocatedBuffer.buffer = buffer;
    allocatedBuffer.allocation = allocation;
    allocatedBuffer.info = allocationInfo;

    return allocatedBuffer;
}

VkDeviceAddress RtEngine::get_buffer_device_address(VkDevice device, VkBuffer buffer)
{
    if (buffer == VK_NULL_HANDLE)
        return 0ULL;

    VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &info);
}
// Creates a device-local buffer and uploads data into it through a staging copy.

AllocatedImage RtEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    VkImageCreateInfo img_info = vk_init::image_create_info(format, usage, size);

    // treat 3D extents as 3D textures
    if (size.depth > 1) {
        img_info.imageType = VK_IMAGE_TYPE_3D;
        img_info.arrayLayers = 1;
    }

    if (mipmapped) {
        const uint32_t mx = std::max({size.width, size.height, size.depth});
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(mx))) + 1;
    }

    VmaAllocationCreateInfo allocinfo{};
    allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vmaCreateImage(_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr));

    VkImageAspectFlags aspectFlag =
        (format == VK_FORMAT_D32_SFLOAT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageViewCreateInfo view_info = vk_init::imageview_create_info(format, newImage.image, aspectFlag);

    // 3D view when needed
    if (size.depth > 1) {
        view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
        view_info.subresourceRange.layerCount = 1;
    }

    view_info.subresourceRange.levelCount = img_info.mipLevels;
    VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));
    return newImage;
}

AllocatedImage RtEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
                                      bool mipmapped)
{
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadbuffer =
        create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(
        size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd) {
        vk_img::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;

        vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &copyRegion);

        if (mipmapped) {
            vk_img::generate_mipmaps(cmd, new_image.image,
                                     VkExtent2D{new_image.imageExtent.width, new_image.imageExtent.height});
        } else {
            vk_img::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    });
    destroy_buffer(uploadbuffer);
    return new_image;
}

void RtEngine::destroy_image(const AllocatedImage& img)
{
    if (img.imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(_device, img.imageView, nullptr);
    }
    if (img.image != VK_NULL_HANDLE) {
        vmaDestroyImage(_allocator, img.image, img.allocation);
    }
}

void RtEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
    }
}

GPUMeshBuffers RtEngine::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    newSurface.vertexCount = static_cast<int>(vertices.size());

    newSurface.vertexBuffer = create_buffer(vertexBufferSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                            VMA_MEMORY_USAGE_GPU_ONLY);

    VkBufferDeviceAddressInfo deviceAdressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                               .buffer = newSurface.vertexBuffer.buffer};
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAdressInfo);

    newSurface.indexBuffer =
        create_buffer(indexBufferSize,
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                      VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging =
        create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.info.pMappedData;

    // copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{0};
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{0};
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    destroy_buffer(staging);

    return newSurface;
}

AllocatedImage RtEngine::load_image_from_file(std::string path)
{
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    VkDeviceSize imageSize = texWidth * texHeight * 4; // RGBA, forced by STBI_rgb_alpha

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = static_cast<uint32_t>(texWidth);
    imageInfo.extent.height = static_cast<uint32_t>(texHeight);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkImage image;
    VmaAllocation allocation;
    vmaCreateImage(_allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr);

    // The image is device-local, so pixels go via a host-visible staging buffer.
    VkBuffer stagingBuffer;
    VmaAllocation stagingBufferAllocation;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo bufferAllocInfo = {};
    bufferAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    vmaCreateBuffer(_allocator, &bufferInfo, &bufferAllocInfo, &stagingBuffer, &stagingBufferAllocation, nullptr);

    void* data;
    vmaMapMemory(_allocator, stagingBufferAllocation, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vmaUnmapMemory(_allocator, stagingBufferAllocation);

    stbi_image_free(pixels);

    immediate_submit([&](VkCommandBuffer cmd) {
        vk_img::transition_image(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vk_img::copy_buffer_to_image(cmd, stagingBuffer, image, static_cast<uint32_t>(texWidth),
                                     static_cast<uint32_t>(texHeight));
        vk_img::transition_image(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    vmaDestroyBuffer(_allocator, stagingBuffer, stagingBufferAllocation);

    VkImageView imageView;
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(_device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    return AllocatedImage{image,
                          imageView,
                          allocation,
                          {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1},
                          VK_FORMAT_R8G8B8A8_SRGB};
}
