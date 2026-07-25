#include "gpu/vk_init.h"

// Designated initializers must follow struct declaration order, and any member
// left out is value-initialized to zero.

VkCommandPoolCreateInfo vk_init::command_pool_create_info(uint32_t queueFamilyIndex,
                                                          VkCommandPoolCreateFlags flags /*= 0*/) {
    return {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = flags,
        .queueFamilyIndex = queueFamilyIndex,
    };
}

VkCommandBufferAllocateInfo vk_init::command_buffer_allocate_info(VkCommandPool pool, uint32_t count /*= 1*/) {
    return {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = count,
    };
}

VkCommandBufferBeginInfo vk_init::command_buffer_begin_info(VkCommandBufferUsageFlags flags /*= 0*/) {
    return {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = flags,
    };
}

VkCommandBufferSubmitInfo vk_init::command_buffer_submit_info(VkCommandBuffer cmd) {
    return {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = cmd,
    };
}

VkFenceCreateInfo vk_init::fence_create_info(VkFenceCreateFlags flags /*= 0*/) {
    return {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = flags,
    };
}

VkSemaphoreCreateInfo vk_init::semaphore_create_info(VkSemaphoreCreateFlags flags /*= 0*/) {
    return {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .flags = flags,
    };
}

VkSemaphoreSubmitInfo vk_init::semaphore_submit_info(VkPipelineStageFlags2 stageMask, VkSemaphore semaphore) {
    return {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = semaphore,
        .value = 1,
        .stageMask = stageMask,
    };
}

VkSubmitInfo2 vk_init::submit_info(VkCommandBufferSubmitInfo* cmd, VkSemaphoreSubmitInfo* signalSemaphoreInfo,
                                   VkSemaphoreSubmitInfo* waitSemaphoreInfo) {
    return {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = waitSemaphoreInfo == nullptr ? 0u : 1u,
        .pWaitSemaphoreInfos = waitSemaphoreInfo,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = cmd,
        .signalSemaphoreInfoCount = signalSemaphoreInfo == nullptr ? 0u : 1u,
        .pSignalSemaphoreInfos = signalSemaphoreInfo,
    };
}

// Counts and pointers are filled in by the caller, which knows the swapchain.
VkPresentInfoKHR vk_init::present_info() {
    return {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
}

VkRenderingAttachmentInfo vk_init::attachment_info(VkImageView view, VkClearValue* clear, VkImageLayout layout) {
    VkRenderingAttachmentInfo info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = view,
        .imageLayout = layout,
        .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    if (clear) {
        info.clearValue = *clear;
    }
    return info;
}

VkRenderingInfo vk_init::rendering_info(VkExtent2D renderExtent, VkRenderingAttachmentInfo* colorAttachment,
                                        VkRenderingAttachmentInfo* depthAttachment) {
    return {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = VkRect2D{VkOffset2D{0, 0}, renderExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = colorAttachment,
        .pDepthAttachment = depthAttachment,
    };
}

VkImageSubresourceRange vk_init::image_subresource_range(VkImageAspectFlags aspectMask) {
    return {
        .aspectMask = aspectMask,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };
}

VkImageCreateInfo vk_init::image_create_info(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent) {
    return {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = extent,
        .mipLevels = 1,
        .arrayLayers = 1,
        // MSAA is off by default: one sample per pixel.
        .samples = VK_SAMPLE_COUNT_1_BIT,
        // Optimal tiling: driver picks the most efficient in-memory layout.
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usageFlags,
    };
}

VkImageViewCreateInfo vk_init::imageview_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags) {
    return {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange =
            {
                .aspectMask = aspectFlags,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
}
