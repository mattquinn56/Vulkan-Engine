#pragma once

#include <gpu_types.h>

// Fills the boilerplate fields of Vulkan create-info structs so call sites only
// state what varies.
namespace vk_init {

VkCommandPoolCreateInfo command_pool_create_info(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags = 0);
VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count = 1);
VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags = 0);
VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer cmd);

VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags = 0);
VkSemaphoreCreateInfo semaphore_create_info(VkSemaphoreCreateFlags flags = 0);
VkSemaphoreSubmitInfo semaphore_submit_info(VkPipelineStageFlags2 stageMask, VkSemaphore semaphore);

VkSubmitInfo2 submit_info(VkCommandBufferSubmitInfo* cmd, VkSemaphoreSubmitInfo* signalSemaphoreInfo,
                          VkSemaphoreSubmitInfo* waitSemaphoreInfo);
VkPresentInfoKHR present_info();

VkRenderingAttachmentInfo attachment_info(VkImageView view, VkClearValue* clear, VkImageLayout layout);
VkRenderingInfo rendering_info(VkExtent2D renderExtent, VkRenderingAttachmentInfo* colorAttachment,
                               VkRenderingAttachmentInfo* depthAttachment);

VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspectMask);
VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent);
VkImageViewCreateInfo imageview_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags);

} // namespace vk_init
