
#pragma once 

#include <vulkan/vulkan.h>

namespace vkutil {

	void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);

	void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination,VkExtent2D srcSize, VkExtent2D dstSize);

	void copy_buffer_to_image(VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

	void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D imageSize);
} // namespace vkutil
