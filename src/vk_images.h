
#pragma once 

#include <vulkan/vulkan.h>

namespace vkutil {

	void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);

	void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination,VkExtent2D srcSize, VkExtent2D dstSize);

	void copy_buffer_to_image(VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

	void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D imageSize);

	void clear_color_image_uint(VkCommandBuffer cmd, VkImage image, uint32_t r, uint32_t g, uint32_t b, uint32_t a);
} // namespace vkutil
