#pragma once

#include <vk_types.h>

namespace vk_shader {
bool load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule);
}
