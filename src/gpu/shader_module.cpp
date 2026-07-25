#include "gpu/shader_module.h"

#include <fstream>

bool vk_shader::load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule) {
    // Opening at the end lets tellg report the size directly.
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return false;
    }

    size_t fileSize = (size_t)file.tellg();

    // SPIR-V is consumed as 32-bit words, and the buffer must satisfy uint32_t
    // alignment, so read into a uint32_t vector rather than a byte array.
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;

    // codeSize is in bytes even though pCode is a word pointer.
    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode = buffer.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return false;
    }
    *outShaderModule = shaderModule;
    return true;
}
