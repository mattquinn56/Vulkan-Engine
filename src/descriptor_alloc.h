#pragma once

#include <vector>
#include <gpu_types.h>
#include <deque>
#include <span>
struct DescriptorLayoutBuilder
{

    std::vector<VkDescriptorSetLayoutBinding> bindings;

    void add_binding(uint32_t binding, VkDescriptorType type, uint32_t descriptorCount = 1);
    void clear();
    VkDescriptorSetLayout build(VkDevice device, VkShaderStageFlags shaderStages);
};
struct DescriptorWriter
{
    int index{0};
    std::vector<std::pair<int, int>> writeArrayIndices; // writes index, imageInfosArray index
    std::deque<VkDescriptorImageInfo> imageInfos;
    std::vector<VkDescriptorImageInfo> imageInfosArray;
    std::deque<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkWriteDescriptorSet> writes;

    void write_image(int binding, VkImageView image, VkSampler sampler, VkImageLayout layout, VkDescriptorType type);
    void write_image_array(int binding, std::vector<VkImageView> images, std::vector<VkSampler> sampler,
                           VkImageLayout layout, VkDescriptorType type);
    void write_buffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type);

    void clear();
    void update_set(VkDevice device, VkDescriptorSet set);
};
struct DescriptorAllocator
{

    struct PoolSizeRatio
    {
        VkDescriptorType type{VK_DESCRIPTOR_TYPE_MAX_ENUM};
        float ratio{0.0f};
    };

    VkDescriptorPool pool{VK_NULL_HANDLE};

    void init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios);
    void clear_descriptors(VkDevice device);
    void destroy_pool(VkDevice device);

    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);
};
struct DescriptorAllocatorGrowable
{
  public:
    struct PoolSizeRatio
    {
        VkDescriptorType type{VK_DESCRIPTOR_TYPE_MAX_ENUM};
        float ratio{0.0f};
    };

    void init_pools(VkDevice device, uint32_t initialSets, std::span<PoolSizeRatio> poolRatios);
    void clear_pools(VkDevice device);
    void destroy_pools(VkDevice device);

    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);

  private:
    VkDescriptorPool get_pool(VkDevice device);
    VkDescriptorPool create_pool(VkDevice device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios);

    std::vector<PoolSizeRatio> ratios;
    std::vector<VkDescriptorPool> fullPools;
    std::vector<VkDescriptorPool> readyPools;
    uint32_t setsPerPool{0};
};
