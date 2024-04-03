﻿#include <vk_descriptors.h>
#include "vk_initializers.h"
#include <memory>

//> descriptor_bind
void DescriptorLayoutBuilder::add_binding(uint32_t binding, VkDescriptorType type, uint32_t descriptorCount /*== 1*/)
{
    VkDescriptorSetLayoutBinding newbind {};
    newbind.binding = binding;
    newbind.descriptorCount = 1;
    newbind.descriptorType = type;

    bindings.push_back(newbind);
}

void DescriptorLayoutBuilder::clear()
{
    bindings.clear();
}
//< descriptor_bind

//> descriptor_layout
VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shaderStages)
{
    for (auto& b : bindings) {
        b.stageFlags |= shaderStages;
    }

    VkDescriptorSetLayoutCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.pNext = nullptr;

    info.pBindings = bindings.data();
    info.bindingCount = (uint32_t)bindings.size();
    info.flags = 0;

    VkDescriptorSetLayout set;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set));

    return set;
}
//< descriptor_layout

//> descriptor_pool_init
void DescriptorAllocator::init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios)
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (PoolSizeRatio ratio : poolRatios) {
        poolSizes.push_back(VkDescriptorPoolSize{
            .type = ratio.type,
            .descriptorCount = uint32_t(ratio.ratio * maxSets)
        });
    }

	VkDescriptorPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	pool_info.flags = 0;
	pool_info.maxSets = maxSets;
	pool_info.poolSizeCount = (uint32_t)poolSizes.size();
	pool_info.pPoolSizes = poolSizes.data();

	vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device)
{
    vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device)
{
    vkDestroyDescriptorPool(device,pool,nullptr);
}
//< descriptor_pool_init
//> descriptor_alloc
VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo allocInfo = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.pNext = nullptr;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));

    return ds;
}
//< descriptor_alloc
//> write_image
void DescriptorWriter::write_image(int binding, VkImageView image, VkSampler sampler,  VkImageLayout layout, VkDescriptorType type)
{
    std::vector<VkDescriptorImageInfo> ordWrites;
    ordWrites.reserve(1); // Seems you're working with a single image here
    VkDescriptorImageInfo info = {
        .sampler = sampler, // Assuming 'sampler' is defined elsewhere
        .imageView = image, // Assuming 'image' is defined elsewhere
        .imageLayout = layout // Assuming 'layout' is defined elsewhere
    };

    // Since you're directly working with the structures, no need for dynamic allocation
    ordWrites.push_back(info);
    imageInfos.push_back(info); // Assuming 'imageInfos' is meant to store these for later use
    imageArrayInfos.push_back(ordWrites); // Extend lifetime by storing in a member variable

    // Note: Ensure 'imageArrayInfos' is a class member that's properly managed to extend the lifetime of 'ordWrites'

    // Now create the write descriptor
    auto write = std::make_shared<VkWriteDescriptorSet>(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = static_cast<uint32_t>(binding),
        .descriptorCount = 1,
        .descriptorType = type,
        .pImageInfo = &imageArrayInfos.back().front() // Point to the recently added 'ordWrites' data
    });


    // Assuming 'writes' is a class member vector of VkWriteDescriptorSet, not pointers
    writes.emplace_back(write);
}
//< write_image
//> write_image_array
void DescriptorWriter::write_image_array(int binding, std::span<VkImageView> images, VkSampler sampler, VkImageLayout layout, VkDescriptorType type)
{
    std::vector<VkDescriptorImageInfo> ordWrites;
    ordWrites.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        auto elem = VkDescriptorImageInfo{
            .sampler = sampler,
            .imageView = images[i],
            .imageLayout = layout
            };
        imageInfos.emplace_back(elem);
        ordWrites.emplace_back(elem);
    }
    imageArrayInfos.push_back(std::move(ordWrites));

    std::shared_ptr<VkWriteDescriptorSet> write_ptr = std::make_shared<VkWriteDescriptorSet>(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = static_cast<uint32_t>(binding),
        .descriptorCount = (uint32_t)images.size(),
        .descriptorType = type,
        .pImageInfo = imageArrayInfos.back().data(), // This is not correct for some reason
    });

    writes.push_back(write_ptr);
}
//< write_image_array
//> write_buffer
void DescriptorWriter::write_buffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type)
{
    auto info = new VkDescriptorBufferInfo{
        .buffer = buffer,
        .offset = offset,
        .range = size
    };

    bufferInfos.emplace_back(*info);

    // create shared ptr with copy of write
    std::shared_ptr<VkWriteDescriptorSet> write_ptr = std::shared_ptr<VkWriteDescriptorSet>(new VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = static_cast<uint32_t>(binding),
        .descriptorCount = 1,
        .descriptorType = type,
        .pBufferInfo = info,
    });

	writes.push_back(write_ptr);
}
//< write_buffer
//> writer_end
void DescriptorWriter::clear()
{
    imageInfos.clear();
    //imageInfosArray.clear();
    writes.clear();
    bufferInfos.clear();
}

void DescriptorWriter::update_set(VkDevice device, VkDescriptorSet set)
{

    // write dstSet for each write pointer
    std::vector<VkWriteDescriptorSet> writes_;
    for (auto& write : writes) {
        write->dstSet = set;
		writes_.push_back(*write);
	}

    vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes_.data(), 0, nullptr);
}
//< writer_end
//> growpool_2
void DescriptorAllocatorGrowable::init(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios)
{
    ratios.clear();
    
    for (auto r : poolRatios) {
        ratios.push_back(r);
    }
	
    VkDescriptorPool newPool = create_pool(device, maxSets, poolRatios);

    setsPerPool = maxSets * 1.5; //grow it next allocation

    readyPools.push_back(newPool);
}

void DescriptorAllocatorGrowable::clear_pools(VkDevice device)
{ 
    for (auto p : readyPools) {
        vkResetDescriptorPool(device, p, 0);
    }
    for (auto p : fullPools) {
        vkResetDescriptorPool(device, p, 0);
        readyPools.push_back(p);
    }
    fullPools.clear();
}

void DescriptorAllocatorGrowable::destroy_pools(VkDevice device)
{
	for (auto p : readyPools) {
		vkDestroyDescriptorPool(device, p, nullptr);
	}
    readyPools.clear();
	for (auto p : fullPools) {
		vkDestroyDescriptorPool(device,p,nullptr);
    }
    fullPools.clear();
}
//< growpool_2

//> growpool_1
VkDescriptorPool DescriptorAllocatorGrowable::get_pool(VkDevice device)
{       
    VkDescriptorPool newPool;
    if (readyPools.size() != 0) {
        newPool = readyPools.back();
        readyPools.pop_back();
    }
    else {
	    //need to create a new pool
	    newPool = create_pool(device, setsPerPool, ratios);

	    setsPerPool = setsPerPool * 1.5;
	    if (setsPerPool > 4092) {
		    setsPerPool = 4092;
	    }
    }   

    return newPool;
}

VkDescriptorPool DescriptorAllocatorGrowable::create_pool(VkDevice device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios)
{
	std::vector<VkDescriptorPoolSize> poolSizes;
	for (PoolSizeRatio ratio : poolRatios) {
		poolSizes.push_back(VkDescriptorPoolSize{
			.type = ratio.type,
			.descriptorCount = uint32_t(ratio.ratio * setCount)
		});
	}

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = 0;
	pool_info.maxSets = setCount;
	pool_info.poolSizeCount = (uint32_t)poolSizes.size();
	pool_info.pPoolSizes = poolSizes.data();

	VkDescriptorPool newPool;
	vkCreateDescriptorPool(device, &pool_info, nullptr, &newPool);
    return newPool;
}
//< growpool_1

//> growpool_3
VkDescriptorSet DescriptorAllocatorGrowable::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
    //get or create a pool to allocate from
    VkDescriptorPool poolToUse = get_pool(device);

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.pNext = nullptr;
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = poolToUse;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &layout;

	VkDescriptorSet ds;
	VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &ds);

    //allocation failed. Try again
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {

        fullPools.push_back(poolToUse);
    
        poolToUse = get_pool(device);
        allocInfo.descriptorPool = poolToUse;

       VK_CHECK( vkAllocateDescriptorSets(device, &allocInfo, &ds));
    }
  
    readyPools.push_back(poolToUse);
    return ds;
}
//< growpool_3