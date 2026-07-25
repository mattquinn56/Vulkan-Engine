#include "core/rt_engine.h"

#include "gpu/descriptor_alloc.h"
#include "scene/gltf_import.h"
#include "gpu/image_utils.h"
#include "passes/ray_tracing_pipeline.h"
#include "platform/screenshot.h"
#include "gpu/shader_module.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include "core/gpu_types.h"
#include "gpu/vk_init.h"

#include "VkBootstrap.h"

#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

#include <stb_image.h>

void RtEngine::draw() {
    // Wait for the previous frame using this FrameContext to finish
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, VK_TRUE, UINT64_MAX));

    // Per-frame cleanup
    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);

    // Acquire next swapchain image, signaling this frame's _swapchainSemaphore when ready
    uint32_t imageIndex = 0;
    VkResult acquire = vkAcquireNextImageKHR(_device, _swapchain, UINT64_MAX, get_current_frame()._swapchainSemaphore,
                                             VK_NULL_HANDLE, &imageIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
        _resizeRequested = true;
        return;
    }
    if (acquire == VK_SUBOPTIMAL_KHR) {
        _resizeRequested = true;
    } // continue; we'll still draw this frame

    // We are going to submit new work for this frame
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    // Reset and begin command buffer recording
    VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));
    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    VkCommandBufferBeginInfo beginInfo =
        vk_init::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Make draw/depth images writable for compute/graphics
    vk_img::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // Ray tracing writes _drawImage directly, in GENERAL layout.
    _rayTracer->raytrace(cmd);

    // If UI changed something that affects appearance, clear histories now
    if (_resetAccumNextFrame) {
        // Reset MC accumulators (count=0, copy current _drawImage into accumColor)
        reset_monte_carlo_history(cmd);

        // Reset/seed TAA history from current frame so blending starts clean
        if (_aaMode == AAMode::TAA) {
            seed_taa_history(this, cmd);
            _taaInitialized = true;
        }
        _resetAccumNextFrame = false;
    }

    bool doProgressive = _progressiveMonteCarlo && (_aaMode == AAMode::TAA ? !_cameraMoving : true);

    // Bind descriptors for MC accumulation
    {
        DescriptorWriter w;
        w.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.write_image(1, _mcAccumColor.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.write_image(2, _mcAccumCount.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.write_image(3, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.update_set(_device, _mcSet);
    }

    if (doProgressive) {
        // Optional: delay reset for a couple frames after movement ends
        static int resetCooldown = 0;
        if (_cameraMoving)
            resetCooldown = _monteCarloResetFrames;
        if (resetCooldown > 0) {
            reset_monte_carlo_history(cmd);
            resetCooldown--;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _mcPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _mcPipeLayout, 0, 1, &_mcSet, 0, nullptr);

        struct
        {
            float perFrameSpp;
            float movingFlag;
        } pc{float(_monteCarloSamplesPerFrame), _cameraMoving ? 1.f : 0.f};
        vkCmdPushConstants(cmd, _mcPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t gx = (_windowExtent.width + 7) / 8;
        uint32_t gy = (_windowExtent.height + 7) / 8;
        vkCmdDispatch(cmd, gx, gy, 1);

        // Note: mc resolve writes back into _drawImage (binding 3), so TAA will consume it next
    } else {
        reset_monte_carlo_history(cmd);
    }

    if (_aaMode == AAMode::TAA) {
        // Make RT writes visible to compute reads
        VkImageMemoryBarrier2 imgBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        imgBarrier.srcStageMask =
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR; // or COLOR_ATTACHMENT_OUTPUT if raster
        imgBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        imgBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        imgBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imgBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imgBarrier.image = _drawImage.image;
        imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &imgBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);

        if (!_taaInitialized) {
            seed_taa_history(this, cmd);
            _taaInitialized = true;
        }
        int prev = _taaIndex;
        int next = 1 - _taaIndex;

        // seed history when we first switch to TAA or on strong movement with zero alpha
        if (_cameraMoving && _taaMovingAlpha == 0.0f) {
            seed_taa_history(this, cmd);
            prev = _taaIndex;
            next = 1 - _taaIndex;
        }

        {
            DescriptorWriter w;
            w.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.write_image(1, _taaHistory[prev].imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.write_image(2, _taaHistory[next].imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.update_set(_device, _taaSet[next]);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _taaPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _taaPipelineLayout, 0, 1, &_taaSet[next], 0,
                                nullptr);

        struct
        {
            float alpha;
            float clampK;
        } pc{_cameraMoving ? _taaMovingAlpha : _taaAlpha, _taaClamp};
        vkCmdPushConstants(cmd, _taaPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t gx = (_windowExtent.width + 7) / 8;
        uint32_t gy = (_windowExtent.height + 7) / 8;
        vkCmdDispatch(cmd, gx, gy, 1);

        VkImageMemoryBarrier2 histBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        histBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        histBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        histBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        histBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        histBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        histBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        histBarrier.image = _taaHistory[next].image;
        histBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        dep = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &histBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);

        vk_img::transition_image(cmd, _taaHistory[next].image, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        vk_img::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vk_img::copy_image_to_image(cmd, _taaHistory[next].image, _drawImage.image, _windowExtent, _windowExtent);
        vk_img::transition_image(cmd, _taaHistory[next].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 VK_IMAGE_LAYOUT_GENERAL);
        vk_img::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

        _taaIndex = next;
    }

    // --- POST: ACES + sRGB (optional) ---
    if (_enableTonemap) {
        // Make sure _drawImage writes are visible to compute
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.image = _drawImage.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // Transition _ldrImage to GENERAL on first use (or from last frame's TRANSFER_SRC)
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
            b.srcAccessMask = 0;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            b.oldLayout = _ldrNeedsInit ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.image = _ldrImage.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
            _ldrNeedsInit = false;
        }

        // Bind descriptors and dispatch post compute
        {
            DescriptorWriter w;
            w.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.write_image(1, _ldrImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.update_set(_device, _postSet);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _postPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _postPipeLayout, 0, 1, &_postSet, 0, nullptr);

        vkCmdPushConstants(cmd, _postPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &_exposure);

        uint32_t gx = (_windowExtent.width + 7) / 8;
        uint32_t gy = (_windowExtent.height + 7) / 8;
        vkCmdDispatch(cmd, gx, gy, 1);

        // Make post-process writes visible before copying from the LDR image.
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.image = _ldrImage.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // Transition the swapchain image before copying the LDR result into it.
        vk_img::transition_image(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkExtent2D extent{_windowExtent.width, _windowExtent.height};
        vk_img::copy_image_to_image(cmd, _ldrImage.image, _swapchainImages[imageIndex], extent, extent);
    } else {
        // No tonemap: copy HDR directly (debug)
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.image = _drawImage.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        vk_img::transition_image(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkExtent2D extent{_windowExtent.width, _windowExtent.height};
        vk_img::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[imageIndex], extent, extent);
    }

    // Draw ImGui on the swapchain image
    vk_img::transition_image(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    draw_imgui(cmd, _swapchainImageViews[imageIndex]);

    // Diagnostic capture reads the presented image, so it runs after ImGui and
    // before the present transition.
    const bool capturing = !_screenshotPath.empty() && !_screenshotDone && _frameNumber >= _screenshotFrame;
    AllocatedBuffer captureBuffer{};
    if (capturing) {
        capture_swapchain(cmd, imageIndex, captureBuffer);
    } else {
        vk_img::transition_image(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    }

    VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit: wait on image-available for this frame, signal render-finished for this frame
    VkCommandBufferSubmitInfo cmdInfo = vk_init::command_buffer_submit_info(cmd);
    VkSemaphoreSubmitInfo waitInfo = vk_init::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
                                                                    get_current_frame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo =
        vk_init::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame()._renderSemaphore);
    VkSubmitInfo2 submitInfo = vk_init::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submitInfo, get_current_frame()._renderFence));

    // Present: wait on this frame's render-finished semaphore
    VkPresentInfoKHR presentInfo = vk_init::present_info();
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;

    VkResult present = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (present == VK_ERROR_OUT_OF_DATE_KHR || present == VK_SUBOPTIMAL_KHR) {
        _resizeRequested = true;
    }

    if (capturing) {
        // Idle so the copy has landed before the buffer is read.
        VK_CHECK(vkDeviceWaitIdle(_device));
        write_capture(captureBuffer);
        destroy_buffer(captureBuffer);
        _screenshotDone = true;
    }

    _frameNumber++;
}

void RtEngine::update_global_descriptor() {

    // Allocated per frame rather than reused, so a frame in flight never has its
    // scene data overwritten. The frame's deletion queue reclaims it.
    AllocatedBuffer gpuSceneDataBuffer =
        create_buffer(sizeof(GPUFrameConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    get_current_frame()._deletionQueue.push_function([=, this]() { destroy_buffer(gpuSceneDataBuffer); });

    GPUFrameConstants* sceneUniformData = (GPUFrameConstants*)gpuSceneDataBuffer.info.pMappedData;
    *sceneUniformData = _sceneData;

    _globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);

    {
        DescriptorWriter writer;
        writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUFrameConstants), 0,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(_device, _globalDescriptor);
    }

    // Same per-frame lifetime for the object description storage buffer.
    _objectDescriptionBuffer = create_buffer_data(sizeof(ObjDesc) * _drawContext.objectDescriptions.size(),
                                                  _drawContext.objectDescriptions.data(),
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Delete it only after this frame's fence signals and the descriptor is no longer in use.
    const AllocatedBuffer objectDescriptionBuffer = _objectDescriptionBuffer;
    get_current_frame()._deletionQueue.push_function(
        [this, objectDescriptionBuffer]() { destroy_buffer(objectDescriptionBuffer); });

    _objDescSet = get_current_frame()._frameDescriptors.allocate(_device, _objDescLayout);

    {
        DescriptorWriter writer;
        writer.write_buffer(0, _objectDescriptionBuffer.buffer, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(_device, _objDescSet);
    }
}

// Copies the just-rendered swapchain image into a host-visible buffer and
// leaves the image ready to present.
void RtEngine::capture_swapchain(VkCommandBuffer cmd, uint32_t imageIndex, AllocatedBuffer& dst) {
    const VkDeviceSize bytes = VkDeviceSize(_windowExtent.width) * _windowExtent.height * 4;
    dst = create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

    vk_img::transition_image(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {_windowExtent.width, _windowExtent.height, 1};
    vkCmdCopyImageToBuffer(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.buffer, 1,
                           &region);

    vk_img::transition_image(cmd, _swapchainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

void RtEngine::write_capture(const AllocatedBuffer& src) {
    const uint8_t* mapped = static_cast<const uint8_t*>(src.info.pMappedData);
    if (mapped == nullptr) {
        fmt::print(stderr, "Screenshot failed: capture buffer is not host-visible\n");
        return;
    }

    // The swapchain is VK_FORMAT_B8G8R8A8_UNORM; PNG wants RGBA.
    const size_t pixels = size_t(_windowExtent.width) * _windowExtent.height;
    std::vector<uint8_t> rgba(pixels * 4);
    for (size_t i = 0; i < pixels; i++) {
        rgba[i * 4 + 0] = mapped[i * 4 + 2];
        rgba[i * 4 + 1] = mapped[i * 4 + 1];
        rgba[i * 4 + 2] = mapped[i * 4 + 0];
        rgba[i * 4 + 3] = 255;
    }

    if (screenshot::write_png(_screenshotPath, _windowExtent.width, _windowExtent.height, rgba.data())) {
        fmt::println("Screenshot written: {} ({}x{}, frame {})", _screenshotPath, _windowExtent.width,
                     _windowExtent.height, _frameNumber);
    } else {
        fmt::print(stderr, "Screenshot failed: could not write {}\n", _screenshotPath);
    }
}
