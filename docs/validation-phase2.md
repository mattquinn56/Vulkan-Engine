# Phase 2 Validation Results

Date: 2026-07-11

## Changes

- Ray-tracing descriptor set 0 binding 2 now consistently uses
  `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` in its layout, pool, and write.
- The environment-map descriptor now includes `_defaultSamplerLinear`, matching
  the miss shader's `sampler2D` declaration.
- The initial `_drawImage` now includes `VK_IMAGE_USAGE_TRANSFER_DST_BIT`,
  matching the TAA copy-back operation.

`src/vk_raytracer.cpp` also had one Windows-1252 smart-apostrophe byte normalized
to an ASCII apostrophe so the file is valid UTF-8 and can be patched reliably.

## Verification

The Debug engine built successfully. The existing compiler warnings remain, but
this phase introduced no build errors.

The six deterministic configurations from the Phase 1 matrix each created a
window and rendered for four seconds:

| Render path | AA | Tonemap | Target VUIDs |
| --- | --- | --- | --- |
| Ray tracing | TAA | On | None |
| Ray tracing | Adaptive | On | None |
| Ray tracing | TAA | Off | None |
| Raster | TAA | On | None |
| Raster | Adaptive | On | None |
| Raster | TAA | Off | None |

A recursive search of `out/validation-phase2/` found no occurrence of:

- `VUID-VkRayTracingPipelineCreateInfoKHR-layout-07990`
- `VUID-VkImageMemoryBarrier2-oldLayout-01213`
- `VUID-VkBlitImageInfo2-dstImage-00224`

This is a stronger result than the pre-fix baseline: TAA and ray tracing no longer
produce rendering-time validation messages during a fixed-size run.

A foregrounded screenshot also confirmed that the ray-traced living-room scene
still renders with TAA and tonemapping after the descriptor change. The screenshot
is stored at `out/validation-phase2/visual-check/`.

## Remaining Failures

Every mode still emits the known invalid-handle/double-destruction messages during
cleanup and then crashes with Windows exception `0xc0000005`. Those errors are
outside the two Phase 2 resource contracts.

Both a single resize and a burst-resize run remained alive but failed to process
a close within 12 seconds. They had to be stopped by their individual test PIDs.
The post-fix runs stalled before the first post-resize frame flushed validation
messages, so they produced neither the fixed VUIDs nor a clean resize result.

Raw logs and result JSON are under the ignored directory
`out/validation-phase2/`. Phase 3 should address extent-dependent ownership,
swapchain recreation, descriptor refresh, and cleanup before this same matrix is
run again.
