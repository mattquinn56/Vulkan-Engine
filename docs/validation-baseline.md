# Validation Baseline

This document preserves the pre-fix Phase 1 baseline. Phase 2 results are in
`docs/validation-phase2.md`.

> Historical record. It describes runs made when the engine still had a
> rasterizer and a `--render-path` flag. Both have since been removed; hardware
> ray tracing is the only render path. References below to raster mode and to
> `--render-path` describe the state at the time of the run, not current
> behavior. The VUID observations are left unedited on purpose.

Date: 2026-07-11

Commit: `a727b06` (`GPU logging`)

## Environment

- Windows, Debug configuration
- NVIDIA GeForce RTX 4080 SUPER
- Vulkan validation layer reporting spec version 1.3.296
- Default engine mode: ray tracing, TAA, progressive Monte Carlo, and ACES/sRGB tonemapping
- Executable: `bin/Debug/engine.exe`
- Working directory: `bin/Debug`

Deterministic startup options used by the matrix are:

```text
--render-path=raytrace|raster
--aa=taa|adaptive
--tonemap=on|off
```

The chosen configuration is printed as the first line of stdout. Unknown options
exit with status 2 before engine initialization.

Raw logs, event timelines, and screenshots are under the ignored directory
`out/validation-baseline/`.

Summarize one or more captured logs from the repository root with:

```powershell
.\tools\summarize-validation.ps1 .\out\validation-baseline\*\stdout.log
```

## Scenario Results

### Startup and normal shutdown

The engine rendered for six seconds and received a normal window-close message. It
then emitted invalid-handle and duplicate-destruction errors during cleanup and
crashed with Windows exception `0xc0000005`. Windows Error Reporting recorded the
fault in `engine.exe` at offset `0x68a91` for test process 25208.

Observed validation identifiers:

| Count | Identifier |
| ---: | --- |
| 778 | `VUID-VkImageMemoryBarrier2-oldLayout-01213` |
| 389 | `VUID-VkBlitImageInfo2-dstImage-00224` |
| 12 | `UNASSIGNED-Threading-Info` |
| 2 | `VUID-vkDestroyPipelineLayout-pipelineLayout-parameter` |
| 2 | `VUID-vkDestroyDescriptorPool-descriptorPool-parameter` |
| 1 | `VUID-vkDestroyDescriptorSetLayout-descriptorSetLayout-parameter` |
| 1 | `VUID-vkDestroyPipeline-pipeline-parameter` |
| 1 | `VUID-VkRayTracingPipelineCreateInfoKHR-layout-07990` |

The `UNASSIGNED-Threading-Info` messages report that the corresponding pipeline,
pipeline-layout, descriptor-pool, or descriptor-set-layout handle could not be
found after an earlier destruction. The log ends mid-message at the crash.

### Single shrink

The client area started at 1250x800 and was resized once to 884x561. The engine
remained alive, but did not process a window close within 15 seconds and had to be
stopped by its test PID.

A resize-specific error appeared:

`VUID-VkBlitImageInfo2-dstOffset-00248` reports a blit destination X extent of
1250 against the newly created swapchain image width of 884. This demonstrates
that rendering retained the old extent after swapchain recreation.

### Minimize and restore

The process remained alive and Windows reported the window as responsive through
minimize and restore. Screenshots confirmed that it presented before minimizing
and after restoring. Restoring entered the resize family of code paths, after
which a window close did not complete within 15 seconds and the test process had
to be stopped.

### Render-mode matrix

Every configuration created a window and remained alive for the five-second
rendering observation. Every configuration then crashed with Windows exception
`0xc0000005` while processing normal shutdown.

| Render path | AA | Tonemap | Messages | Rendering identifiers |
| --- | --- | --- | ---: | --- |
| Ray tracing | TAA | On | 309 | `layout-07990`, `oldLayout-01213`, `dstImage-00224` |
| Ray tracing | Adaptive | On | 20 | `layout-07990` |
| Ray tracing | TAA | Off | 341 | `layout-07990`, `oldLayout-01213`, `dstImage-00224` |
| Raster | TAA | On | 23 | `layout-07990`, `oldLayout-01213`, `dstImage-00224` |
| Raster | Adaptive | On | 20 | `layout-07990` |
| Raster | TAA | Off | 952 | `layout-07990`, `oldLayout-01213`, `dstImage-00224` |

Message totals vary with rendered frame count and crash timing. The identifier
sets are the meaningful result:

- The transfer usage/layout errors are specific to the TAA path.
- Tonemapping does not introduce another unique rendering VUID.
- The ray-tracing pipeline-layout error also occurs in raster mode because the
  ray-tracing pipeline is still created unconditionally.
- All modes emit invalid-handle/double-destruction errors during shutdown. The
  exact final set varies because cleanup crashes before every callback completes.

### Window-state matrix

These tests used raster rendering, Adaptive AA, and tonemapping to keep the known
per-frame TAA errors out of the resize logs.

Maximize/restore changed the client area from 1250x800 to 2560x1417 and back. The
window remained alive and responsive, but it did not process a close within 12
seconds and had to be stopped by its test PID.

A burst of five sizes ended at a client area of 1184x711. The process remained
alive but again timed out during close. Its first resize recorded these additional
identifiers before the log was truncated:

- `VUID-VkBlitImageInfo2-dstOffset-00248`
- `VUID-VkBlitImageInfo2-dstOffset-00249`
- `VUID-VkBlitImageInfo2-pRegions-00216`
- `VUID-VkRenderingInfo-pNext-06079`
- `VUID-VkRenderingInfo-pNext-06080`

They report that both the blit region and ImGui dynamic-rendering area retain the
old extent after the swapchain images have changed size.

## Source Correlation

These are source observations to verify while implementing fixes, not accepted
behavior:

- `resize_swapchain()` replaces the swapchain and LDR/TAA/MC resources, but does
  not recreate `_drawImage` or `_depthImage`.
- The code stores the requested SDL window size rather than the actual extent
  selected by `SwapchainBuilder`.
- `init_taa_resources()` and `init_mc_resources()` recreate persistent layouts,
  pipelines, and descriptor sets on every resize.
- several deletion callbacks capture mutable member handles, allowing multiple
  callbacks to destroy the latest handle rather than the handle existing when
  each callback was registered.
- `destroy_swapchain()` destroys the swapchain before its image views.

## Phase 1 Status

The planned startup, render-path, AA, tonemapping, resize, minimize/restore, and
normal-shutdown scenarios now have unique-identifier inventories and behavioral
outcomes. Phase 1 is complete. Counts should be regenerated after every fix; this
document is a pre-fix baseline, not an allowlist.
