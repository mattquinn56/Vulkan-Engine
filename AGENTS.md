# Local Agent Notes

## Version control

- Commits carry the repository owner's name only. Do not add `Co-Authored-By`
  trailers, agent attribution, or generated-with notices to commit messages.
- Never push. Commit locally and leave publishing to the repository owner.
- Keep each commit to one self-contained change so a bad one can be reverted
  in isolation.

## Build

- This is a Visual Studio multi-config CMake build. Build Debug from the repository root with:
  `cmake --build build --config Debug --target Shaders engine`
- Use the CMake that generated the build tree: `C:\Program Files\CMake\bin\cmake.exe`. A bare
  `cmake` may resolve to the Anaconda copy on `PATH`, which fails compiler-id detection against
  this tree.
- Do not run that command while Visual Studio is already building or debugging the engine. A running session can lock SDL's `SDL2d.pdb` and make the command fail with `LNK1201`.
- Shader source changes require rebuilding the `Shaders` target. Adding *or removing* a shader also
  requires rerunning CMake configure/generate, because the root `CMakeLists.txt` uses a
  configure-time glob; a deleted shader otherwise fails the build with `MSB8066` from a stale rule.
- `src/CMakeLists.txt` globs `*.cpp`/`*.h`, so new source files are picked up by a reconfigure
  without editing it.
- The Debug executable and runtime DLLs are written to `bin/Debug/`, not under `build/`.

## Source layout

Hardware ray tracing is the only render path; there is no rasterizer.

- `rt_engine.h/.cpp` — the `RtEngine` class declaration and its init/cleanup/run shell.
  Everything below implements members of that same class, so these files split the
  implementation, not the coupling.
- `device_context` instance/device/swapchain/render targets, `frame_sync` command pools
  and fences, `gpu_resources` buffer and image helpers, `descriptor_setup` engine-wide
  layouts, `scene_graph` draw-list building, `frame_draw` per-frame orchestration,
  `ui_overlay` ImGui, and the passes: `taa_pass`, `accumulation_pass`,
  `postprocess_pass`, `volumetrics`.
- `ray_tracing_pipeline.h/.cpp` — acceleration structures, RT pipeline, shader binding
  table. This one is over the ~500-line guideline on purpose: it is a single coherent
  subsystem written for this project, not tutorial residue, and splitting it would cut
  across the BLAS/TLAS/SBT build sequence.
- `gltf_import`, `image_utils`, `vk_init`, `descriptor_alloc`, `shader_module` are
  supporting utilities. `meshes.cpp` is generated data.

## C++ style

- Format `src/*.cpp` and `src/*.h` with the repository `.clang-format` file before committing C++ changes.
- `src/meshes.cpp` contains generated mesh tables and is excluded via `.clang-format-ignore`.
- Use four spaces and LF line endings. Braces are attached for anything executable — functions,
  `if`/`for`/`while`, lambdas — and go on their own line for type definitions (`struct`, `class`,
  `enum`, `union`), so a type declaration stays visually distinct from code. Brace initialization
  is unaffected.
- `.editorconfig` defines the whitespace and line-ending defaults for supported editors.
- Use `PascalCase` for types, `snake_case` for functions, `camelCase` for locals and data-struct fields, and
  `_camelCase` for class data members. Boolean names should describe a state or capability.

## Comments

- Keep comments short. One or two lines is the norm. Reserve three or more lines
  for things that genuinely need it, such as a non-obvious invariant, a packed
  GPU layout, or a docstring on a key function.
- Comment the reason, not the mechanics. Do not restate what the next line does,
  and do not narrate a function step by step.
- Prefer no comment over a filler one. Deleting a redundant comment is an
  improvement.

## Running and validation output

- Run with `bin/Debug` as the working directory. Engine resource paths use `../../shaders` and `../../assets`; launching with the repository root as the working directory fails during initialization.
- Validation layers are unconditionally enabled by `bUseValidationLayers` in `src/rt_engine.h`.
- vk-bootstrap's default validation callback writes validation messages to stdout in this project. Capture both streams anyway.
- A verified PowerShell capture recipe from the repository root is:

  ```powershell
  $runDir = (Resolve-Path .\bin\Debug).Path
  $process = Start-Process `
      -FilePath (Join-Path $runDir engine.exe) `
      -WorkingDirectory $runDir `
      -RedirectStandardOutput "$env:TEMP\vulkan-engine-stdout.log" `
      -RedirectStandardError "$env:TEMP\vulkan-engine-stderr.log" `
      -PassThru
  ```

  Let it render several frames, then close the window normally when cleanup behavior matters. For a bounded observation, stop only the returned process ID with `Stop-Process -Id $process.Id`.
- `Stop-Process -Force` kills the process before stdio is flushed, leaving both logs empty. To get
  output, send a normal close (`taskkill /PID <id>`) and then wait on the process, e.g.
  `$process.WaitForExit(20000)`. The engine does not always process a close promptly; see the
  known cleanup issues in `docs/validation-phase2.md`.
- Per-frame validation errors can make the log very large. Inventory distinct errors with:

  ```powershell
  Select-String "$env:TEMP\vulkan-engine-stdout.log" -Pattern 'VUID-[A-Za-z0-9-]+' -AllMatches |
      ForEach-Object { $_.Matches.Value } |
      Group-Object |
      Sort-Object Count -Descending
  ```

## Current observed baseline

On 2026-07-11, a 15-second Debug run initialized on an NVIDIA GeForce RTX 4080 SUPER, loaded `assets/livingroom_vkr.glb` and 11 lights, and remained responsive until the observation process was stopped. The unique validation errors were:

- `VUID-VkRayTracingPipelineCreateInfoKHR-layout-07990`: ray-miss shader binding 2 declares a combined image sampler while the ray-tracing descriptor layout declares a sampled image.
- `VUID-VkImageMemoryBarrier2-oldLayout-01213`: an image is transitioned to/from `TRANSFER_DST_OPTIMAL` without `VK_IMAGE_USAGE_TRANSFER_DST_BIT`.
- `VUID-VkBlitImageInfo2-dstImage-00224`: that same image is used as a blit destination without `VK_IMAGE_USAGE_TRANSFER_DST_BIT`.

Treat this list as a dated baseline, not an allowlist. Re-run and deduplicate validation output after every fix because later errors may be masked by earlier invalid state.

The maintained scenario baseline is in `docs/validation-baseline.md`. Raw logs,
event timelines, and screenshots belong under ignored `out/validation-baseline/`.
Use `tools/summarize-validation.ps1` to count validation message identifiers
without double-counting the VUID repeated in each message's specification URL.

For deterministic diagnostic runs, `engine.exe` accepts `--aa=taa|adaptive` and
`--tonemap=on|off`. Invalid options exit with status 2 before Vulkan starts.
`--render-path` no longer exists: hardware ray tracing is the only render path.

Phase 2 verification is recorded in `docs/validation-phase2.md`. Its raw logs are
under ignored `out/validation-phase2/`.
