# Local Agent Notes

## Build

- This is a Visual Studio multi-config CMake build. Build Debug from the repository root with:
  `cmake --build build --config Debug --target Shaders engine`
- Do not run that command while Visual Studio is already building or debugging the engine. A running session can lock SDL's `SDL2d.pdb` and make the command fail with `LNK1201`.
- Shader source changes require rebuilding the `Shaders` target. Adding a shader also requires rerunning CMake configure/generate because the root `CMakeLists.txt` uses a configure-time glob.
- The Debug executable and runtime DLLs are written to `bin/Debug/`, not under `build/`.

## C++ style

- Format `src/*.cpp` and `src/*.h` with the repository `.clang-format` file before committing C++ changes.
- `src/meshes.cpp` contains generated mesh tables and is excluded via `.clang-format-ignore`.
- Use four spaces, LF line endings, attached control-flow braces, and braces on the next line for functions and types.
- `.editorconfig` defines the whitespace and line-ending defaults for supported editors.
- Use `PascalCase` for types, `snake_case` for functions, `camelCase` for locals and data-struct fields, and
  `_camelCase` for class data members. Boolean names should describe a state or capability.

## Running and validation output

- Run with `bin/Debug` as the working directory. Engine resource paths use `../../shaders` and `../../assets`; launching with the repository root as the working directory fails during initialization.
- Validation layers are unconditionally enabled by `bUseValidationLayers` in `src/vk_engine.cpp`.
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

For deterministic diagnostic runs, `engine.exe` accepts
`--render-path=raytrace|raster`, `--aa=taa|adaptive`, and
`--tonemap=on|off`. Invalid options exit with status 2 before Vulkan starts.

Phase 2 verification is recorded in `docs/validation-phase2.md`. Its raw logs are
under ignored `out/validation-phase2/`.
