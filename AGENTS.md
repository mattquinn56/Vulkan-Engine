# Local Agent Notes

## General

- Prefer an existing library over writing the logic yourself. Search first —
  vcpkg and `third_party/` already cover a lot — and only hand-roll when no
  reasonable option exists, or when the dependency costs more than it saves.
  Say which you did and why.
- Use US spelling everywhere: code, comments, commit messages, docs.

## Version control

- Commits carry the repository owner's name only. Do not add `Co-Authored-By`
  trailers, agent attribution, or generated-with notices to commit messages.
- Never push. Commit locally and leave publishing to the repository owner.
- One self-contained change per commit, so a bad one can be reverted in
  isolation. Build and verify before each commit, not just at the end.
- Commit messages are a subject line. Default to no body at all.
- Write a body only if, without it, a reader would revert the change believing
  it was a mistake. Wanting to share what you learned is not that reason.
- Same test as comments: if the body's job is to contrast the new code with the
  old, drop it. The diff already shows both sides.

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
- The Debug executable and runtime DLLs are written to `bin/Debug/`, not under `build/`.

## Source layout

Hardware ray tracing is the only render path; there is no rasterizer.
`src/` is the single include root, so headers are included by their subfolder
path: `#include "core/rt_engine.h"`.

- `core/` — `RtEngine` itself. `rt_engine` is the class declaration plus the
  init/cleanup/run shell; `device_context` (instance, device, swapchain, render
  targets), `frame_sync` (command pools, fences, `immediate_submit`) and
  `frame_draw` (per-frame orchestration) implement members of that same class,
  so these files split the implementation, not the coupling. `gpu_types.h` holds
  the shared GPU-facing structs.
- `passes/` — `ray_tracing_pipeline` (acceleration structures, RT pipeline,
  shader binding table), plus `taa_pass`, `accumulation_pass`,
  `postprocess_pass` and `volumetrics`. `ray_tracing_pipeline.cpp` is over the
  ~500-line guideline on purpose: splitting it would cut across the BLAS/TLAS/SBT
  build sequence.
- `scene/` — `gltf_import`, `scene_graph` draw-list building, `camera`, and
  generated `meshes` data.
- `gpu/` — Vulkan plumbing shared by everything: `gpu_resources`,
  `descriptor_alloc`, `descriptor_setup`, `image_utils`, `vk_init`,
  `shader_module`.
- `platform/` — host-side concerns: `resource_path`, `screenshot`, `ui_overlay`.

`src/CMakeLists.txt` uses `GLOB_RECURSE`, so new files in any subfolder are
picked up by a reconfigure.

## C++ style

- Format `src/*.cpp` and `src/*.h` with the repository `.clang-format` file before committing C++ changes.
- `src/scene/meshes.cpp` contains generated mesh tables and is excluded via `.clang-format-ignore`.
- Use four spaces and LF line endings. Braces are attached for anything executable — functions,
  `if`/`for`/`while`, lambdas — and go on their own line for type definitions (`struct`, `class`,
  `enum`, `union`), so a type declaration stays visually distinct from code. Brace initialization
  is unaffected.
- `.editorconfig` defines the whitespace and line-ending defaults for supported editors.
- Use `PascalCase` for types, `snake_case` for functions, `camelCase` for locals and data-struct fields, and
  `_camelCase` for class data members. Boolean names should describe a state or capability.

## GUIDE.md

Do not document anything about `GUIDE.md` here. Its conventions live in
`GUIDE.md` itself — read that file's header before editing it.

## Comments

- Keep comments short. One or two lines is the norm. Reserve three or more lines
  for things that genuinely need it, such as a non-obvious invariant, a packed
  GPU layout, or a docstring on a key function.
- Comment the reason, not the mechanics. Do not restate what the next line does,
  and do not narrate a function step by step.
- A comment may not name anything that is not in the codebase. No previous
  implementation, no library you rejected, no approach you considered and
  dropped. Test: point at every thing your comment names. If a reader cannot
  find it in the tree, delete that clause. "PCG hash" passes; "PCG hash, better
  than the sin-based one" does not, because there is no sin-based one to look at.
- Explaining why the current code is shaped this way is fine and often useful.
  Explaining why you changed it is not — that is what the commit is for.
- Prefer no comment over a filler one. Deleting a redundant comment is an
  improvement.

## Running and validation output

- The working directory no longer matters: shaders and assets are resolved relative to the
  executable's own location, so `engine.exe` can be launched from anywhere.
- Validation layers are enabled by `bUseValidationLayers` in `src/rt_engine.h`: on for Debug,
  off for Release. Validation work must therefore be done against a Debug build.
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
  `$process.WaitForExit(20000)`.
- Per-frame validation errors can make the log very large. Inventory distinct errors with:

  ```powershell
  Select-String "$env:TEMP\vulkan-engine-stdout.log" -Pattern 'VUID-[A-Za-z0-9-]+' -AllMatches |
      ForEach-Object { $_.Matches.Value } |
      Group-Object |
      Sort-Object Count -Descending
  ```

## Current observed baseline

As of 2026-07-26, a Debug run on an NVIDIA GeForce RTX 4080 SUPER loads
`assets/livingroom_vkr.glb` with 11 lights, produces **no validation messages**,
and exits with code 0.

That is the bar to hold. Any validation output is a regression from the change
under test. Re-check after every change, and deduplicate before concluding,
because later errors can be masked by earlier invalid state.

A descriptor set that names only render targets should be written once, when
those targets are created, not per frame. Writing per frame touches a set the
previous frame still has pending, which is `VUID-vkUpdateDescriptorSets-None-03047`.

Use `tools/summarize-validation.ps1` to count validation message identifiers
without double-counting the VUID repeated in each message's specification URL.

For deterministic diagnostic runs, `engine.exe` accepts `--tonemap=on|off` and
`--debug-view=<n>`. Invalid options exit with status 2 before Vulkan starts.
`--render-path` no longer exists: hardware ray tracing is the only render path.

## Verifying a frame

Prefer this over launching interactively — it needs no manual interaction and
gives you an image to inspect:

```powershell
.in\Debug\engine.exe --screenshot=C:\path	o\shot.png --frames=45
```

It renders `--frames` frames, writes the presented swapchain image as a PNG, and
exits 0 on its own. Raise `--frames` when the Monte Carlo accumulation needs
longer to converge; the image is visibly noisy at low frame counts.

A capture run creates its window hidden and does not grab the mouse, so nothing
appears on screen and it can run while you work. It is not surface-less
rendering: a window and swapchain still exist, they are just never shown.

## Golden image tests

```
ctest --test-dir build -C Debug --output-on-failure
```

Run after any change that could affect the rendered image, and treat a failure
as a regression until proven otherwise. On failure the run writes `.actual.png`
and `.diff.png` into `build/golden-output/`; look at the diff before assuming
the reference is stale.

Regenerate references only for an intentional output change:
`cmake --build build --config Debug --target golden-update`.

Comparison lives in the standalone `imagediff` target rather than in the engine,
so it can be run against any two PNGs and exercised on its own. A pixel counts
as differing when any channel is more than 4 apart, and up to 0.2% of pixels may
differ before a test fails; back-to-back runs on one GPU agree to within a
single channel value, so that slack is for driver and hardware variation.

Golden cases must pass `--no-ui`. The overlay prints a frame time that changes
every run, so a reference including it would fail against itself. Cases are
declared in the root `CMakeLists.txt`.

`livingroom` renders 60 frames, `livingroom_converged` renders 480. The long case
exists to catch changes in how fast the image converges, not just what it
converges to; a change that only slows accumulation passes the short case and
fails the long one. Back-to-back runs are bit-identical on one GPU, so a
non-trivial diff is a real change.

Phase 2 verification is recorded in `docs/validation-phase2.md`. Its raw logs are
under ignored `out/validation-phase2/`.
