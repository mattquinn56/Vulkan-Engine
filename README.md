# Vulkan-Engine

A Vulkan renderer built around hardware ray tracing, working toward real-time
volumetrics. Ray tracing is the only render path — there is no rasterizer
fallback.

Current features: hardware-accelerated ray tracing with BLAS compaction,
progressive Monte Carlo accumulation, temporal antialiasing, ACES tonemapping,
glTF scene import, and an ImGui debug overlay.

## Requirements

| | |
|---|---|
| OS | Windows 10 or 11 (x64) |
| GPU | Must support `VK_KHR_acceleration_structure` and `VK_KHR_ray_tracing_pipeline` — NVIDIA RTX 20-series or newer, AMD RX 6000-series or newer, or Intel Arc |
| [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) | 1.3 or newer. Supplies `glslc` and the validation layers |
| [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) | With the "Desktop development with C++" workload |
| [CMake](https://cmake.org/download/) | 3.15 or newer, on `PATH` |
| [Git](https://git-scm.com/downloads) | For cloning, and for fetching vcpkg during setup |

The GPU requirement is hard: device selection fails at startup without those two
extensions. If you are unsure whether your card qualifies, run `vulkaninfo` from
the SDK and look for them in the device extension list.

## Build

```powershell
git clone https://github.com/mattquinn56/Vulkan-Engine.git
cd Vulkan-Engine
.\tools\setup.ps1 -Build
```

`setup.ps1` checks your prerequisites, fetches and bootstraps vcpkg at a pinned
commit, installs the dependencies, and configures CMake. It is safe to re-run —
every step is skipped if already done.

Then run it:

```powershell
.\bin\Debug\engine.exe
```

The executable resolves its shaders and assets relative to its own location, so
you can launch it from anywhere, including by double-clicking it in Explorer.

For a release build:

```powershell
.\tools\setup.ps1 -Build -Config Release
.\bin\Release\engine.exe
```

Release omits the validation layers, so it is the faster of the two and the one
to use for anything resembling a benchmark.

### Rebuilding after changes

```powershell
cmake --build build --config Debug --target Shaders engine
```

Editing a shader only needs the `Shaders` target rebuilt. *Adding or removing* a
shader file also needs `cmake -S . -B build` re-run first, because the shader
list is gathered at configure time.

## Controls

| Input | Action |
|---|---|
| `W` `A` `S` `D` | Move horizontally |
| `E` / `Q` | Move up / down |
| `Shift` / `Ctrl` | Move faster / slower |
| Mouse | Look |
| `Alt` | Release the mouse to interact with the overlay |

## Command line

```
--tonemap=on|off      ACES + sRGB tonemapping
--screenshot=<path>   render, write a PNG, and exit
--frames=<n>          which frame to capture on (default 30)
--no-ui               render without the ImGui overlay
```

Unrecognized options exit with status 2 before Vulkan starts.

Capture runs keep their window hidden and exit on their own, which makes them
usable for scripted checks:

```powershell
.\bin\Debug\engine.exe --screenshot=shot.png --frames=60
```

Because the image is accumulated progressively, a higher `--frames` gives a
cleaner result; it is visibly noisy below about 30.

## Tests

```
ctest --test-dir build -C Debug --output-on-failure
```

Each golden test renders a scene and compares it against a stored reference in
`tests/golden/`, failing if too many pixels differ. A failure writes an
`.actual.png` and a color-coded `.diff.png` into `build/golden-output/` so you
can see exactly what moved.

Regenerate the references after an intentional change to output:

```
cmake --build build --config Debug --target golden-update
```

The comparison is a separate `imagediff` executable, usable on any two PNGs:

```
.\bin\Debug\imagediff.exe reference.png actual.png --diff=diff.png
```

It exits 0 when the images match, 1 when they differ, and 2 if an image could
not be read. In the diff, unchanged pixels are dimmed grayscale and differing
ones are tinted blue through red by how far apart they are.

## Troubleshooting

**`VULKAN_SDK is not set`** — install the Vulkan SDK and open a new terminal so
the environment variable is picked up.

**CMake fails during configure with a compiler-detection error** — another CMake
earlier on `PATH` (Anaconda ships one) is being used against a Visual Studio
build tree. `setup.ps1` prefers a real CMake install; if you invoke CMake
yourself, use the full path to `C:\Program Files\CMake\bin\cmake.exe`.

**`LNK1201` while building** — Visual Studio is holding the debug PDB. Close any
running or debugging session and rebuild.

**Cannot find `vk_enum_string_helper.h`** — the pinned vcpkg revision does not
always install this SDK header. `setup.ps1` copies it in automatically; re-run it
if you configured by hand.

**The window opens but nothing renders, or startup fails on device selection** —
your GPU almost certainly lacks the ray tracing extensions. See Requirements.
