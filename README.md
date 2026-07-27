# Vulkan Engine

A Windows renderer built around Vulkan hardware ray tracing.

## Build and run

You need:

- Windows 10/11 and a ray-tracing-capable GPU
- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) with
  **Desktop development with C++**
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
- [CMake 3.24+](https://cmake.org/download/) and
  [Git](https://git-scm.com/downloads)

Then:

```powershell
git clone https://github.com/mattquinn56/Vulkan-Engine.git
cd Vulkan-Engine
.\setup.bat
.\bin\Release\engine.exe
```

`setup.bat` downloads pinned dependencies, configures CMake, and builds the
engine. It is safe to run again.

## Controls

| Input | Action |
|---|---|
| `W` `A` `S` `D` | Move |
| `Q` / `E` | Move down / up |
| `Shift` / `Ctrl` | Move faster / slower |
| Mouse | Look |
| `Tab` | Open or close renderer settings |

## Development

```powershell
.\tools\setup.ps1 -Build
cmake --build build --config Debug --target Shaders engine
ctest --test-dir build -C Debug --output-on-failure
```

Run `.\bin\Debug\engine.exe --help` for capture and diagnostic options.
