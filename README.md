# Vulkan-Engine
 A vulkan-based engine

## Build instructions:
1. Clone the repository
2. In the repository root directory, clone vcpkg using URL `https://github.com/microsoft/vcpkg.git`
3. Open the directory `Vulkan-Engine/vcpkg/` using cmd. Then, run the following: `bootstrap-vcpkg.bat`; `vcpkg install vulkan`; `vcpkg install nlohmann-json`
4. Install CMake and run the GUI
5. For the "Where is the source code" field put "<parent-directory>/Vulkan-Engine/" and for the "Where to build the binaries" put "<parent-directory>/Vulkan-Engine/build/"
6. Run Configure then Generate, each with the default options
7. Open the generated project file in the build directory with Visual Studio
8. Build target "ALL_BUILD". Then, build target "Shaders". Note: for any edits to existing shader files, rebuild the "Shaders" target. For any new shader files, rerun CMake Configure and Generate
    - I have found that vcpkg may not properly include file. `vk_enum_string_helper.h`. If so, copy file `...\Vulkan-Engine\third_party\vk_enum_string_helper.h` and paste it insdie of directory `...\Vulkan-Engine\vcpkg\installed\x64-windows\include\vulkan\`
9. Right-click on target "engine" and select "Set as Startup Project". Now you can run the engine by pressing F5 or clicking "Local Windows Debugger" in the toolbar. Press alt to access GUI options