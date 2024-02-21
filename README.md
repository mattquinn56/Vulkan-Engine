# Vulkan-Engine
 A vulkan-based engine

## Build instructions:
1. Clone the repository
2. Install CMake and run the GUI
3. For the "Where is the source code" field put "<parent-directory>/Vulkan-Engine/" and for the "Where to build the binaries" put "<parent-directory>/Vulkan-Engine/build/"
4. Run Configure then Generate, each with the default options
5. Open the generated project file in the build directory with Visual Studio
6. Build target "ALL_BUILD". Then, build target "Shaders". Note: for any edits to existing shader files, rebuild the "Shaders" target. For any new shader files, rerun CMake Configure and Generate
7. Right-click on target "engine" and select "Set as Startup Project". Now you can run the engine by pressing F5 or clicking "Local Windows Debugger" in the toolbar