# CMakeLists.txt for the project directory

cmake_minimum_required(VERSION 3.8)
project("vulkan_guide")

find_package(Vulkan REQUIRED)

# Add the third_party directory where glslang and other dependencies are located
add_subdirectory(third_party)

# Set output directories for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

# Add the src directory where the main application code is located
add_subdirectory(src)
