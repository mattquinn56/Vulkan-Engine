# CMakeList.txt : CMake project for vulkan_hardware_rt, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.15)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
  CACHE STRING "Vcpkg toolchain file")

project ("vulkan_hardware_rt")

find_package(Vulkan REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_subdirectory(third_party)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_subdirectory(src)
add_subdirectory(tools/imagediff)


file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    "${PROJECT_SOURCE_DIR}/shaders/*.rchit"
    "${PROJECT_SOURCE_DIR}/shaders/*.rmiss"
    "${PROJECT_SOURCE_DIR}/shaders/*.rgen"
    )

# Shared .glsl files are #included by the shaders above. glslc resolves them at
# compile time, but the build system cannot see them, so list them as
# dependencies of every shader. Coarse — touching one rebuilds all — but a
# stale SPIR-V that silently ignores an edit is far worse.
file(GLOB GLSL_INCLUDE_FILES "${PROJECT_SOURCE_DIR}/shaders/*.glsl")

foreach(GLSL ${GLSL_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
  message(STATUS ${GLSL})

  # Determine shader type to apply appropriate glslangValidator flags
  get_filename_component(SHADER_EXT ${GLSL} EXT)

  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND glslc  --target-env=vulkan1.2 ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL} ${GLSL_INCLUDE_FILES})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
    )

enable_testing()

# Each golden case renders with fixed settings and is compared against a stored
# reference. --no-ui is mandatory: the overlay prints a frame time that changes
# every run, so a reference including it would fail against itself.
#
# livingroom is the fast case: 60 frames is enough to be stable but still carries
# visible sampling noise. livingroom_converged runs long enough that the
# accumulator has settled, so it fails on anything that changes how quickly the
# image converges rather than only what it converges to. Frames are cheap here;
# startup dominates both cases.
#
# livingroom_panning is the only case where reprojection does any work: a still
# camera reprojects every pixel onto itself.
set(GOLDEN_CASES "livingroom" "livingroom_converged" "livingroom_panning")
set(GOLDEN_ARGS_livingroom "--no-ui;--frames=60")
set(GOLDEN_ARGS_livingroom_converged "--no-ui;--frames=480")
set(GOLDEN_ARGS_livingroom_panning "--no-ui;--orbit=0.5;--orbit-frames=60;--frames=60")

foreach(case ${GOLDEN_CASES})
    add_test(NAME golden.${case}
        COMMAND ${CMAKE_COMMAND}
            -DENGINE=$<TARGET_FILE:engine>
            -DIMAGEDIFF=$<TARGET_FILE:imagediff>
            -DREFERENCE=${PROJECT_SOURCE_DIR}/tests/golden/${case}.png
            -DOUTPUT_DIR=${PROJECT_BINARY_DIR}/golden-output
            -DCASE=${case}
            "-DENGINE_ARGS=${GOLDEN_ARGS_${case}}"
            -P ${PROJECT_SOURCE_DIR}/tests/run_golden.cmake)
    set_tests_properties(golden.${case} PROPERTIES TIMEOUT 180)
endforeach()

# Regenerates every reference image. Run after an intentional change to output.
set(update_commands "")
foreach(case ${GOLDEN_CASES})
    list(APPEND update_commands
        COMMAND $<TARGET_FILE:engine>
            "--screenshot=${PROJECT_SOURCE_DIR}/tests/golden/${case}.png"
            ${GOLDEN_ARGS_${case}})
endforeach()
add_custom_target(golden-update ${update_commands}
    DEPENDS engine
    COMMENT "Regenerating golden reference images")
