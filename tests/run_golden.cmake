# Drives one golden image test: render a frame, then diff it against the
# reference. Written as a CMake script so `ctest` can run it on any platform
# without depending on a shell.
#
# Required: -DENGINE=  -DIMAGEDIFF=  -DREFERENCE=  -DOUTPUT_DIR=  -DCASE=
# Optional: -DENGINE_ARGS=  (semicolon-separated)

foreach(required ENGINE IMAGEDIFF REFERENCE OUTPUT_DIR CASE)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "run_golden.cmake requires -D${required}=")
    endif()
endforeach()

if(NOT EXISTS "${REFERENCE}")
    message(FATAL_ERROR "No reference image at ${REFERENCE}. Regenerate with the golden-update target.")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(actual "${OUTPUT_DIR}/${CASE}.actual.png")
set(diff "${OUTPUT_DIR}/${CASE}.diff.png")

execute_process(
    COMMAND "${ENGINE}" "--screenshot=${actual}" ${ENGINE_ARGS}
    RESULT_VARIABLE render_result
    OUTPUT_VARIABLE render_output
    ERROR_VARIABLE render_output)

if(NOT render_result EQUAL 0)
    message(FATAL_ERROR "Render failed (${render_result}):\n${render_output}")
endif()

execute_process(
    COMMAND "${IMAGEDIFF}" "${REFERENCE}" "${actual}" "--diff=${diff}"
    RESULT_VARIABLE diff_result
    OUTPUT_VARIABLE diff_output
    ERROR_VARIABLE diff_output)

message("${diff_output}")

if(NOT diff_result EQUAL 0)
    message(FATAL_ERROR "Golden test '${CASE}' failed. Compare:\n  reference ${REFERENCE}\n  actual    ${actual}\n  diff      ${diff}")
endif()
