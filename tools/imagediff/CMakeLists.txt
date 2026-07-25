add_executable(imagediff main.cpp)

set_property(TARGET imagediff PROPERTY CXX_STANDARD 20)
target_link_libraries(imagediff PRIVATE stb_image)
