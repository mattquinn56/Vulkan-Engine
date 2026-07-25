#pragma once

#include <string>
#include <string_view>

// Resolves bundled resources without depending on the working directory, so the
// executable can be launched from anywhere (including by double-clicking it).
namespace resource {

// Absolute path to a file under shaders/ or assets/. Falls back to a
// working-directory-relative path if no candidate root is found, which keeps
// the failure reported by the caller rather than here.
std::string shader(std::string_view name);
std::string asset(std::string_view name);

} // namespace resource
