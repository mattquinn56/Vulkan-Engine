#include "platform/resource_path.h"

#include <SDL.h>

#include <filesystem>

namespace {

namespace fs = std::filesystem;

// The executable lives in bin/<config>/, so the repository root is two levels
// up; a deployed copy may instead sit alongside its resources. Try both, plus
// the working directory, and keep the first that actually contains the folder.
fs::path locate_root(const char* folder) {
    std::vector<fs::path> candidates;

    if (char* base = SDL_GetBasePath()) {
        const fs::path exeDir(base);
        SDL_free(base);
        candidates.push_back(exeDir);
        candidates.push_back(exeDir.parent_path().parent_path());
        candidates.push_back(exeDir.parent_path().parent_path().parent_path());
    }

    std::error_code ec;
    candidates.push_back(fs::current_path(ec));

    for (const fs::path& candidate : candidates) {
        if (!candidate.empty() && fs::is_directory(candidate / folder, ec)) {
            return candidate / folder;
        }
    }
    return fs::path("..") / ".." / folder;
}

std::string resolve(const char* folder, std::string_view name) {
    static const fs::path shaderRoot = locate_root("shaders");
    static const fs::path assetRoot = locate_root("assets");

    const fs::path& root = (std::string_view(folder) == "shaders") ? shaderRoot : assetRoot;
    return (root / name).string();
}

} // namespace

std::string resource::shader(std::string_view name) {
    return resolve("shaders", name);
}

std::string resource::asset(std::string_view name) {
    return resolve("assets", name);
}
