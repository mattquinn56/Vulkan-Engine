# One-shot setup: fetches and bootstraps vcpkg, installs dependencies, and
# configures the CMake build. Safe to re-run; each step is skipped if already done.
#
#   .\tools\setup.ps1
#   .\tools\setup.ps1 -Build          # also compile once configured

param(
    [switch] $Build,
    [string] $Config = 'Debug'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$vcpkgDir = Join-Path $repoRoot 'vcpkg'
$vcpkgExe = Join-Path $vcpkgDir 'vcpkg.exe'

# Pinned so everyone resolves the same dependency versions.
$vcpkgCommit = 'b322364f06308bdd24823f9d8f03fe0cc86fd46f'

function Require-Command($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Error "'$name' was not found on PATH. $hint"
    }
}

# Native tools write progress and warnings to stderr, which PowerShell would
# otherwise turn into terminating errors. Judge success by exit code instead.
function Invoke-Native {
    param([string] $Exe, [string[]] $Arguments, [string] $What)

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host "    $_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous

    if ($code -ne 0) {
        Write-Error "$What failed with exit code $code."
    }
}

# A conda or MSYS cmake earlier on PATH can fail against a Visual Studio build
# tree, so prefer a real CMake install when one exists.
function Resolve-CMake {
    $candidates = @(
        "$env:ProgramFiles\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    $onPath = Get-Command cmake -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    return $null
}

Write-Host '== Checking prerequisites ==' -ForegroundColor Cyan
Require-Command 'git' 'Install Git: https://git-scm.com/downloads'

$cmake = Resolve-CMake
if (-not $cmake) {
    Write-Error 'CMake was not found. Install it and tick "Add to PATH": https://cmake.org/download/'
}
Write-Host "  CMake: $cmake"

if (-not $env:VULKAN_SDK) {
    Write-Error @'
VULKAN_SDK is not set. Install the Vulkan SDK from https://vulkan.lunarg.com/sdk/home
and open a new terminal so the environment variable is picked up. The SDK supplies
glslc (shader compilation) and the validation layers.
'@
}
Write-Host "  Vulkan SDK: $env:VULKAN_SDK"

Require-Command 'glslc' 'glslc ships with the Vulkan SDK; ensure its Bin directory is on PATH.'

Write-Host '== vcpkg ==' -ForegroundColor Cyan
if (-not (Test-Path $vcpkgDir)) {
    Write-Host '  cloning...'
    Invoke-Native 'git' @('clone', 'https://github.com/microsoft/vcpkg.git', $vcpkgDir) 'git clone'
    Invoke-Native 'git' @('-C', $vcpkgDir, 'checkout', $vcpkgCommit) 'git checkout'
} else {
    Write-Host '  already present'
}

if (-not (Test-Path $vcpkgExe)) {
    Write-Host '  bootstrapping...'
    Invoke-Native (Join-Path $vcpkgDir 'bootstrap-vcpkg.bat') @('-disableMetrics') 'vcpkg bootstrap'
}

Write-Host '  installing dependencies...'
Invoke-Native $vcpkgExe @('install', 'vulkan', 'nlohmann-json', '--triplet', 'x64-windows') 'vcpkg install'

# The pinned vcpkg revision does not always lay down this SDK header, and the
# engine includes it for readable VkResult names.
$enumHelper = Join-Path $vcpkgDir 'installed\x64-windows\include\vulkan\vk_enum_string_helper.h'
if (-not (Test-Path $enumHelper)) {
    Write-Host '  patching missing vk_enum_string_helper.h'
    $target = Split-Path -Parent $enumHelper
    if (-not (Test-Path $target)) { New-Item -ItemType Directory -Force -Path $target | Out-Null }
    Copy-Item (Join-Path $repoRoot 'third_party\vk_enum_string_helper.h') $enumHelper
}

Write-Host '== Configuring ==' -ForegroundColor Cyan
Invoke-Native $cmake @('-S', $repoRoot, '-B', (Join-Path $repoRoot 'build')) 'cmake configure'

if ($Build) {
    Write-Host "== Building ($Config) ==" -ForegroundColor Cyan
    Invoke-Native $cmake @('--build', (Join-Path $repoRoot 'build'), '--config', $Config, '--target', 'Shaders', 'engine') 'cmake build'
}

Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host "  Build:  cmake --build build --config $Config --target Shaders engine"
Write-Host "  Run:    .\bin\$Config\engine.exe"
