# One-shot setup: fetches and bootstraps vcpkg, installs dependencies, and
# configures the CMake build. Safe to re-run; each step is skipped if already done.
#
#   .\tools\setup.ps1
#   .\tools\setup.ps1 -Build          # also compile once configured

param(
    [switch] $Build,
    [ValidateSet('Debug', 'Release')]
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

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error 'Visual Studio 2022 was not found. Install it with the "Desktop development with C++" workload.'
}
$vswhereArguments =
    @('-latest', '-version', '[17.0,18.0)', '-products', '*', '-requires',
      'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath')
$visualStudio = (& $vswhere @vswhereArguments | Out-String).Trim()
if (-not $visualStudio) {
    Write-Error 'Visual Studio is missing the "Desktop development with C++" workload.'
}
Write-Host "  Visual Studio: $visualStudio"

Write-Host '== vcpkg ==' -ForegroundColor Cyan
if (-not (Test-Path $vcpkgDir)) {
    Write-Host '  cloning...'
    Invoke-Native 'git' @('clone', 'https://github.com/microsoft/vcpkg.git', $vcpkgDir) 'git clone'
} else {
    Write-Host '  already present'
}

$currentVcpkgCommit = (& git -C $vcpkgDir rev-parse HEAD 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Error "'$vcpkgDir' exists but is not a valid vcpkg Git checkout. Remove or rename it, then run setup again."
}
if ($currentVcpkgCommit -ne $vcpkgCommit) {
    Write-Host '  selecting pinned revision...'
    Invoke-Native 'git' @('-C', $vcpkgDir, 'fetch', 'origin', $vcpkgCommit, '--depth', '1') 'vcpkg fetch'
    Invoke-Native 'git' @('-C', $vcpkgDir, 'checkout', '--detach', $vcpkgCommit) 'vcpkg checkout'
}

if (-not (Test-Path $vcpkgExe)) {
    Write-Host '  bootstrapping...'
    Invoke-Native (Join-Path $vcpkgDir 'bootstrap-vcpkg.bat') @('-disableMetrics') 'vcpkg bootstrap'
}

Write-Host '== Configuring ==' -ForegroundColor Cyan
$buildDir = Join-Path $repoRoot 'build'
$configureArguments = @(
    '-S', $repoRoot,
    '-B', $buildDir,
    '-G', 'Visual Studio 17 2022',
    '-A', 'x64',
    '-DVCPKG_MANIFEST_MODE=ON',
    "-DVCPKG_MANIFEST_DIR=$repoRoot",
    '-DVCPKG_TARGET_TRIPLET=x64-windows'
)

# vcpkg cannot switch an existing CMake cache from classic to manifest mode.
# --fresh regenerates only CMake's configuration state and leaves outputs intact.
$cache = Join-Path $buildDir 'CMakeCache.txt'
if (Test-Path $cache) {
    $legacyManifest = Select-String -Path $cache -Pattern '^(VCPKG_MANIFEST_MODE:BOOL|Z_VCPKG_CHECK_MANIFEST_MODE:INTERNAL)=OFF$' -Quiet
    $implicitPlatform = Select-String -Path $cache -Pattern '^CMAKE_GENERATOR_PLATFORM:INTERNAL=$' -Quiet
    $differentGenerator =
        -not (Select-String -Path $cache -Pattern '^CMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022$' -Quiet)
    if ($legacyManifest -or $implicitPlatform -or $differentGenerator) {
        Write-Host '  migrating the existing build to the portable configuration...'
        $configureArguments = @('--fresh') + $configureArguments
    }
}

Invoke-Native $cmake $configureArguments 'cmake configure'

if ($Build) {
    Write-Host "== Building ($Config) ==" -ForegroundColor Cyan
    Invoke-Native $cmake @('--build', $buildDir, '--config', $Config, '--target', 'Shaders', 'engine') 'cmake build'
}

Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host "  Build:  cmake --build build --config $Config --target Shaders engine"
Write-Host "  Run:    .\bin\$Config\engine.exe"
