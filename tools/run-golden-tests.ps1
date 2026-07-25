# Renders each golden scene and compares it against its reference image.
# Exits non-zero if any comparison fails, so it can gate a commit or CI run.
#
#   .\tools\run-golden-tests.ps1
#   .\tools\run-golden-tests.ps1 -Config Release
#   .\tools\run-golden-tests.ps1 -Update      # regenerate references instead

param(
    [string] $Config = 'Debug',
    [switch] $Update
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repoRoot "bin\$Config\engine.exe"
$goldenDir = Join-Path $repoRoot 'tests\golden'

if (-not (Test-Path $exe)) {
    Write-Error "engine.exe not found at $exe. Build the '$Config' configuration first."
}

# Each case is a reference image plus the flags that must reproduce it.
$cases = @(
    @{ Name = 'livingroom'; Args = '--no-ui --frames=60' }
)

$failed = 0
foreach ($case in $cases) {
    $reference = Join-Path $goldenDir "$($case.Name).png"

    if ($Update) {
        $arguments = "--screenshot=$reference $($case.Args)"
    } else {
        if (-not (Test-Path $reference)) {
            Write-Host "MISSING  $($case.Name): no reference at $reference" -ForegroundColor Red
            $failed++
            continue
        }
        $arguments = "--compare=$reference $($case.Args)"
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = $arguments
    # Resource paths resolve against the executable, so any directory works.
    $psi.WorkingDirectory = $repoRoot
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    $process = [System.Diagnostics.Process]::Start($psi)
    if (-not $process.WaitForExit(180000)) {
        $process.Kill()
        Write-Host "TIMEOUT  $($case.Name)" -ForegroundColor Red
        $failed++
        continue
    }

    $output = $process.StandardOutput.ReadToEnd() + $process.StandardError.ReadToEnd()
    $summary = ($output -split "`n" | Select-String 'Compare|Screenshot written') -join '; '

    if ($process.ExitCode -eq 0) {
        Write-Host "OK       $($case.Name): $summary" -ForegroundColor Green
    } else {
        Write-Host "FAIL     $($case.Name) (exit $($process.ExitCode)): $summary" -ForegroundColor Red
        $failed++
    }
}

if ($failed -gt 0) {
    Write-Host "`n$failed golden test(s) failed." -ForegroundColor Red
    exit 1
}

Write-Host "`nAll golden tests passed." -ForegroundColor Green
