# Molt post-task checks for Windows (Windows PowerShell 5.1 or PowerShell 7+).
# Runs ruff and pytest. Skips cleanly until Task 2 creates pyproject.toml.
# Mirrors the sh hook in .kiro/hooks/checks-after-task.json.

$ErrorActionPreference = 'Stop'

# This script lives in <repo>/.kiro/hooks/scripts/, so the repo root is three levels up.
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $repoRoot

if (-not (Test-Path (Join-Path $repoRoot 'pyproject.toml'))) {
    Write-Output 'molt checks: no pyproject.toml yet (before Task 2), skipping'
    exit 0
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Output 'molt checks: uv is not on PATH; install it from https://docs.astral.sh/uv/'
    exit 1
}

$steps = @(
    @('run', 'ruff', 'check', '.'),
    @('run', 'ruff', 'format', '--check', '.'),
    @('run', 'pytest', '-q')
)

foreach ($step in $steps) {
    & uv @step
    if ($LASTEXITCODE -ne 0) {
        Write-Output ("molt checks: 'uv {0}' failed with exit code {1}" -f ($step -join ' '), $LASTEXITCODE)
        exit $LASTEXITCODE
    }
}

Write-Output 'molt checks: all passed'
exit 0
