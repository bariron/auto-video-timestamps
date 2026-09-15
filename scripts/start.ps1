param([int]$Port = 7860)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonExecutable = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { $pythonExecutable = 'python' }
Push-Location $projectRoot
try {
    & $pythonExecutable -m video_timestamps --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Application failed to start.' }
} finally { Pop-Location }
