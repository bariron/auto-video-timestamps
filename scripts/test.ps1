$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonExecutable = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { $pythonExecutable = 'python' }
Push-Location $projectRoot
try {
    & $pythonExecutable -m ruff check video_timestamps tests
    if ($LASTEXITCODE -ne 0) { throw 'Lint failed.' }
    & $pythonExecutable -m ruff format --check video_timestamps tests
    if ($LASTEXITCODE -ne 0) { throw 'Formatting check failed.' }
    & $pythonExecutable -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
} finally { Pop-Location }
