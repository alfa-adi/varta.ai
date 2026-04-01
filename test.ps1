# test.ps1
# Automates running tests without needing manual venv activation.

$ErrorActionPreference = "Stop"

Write-Host "🧪 Running Varta.ai Tests..." -ForegroundColor Cyan

# Ensure we're in the right directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Set PYTHONPATH so Python can find the root files
$env:PYTHONPATH = $ScriptDir

# Check if venv exists
$PythonPath = Join-Path $ScriptDir "venv\Scripts\python.exe"
if (-Not (Test-Path $PythonPath)) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    exit 1
}

# Run the buffering API test script
& $PythonPath tests/test_buffering_api.py
