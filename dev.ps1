# dev.ps1
# Automates starting the server so you never have to manually activate venv or set paths.

$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting Varta.ai Server..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray

# Ensure we're in the right directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Set PYTHONPATH so Python can find the 'adapter' and 'pipeline' folders
$env:PYTHONPATH = $ScriptDir

# Check if venv exists
$UvicornPath = Join-Path $ScriptDir "venv\Scripts\uvicorn.exe"
if (-Not (Test-Path $UvicornPath)) {
    Write-Host "❌ Virtual environment not found or uvicorn not installed!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv; .\venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Run the server using the virtual environment's exact executable
& $UvicornPath web.server:app --reload --host 0.0.0.0 --port 8000
