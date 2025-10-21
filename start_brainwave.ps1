# Brainwave Server Startup Script
# Run this as Administrator to use port 80

# Set the project directory
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

# Activate virtual environment if exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
}

# Start the server
Write-Host "Starting Brainwave Server on port 80..." -ForegroundColor Green
python -m uvicorn realtime_server:app --host 0.0.0.0 --port 80

# Keep window open if there's an error
Read-Host -Prompt "Press Enter to exit"
