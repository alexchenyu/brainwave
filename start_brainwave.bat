@echo off
REM Brainwave Server Startup Script

REM Change to the project directory
cd /d "%~dp0"

REM Start the uvicorn server using uv
uv run uvicorn realtime_server:app --host 0.0.0.0 --port 80

REM Uncomment the line below if you want the window to stay open after an error
REM pause
