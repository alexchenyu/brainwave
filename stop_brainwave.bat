@echo off
REM Stop Brainwave Server

echo Stopping Brainwave Server...

REM Kill all uvicorn processes
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *uvicorn*" 2>nul

REM Alternative: kill by command line pattern
wmic process where "commandline like '%%uvicorn%%realtime_server%%'" delete 2>nul

echo Server stopped.
pause
