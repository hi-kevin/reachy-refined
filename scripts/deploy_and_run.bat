@echo off
setlocal
call "%~dp0deploy.bat"
if errorlevel 1 (
    echo Skipping run because deploy failed.
    endlocal
    exit /b 1
)

echo ****************************************************************
echo [5/5] Starting remote application...
echo ****************************************************************
ssh -t pollen@reachy-mini.local "cd ~/reachy-refined && source .venv/bin/activate && python -m src.main"
endlocal
