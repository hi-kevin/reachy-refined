@echo off
call deploy.bat

echo ****************************************************************
echo [5/5] Starting remote application...
echo ****************************************************************
ssh -t pollen@reachy-mini.local "cd ~/reachy-refined && source .venv/bin/activate && python -m src.main"
