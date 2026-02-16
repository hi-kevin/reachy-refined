@echo off
echo ****************************************************************
echo [1/4] Killing existing Refined Reachy processes on robot...
echo ****************************************************************
ssh pollen@reachy-mini.local "pkill -f 'src.main'"

echo ****************************************************************
echo [2/4] Removing local cache...
echo ****************************************************************
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo ****************************************************************
echo [3/4] Syncing source code...
echo ****************************************************************
ssh pollen@reachy-mini.local "mkdir -p ~/reachy-refined"
scp -r src pollen@reachy-mini.local:~/reachy-refined/
scp .env pollen@reachy-mini.local:~/reachy-refined/
scp requirements.txt pollen@reachy-mini.local:~/reachy-refined/
scp scripts/check_encoding.py pollen@reachy-mini.local:~/reachy-refined/

echo ****************************************************************
echo [4/4] Checking remote encoding...
echo ****************************************************************
ssh pollen@reachy-mini.local "cd ~/reachy-refined && python check_encoding.py"

echo Deployment complete. Run 'python -m src.main' on the robot to start.
