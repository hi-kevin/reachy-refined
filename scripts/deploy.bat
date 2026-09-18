@echo off
setlocal
set ROOT=%~dp0..
set REMOTE=pollen@reachy-mini.local
set RPATH=~/reachy-refined

echo ****************************************************************
echo [1/4] Killing existing Refined Reachy processes on robot...
echo ****************************************************************
ssh %REMOTE% "pkill -f 'src.main'"

echo ****************************************************************
echo [2/4] Removing local cache...
echo ****************************************************************
for /d /r "%ROOT%" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo ****************************************************************
echo [3/4] Syncing source code...
echo ****************************************************************
ssh %REMOTE% "mkdir -p %RPATH%/scripts"
if errorlevel 1 goto :fail
scp -r "%ROOT%\src" %REMOTE%:%RPATH%/
if errorlevel 1 goto :fail
scp "%ROOT%\.env" %REMOTE%:%RPATH%/
if errorlevel 1 goto :fail
scp "%ROOT%\requirements.txt" %REMOTE%:%RPATH%/
if errorlevel 1 goto :fail
scp "%ROOT%\scripts\check_encoding.py" %REMOTE%:%RPATH%/
if errorlevel 1 goto :fail

echo ****************************************************************
echo [4/4] Checking remote encoding...
echo ****************************************************************
ssh %REMOTE% "cd %RPATH% && python check_encoding.py"
if errorlevel 1 goto :fail

echo.
echo Deployment complete. Run 'python -m src.main' on the robot to start.
endlocal
exit /b 0

:fail
echo.
echo *** DEPLOY FAILED (errorlevel %errorlevel%) - robot NOT updated. ***
endlocal
exit /b 1
