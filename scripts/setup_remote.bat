@echo off
setlocal
set ROOT=%~dp0..
set REMOTE=pollen@reachy-mini.local
set RPATH=~/reachy-refined

echo Creating remote project dir and copying requirements...
ssh %REMOTE% "mkdir -p %RPATH%"
if errorlevel 1 goto :fail
scp "%ROOT%\requirements.txt" %REMOTE%:%RPATH%/
if errorlevel 1 goto :fail

echo Setting up remote venv and installing dependencies...
ssh -t %REMOTE% "cd %RPATH% && python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"
if errorlevel 1 goto :fail

echo Downloading ArcFace face-recognition model...
scp "%ROOT%\scripts\fetch_face_model.py" %REMOTE%:%RPATH%/
if errorlevel 1 goto :fail
ssh %REMOTE% "cd %RPATH% && source .venv/bin/activate && python fetch_face_model.py"
if errorlevel 1 goto :fail

echo Setup complete.
endlocal
exit /b 0

:fail
echo *** SETUP FAILED (errorlevel %errorlevel%) ***
endlocal
exit /b 1
