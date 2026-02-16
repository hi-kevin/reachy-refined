@echo off
echo Setting up remote environment...
ssh -t pollen@reachy-mini.local "cd reachy-refined && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .[yolo_vision]"
echo Setup complete.
