@echo off
echo Killing Refined Reachy processes on Reachy Mini...
ssh pollen@reachy-mini.local "pkill -f 'src.main'"
echo Done.
