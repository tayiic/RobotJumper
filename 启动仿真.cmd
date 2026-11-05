@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "run_simulation.py"
if errorlevel 1 pause
