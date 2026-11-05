@echo off
cd /d "%~dp0"
if exist ".conda-env\python.exe" (
  ".conda-env\python.exe" run_basketball.py
) else (
  python run_basketball.py
)
pause
