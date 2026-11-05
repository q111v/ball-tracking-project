@echo off
cd /d "%~dp0"
if exist ".conda-env\python.exe" (
  ".conda-env\python.exe" run_verified_demo.py
) else (
  python run_verified_demo.py
)
pause
