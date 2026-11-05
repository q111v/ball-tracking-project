@echo off
cd /d "%~dp0"
if exist ".conda-env\python.exe" (
  ".conda-env\python.exe" run_nba.py
) else (
  python run_nba.py
)
pause
