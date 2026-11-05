@echo off
setlocal
cd /d "%~dp0.."
where conda >nul 2>nul
if errorlevel 1 (
  echo Please run this file from Anaconda Prompt. Conda was not found.
  pause
  exit /b 1
)
set "PROJECT_ENV=%CD%\.conda-env"
if not exist "%PROJECT_ENV%\python.exe" (
  call conda create --prefix "%PROJECT_ENV%" python=3.12 -y
  if errorlevel 1 goto failed
)
call conda run --prefix "%PROJECT_ENV%" python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
call conda run --prefix "%PROJECT_ENV%" python -m pip install -r requirements-dev.txt
if errorlevel 1 goto failed
call conda run --prefix "%PROJECT_ENV%" python -m pytest -q
if errorlevel 1 goto failed
set "TEST_RUN=runs\windows_online_%RANDOM%_%RANDOM%"
call conda run --prefix "%PROJECT_ENV%" python run_nba.py --no-display --basketball-pass forward --max-frames 10 --output "%TEST_RUN%"
if errorlevel 1 goto failed
echo Finished. See %TEST_RUN% and docs\BASKETBALL_GUIDE.md.
echo In PyCharm, select .conda-env\python.exe as the interpreter.
pause
exit /b 0
:failed
echo Installation or test failed. Review the error above.
pause
exit /b 1
