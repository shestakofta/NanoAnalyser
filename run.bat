@echo off
rem Run NPA from source code (Python 3.10-3.12 with libraries from requirements.txt)
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe app.py
) else (
  python app.py
)
if errorlevel 1 pause
