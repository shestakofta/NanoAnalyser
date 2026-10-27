@echo off
setlocal
rem ================================================================
rem  Build portable NPA (NanoAnalyzer): result in dist\NanoAnalyzer
rem  1) Install Python 3.10-3.12 from python.org ("Add to PATH").
rem  2) Edit !infoabout.txt or resources\!infoabout.txt ("About" text) if needed.
rem  3) Double-click this file. Build takes 3-10 minutes.
rem ================================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Install Python 3.10-3.12 from python.org
  echo         and tick "Add python.exe to PATH" during installation.
  pause
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment .venv ...
  python -m venv .venv
  if errorlevel 1 goto :fail
)
set PY=.venv\Scripts\python.exe

echo Installing libraries ...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo Embedding "About" text from !infoabout.txt ...
"%PY%" make_about.py
if errorlevel 1 goto :fail

echo Building the program ...
rem Icon: resources\NPA.ico (GitHub layout) or NPA.ico next to this file
set ICON=NPA.ico
if exist resources\NPA.ico set ICON=resources\NPA.ico
"%PY%" -m PyInstaller --noconfirm --clean --onedir --windowed --name NanoAnalyzer --icon "%ICON%" ^
  --collect-submodules skimage --collect-data skimage ^
  --hidden-import tifffile --hidden-import openpyxl ^
  --hidden-import matplotlib.backends.backend_tkagg --hidden-import matplotlib.backends.backend_agg ^
  --hidden-import PIL._tkinter_finder --hidden-import PIL.ImageTk --hidden-import about_text --hidden-import ml_annot ^
  app.py
if errorlevel 1 goto :fail

echo.
echo DONE. Program: dist\NanoAnalyzer\NanoAnalyzer.exe
echo The whole folder dist\NanoAnalyzer can be copied to a USB stick or another PC.
pause
exit /b 0

:fail
echo.
echo [ERROR] Build failed. Scroll up to see the message, or send a screenshot.
pause
exit /b 1
