@echo off
REM One-time setup for Windows. Needs Python 3.10-3.13 on PATH.
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv || goto :fail )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip || goto :fail
python -m pip install -r requirements.txt || goto :fail
if not exist .env copy .env.example .env >nul
echo.
echo Setup finished. Start the app with run.bat
exit /b 0
:fail
echo Setup failed. See the messages above.
exit /b 1
