@echo off
REM Starts SonicCaption (API + web UI on one port). Open http://localhost:8000
cd /d "%~dp0"
if not exist .venv ( echo Run setup.bat first & exit /b 1 )
call .venv\Scripts\activate.bat
if "%PORT%"=="" set PORT=8000
echo SonicCaption on http://localhost:%PORT%
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port %PORT%
