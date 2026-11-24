# Antigravity Setup Guide (Windows-first)

## 1. Prerequisites
Install:
- Python 3.10 or 3.11
- Node.js LTS (includes npm)
- Git (recommended)
- Antigravity

Verify in PowerShell:
```powershell
python --version
node --version
npm --version
git --version
```

## 2. Open the project
1. Extract `SonicCaption_Member3.zip`.
2. Open Antigravity.
3. Choose the extracted `SonicCaption_Member3` folder as the workspace root.
4. Open two terminal windows in Antigravity.

## 3. Start the backend
In terminal 1:
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

If PowerShell blocks activation, run this in that terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Backend health: http://localhost:8000/api/health  
Swagger docs: http://localhost:8000/docs

## 4. Start the frontend
In terminal 2, from the project root:
```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally http://localhost:5173.

## 5. Test the application
- Try Upload WAV with a `.wav` file.
- Try Live Recording in a browser that supports microphone access.
- Confirm the app shows Demo mode.
- Open Signal Lab, Pattern Lab, HMM Trellis, and Model Card from the navigation.
- Try exporting SRT, VTT, JSON, CSV, and TXT from Results.

Microphone access generally requires localhost or HTTPS and browser permission.

## 6. Run backend tests
From `backend`, with the virtual environment active:
```powershell
pytest
```

## 7. macOS / Linux notes
Create and activate the virtual environment:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
In another terminal:
```bash
cd frontend
npm install
npm run dev
```

## 8. Troubleshooting
- **Port 8000 already in use:** stop the other process or run Uvicorn on another port and update `VITE_API_BASE_URL`.
- **Port 5173 already in use:** Vite will usually offer another port; use the URL it prints.
- **Python not found:** reinstall Python and enable the PATH option.
- **npm not found:** install Node.js LTS and reopen the terminal.
- **CORS/network error:** ensure the backend is running on port 8000.
- **Microphone unavailable:** grant browser microphone permission and use localhost/HTTPS.
- **Upload rejected:** use a valid `.wav` file.

## 9. Stop servers
Press `Ctrl+C` in each terminal. Deactivate the backend virtual environment with:
```powershell
deactivate
```

## 10. Mock mode
The starter defaults to mock data. The backend returns demonstration responses and does not call real models. Keep `USE_MOCK_DATA=true` until the team supplies and integrates real implementations.
