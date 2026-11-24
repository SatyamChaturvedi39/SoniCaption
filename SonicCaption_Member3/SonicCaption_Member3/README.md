# SonicCaption — Member 3 Full-Stack Starter

A runnable mock-mode implementation for the SonicCaption speech and environmental sound captioning project.

## Features
- Live microphone recording using the browser MediaRecorder API.
- WAV file upload and preview.
- Simulated processing stages, explicitly labelled as demo mode.
- FastAPI mock analysis endpoint.
- Shared results view with waveform/audio player and synchronized speech/event timeline.
- Signal Lab, Pattern Lab, HMM Trellis, and Model Card demonstration pages.
- SRT, VTT, JSON, CSV and TXT exports.
- Clear adapter placeholders for future Member 1 / Member 2 integration.

## Quick start
See [ANTIGRAVITY_SETUP_GUIDE.md](ANTIGRAVITY_SETUP_GUIDE.md) for prerequisites and exact commands.

Backend: `http://localhost:8000`  
API docs: `http://localhost:8000/docs`  
Frontend: `http://localhost:5173`

## Important
All analysis and lab values are synthetic demonstration data. No trained model, DSP algorithm, ASR provider, or VAD engine is included in this initial Member 3 build.
