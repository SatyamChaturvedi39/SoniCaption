# SonicCaption — Project Requirements (Member 3)

## Project
Build a web application for speech transcription and environmental sound captioning. The complete team system will combine speech segments and environmental sound events in a synchronized audio timeline.

## Current scope
Implement only the frontend, backend orchestration, mock APIs, visualizations, exports, and integration-ready structure. Do not implement or train DSP/ML models. All generated analysis values must be clearly identified as synthetic demo data.

## Design
- Light, clean, responsive interface.
- Dark blue `#172D52`, secondary blue `#233E68`, pastel yellow `#F7EFCB`, light background `#F7F8FC`.
- Main workspace has two tabs: **Live Recording** and **Upload WAV File**.
- Both input flows lead to the same Results view.

## Live recording
Use browser MediaRecorder. Show ready, recording, completed, and processing states. During recording show a recording indicator, timer, and audio-level/waveform feedback. Stop microphone tracks after recording. Handle denied permissions, missing microphone, unsupported browser, empty recording, and network errors.

## Upload
Accept `.wav` only, support drag/drop and file browsing, validate file type, show filename and audio preview, and provide an Analyze action.

## Processing
Show staged messages: preparing audio, analyzing speech, identifying sounds, building captions, preparing results. In mock mode these stages are simulated, not real model progress. Do not fabricate percentage progress.

## Results
Show audio metadata, player, speech segments, environmental events, confidence when available, unified chronological timeline, click-to-seek, and export actions.

## Lab pages
- Signal Lab: waveform, FFT, STFT, RMS/energy, ZCR, MFCC, LPC, VQ demonstration.
- Pattern Lab: DTW cost matrix, alignment path, distance.
- HMM Trellis: states, transitions, probabilities, best path.
- Model Card: placeholder/demo model information and illustrative metrics, never presented as actual model results.

## Stack
Frontend: React, TypeScript, Vite, Tailwind CSS, React Router, WaveSurfer.js, Recharts/Lucide as useful.
Backend: Python 3.10/3.11, FastAPI, Uvicorn, Pydantic, pytest.

## API
- `GET /api/health`
- `POST /api/v1/audio/analyze`
- `POST /api/v1/labs/signal`
- `POST /api/v1/labs/dtw`
- `POST /api/v1/labs/hmm/viterbi`
- `POST /api/v1/export/{format}`

Use stable response schemas and `USE_MOCK_DATA=true` by default. Keep adapters for Member 1's DSP/DTW and Member 2's ASR/VAD/CNN/HMM. Do not assume their eventual filenames or function signatures. Do not silently substitute mock output if live integration fails.

## Exports
Support SRT, VTT, JSON, CSV, TXT.

## Deliverables
Monorepo source, setup guide, README, tests, screenshot PDF compilation support (to be completed after UI review), five-page report contributions (Member 3: Page 1 and Page 5), and deployment preparation. Do not invent test results, model metrics, URLs, or contributions.
