# SonicCaption: notes for coding agents

* Run: `./setup.sh` then `./run.sh` (http://localhost:8000). Tests: `.venv/bin/python -m pytest backend/tests -q` (run from repo root).
* Backend `backend/app/main.py` (FastAPI) + library `backend/src/soniccaption/`. Frontend `frontend/src` (React + Vite); `frontend/dist` is committed and served by the backend, so rebuild it with `cd frontend && npm install && npm run build` after UI changes.
* Pipeline order: VAD -> scene classifier (raw audio) -> diarization (denoised audio) -> ASR -> NLU. Live path: `services/live_session.py`; final pass reuses `analysis_pipeline.analyse_audio`.
* ASR is injectable (`asr_fn`) so tests never need network. Never fabricate transcript text when ASR fails; surface it.
* Product requirements (do not change): two tabs (Live recording, Upload audio); dark blue + light yellow on white, red/green for status; speaker-attributed tasks; result sub-tabs for speech per person, MFCC/spectrogram, HMM, patterns.
