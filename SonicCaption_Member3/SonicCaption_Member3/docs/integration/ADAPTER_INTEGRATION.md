# Future Integration Adapters

This starter intentionally does not assume the final implementation signatures from teammates.

## Member 1 — DSP / DTW
Expected responsibilities to connect:
- WAV loading, decoding, mono conversion and resampling.
- Framing/windowing/pre-emphasis.
- Short-time energy, ZCR, FFT, STFT, MFCC.
- LPC, VQ, DTW distance/cost matrix/backtracking path.

Suggested integration: implement a service in `backend/app/adapters/member1_adapter.py` that maps Member 1's outputs into the API response contracts. Preserve the frontend response schema.

## Member 2 — Recognition / HMM
Expected responsibilities to connect:
- ASR transcription and word/segment timestamps.
- VAD speech intervals.
- Environmental CNN event labels and confidence scores.
- HMM Forward likelihood and Viterbi path/trellis.
- Model metadata and verified evaluation metrics.

Suggested integration: implement `backend/app/adapters/member2_adapter.py`. Keep credentials in environment variables, never in source code.

## Live-mode behavior
When real adapters are ready, add explicit configuration and health/status reporting. If real inference fails, return an actionable error; do not silently fall back to mock output.
