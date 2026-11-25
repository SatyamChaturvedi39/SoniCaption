# SonicCaption — Work Split & Deliverables Ownership
**SonicCaption: A Multimodal Corporate Meeting Intelligence and Workplace Accessibility Assistant Using Hybrid DSP, Acoustic Sensing, and LLM Discourse Analysis**  
**Speech Processing & Recognition (MAI522-5) | MSc AIML | Group of 3**

---

## 0. Group Roster & Architectural Model

| Member | Reg. No. | Primary Engineering & Research Ownership | Report Sections |
|---|---|---|---|
| **Member 1: Satyam Chaturvedi** | `2548547` | Audio DSP Core, Noise Enhancement, ESC-50 ResNet-18 Training, Multi-Speaker Diarization, Prosody/Emphasis & Question Inflection, AI Voice Authenticator, Subsequence DTW Spotter, SQLite Store, Corporate Action-Item & Scenario Copilot | Sections 3, 4.1–4.5, 4.7, 5.1–5.3 |
| **Member 2: Huda Maniyar** | `2548526` | Silero VAD Segmentation, Groq Whisper & `faster-whisper` ASR Adapter, ResNet/SonicCNN Sliding-Window Inference (`sound_cnn.py`), Backup 5-Fold `SonicCNN`, Discrete HMM & Viterbi Sequence Modeling, Timeline Fusion & Workplace Cues | Sections 4.5 (HMM), 4.6, 5.2 |
| **Member 3: J.V. Baarathi** | `2548514` | FastAPI Unified Orchestrator Contracts & Exporters, React 19 + TypeScript Corporate Suite UI (**Meeting Studio**, **Live Meeting Radar**, **AI Voice Authenticator**, **Voice Copilot**), WaveSurfer.js Playback, HTML5 Canvas Visualizers & CI/Packaging | Abstract, Sections 1–2, 4.7 (Web Stack), 6 |

### Core Design Principle: Notebook-First Math + Modular Service Adapters
1. **Reproducible Jupyter Notebooks (`notebooks/`) Hold Core Equations & Training Loops:** Every DSP transform, model training run, and mathematical verification lives cleanly inside notebooks `01`, `02`, `02b`, `03`, and `04` with exported figures in `report/Figures/` and `notebooks/screenshots/`.
2. **Modular Python Services (`backend/src/soniccaption/`) Package Verified Logic:** Zero research code duplication—the FastAPI server (`backend/app/main.py`) orchestrates clean domain packages (`audio/`, `algorithms/`, `services/`, `timeline/`).

---

## 1. Detailed Deliverable & Code Ownership Breakdown

### Member 1 — Satyam Chaturvedi (`2548547`): Signal Processing, Acoustic Model, Forensics & Corporate Copilot
- **Research & Training Notebooks:**
  - `notebooks/01_audio_feature_extraction_pipeline.ipynb`: From-scratch DSP pipeline (16 kHz resampling, pre-emphasis $\alpha=0.97$, Hamming framing, STFT, 40-channel Mel filter-banks, 13-MFCC + $\Delta/\Delta\Delta$, order-16 Levinson-Durbin LPC & PARCOR formants, K-Means/LBG VQ, DTW alignment) + mathematical proof matching PyTorch/TorchAudio tensors.
  - `notebooks/02_environmental_sound_classifier_training.ipynb`: Leak-free ESC-50 10-class ResNet-18 transfer learning (Folds 1–3 train, Fold 4 select, Fold 5 strict held-out test) with SpecAugment, achieving **88.75% Fold-4 validation accuracy**, **80.00% Fold-5 test accuracy**, and **0.787 macro-F1**; exports `soniccaption_esc50_resnet.pt` & metrics JSONs.
  - `notebooks/04_acoustic_enhancement_diarization_and_agent.ipynb`: Boll Spectral Subtraction denoising, $F_0$ pitch contour tracking, VQ diarization, SQLite persistence, and Copilot workflow validation.
- **Production Backend Modules:**
  - `backend/src/soniccaption/audio/io.py` & `preprocessing.py` (decode, mono mix, 16 kHz resampling, framing/windows).
  - `backend/src/soniccaption/algorithms/spectral.py`, `lpc.py`, `vq.py`, `dtw.py`.
  - `backend/src/soniccaption/algorithms/enhancement.py`: Boll/Berouti OLA spectral subtraction (with sub-65 Hz stationary fan-rumble attenuation), subharmonic-guarded $F_0$ estimation, `split_sentences_with_words`, `diarize_speakers` (4.0$\times$ L2-timbre + semitone $F_0$), `analyze_segment_prosody` (strictly one-word `emphasis_tag` + vocal rising-pitch question detection), `subsequence_dtw_search`, and `find_similar_sessions_dtw`.
  - `backend/src/soniccaption/algorithms/authenticator.py`: Forensic AI voice authenticity detector (parabolic micro-pitch jitter, 10 ms sub-frame shimmer, LPC residual kurtosis/flatness, $\ge 7\text{ kHz}$ spectral cutoff).
  - `backend/src/soniccaption/services/session_store.py` & `agent_service.py`: SQLite database + Groq `openai/gpt-oss-20b` JSON Copilot + deterministic regex action-item extraction (`owner`, `task`, `deadline`, `priority`) and word-boundary corporate scenario derivation (`_has_any_term`).
  - **Automated Tests:** `test_io.py`, `test_spectral.py`, `test_lpc.py`, `test_vq.py`, `test_dtw.py`, `test_enhancement.py`, `test_agent_and_store.py`, and end-to-end corporate/API cases in `test_api.py`.

### Member 2 — Huda Maniyar (`2548526`): Speech Recognition, Sequence HMMs & Timeline Fusion
- **Research & Backup Training Notebooks:**
  - `notebooks/03_speech_recognition_and_sequence_fusion.ipynb`: Silero VAD segmentation, Groq Whisper speech-to-text evaluation, log-space Discrete HMM formulation ($\pi, A, B$), Forward likelihood, Viterbi trellis decoding, and chronological dual-stream timeline fusion + `.srt` export.
  - `notebooks/02b_sound_classifier_cnn_5fold_backup.ipynb`: Compact 4-block CNN (`SonicCNN`) trained under strict 5-fold CV across 10 balanced ESC-50 classes (`0.843` accuracy, `0.842` macro-F1), published at Hugging Face Hub (`hudhud26/soniccaption-esc50-cnn`).
- **Production Backend Modules:**
  - `backend/src/soniccaption/audio/vad.py`: Silero VAD speech boundary detector & speech ratio calculation.
  - `backend/src/soniccaption/services/asr_service.py`: Groq Whisper API provider (`whisper-large-v3-turbo` with chunking and flat word-timestamp mapping) plus offline `faster-whisper` fallback behind a unified contract.
  - `backend/src/soniccaption/services/sound_cnn.py`: Dynamic checkpoint loader & sliding-window detector supporting both `SonicAcousticResNet` (default) and `SonicCNN` backup with ambient noise/flatness gating.
  - `backend/src/soniccaption/algorithms/hmm.py`: `DiscreteHMM` (Forward, Backward, Posteriors, Viterbi trellis `decode_for_ui`) plus production presets `acoustic_scene` (`decode_acoustic_scene_hmm`) and `prosody_discourse` (`decode_prosody_hmm`).
  - `backend/src/soniccaption/timeline/fusion.py`: Deterministic chronological timeline merger (`fuse_timeline`, `FusionConfig`), Silero VAD speech-overlap suppression, competing-event resolution, and workplace cue definitions (`SOUND_TAGS`, `WORKPLACE_CUES`).
  - **Automated Tests:** `test_vad.py`, `test_asr_service.py`, `test_model_inference.py`, `test_hmm.py`, `test_fusion.py`.

### Member 3 — J.V. Baarathi (`2548514`): Full-Stack API Integration, Corporate Accessibility UI & Exports
- **FastAPI Backend Orchestration (`backend/app/`):**
  - `backend/app/main.py`, `schemas.py`, `mock_data.py`: Unified API server exposing `/api/health`, `/api/v1/audio/analyze`, `/api/v1/audio/query-dtw`, `/api/v1/voice/authenticate`, `/api/v1/agent/turn`, `/api/v1/watch/chunk`, `/api/v1/meeting/live-intelligence`, `/api/v1/sessions*`, `/api/v1/reminders*`, `/api/v1/alerts`, `/api/v1/labs/{signal,dtw,hmm/viterbi}`, and 5-format exporter `/api/v1/export/{srt,vtt,json,csv,txt}`.
- **React 19 + TypeScript Web Product (`frontend/src/`):**
  - `App.tsx` & `style.css`: Corporate Suite sidebar navigation, upload & raw 16 kHz mic recording, AI authenticity chip, **Overarching Corporate Meeting Scenario** card, **Automated Action Items & Spoken Commitments** table with seek cues, color-coded **Chronological Meeting Captions, Emphasis & Room Cues** timeline, Query-by-Voice DTW Spotter, Acoustic & Prosody HMM Viterbi badges, DTW similar-session matches, and collapsible 4-panel Signal & Formant Diagnostics.
  - `components/StudioTabs.tsx`:
    - `WatchView` (**Live Meeting Radar**): Real-time 3s PCM monitoring, room floor calibration, semitone speaker clustering (`assignLiveSpeaker`), `QUESTION ASKED IN ROOM` banner, urgent room alerts, continuous live scenario updates, live spoken action items, and 1-click **Test Sample Meeting** demo.
    - `VoiceAuthenticatorView`: Forensic voice scanner displaying Authenticity Index, verdict badge, and 4 biometric metrics.
    - `AgentView` (**Voice Copilot**): Spoken/text Q&A grounded in saved SQLite sessions and ambient detections.
    - `EngineSpecsModal`: ESC-50 10-class ResNet-18 quantitative benchmarks modal.
  - `components/WavePlayer.tsx`, `components/VisualCanvas.tsx`, `utils/wavEncoder.ts`, `api.ts`, `types.ts`: WaveSurfer.js v8.1 scrubber, HTML5 Canvas signal/spectrogram plotters, raw PCM microphone encoder, and typed API layer.

---

## 2. Final Submission Deliverables Verification Status

| Deliverable | Contents | Current Status |
|---|---|---|
| **1. GitHub Monorepo** | Full `backend/`, `frontend/`, `notebooks/` (`01`, `02`, `02b`, `03`, `04`), trained weights, and **13 pytest suites (`99/99 passed`)** + `tsc -b && vite build` clean build | Verified Complete |
| **2. CIA III Academic Report (`report/main.tex`)** | Updated title, corporate accessibility framing, Boll denoising + sub-65 Hz attenuation, diarization, 1-word emphasis + rising-pitch questions, AI voice forensics, DTW/HMM, ESC-50 ResNet-18 benchmarks, and verified corporate API evaluation | Verified Updated |
| **3. Plan & Worksplit (`*.md`)** | Complete v4.0 engineering blueprint matching actual file paths, Corporate Suite UI tabs, API routes, weights, test count (`99`), and exact 3-member split | Verified Updated |
| **4. Live Application Demo** | Running Vite SPA (`http://localhost:5173`) + Uvicorn FastAPI (`http://localhost:8000`) tested live on `/api/v1/audio/analyze` and `/api/v1/meeting/live-intelligence` with Groq `openai/gpt-oss-20b` | Verified Live |

---

## 3. Viva Defense Topic Mapping (All 3 Members Ready)

- **Member 1 (DSP, ResNet-18 & Corporate Copilot):** Pre-emphasis, Mel scale & MFCCs, Levinson-Durbin LPC stability (`|k_i| < 1`), Boll spectral subtraction gain + sub-65 Hz fan attenuation, timbre+semitone diarization without pitch-flip errors, AI voice jitter/shimmer/kurtosis forensics, ResNet-18 `conv1` RGB adaptation & leak-free Fold 1–3/4/5 split, Groq `openai/gpt-oss-20b` action extraction & word-boundary fallback.
- **Member 2 (ASR, VAD, HMM & Fusion):** Silero VAD energy/probability gating, Groq Whisper word-timestamp alignment vs `faster-whisper` fallback, log-space Forward & Viterbi dynamic programming ($O(N^2 T)$ vs $O(N^T)$), production `acoustic_scene` and `prosody_discourse` state transitions, speech-overlap suppression rules preventing voice-triggered ESC-50 false positives, and `SonicCNN` 5-fold backup.
- **Member 3 (Full-Stack Product & Accessibility UX):** FastAPI decoupled REST pipeline & SQLite persistence, raw 16 kHz PCM capture avoiding browser auto-gain compression, WaveSurfer.js seek synchronization, semitone live speaker tracking in **Live Meeting Radar**, HTML5 Canvas spectrogram/LPC rendering, 5-format closed-caption exporters, and `$0.00` deployment architecture.
