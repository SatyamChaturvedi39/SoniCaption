# SonicCaption — Complete Product Implementation Plan (v4.0)
**SonicCaption: A Multimodal Corporate Meeting Intelligence and Workplace Accessibility Assistant Using Hybrid DSP, Acoustic Sensing, and LLM Discourse Analysis**  
**Speech Processing & Recognition (MAI522-5) | MSc AI & ML | Group of 3**  
**Team:** Satyam Chaturvedi (2548547) · Huda Maniyar (2548526) · J.V. Baarathi (2548514)

---

## 0. Executive Summary & Evolution to Corporate Meeting Intelligence

SonicCaption began as a dual-stream multimodal captioning system combining Whisper ASR with custom environmental sound classification. It has been evolved and re-architected into an end-to-end **Corporate Meeting Assistant and Workplace Accessibility Intelligence Platform** tailored for deaf and hard-of-hearing professionals—while strictly maintaining a **$0.00 infrastructure cost**.

Beyond transcribing what was said and tagging background events, the implemented system gives deaf employees full situational, social, paralinguistic, and operational visibility during live or recorded workplace meetings:

### Core Implemented Capabilities
1. **Signal Conditioning & Noise Suppression:** Multi-format audio ingest (WAV, MP3, OGG, M4A, FLAC, WebM) downmixed to 16 kHz mono; Boll/Berouti magnitude spectral subtraction (OLA) with smoothed quiet-frame noise floor tracking and sub-65 Hz stationary fan/mains attenuation (`backend/src/soniccaption/algorithms/enhancement.py`).
2. **Word-Level Diarization & Sentence Prosody:** Splits Whisper transcripts cleanly at word-timestamp punctuation boundaries (`split_sentences_with_words`), segregates speakers via L2-normalized 12-DCT active-frame vocal-tract timbre + octave-guarded semitone $F_0$ (`diarize_speakers` in batch mode; `assignLiveSpeaker` 2.8-semitone clustering up to 4 speakers in live streaming), annotates **every sentence with a strictly one-word acoustic emphasis tag** (`Urgent`, `Emphatic`, `Assertive`, `Hesitant`, `Soft`, `Neutral`), and explicitly flags rising-pitch interrogative delivery (`is_question`, `question_reason`).
3. **Deep Workplace Acoustic Event Sensing (`ESC-50`):** Fine-tuned 1-channel adapted **ResNet-18** (`SonicAcousticResNet`, default) achieving **88.75% Fold-4 validation accuracy**, **80.00% held-out Fold-5 test accuracy**, and **0.787 macro-F1** across 10 balanced workplace/safety classes (`clapping`, `coughing`, `dog`, `door_wood_knock`, `footsteps`, `glass_breaking`, `laughing`, `rain`, `siren`, `sneezing`), plus optional 4-block 5-fold backup **SonicCNN** (`hudhud26/soniccaption-esc50-cnn`, 84.3% CV accuracy). Speech-overlap masking via Silero VAD prevents voice harmonics from misfiring environmental labels, while surviving detections map to clear workplace cues (`WORKPLACE_CUES`, e.g. colleague laughter, door knock/desk tap, sharp bang, group applause, emergency siren).
4. **Automated Action-Item Extraction & Dynamic Meeting Scenario:** `agent_service.py` combines compact scalar prosody turns + workplace room cues through Groq's `openai/gpt-oss-20b` JSON Copilot (with word-boundary fallback `_has_any_term` and regex commitment extractors `_SELF_COMMIT_RE`, `_DELEGATED_COMMIT_RE`, `_TIME_RE` offline) to structure commitments into `owner`, `task`, `deadline`, `priority`, and `timestamp_s`, while continuously deriving the overarching `meeting_scenario` (label, mood, speaker/question/action counts, and active room cues).
5. **AI Synthetic Voice Authenticator (`authenticator.py`):** Evaluates sub-sample parabolic micro-pitch jitter, 10 ms non-overlapping amplitude shimmer, 16th-order LPC inverse-filter glottal residual kurtosis/flatness, and high-frequency spectral cutoff ($\ge 7\text{ kHz}$ vs $0.3\text{--}3.5\text{ kHz}$) on VAD-active speech to score synthetic/TTS audio vs natural human vocal tracts (`Authentic Human Voice`, `Borderline`, `Likely AI-Generated / Synthetic`).
6. **Subsequence DTW Spotter & Discrete HMM Sequence Smoothing:** Query-by-Voice acoustic search (`subsequence_dtw_search`) over Z-normalized 13-MFCC sequences, cross-session DTW archival ranking (`find_similar_sessions_dtw`), and log-space Viterbi decoding (`DiscreteHMM`) running automatically across `acoustic_scene` (`Ambient`, `Speech`, `SoundEvent`) and `prosody_discourse` (`TurnPause`, `Explanation`, `KeyEmphasis`, `QuestionPrompt`) states in addition to textbook presets.
7. **Full-Stack Corporate Web Suite (`React 19 + FastAPI + SQLite`):** Responsive dark studio web application with **Meeting Studio**, **Live Meeting Radar** (3s PCM streaming + 1-click Sample Corporate Meeting Demo), **AI Voice Authenticator**, and **Voice Copilot** (conversational SQLite session Q&A), backed by 5-format exports (`.srt`, `.vtt`, `.json`, `.csv`, `.txt`) and `99/99` passing backend tests.

---

## 1. Speech Processing Foundations & System Alignment

| Technical Area | Mathematical / ML Methods Implemented | Backend Module & Frontend Evidence |
|---|---|---|
| **Audio Ingest & Time-Frequency DSP** | 16 kHz mono resampling, pre-emphasis ($\alpha=0.97$), Hamming framing ($25\text{ ms}/10\text{ ms}$), short-time energy, ZCR, FFT, STFT spectrogram, Boll OLA denoising with sub-65 Hz rumble mask | `audio/io.py`, `audio/preprocessing.py`, `algorithms/spectral.py`, `algorithms/enhancement.py`; Collapsible **Audio Signal & Formant Diagnostics** Canvas charts in **Meeting Studio** |
| **Spectral, Formant & VQ Modeling** | 40-band Mel filter-banks, 13 MFCCs + $\Delta, \Delta\Delta$, 16th-order Levinson-Durbin LPC & PARCOR poles ($F_1\text{--}F_3$), K-Means & LBG Vector Quantization | `algorithms/spectral.py`, `algorithms/lpc.py`, `algorithms/vq.py`, `algorithms/authenticator.py`; **FFT vs LPC Envelope** Canvas plot & **AI Voice Authenticator** biometric strip |
| **VAD & Dynamic Pattern Alignment** | Silero VAD active speech segmentation, subharmonic-guarded F0 tracking, Timbre+Pitch multi-speaker diarization, Subsequence DTW query search & session similarity | `audio/vad.py`, `algorithms/dtw.py`, `algorithms/enhancement.py`; **Query-by-Voice Acoustic Spotter** & **Similar Recordings (DTW Match)** in **Meeting Studio** |
| **Discrete HMM & Sequence Smoothing** | Log-space Forward/Backward & Viterbi decoding over $\lambda=(S, V, A, B, \pi)$ for pedagogical presets and live `acoustic_scene` + `prosody_discourse` flows | `algorithms/hmm.py`, `/api/v1/labs/hmm/viterbi`; **Acoustic Scene Flow** & **Discourse & Vocal Emphasis Flow** badges in **Meeting Studio** |
| **Deep Sound Classification & Multimodal Copilot** | ResNet-18 (default) / SonicCNN (backup) ESC-50 sliding inference, Groq Whisper ASR (`faster-whisper` fallback), VAD overlap suppression (`WORKPLACE_CUES`), Groq LLM + regex action/scenario engine | `services/{asr_service,sound_cnn,agent_service,session_store}.py`, `timeline/fusion.py`, `app/main.py`; **Overarching Scenario**, **Action Items Table**, **Live Meeting Radar**, **Voice Copilot** |

---

## 2. Actual Technology Stack ($0.00 Cost Architecture)

| Tier | Component | Actual Implementation | Infrastructure Cost |
|---|---|---|---|
| **Frontend Client** | React 19, Vite, TypeScript | Single-page workstation (`frontend/src/App.tsx`, `StudioTabs.tsx`, `style.css` dark theme — zero bloated CSS frameworks) | Free (Vercel / Local Vite) |
| **Audio UI & Plotting** | WaveSurfer.js v8.1 + HTML5 Canvas + MediaRecorder | `WavePlayer.tsx` scrubbing & timestamp seeking; `VisualCanvas.tsx` dual-line & STFT/MFCC heatmaps; `wavEncoder.ts` 16 kHz PCM capture | Free (Client-side WebAudio) |
| **Backend API** | FastAPI + Uvicorn + Pydantic | `backend/app/main.py` REST endpoints + OpenAPI `/docs` + SQLite (`backend/data/soniccaption.db`) | Free (HF Spaces / Local) |
| **Speech ASR & LLM** | Groq Cloud API + `faster-whisper` | `whisper-large-v3-turbo` ASR + `openai/gpt-oss-20b` JSON Copilot, with offline `faster-whisper` & heuristic/regex fallback | Free Tier / Offline |
| **Sound Classification** | PyTorch + TorchAudio + Torchvision | `SonicAcousticResNet` (1-ch ResNet-18 in `services/weights/`) + optional HF `SonicCNN` backup | Free (Colab T4 / Local CPU) |
| **Testing & QA** | Pytest + TypeScript `tsc` / Vite build | 13 Python test suites (**99/99 tests passing**) + strict TypeScript compilation | Free (GitHub Actions / Local) |

---

## 3. System Architecture & REST API Surface

```text
+---------------------------------------------------------------------------------------------------+
|                        CLIENT TIER: React 19 + Vite + TypeScript (frontend/)                      |
|  Sidebar Corporate Suite:                                                                         |
|   [Meeting Studio]  [Live Meeting Radar]  [AI Voice Authenticator]  [Voice Copilot]               |
|   + Engine Architecture Modal (ESC-50 10-class ResNet-18 specs)                                   |
+--------------------+--------------------------+----------------------------+----------------------+
                     | multipart / JSON REST    | 3s PCM chunks              | biometric / archive
                     v                          v                            v
+---------------------------------------------------------------------------------------------------+
|                             BACKEND TIER: FastAPI (backend/app/main.py)                           |
|                                                                                                   |
|  POST /api/v1/audio/analyze ──> Spectral Sub OLA ──> Silero VAD ──> Whisper + Sentence Split      |
|       │                                                                    │                      |
|       ├──> diarize_speakers (12-DCT timbre + F0) + analyze_segment_prosody (1-word tag + ? flag)  |
|       ├──> detect_sound_events (ResNet-18 / SonicCNN) + fuse_timeline (VAD mask + WORKPLACE_CUES) |
|       ├──> decode_acoustic_scene_hmm + decode_prosody_hmm + extract_clip_signature (DTW)          |
|       ├──> detect_synthetic_voice (jitter / shimmer / LPC kurtosis / HF cutoff)                   |
|       └──> extract_session_insights (Groq openai/gpt-oss-20b + word-boundary/regex fallback)      |
|                 └─> SQLite persistence (sessions, reminders/action-notes, urgent room alerts)     |
|                                                                                                   |
|  Additional Live & Research Routes:                                                               |
|   • POST /api/v1/watch/chunk             (Live streaming radar: VAD, ResNet cues, ASR, prosody)   |
|   • POST /api/v1/meeting/live-intelligence (Continuous live scenario + action item synthesis)      |
|   • POST /api/v1/audio/query-dtw         (Query-by-Voice subsequence MFCC search)                 |
|   • POST /api/v1/voice/authenticate      (Biometric deepfake / synthetic speech scan)             |
|   • POST /api/v1/agent/turn              (Voice Copilot archive Q&A across SQLite sessions)       |
|   • POST /api/v1/labs/{signal,dtw,hmm/viterbi} + /api/v1/sessions* + /api/v1/export/{fmt}         |
+---------------------------------------------------------------------------------------------------+
```

### Primary Implemented Endpoints (`backend/app/main.py`)
| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Runtime mode (`live`/`mock`), `groq_configured`, model & FFmpeg readiness |
| `POST` | `/api/v1/audio/analyze` | End-to-end corporate meeting pipeline returning timeline, speakers, 1-word emphasis tags, questions, room cues, action items, scenario, HMM paths, DTW signature, diagnostics & voice authenticity |
| `POST` | `/api/v1/watch/chunk` | Low-latency 3s microphone chunk processor returning RMS dB, baseline calibration, speech turns with emphasis/question metadata, speech-masked workplace cues, and urgent alerts |
| `POST` | `/api/v1/meeting/live-intelligence` | Continuously updates live `meeting_scenario` and `action_items` from streamed speaker turns & sound events |
| `POST` | `/api/v1/audio/query-dtw` | Query-by-Voice Subsequence DTW locating spoken words/phrases in the active clip |
| `POST` | `/api/v1/voice/authenticate` | Forensic deepfake/TTS detection via jitter, shimmer, LPC residual kurtosis & spectral cutoff |
| `POST` | `/api/v1/agent/turn` | Conversational Voice Copilot grounded in SQLite session history and live acoustic events |
| `GET/POST/DELETE` | `/api/v1/sessions*`, `/reminders*`, `/alerts` | Persistent SQLite CRUD and DTW similar-session retrieval (`/sessions/{id}/similar`) |
| `POST` | `/api/v1/labs/{signal,dtw,hmm/viterbi}` | Interactive DSP, DTW matrix, and Viterbi trellis JSON endpoints |
| `POST` | `/api/v1/export/{fmt}` | Export subtitles and structured logs in `.srt`, `.vtt`, `.json`, `.csv`, `.txt` |

---

## 4. Actual Monorepo Layout

```text
SoniCaption/
├── backend/
│   ├── app/
│   │   ├── main.py                      # Unified FastAPI server, pipeline orchestrator & exporters
│   │   ├── mock_data.py                 # Fallback mock payloads when ML deps are disabled
│   │   └── schemas.py                   # API Pydantic models
│   ├── data/
│   │   └── soniccaption.db              # Persistent SQLite database (sessions, action notes, alerts)
│   ├── src/soniccaption/
│   │   ├── core/schemas.py              # Shared dataclass contracts
│   │   ├── audio/
│   │   │   ├── io.py                    # Multi-format decode, mono mix, 16 kHz resampling
│   │   │   ├── preprocessing.py         # Pre-emphasis, framing, Hamming/Hanning windowing
│   │   │   └── vad.py                   # Silero VAD speech interval detector
│   │   ├── algorithms/
│   │   │   ├── spectral.py              # Energy, ZCR, FFT, STFT, 40-Mel, 13-MFCC + deltas
│   │   │   ├── lpc.py                   # Order-16 Levinson-Durbin LPC, PARCOR & formants
│   │   │   ├── vq.py                    # K-Means & LBG Vector Quantization
│   │   │   ├── dtw.py                   # Dynamic Time Warping recurrence & backtracking
│   │   │   ├── hmm.py                   # DiscreteHMM + acoustic_scene & prosody_discourse decoders
│   │   │   ├── enhancement.py           # Boll OLA denoising, F0, sentence split, diarization, prosody, sub-DTW
│   │   │   └── authenticator.py         # Forensic jitter/shimmer/LPC/cutoff AI voice detector
│   │   ├── services/
│   │   │   ├── asr_service.py           # Groq Whisper (large-v3-turbo) + faster-whisper local fallback
│   │   │   ├── sound_cnn.py             # SonicAcousticResNet (default) + SonicCNN (HF backup) + ambient gating
│   │   │   ├── agent_service.py         # Groq gpt-oss-20b Copilot + word-boundary/regex actions & scenario
│   │   │   ├── session_store.py         # SQLite persistence layer
│   │   │   └── weights/                 # Trained ResNet-18 weights, config, labels & metrics JSONs
│   │   └── timeline/
│   │       └── fusion.py                # Chronological fusion, speech-overlap masking & WORKPLACE_CUES
│   └── tests/                           # 13 pytest suites (99 passing tests)
├── frontend/
│   ├── package.json                     # React, Vite, TypeScript, Lucide, WaveSurfer.js 8.1
│   └── src/
│       ├── App.tsx                      # Corporate Suite shell & Meeting Studio + ResultsView
│       ├── api.ts & types.ts            # Typed REST client & TypeScript schemas
│       ├── style.css                    # Custom dark-theme responsive workstation styling
│       ├── components/
│       │   ├── StudioTabs.tsx           # Live Meeting Radar (WatchView), Voice Authenticator, Copilot, Specs Modal
│       │   ├── WavePlayer.tsx           # WaveSurfer.js audio scrubber with seek support
│       │   └── VisualCanvas.tsx         # HTML5 Canvas waveform, FFT/LPC, STFT & MFCC renderers
│       └── utils/wavEncoder.ts          # Raw 16 kHz PCM capture & WAV encoder
├── notebooks/
│   ├── 01_audio_feature_extraction_pipeline.ipynb
│   ├── 02_environmental_sound_classifier_training.ipynb
│   ├── 02b_sound_classifier_cnn_5fold_backup.ipynb
│   ├── 03_speech_recognition_and_sequence_fusion.ipynb
│   └── 04_acoustic_enhancement_diarization_and_agent.ipynb
├── report/
│   ├── main.tex                         # CHRIST University CIA III IEEE-style LaTeX report
│   ├── mybibliography.bib               # Verified IEEE/ISCA/CVPR bibliography
│   └── Figures/                         # Training curves, confusion matrix, DSP, LPC, DTW, HMM visuals
├── SonicCaption_Parallel_Execution_and_Work_Split.md
└── SonicCaption_Updated_Complete_Implementation_Plan.md
```

---

## 5. Verification Results & Live Corporate Demo Flow

### Verified Test & Build Gates
- **Backend Automated Suite (`pytest backend/tests/ -q`):** **99 / 99 passed** across all 13 test modules (covering Spectral DSP, LPC/VQ, DTW & Subsequence DTW, Discrete HMM & scene/prosody decoders, Boll Spectral Subtraction SNR gain, 3-turn diarization stability `Speaker 1 -> Speaker 2 -> Speaker 1`, one-word emphasis & rising-pitch question detection, AI voice authentication, Fusion speech suppression, SQLite CRUD, word-boundary scenario classification, and FastAPI integration endpoints).
- **Frontend Build (`npm run build --prefix frontend`):** TypeScript check (`tsc -b`) + Vite bundle builds cleanly with 0 errors.
- **Live Server API Checks:**
  - `POST /api/v1/audio/analyze` (`speech_sample.ogg`) returns HTTP 200 with `insights_provider: "groq-openai/gpt-oss-20b"`, accurate `Executive Briefing` scenario (zero substring false positives), and annotated one-word emphasis badges.
  - `POST /api/v1/meeting/live-intelligence` returns structured speaker-assigned commitments (e.g., `Speaker 3` $\to$ `"Update GitHub with the pull request"` due `"Tomorrow at 4 PM"`) alongside dynamically derived `"Engineering Sprint Standup"` mood and active workplace cues.

### Live Presentation Walkthrough (5–6 Minutes)
1. **Corporate Meeting Studio (`Meeting Studio` tab):** Upload or record a multi-speaker clip containing a question, a commitment (*"I will update GitHub tomorrow at 4 PM"*), and background applause/door knock. Inspect the **Overarching Corporate Meeting Scenario**, **Automated Action Items** table, color-coded speakers, one-word emphasis tags (`Urgent`, `Emphatic`, `Assertive`), `? QUESTION` pitch flags, and workplace room cues. Click any row or action item cue to seek **WaveSurfer.js**.
2. **Acoustic Pattern & Sequence Analysis:** Run a quick spoken query via **Query-by-Voice Acoustic Spotter (Subsequence DTW)**, inspect **Acoustic Scene & Discourse Prosody HMM Viterbi Flows**, and toggle **Inspect Plots** to examine Denoised vs Raw Waveform, 2D STFT Spectrogram, 13-MFCC profile, and Peak FFT vs LPC Formants.
3. **Live Corporate Meeting Radar (`Live Meeting Radar` tab):** Demonstrate real-time 3s PCM room monitoring (`Calibrate Room Floor` in noisy classrooms) and click **Test Sample Meeting** to show immediate question alerts, multi-speaker live transcripts, action-item capture, and scenario updates.
4. **AI Voice Authenticator & Voice Copilot:** Scan a voice clip in **AI Voice Authenticator** to inspect jitter/shimmer/LPC kurtosis forensics, query saved meetings in **Voice Copilot**, open **Engine Architecture** for ESC-50 per-class F1 scores (`0.787` macro-F1 / `80.0%` held-out test), and download `.srt` / `.vtt` / `.json` captions.
