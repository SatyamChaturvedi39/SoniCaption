# SonicCaption — Parallel Work Split & Deliverables Execution Plan
**Speech Processing and Recognition (MAI522-5) | MSc AI & ML | Group of 3**  
**Autonomous, Non-Linear Execution Model Powered by Antigravity**

---

## 0. Executive summary & deliverables contract

This document defines the **parallel, non-linear work-breakdown structure** for the 3-member group. Traditional software projects stall because of sequential dependencies (e.g., frontend waiting for API endpoints, API waiting for machine learning models). Under this plan, all three team members work **simultaneously from Hour 1** using a **Contract-First Architecture** and independent **Antigravity agent workflows**.

### The 3 Required Submission Deliverables

```text
+----------------------------------------------------------------------------------------------------+
|                                    FINAL SUBMISSION DELIVERABLES                                   |
+------------------------------------+----------------------------------+----------------------------+
| 1. GitHub Repository Link          | 2. App Screenshots PDF           | 3. Five-Page Project Report|
|                                    |                                  |                            |
| - Complete source code (Monorepo)  | - High-res annotated screenshots | - Page 1: Intro & Arch     |
| - Frontend (React + TS + Vite)     | - Studio Workspace & Timeline    | - Page 2: U1 & U2 DSP Math |
| - Backend (FastAPI + DSP + ML)     | - Signal Lab (FFT, STFT, MFCC)   | - Page 3: Acoustic CNN     |
| - Pytest & Vitest test suites      | - Pattern Lab (DTW Matrix Trace) | - Page 4: U3 DTW & U4 HMM  |
| - Live Vercel & HF Space links     | - HMM Trellis Graph & Decoder    | - Page 5: Results & Viva   |
| - Clean Git PR & commit history    | - Model Card & Subtitle Exports  | - Individual contributions |
+------------------------------------+----------------------------------+----------------------------+
```

---

## 1. The non-linear simultaneous execution model

### 1.1 Why Traditional Workflow Fails vs How We Work in Parallel

```text
TRADITIONAL WATERFALL (BLOCKING & SLOW):
[Member 1: Audio Core] ---> [Member 2: Train Model & ASR] ---> [Member 3: Build UI] ---> [Merge Chaos]
     (Days 1-2)                      (Days 2-3)                      (Days 4-5)              (Late)

SONICCAPTION PARALLEL SPRINT (SIMULTANEOUS & ZERO-WAIT):
Day 0: Shared Data Contracts agreed (JSON / Pydantic schemas)
Day 1-4:
+---------------------------------------------------------------------------------------------------+
| MEMBER 1 (DSP Specialist)  --> Owns DSP & syllabus algorithms (U1, U2, DTW) + Pytest tests        |
| MEMBER 2 (ML Engineer)     --> Owns Colab training (ESC-50 CNN) + Hugging Face + HMM/Viterbi      |
| MEMBER 3 (Full-Stack Lead) --> Owns React Studio UI + Mock FastAPI endpoints + WaveSurfer player |
+---------------------------------------------------------------------------------------------------+
Day 4-5: One-click integration (Toggle USE_MOCKS=false), export verification, report compilation.
```

### 1.2 The Parallel Enabler: Contract-First Mock API
Member 3 does **not** wait for Member 1's signal-processing code or Member 2's trained CNN. On Day 0, the team freezes the JSON data schemas. Member 3 builds a lightweight mock router (`backend/api/mock_data.py`) returning pre-computed JSON responses. 
- **Member 3** immediately develops and tests the full React frontend, audio scrubber, canvas spectrogram, DTW heatmap, and Trellis visualizer using mock data.
- **Member 1** develops mathematical DSP algorithms in pure Python/NumPy, validated locally using synthetic waveforms and `pytest`.
- **Member 2** trains the CNN in Google Colab, hosts weights on Hugging Face, and builds the discrete HMM decoder in isolation.
- On Day 4, Member 3 switches `USE_MOCK_DATA = False` in `config.py` to seamlessly connect the live backend implementations.

---

## 2. Equal work distribution among 3 group members

Each member owns an equal, mathematically sound, and technically rigorous portion of the project, complete with independent code scope, report ownership, and live viva defense areas.

```text
========================================================================================================
MEMBER 1: DSP & ACOUSTIC FEATURES (Signal Processing Specialist)
Focus: Units 1, 2, and 3 (Signal fundamentals, spectral analysis, LPC, VQ, DTW algorithm)
========================================================================================================
- Codebase Scope:
  * backend/src/soniccaption/audio/io.py (validation, decoding, mono conversion, 16kHz resampling)
  * backend/src/soniccaption/audio/preprocessing.py (framing, Hamming windowing, pre-emphasis)
  * backend/src/soniccaption/algorithms/spectral.py (short-time energy, ZCR, FFT, STFT spectrogram)
  * backend/src/soniccaption/algorithms/lpc.py (Levinson-Durbin autocorrelation, all-pole vocal tract)
  * backend/src/soniccaption/algorithms/vq.py (K-means / LBG vector quantisation codebook generator)
  * backend/src/soniccaption/algorithms/dtw.py (Dynamic Time Warping distance recurrence & backtracking)
  * backend/tests/test_spectral.py, test_lpc.py, test_dtw.py
- Deliverable Ownership:
  * Project Report: Page 2 (DSP & Speech Analysis) + Page 4 (DTW Formulation & Path Trace).
  * Screenshots PDF: High-res captures of Signal Lab (Spectrogram, FFT, MFCC, LPC) and Pattern Lab.
- Viva Specialisation: Sampling theory, Mel filter-banks, LPC formant extraction, DTW constraints.

========================================================================================================
MEMBER 2: ACOUSTIC MODELING & RECOGNITION (ML & Pattern Recognition Specialist)
Focus: Units 3 & 4 (Environmental CNN, Hugging Face, Whisper ASR, VAD, Discrete HMM & Viterbi)
========================================================================================================
- Codebase Scope:
  * notebooks/01_esc50_exploration.ipynb, 02_train_sonic_cnn.ipynb, 03_evaluation.ipynb
  * Custom PyTorch CNN (4-layer Conv2D + BatchNorm + SpecAugment) trained on ESC-50
  * Model publishing: Checkpoint, `config.json`, `labels.json`, and Model Card on Hugging Face Hub
  * backend/src/soniccaption/services/asr_service.py (Groq Whisper API + local faster-whisper fallback)
  * backend/src/soniccaption/audio/vad.py (Silero VAD speech interval segmentation)
  * backend/src/soniccaption/algorithms/hmm.py (Discrete HMM, Forward likelihood, Viterbi dynamic trellis)
  * backend/tests/test_hmm.py, test_model_inference.py
- Deliverable Ownership:
  * Project Report: Page 3 (Acoustic Modeling & CNN) + Page 4 (Discrete HMM & Viterbi Decoding).
  * Screenshots PDF: Model Card metrics, Colab training curves, confusion matrix, and HMM Trellis.
- Viva Specialisation: SpecAugment, CNN architectures, Whisper architecture, HMM evaluation vs decoding.

========================================================================================================
MEMBER 3: FULL-STACK PRODUCT & INTEGRATION (Lead Product & Deployment Engineer)
Focus: System Architecture, React SPA, WaveSurfer Audio Engine, FastAPI Orchestrator, Deployment
========================================================================================================
- Codebase Scope:
  * frontend/ (Vite + React 18/19 + TypeScript + Tailwind CSS + Lucide Icons)
  * frontend/src/components/AudioPlayer.tsx (WaveSurfer.js waveform scrubbing, regions, playback)
  * frontend/src/components/TimelineView.tsx (Dual-track speech + acoustic event timeline, click-to-seek)
  * frontend/src/pages/ (Workspace, SignalLab, PatternLab, HMMTrellis, ModelCard)
  * backend/main.py, backend/api/ (FastAPI routing, mock endpoints, timeline fusion, export engine)
  * Deployment: Vercel (Frontend CDN) + Hugging Face Spaces / Render (Docker/FastAPI Backend)
  * GitHub CI: .github/workflows/ (Frontend Vitest & Backend Pytest automation)
- Deliverable Ownership:
  * Project Report: Page 1 (Introduction & System Architecture) + Page 5 (Evaluation, Viva, Conclusion).
  * Screenshots PDF: Master compilation of the PDF document; UI aesthetics, styling, and navigation.
- Viva Specialisation: Monorepo full-stack architecture, VAD/ASR/CNN timeline fusion, latency optimization.
========================================================================================================
```

---

## 3. The 3 core deliverables: production specification

### Deliverable 1: GitHub Repository (`github.com/<user>/SoniCaption`)
- **Repository Structure:** Decoupled monorepo with `frontend/`, `backend/`, `notebooks/`, and `data/samples/`.
- **Branching Policy:**
  - `main`: Protected production branch; only merged through Pull Requests.
  - `feature/dsp-algorithms` (Owned by Member 1)
  - `feature/ml-recognition` (Owned by Member 2)
  - `feature/fullstack-ui` (Owned by Member 3)
- **CI/CD Quality Gates:** GitHub Actions run linting, TypeScript build, and Pytest on all PRs before merge.
- **Documentation:** Root `README.md` containing single-command setup instructions, architecture diagram, deployed URLs, and video walkthrough.

---

### Deliverable 2: App Screenshots PDF (`SonicCaption_App_Screenshots.pdf`)
A clean, 6–8 page visual walkthrough demonstrating a complete working product:

| Page | Screen / Feature | Key Visual Elements |
|---|---|---|
| **Cover** | Title & Architecture | Project Title, Authors, Group Members, High-Level Architecture Diagram |
| **Page 1** | **Studio Workspace** | Audio dropzone, WaveSurfer waveform player, playback controls, metadata drawer |
| **Page 2** | **Dual-Track Timeline** | Synchronized speech subtitles and acoustic event chips (`[Applause]`, `[Door Knock]`) with click-to-seek |
| **Page 3** | **Signal Lab (U1 & U2)** | Time-aligned STFT Spectrogram, FFT Spectrum, RMS Energy, and ZCR plots |
| **Page 4** | **Feature Lab (U2)** | 13-coefficient MFCC Heatmap with deltas, LPC Formant spectral overlay, VQ cluster scatter |
| **Page 5** | **Pattern Lab (U3)** | Dual audio waveform alignment, 2D DTW accumulated cost matrix heatmap, red optimal path |
| **Page 6** | **HMM Trellis (U4)** | Interactive state transition graph, editable probability matrices, dynamic Viterbi Trellis |
| **Page 7** | **Model & Exports** | ESC-50 CNN confusion matrix, Hugging Face model card, Subtitle Export modal (`.srt`, `.vtt`, `.json`) |

*How to generate:* Member 3 captures full-resolution browser screenshots, inserts them into a styled Markdown/HTML template, and compiles via Chrome/Puppeteer to PDF.

---

### Deliverable 3: 5-Page Academic Project Report (`SonicCaption_Final_Report.pdf`)
A dense, rigorous IEEE/ACM-style 5-page conference format report structured as follows:

```text
+----------------------------------------------------------------------------------------------------+
|                               FIVE-PAGE REPORT DETAILED BLUEPRINT                                  |
+----------------------------------------------------------------------------------------------------+
| PAGE 1: TITLE, ABSTRACT, INTRODUCTION & SYSTEM ARCHITECTURE (Owned by Member 3)                   |
| - Abstract: Motivation for ambient-aware captioning, methodology, and key quantitative findings.   |
| - 1. Introduction: Limitations of speech-only ASR; syllabus alignment with Units 1, 2, 3, and 4.   |
| - 2. System Architecture: Decoupled React + FastAPI design, data contracts, $0.00 infrastructure.   |
| - Figure 1: End-to-end system architectural flow diagram.                                          |
| - Table 1: Tech stack, computational complexity, and free-tier service allocations.               |
+----------------------------------------------------------------------------------------------------+
| PAGE 2: SPEECH SIGNAL PROCESSING & ANALYSIS PIPELINE (Owned by Member 1)                          |
| - 3. Audio Preprocessing: 16 kHz mono resampling, framing, Hamming windowing, pre-emphasis filter.  |
| - 4. Time & Spectral Analysis: Mathematical derivations of Short-Time Energy, ZCR, FFT, and STFT.  |
| - 5. Acoustic Feature Representation: Mel filter-bank design, 13-MFCC extraction with DCT-II.     |
| - 6. Vocal Tract Modeling (LPC) & Vector Quantisation: Levinson-Durbin recursion and LBG/K-means.  |
| - Figure 2: STFT Spectrogram with harmonic formants vs instantaneous Zero-Crossing Rate.           |
| - Figure 3: LPC spectral envelope fitted against discrete FFT peaks.                               |
+----------------------------------------------------------------------------------------------------+
| PAGE 3: ACOUSTIC EVENT MODELING & RECOGNITION (Owned by Member 2)                                  |
| - 7. Environmental Audio Classification: ESC-50 dataset characteristics and 10-class subset.       |
| - 8. Neural Network Architecture: 4-layer 2D-CNN with BatchNorm, Dropout, and SpecAugment.         |
| - 9. Training Methodology: 5-fold cross-validation protocol, Adam optimizer, cross-entropy loss.  |
| - 10. Foundation ASR & VAD Integration: Groq Whisper Large v3 turbo and Silero VAD integration.    |
| - Figure 4: PyTorch CNN training/validation loss curves over 30 epochs.                            |
| - Figure 5: Normalized 10-class confusion matrix on the held-out test fold.                       |
+----------------------------------------------------------------------------------------------------+
| PAGE 4: PATTERN MATCHING & SYLLABUS ALGORITHMS (Jointly Owned: Member 1 & Member 2)                |
| - 11. Dynamic Time Warping (DTW) (Member 1): Formulation of local Euclidean/Cosine distortion,     |
|   accumulated cost matrix recursion, monotonicity, continuity constraints, and optimal trace.     |
| - 12. Discrete Hidden Markov Models (HMM) (Member 2): 5-tuple parameterization (pi, A, B, S, V).   |
| - 13. Forward Evaluation & Viterbi Decoding: Forward variable alpha_t(i) and Viterbi dynamic       |
|   programming trellis recursion with state backpointers.                                           |
| - Figure 6: DTW 2D accumulated cost matrix showing non-linear alignment of two spoken words.      |
| - Figure 7: Dynamic Viterbi Trellis diagram showing optimal state path decoding.                   |
+----------------------------------------------------------------------------------------------------+
| PAGE 5: PERFORMANCE EVALUATION, VIVA DEFENSE & CONCLUSION (Owned by Member 3 & All)                |
| - 14. Quantitative Results: ASR Word Error Rate (WER/CER) and Sound CNN Macro-F1 / Accuracy.       |
| - 15. Latency & System Benchmarks: End-to-end wall-clock timings (<1.5s per 10s audio clip).       |
| - 16. Accessibility Impact & Limitations: Practical utility for hearing-impaired users; challenges |
|   in overlapping acoustic events and background noise.                                             |
| - 17. Individual Member Contribution Breakdown: Auditable table of Git commits, modules, & roles.  |
| - 18. References: Official course syllabus, ESC-50 paper, Whisper paper, and Rabiner's HMM guide.  |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. Antigravity prompt instructions for each member

Each team member can open Antigravity in their workspace, checkout their assigned branch, and paste their dedicated execution prompt to proceed autonomously:

### Antigravity Prompt for Member 1 (Branch: `feature/dsp-algorithms`)
```text
You are Member 1 (DSP & Speech Signal Processing Specialist) on the SonicCaption team.
Your objective is to implement and unit-test all audio preprocessing and Units 1, 2, and 3 algorithms.

Your assigned scope:
1. backend/src/soniccaption/audio/io.py: Robust loading (WAV, MP3, FLAC), mono conversion, 16kHz resampling.
2. backend/src/soniccaption/audio/preprocessing.py: Short-time framing (25ms window, 10ms hop), Hamming window.
3. backend/src/soniccaption/algorithms/spectral.py: Short-time energy, ZCR, FFT magnitude, STFT spectrogram.
4. backend/src/soniccaption/algorithms/lpc.py: Autocorrelation, Levinson-Durbin recursion for LPC coefficients.
5. backend/src/soniccaption/algorithms/vq.py: K-means vector quantisation codebook generator and quantization error.
6. backend/src/soniccaption/algorithms/dtw.py: Dynamic Time Warping with Euclidean/Cosine metric, cost matrix, backtracking path.
7. Comprehensive Pytest tests under backend/tests/ validating mathematical correctness and handling silence/zeros.

Follow the data contracts in SonicCaption_Updated_Complete_Implementation_Plan.md. Do not edit UI or ML files.
When done, run pytest and report the exact passing test output.
```

### Antigravity Prompt for Member 2 (Branch: `feature/ml-recognition`)
```text
You are Member 2 (ML & Pattern Recognition Specialist) on the SonicCaption team.
Your objective is to train the ESC-50 acoustic classifier, build the ASR/VAD service, and implement Unit 4 HMM.

Your assigned scope:
1. notebooks/02_train_sonic_cnn.ipynb: Google Colab notebook downloading ESC-50, filtering 10 classes, extracting log-Mel spectrograms, applying SpecAugment, training a 4-layer CNN with PyTorch, evaluating 5-fold cross-validation, and saving model.pt + metrics.
2. backend/src/soniccaption/services/sound_cnn.py: Model loader fetching weights from Hugging Face Hub, sliding window inference.
3. backend/src/soniccaption/services/asr_service.py: Groq Whisper API provider with faster-whisper local CPU fallback.
4. backend/src/soniccaption/audio/vad.py: Silero VAD speech interval detection.
5. backend/src/soniccaption/algorithms/hmm.py: Discrete HMM (pi, A, B), Forward likelihood evaluation, and Viterbi decoding with backpointers.
6. Pytest test cases under backend/tests/ for HMM trellis correctness and CNN inference smoke test.

Follow the data contracts in SonicCaption_Updated_Complete_Implementation_Plan.md. Do not edit UI or DSP files.
When done, report the Colab metrics (Accuracy, F1) and Pytest results.
```

### Antigravity Prompt for Member 3 (Branch: `feature/fullstack-ui`)
```text
You are Member 3 (Full-Stack Product & Integration Lead) on the SonicCaption team.
Your objective is to build the commercial-grade React SPA, FastAPI orchestrator with mock endpoints, and deployment setup.

Your assigned scope:
1. frontend/: Vite + React 18/19 + TypeScript + Tailwind CSS + Lucide Icons.
   - components/AudioPlayer.tsx: WaveSurfer.js waveform visualizer, zoom, timecode, region markers.
   - components/TimelineView.tsx: Dual-track synchronized timeline (Speech + Sound events) with click-to-seek.
   - pages/WorkspacePage.tsx, SignalLabPage.tsx (Canvas spectrogram & FFT), PatternLabPage.tsx (DTW heatmap), HMMPage.tsx (Interactive Trellis visualizer), ModelCardPage.tsx.
2. backend/api/: FastAPI routes (/api/v1/audio/analyze, /labs/signal, /labs/dtw, /labs/hmm/viterbi, /export/{format}).
   - Provide high-fidelity mock fallback data so the entire frontend runs and animates immediately!
   - Timeline fusion logic merging speech segments and acoustic events chronologically.
   - Subtitle exporters (.srt, .vtt, .json, .csv, .txt).
3. backend/Dockerfile and deployment configurations for Vercel and Hugging Face Spaces.
4. Compile the App Screenshots PDF script and layout.

Follow the data contracts in SonicCaption_Updated_Complete_Implementation_Plan.md. Do not edit algorithm internals.
When done, verify frontend with `npm run build` and backend with `pytest`.
```

---

## 5. Contract-first mock data payloads (The zero-dependency bridge)

To guarantee that Member 3 can build the entire UI without waiting for Members 1 and 2, the FastAPI backend provides mock responses under `backend/api/mock_data.py`:

### Mock Analysis Payload (`POST /api/v1/audio/analyze`)
```json
{
  "filename": "speech_with_applause.wav",
  "duration": 12.5,
  "sample_rate": 16000,
  "speech_segments": [
    {
      "id": 1,
      "start": 0.5,
      "end": 4.2,
      "text": "Good morning everyone, welcome to the SonicCaption product launch.",
      "words": [
        {"word": "Good", "start": 0.5, "end": 0.8},
        {"word": "morning", "start": 0.9, "end": 1.4},
        {"word": "everyone", "start": 1.5, "end": 2.1}
      ]
    },
    {
      "id": 2,
      "start": 6.8,
      "end": 11.9,
      "text": "Today we demonstrate how multi-modal acoustic captioning enhances accessibility.",
      "words": []
    }
  ],
  "sound_events": [
    {
      "start": 4.3,
      "end": 6.7,
      "label": "clapping",
      "confidence": 0.94,
      "model_source": "soniccaption-esc50-cnn"
    }
  ],
  "timeline": [
    {
      "id": "t1",
      "start": 0.5,
      "end": 4.2,
      "category": "speech",
      "label": "Speech",
      "display_text": "Good morning everyone, welcome to the SonicCaption product launch.",
      "confidence": null
    },
    {
      "id": "t2",
      "start": 4.3,
      "end": 6.7,
      "category": "sound_event",
      "label": "Clapping / Applause",
      "display_text": "[Applause]",
      "confidence": 0.94
    },
    {
      "id": "t3",
      "start": 6.8,
      "end": 11.9,
      "category": "speech",
      "label": "Speech",
      "display_text": "Today we demonstrate how multi-modal acoustic captioning enhances accessibility.",
      "confidence": null
    }
  ],
  "metrics": {
    "vad_speech_ratio": 0.69,
    "transcription_time_ms": 520,
    "cnn_inference_time_ms": 75
  }
}
```

### Mock DTW Alignment Payload (`POST /api/v1/labs/dtw`)
```json
{
  "distance": 14.82,
  "normalized_distance": 0.185,
  "path_length": 80,
  "path": [
    [0, 0], [1, 1], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]
  ],
  "cost_matrix_sample": [
    [0.1, 1.2, 2.5],
    [1.4, 0.3, 1.8],
    [2.8, 1.5, 0.4]
  ]
}
```

### Mock HMM Trellis Payload (`POST /api/v1/labs/hmm/viterbi`)
```json
{
  "log_likelihood": -14.285,
  "best_path": ["Sunny", "Sunny", "Rainy", "Rainy", "Sunny"],
  "state_probabilities": [
    {"step": 0, "state": "Sunny", "prob": 0.6},
    {"step": 0, "state": "Rainy", "prob": 0.4},
    {"step": 1, "state": "Sunny", "prob": 0.72},
    {"step": 1, "state": "Rainy", "prob": 0.28}
  ],
  "trellis": [
    {"from": "Sunny", "to": "Sunny", "prob": 0.7, "chosen": true},
    {"from": "Sunny", "to": "Rainy", "prob": 0.3, "chosen": false}
  ]
}
```

---

## 6. Day-by-day synchronized parallel schedule

| Day & Phase | Member 1 (DSP Specialist) | Member 2 (ML Specialist) | Member 3 (Product & Full-Stack) |
|---|---|---|---|
| **Day 0: Kickoff (4 Hours)** | Review DSP syllabus formulas; agree on sample rate (16 kHz) & FFT params. | Create Colab notebook; mount Drive; download ESC-50; filter 10 classes. | Initialize Git repo, monorepo folders, Vite React app, and FastAPI mock server. |
| **Day 1: Independent Build** | Implement `io.py`, `preprocessing.py`, short-time energy, ZCR, FFT, STFT. | Build PyTorch ESC-50 dataset loader, log-Mel extractor, and CNN architecture. | Build React navbar, dark theme, WaveSurfer player, and audio file dropzone. |
| **Day 2: Features & Models** | Implement filter-banks, MFCC, Levinson-Durbin LPC, and VQ K-means codebooks. | Run Colab 5-fold training; evaluate macro-F1; upload weights to Hugging Face Hub. | Build dual-track timeline component, seek-to-timestamp sync, and Signal Lab UI. |
| **Day 3: Syllabus Algorithms** | Implement custom Dynamic Time Warping (DTW) distance and backtracking path. | Implement Discrete HMM, Forward evaluation, and Viterbi dynamic programming trellis. | Build Pattern Lab (DTW heatmap visualizer) and HMM Sandbox (Trellis graph). |
| **Day 4: Integration & Exporters** | Connect real DSP modules to FastAPI `/labs/signal` and `/labs/dtw`. PR merge. | Connect real CNN loader & HMM to `/api/v1/audio/analyze` and `/labs/hmm`. PR merge. | Switch `USE_MOCK_DATA=false`; verify live end-to-end flow; implement `.srt`/`.vtt` exports. |
| **Day 5: Deliverables Polish** | Write Report Page 2 & 4 (DSP/DTW math); verify Pytest test suite (>85% coverage). | Write Report Page 3 & 4 (ML/HMM math); export confusion matrix & loss curves. | Deploy frontend to Vercel and backend to HF Spaces; compile Screenshots PDF; write Pages 1 & 5. |

---

## 7. Submission verification checklist

Before submission, all three members review and check off all items:

- [ ] **Deliverable 1: GitHub Repository**
  - [ ] Monorepo contains functional `frontend/` and `backend/`.
  - [ ] GitHub Actions CI badge is passing green.
  - [ ] Clean commit history showcasing balanced contributions from all 3 members.
  - [ ] `README.md` includes live Vercel and Hugging Face deployment URLs.
- [ ] **Deliverable 2: App Screenshots PDF**
  - [ ] High-resolution screenshots of all 6 primary views (Workspace, Timeline, Signal Lab, Pattern Lab, HMM Trellis, Model Card).
  - [ ] Captions and visual annotations explaining features and syllabus alignment.
- [ ] **Deliverable 3: Five-Page Academic Report**
  - [ ] Conforms strictly to 5-page standard (IEEE/ACM two-column or clean university format).
  - [ ] Full coverage of Units 1, 2, 3, and 4 with mathematical rigor.
  - [ ] Real empirical metrics reported (WER for speech, macro-F1 & confusion matrix for sound events).
  - [ ] Detailed contribution table explicitly documenting equal workload distribution.
- [ ] **Viva Readiness**
  - [ ] All 3 members can run and demonstrate the live system.
  - [ ] All 3 members can answer theoretical DSP, ML, and HMM questions across all four units.
