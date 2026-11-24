# SonicCaption — Parallel Product Work Split & Deliverables Execution Plan
**Intelligent Audio Captioning System | MSc AI & Machine Learning | Group of 3**  
**Autonomous, Non-Linear Execution Model Powered by Antigravity**

---

## 0. Executive Summary & Architectural Philosophy

This document defines the **product-centric, non-linear work-breakdown structure** for the 3-member SonicCaption group.

### Core Architectural Principle: Notebook-First Logic & Lean Service Adapters
In a real-world AI audio product, production `.py` files should not obscure core research math; rather:
1. **Interactive Notebooks (`notebooks/`) Hold the Core Algorithmic & Scientific Logic:**  
   All signal processing equations, psychoacoustic transforms, neural network architectures, training loops, hyperparameter tuning, metric evaluations, and rich visual interpretations live directly inside reproducible Jupyter Notebooks (`01_*`, `02_*`, `03_*`).
2. **Backend Modules (`backend/src/`) Act as Clean Service Adapters:**  
   The Python files under `backend/` are lightweight, modular application wrappers and FastAPI endpoints that package the notebook-proven algorithms into production services for the React web application.
3. **Contract-First Independence:**  
   All 3 team members work **simultaneously from Hour 1** using frozen JSON data contracts and mock APIs, eliminating sequential blocking.

---

### The 3 Required Submission Deliverables

```text
+----------------------------------------------------------------------------------------------------+
|                                    FINAL SUBMISSION DELIVERABLES                                   |
+------------------------------------+----------------------------------+----------------------------+
| 1. GitHub Repository Link          | 2. App Screenshots PDF           | 3. Five-Page Product Report|
|                                    |                                  |                            |
| - Complete source code (Monorepo)  | - High-res annotated screenshots | - Page 1: Product & Arch   |
| - Research Notebooks (01, 02, 03)  | - Studio Workspace & Timeline    | - Page 2: DSP Pipeline     |
| - Frontend (React + TS + Vite)     | - Signal Lab (FFT, STFT, MFCC)   | - Page 3: Acoustic ResNet  |
| - Backend (FastAPI + Services)     | - Pattern Lab (DTW Matrix Trace) | - Page 4: Sequence Fusion  |
| - Pytest test suite (17/17 passed) | - HMM Trellis Graph & Decoder    | - Page 5: Benchmarks & Viva|
| - Live Vercel & HF Space links     | - Subtitle & Timeline Exports    | - Contribution breakdown   |
+------------------------------------+----------------------------------+----------------------------+
```

---

## 1. The Non-Linear Simultaneous Execution Model

```text
TRADITIONAL WATERFALL (BLOCKING & SLOW):
[Audio DSP] ---> [Train Model & ASR] ---> [Build UI] ---> [Merge Chaos]
 (Days 1-2)          (Days 2-3)            (Days 4-5)       (Late)

SONICCAPTION PRODUCT SPRINT (SIMULTANEOUS & ZERO-WAIT):
Day 0: Shared Data Contracts frozen (Pydantic schemas in core/schemas.py)
Day 1-4:
+---------------------------------------------------------------------------------------------------+
| MEMBER 1 (Audio DSP & ML Lead)   --> Notebooks 01 & 02 + DSP algorithms + Pytest tests           |
| MEMBER 2 (Speech & Fusion Lead)  --> Notebook 03 + Whisper ASR + VAD + HMM Viterbi + Fusion       |
| MEMBER 3 (Full-Stack Lead)       --> React Studio UI + Mock FastAPI endpoints + WaveSurfer player |
+---------------------------------------------------------------------------------------------------+
Day 4-5: One-click integration (Toggle USE_MOCK_DATA=False), export verification, report compilation.
```

### The Contract-First Mock API (The Zero-Dependency Bridge)
Member 3 does **not** wait for Member 1's audio processing code or Member 2's trained model. On Day 0, the team freezes the JSON data schemas. Member 3 builds a lightweight mock router (`backend/api/mock_data.py`) returning pre-computed JSON responses:
* **Member 3** immediately develops and tests the full React frontend, audio scrubber, canvas spectrogram, DTW heatmap, and Trellis visualizer using mock data.
* **Member 1** develops the mathematical DSP pipeline and acoustic neural classifier in pure Python/NumPy and PyTorch, validated in Notebooks 01 and 02 with `pytest`.
* **Member 2** develops the speech-to-text integration, VAD segmenter, sequence modeling (HMM/Viterbi), and chronological fusion in Notebook 03 and backend services.
* On Day 4, Member 3 switches `USE_MOCK_DATA = False` in `config.py` to seamlessly connect the live backend implementations.

---

## 2. Product-Centric Work Distribution Among 3 Members

Each member owns an equal, technically rigorous pillar of the product, with distinct notebook ownership, code scope, report pages, and live viva defense areas:

```text
========================================================================================================
MEMBER 1: AUDIO SIGNAL PROCESSING & ACOUSTIC INTELLIGENCE LEAD (Member 1 — YOU)
Focus: Audio Ingestion, Spectral Transformations, Psychoacoustic Features & Environmental Classifier
========================================================================================================
- Core Research & Engineering Notebooks (Where the Real Logic Lives):
  * notebooks/01_audio_feature_extraction_pipeline.ipynb
    - Complete from-scratch DSP pipeline: pre-emphasis filter (alpha=0.97), short-time framing,
      Hamming windowing, FFT spectrum, STFT power spectrogram, 40-channel Mel filter-banks,
      13 static MFCCs + Delta + Delta-Delta dynamic features.
    - Levinson-Durbin autocorrelation recursion for all-pole LPC coefficients & formant tracking.
    - K-Means and Linde-Buzo-Gray (LBG) Vector Quantisation codebooks.
    - Dynamic Time Warping (DTW) with local continuity constraints and optimal backtracking warp path.
    - Contains 7 high-DPI engineering figures with mathematical interpretations.
  * notebooks/02_environmental_sound_classifier_training.ipynb
    - ESC-50 dataset ingestion & balanced 10-class curation.
    - Mathematical bridge verifying that custom DSP features match GPU-accelerated TorchAudio tensors.
    - SpecAugment (double time/frequency masking) + random gain regularisation.
    - Pretrained ResNet-18 fine-tuning adapted for 1-channel log-Mel spectrograms.
    - Achieves 93.06% peak validation accuracy and 0.837 macro F1 in 25 epochs.
    - Full export of model weights (soniccaption_esc50_resnet.pt), config.json, labels.json, and figures.
- Backend Service Adapter Modules:
  * backend/src/soniccaption/audio/io.py (validation, decoding, mono conversion, 16kHz resampling)
  * backend/src/soniccaption/audio/preprocessing.py (framing, Hamming windowing, pre-emphasis)
  * backend/src/soniccaption/algorithms/spectral.py (energy, ZCR, FFT, STFT, Mel, MFCCs)
  * backend/src/soniccaption/algorithms/lpc.py (Levinson-Durbin, PARCOR, formant root-solving)
  * backend/src/soniccaption/algorithms/vq.py (K-Means & LBG codebook generators)
  * backend/src/soniccaption/algorithms/dtw.py (Dynamic Time Warping dynamic programming)
  * backend/tests/ (Full automated test suite — 17/17 passing unit tests)
- Deliverables Ownership:
  * Notebooks 01 and 02 (fully executed with outputs and all 11 figures in notebooks/screenshots/).
  * Project Report: Page 2 (DSP Feature Extraction) + Page 3 (Acoustic Event Model Training).
- Viva Specialisation: Sampling theory, Mel filter-banks, LPC vocal tract physics, SpecAugment, ResNet-18.

========================================================================================================
MEMBER 2: SPEECH RECOGNITION, SEQUENCE MODELING & TIMELINE FUSION LEAD (Member 2)
Focus: Speech-to-Text Transcription, Voice Activity Detection, HMM Sequence Decoding & Timeline Fusion
========================================================================================================
- Core Research & Engineering Notebook (Where the Real Logic Lives):
  * notebooks/03_speech_recognition_and_sequence_fusion.ipynb
    - Voice Activity Detection (VAD) audio interval segmentation (speech vs ambient gap detection).
    - Speech-to-Text evaluation with Groq Whisper API (Word Error Rate / Character Error Rate).
    - Discrete Hidden Markov Model (HMM) formulation: transition matrix A, emission matrix B, initial pi.
    - Forward probability evaluation and Viterbi dynamic programming trellis decoding with backpointers.
    - Dual-stream timeline fusion algorithm: chronological interleaving of speech segments and acoustic tags.
- Backend Service Adapter Modules:
  * backend/src/soniccaption/audio/vad.py (Silero VAD speech interval segmentation)
  * backend/src/soniccaption/services/asr_service.py (Groq Whisper API + faster-whisper local fallback)
  * backend/src/soniccaption/services/sound_cnn.py (Sliding-window inference using Member 1's ResNet weights)
  * backend/src/soniccaption/algorithms/hmm.py (Discrete HMM, Forward evaluation, Viterbi dynamic trellis)
  * backend/src/soniccaption/timeline/fusion.py (Chronological speech + acoustic event alignment engine)
  * backend/tests/test_hmm.py, test_model_inference.py, test_fusion.py
- Deliverables Ownership:
  * Notebook 03 (fully executed with Viterbi Trellis and timeline fusion outputs).
  * Project Report: Page 4 (Sequence Modeling, Dynamic Alignment & Dual-Stream Timeline Fusion).
- Viva Specialisation: Whisper transformer encoder-decoder, Silero VAD energy thresholds, HMM Viterbi decoding.

========================================================================================================
MEMBER 3: FULL-STACK PRODUCT, UI/UX & DEPLOYMENT LEAD (Member 3)
Focus: Commercial Web Interface, Audio Visualization Studio, Interactive Labs, Exporters & Deployment
========================================================================================================
- Frontend Product Scope:
  * frontend/ (Vite + React 18/19 + TypeScript + Tailwind CSS + Lucide Icons)
  * frontend/src/components/AudioPlayer.tsx (WaveSurfer.js waveform visualizer, zoom, timecode, scrub)
  * frontend/src/components/TimelineView.tsx (Dual-track speech + acoustic event timeline, click-to-seek)
  * frontend/src/pages/WorkspacePage.tsx (Audio upload zone, analyze runner, synchronized subtitle preview)
  * frontend/src/pages/SignalLabPage.tsx (Interactive STFT spectrogram & FFT spectrum canvas)
  * frontend/src/pages/PatternLabPage.tsx (Interactive DTW cost matrix heatmap & warp path trace)
  * frontend/src/pages/HMMPage.tsx (Interactive HMM Viterbi Trellis sandbox)
  * frontend/src/pages/ModelCardPage.tsx (ESC-50 ResNet performance dashboard & confusion matrix)
- Backend Application & Deployment Scope:
  * backend/main.py, backend/api/ (FastAPI routing, CORS, mock endpoints, execution orchestrator)
  * backend/src/soniccaption/timeline/exporters.py (Multi-format subtitle exports: .srt, .vtt, .json, .csv)
  * backend/Dockerfile & cloud deployment configs (Vercel Frontend + HF Spaces / Render Backend)
  * GitHub CI: .github/workflows/ (Automated test runner)
- Deliverables Ownership:
  * App Screenshots PDF (Annotated 6–8 page master presentation of the live running product).
  * Project Report: Page 1 (Product Overview & Architecture) + Page 5 (Evaluation, Benchmarks & Viva).
- Viva Specialisation: Monorepo full-stack architecture, WaveSurfer WebAudio synchronization, API latency.
========================================================================================================
```

---

## 3. Five-Page Academic Report: Product Allocation

A dense, rigorous IEEE/ACM-style 5-page conference report structured around product engineering:

```text
+----------------------------------------------------------------------------------------------------+
|                               FIVE-PAGE REPORT DETAILED BLUEPRINT                                  |
+----------------------------------------------------------------------------------------------------+
| PAGE 1: TITLE, ABSTRACT, INTRODUCTION & SYSTEM ARCHITECTURE (Owned by Member 3)                   |
| - Abstract: Motivation for ambient-aware captioning, real-world accessibility impact, key findings. |
| - 1. Introduction: Limitations of speech-only ASR; commercial need for non-speech acoustic tags.   |
| - 2. System Architecture: Decoupled React + FastAPI design, data contracts, $0.00 infrastructure.   |
| - Figure 1: End-to-end system architectural flow diagram.                                          |
| - Table 1: Tech stack, computational complexity, and free-tier service allocations.               |
+----------------------------------------------------------------------------------------------------+
| PAGE 2: AUDIO PREPROCESSING & SPECTRAL FEATURE EXTRACTION (Owned by Member 1)                      |
| - 3. Audio Preprocessing: 16 kHz mono polyphase resampling, framing, Hamming windowing, pre-emphasis.|
| - 4. Time & Spectral Analysis: Mathematical derivations of Short-Time Energy, ZCR, FFT, and STFT.  |
| - 5. Acoustic Feature Representation: 40-channel Mel filter-bank design, 13-MFCC extraction (DCT). |
| - 6. Vocal Tract Modeling (LPC) & Vector Quantisation: Levinson-Durbin recursion and LBG/K-means.  |
| - Figure 2: STFT Spectrogram with harmonic formants vs instantaneous Zero-Crossing Rate.           |
| - Figure 3: LPC spectral envelope fitted against discrete FFT peaks.                               |
+----------------------------------------------------------------------------------------------------+
| PAGE 3: DEEP ACOUSTIC EVENT CLASSIFICATION & TRAINING (Owned by Member 1)                          |
| - 7. Environmental Audio Classification: ESC-50 dataset characteristics and balanced curation.     |
| - 8. Mathematical Feature Bridge: Proving custom DSP equivalence to GPU TorchAudio transforms.      |
| - 9. Neural Network Architecture: Pretrained ResNet-18 adapted for 1-channel log-Mel spectrograms. |
| - 10. Training Methodology: SpecAugment regularisation, Cosine Annealing, 93.06% accuracy, 0.837 F1|
| - Figure 4: PyTorch ResNet training & validation loss/accuracy curves over 25 epochs.              |
| - Figure 5: Normalized 10-class confusion matrix on the held-out test fold.                       |
+----------------------------------------------------------------------------------------------------+
| PAGE 4: SEQUENCE MODELING, ALIGNMENT & DUAL-STREAM FUSION (Owned by Member 2)                      |
| - 11. Speech Activity Segmentation: Silero VAD energy thresholds for speech vs ambient intervals.  |
| - 12. Foundation Speech Recognition: Cloud Whisper API integration and transcription formatting.   |
| - 13. Dynamic Time Warping (DTW): Pairwise cost matrix, continuity constraints, optimal warp trace.|
| - 14. Discrete HMM & Viterbi Decoding: State transitions, emission matrix, dynamic programming.    |
| - 15. Dual-Stream Timeline Fusion: Chronological alignment of speech and acoustic event tags.      |
| - Figure 6: DTW 2D accumulated cost matrix showing non-linear alignment of two spoken words.      |
| - Figure 7: Dynamic Viterbi Trellis diagram and fused subtitle timeline output.                    |
+----------------------------------------------------------------------------------------------------+
| PAGE 5: PRODUCT PERFORMANCE, VIVA DEFENSE & CONCLUSION (Owned by Member 3 & All)                  |
| - 16. Quantitative Results: Whisper ASR WER/CER and ResNet-18 Sound Classifier per-class metrics.  |
| - 17. Latency & System Benchmarks: End-to-end wall-clock timings (<1.5s for 10s audio clip).       |
| - 18. Accessibility Impact & Subtitle Conformance: FCC Closed Captioning compliance and .srt export|
| - 19. Individual Member Contribution Breakdown: Auditable table of Git commits, modules, & roles.  |
| - 20. References: Academic citations (Rabiner, Piczak, Radford/Whisper, He/ResNet).                |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. Antigravity Prompt Instructions for Each Member

### Antigravity Prompt for Member 1 (Branch: `feature/dsp-algorithms`)
```text
You are Member 1 (Audio Signal Processing & Acoustic Intelligence Lead) on the SonicCaption team.
Your objective is to own the foundational audio research, from-scratch DSP transforms, and the acoustic neural classifier.

Your assigned scope:
1. notebooks/01_audio_feature_extraction_pipeline.ipynb: Complete from-scratch DSP pipeline (resampling, framing, Hamming window, FFT, STFT, 40-channel Mel filterbanks, 13 MFCCs + Deltas, Levinson-Durbin LPC formants, VQ, DTW).
2. notebooks/02_environmental_sound_classifier_training.ipynb: ESC-50 dataset ingestion, DSP feature bridge, SpecAugment, ResNet-18 transfer learning achieving >90% validation accuracy, model weights export.
3. backend/src/soniccaption/audio/ and algorithms/: Modular service implementations of all audio DSP transforms.
4. backend/tests/: Pytest test suite validating numerical stability, silence handling, and algorithm accuracy.

Follow the data contracts in core/schemas.py. Do not edit UI or ASR service files.
Run pytest and ensure 100% of test assertions pass.
```

### Antigravity Prompt for Member 2 (Branch: `feature/ml-recognition`)
```text
You are Member 2 (Speech Recognition, Sequence Modeling & Timeline Fusion Lead) on the SonicCaption team.
Your objective is to build the speech-to-text pipeline, sequence alignment algorithms, and timeline fusion engine.

Your assigned scope:
1. notebooks/03_speech_recognition_and_sequence_fusion.ipynb: VAD speech segmentation, Whisper ASR evaluation, Discrete HMM formulation, Viterbi dynamic trellis decoding, and timeline fusion algorithm.
2. backend/src/soniccaption/audio/vad.py: Silero VAD speech interval detector.
3. backend/src/soniccaption/services/asr_service.py: Groq Whisper API provider with faster-whisper local fallback.
4. backend/src/soniccaption/services/sound_cnn.py: Sliding-window inference service loading Member 1's ResNet weights.
5. backend/src/soniccaption/algorithms/hmm.py: Discrete HMM (pi, A, B), Forward likelihood, and Viterbi dynamic programming decoder.
6. backend/src/soniccaption/timeline/fusion.py: Chronological merger of Whisper speech segments and ResNet acoustic tags.
7. Pytest test cases under backend/tests/ validating HMM Viterbi and timeline fusion.

Follow the data contracts in core/schemas.py. Do not edit UI or DSP internals.
```

### Antigravity Prompt for Member 3 (Branch: `feature/fullstack-ui`)
```text
You are Member 3 (Full-Stack Product, UI/UX & Deployment Lead) on the SonicCaption team.
Your objective is to build the commercial-grade React web application, interactive visualizer labs, and deployment setup.

Your assigned scope:
1. frontend/: Vite + React 18/19 + TypeScript + Tailwind CSS + Lucide Icons.
   - components/AudioPlayer.tsx: WaveSurfer.js waveform visualizer, zoom, timecode, region markers.
   - components/TimelineView.tsx: Dual-track synchronized timeline (Speech + Sound events) with click-to-seek.
   - pages/WorkspacePage.tsx, SignalLabPage.tsx (Canvas spectrogram & FFT), PatternLabPage.tsx (DTW heatmap), HMMPage.tsx (Interactive Trellis visualizer), ModelCardPage.tsx.
2. backend/main.py & backend/api/: FastAPI routes (/api/v1/audio/analyze, /labs/signal, /labs/dtw, /labs/hmm/viterbi, /export/{format}).
   - Provide high-fidelity mock fallback data so the entire frontend runs and animates immediately!
   - Subtitle exporters (.srt, .vtt, .json, .csv, .txt).
3. backend/Dockerfile and deployment configurations for Vercel and Hugging Face Spaces / Render.
4. Compile the App Screenshots PDF script and layout.

Follow the data contracts in core/schemas.py. Do not edit algorithm internals.
Verify frontend with `npm run build` and backend with `pytest`.
```

---

## 5. Day-by-Day Synchronized Parallel Schedule

| Day & Phase | Member 1 (Audio DSP & ML Lead) | Member 2 (Speech & Fusion Lead) | Member 3 (Full-Stack Product Lead) |
|---|---|---|---|
| **Day 0: Contract Freeze** | Freeze `AudioData`, `SpectralFeatures`, `LPCResult`, `DTWResult`, and `TimelineItem` schemas. | Agree on Whisper transcript format & event tag schema `[Event]`. | Create `backend/api/mock_data.py` with mock payloads for all endpoints. |
| **Day 1: Independent Core** | Implement `audio/io.py`, `preprocessing.py`, `spectral.py`. Run Notebook 01 in Colab. | Implement `audio/vad.py` and `services/asr_service.py` (Groq Whisper client). | Initialize React + Vite + Tailwind shell. Build WaveSurfer `AudioPlayer.tsx`. |
| **Day 2: Algorithms & Training** | Implement `lpc.py`, `vq.py`, `dtw.py`. Fine-tune ResNet-18 in Notebook 02 (93% acc). | Implement `algorithms/hmm.py` (Viterbi trellis) and `services/sound_cnn.py`. | Build `WorkspacePage.tsx` and dual-track `TimelineView.tsx` with click-to-seek. |
| **Day 3: Labs & Fusion** | Extract model weights & 11 figures. Verify 17/17 passing unit tests in `pytest`. | Implement `timeline/fusion.py` to merge speech text & sound tags chronologically. | Build `SignalLabPage.tsx`, `PatternLabPage.tsx`, `HMMPage.tsx`, and exporters. |
| **Day 4: Integration** | Validate live DSP & CNN outputs against API routes. | Connect live VAD $\to$ Whisper $\to$ ResNet $\to$ Timeline Fusion in `/audio/analyze`. | Toggle `USE_MOCK_DATA = False`. Test end-to-end audio upload $\to$ playback $\to$ `.srt` export. |
| **Day 5: Deliverables** | Draft Report Page 2 (DSP) and Page 3 (Acoustic ResNet). | Draft Report Page 4 (Sequence Modeling & Timeline Fusion). | Capture high-res UI screenshots $\to$ Compile Screenshots PDF; draft Page 1 & 5. |
