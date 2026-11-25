# SonicCaption — Updated Complete Product Implementation Plan
**Speech Processing and Recognition | MSc AI & ML | Group of 3**  
**Version 3.0 | Modern React + FastAPI Architecture, 100% Free Stack, Product UX, and Course Alignment**

---

## 0. Executive decision

SonicCaption is an intelligent, production-grade audio understanding and captioning web application. A user uploads or records an audio clip; the system analyses the raw acoustic signal, transcribes speech with word/segment timestamps, detects environmental acoustic events (e.g., clapping, laughter, door knocks, sirens) using a custom-trained model, and synthesises both into a unified, interactive caption timeline. Furthermore, the platform features dedicated interactive visual laboratories demonstrating fundamental algorithms from **Units 1–4** of the Speech Processing and Recognition syllabus (MAI522-5).

### Key Architectural Evolution (v2.0 Streamlit → v3.0 React + FastAPI)
To transition from a standard prototype to a genuine, commercially viable product while strictly maintaining a **$0.00 infrastructure cost**:
- **Frontend:** A responsive Single Page Application (SPA) built with **React (Vite + TypeScript)**, styled with **Tailwind CSS**, featuring interactive audio waveform scrubbing via **WaveSurfer.js**, dynamic SVG/Canvas-based acoustic visualisations, and dual-track timeline editing. Deployed freely on **Vercel**.
- **Backend:** A high-performance asynchronous **FastAPI** service with OpenAPI/Swagger documentation, modular signal-processing pipelines, and secure CORS handling. Deployed freely on **Hugging Face Spaces (Docker/FastAPI)** or **Render Free Tier**.
- **Model Training & Storage:** Google Colab (Free T4 GPU) for training the custom environmental CNN on ESC-50, hosted on **Hugging Face Hub** (free public model repository).
- **Speech Recognition:** Pretrained Whisper inference via **Groq Cloud API** (free developer tier) with a local CPU fallback via **faster-whisper**.

### Core Product Capabilities

1. **Audio Workspace & Ingestion:** Upload WAV, MP3, M4A, FLAC, and OGG formats (or optional direct microphone recording in-browser via MediaRecorder API); client-side audio playback with synchronized cursor scrubbing.
2. **Signal Validation & Conditioning:** Server-side decoding, channel reduction to mono, resampling (16 kHz standard), duration and RMS checks.
3. **Signal Laboratory (U1 & U2):** Interactive inspection of waveform, RMS energy, zero-crossing rate (ZCR), FFT magnitude spectrum, STFT spectrogram (log-Mel and linear), filter-bank energies, MFCC heatmaps, LPC spectrum/poles, and K-means vector quantisation (VQ) codebook assignments.
4. **VAD Speech Segmentation (U3):** Speech region detection using Silero VAD / WebRTC VAD to isolate voice activity from acoustic background.
5. **ASR Speech Transcription:** Automatic speech transcription with segment and word-level timestamps powered by Whisper.
6. **Acoustic Event Detection:** Multi-class classification of environmental events using the team's custom CNN trained on ESC-50.
7. **Unified Caption Timeline:** Synchronised dual-track timeline merging speech transcript text and sound event tags (e.g., `[Applause]`, `[Laughter]`, `[Door Knock]`) with interactive seek-to-timestamp capabilities.
8. **Pattern Matching Studio (U3):** Pairwise audio comparison, acoustic feature distortion distance (Euclidean/Cosine), and dynamic time warping (DTW) cost matrix heatmap with optimal warp path overlay.
9. **HMM & Viterbi Interactive Sandbox (U4):** Discrete Hidden Markov Model sandbox with user-configurable transition/emission matrices, forward algorithm likelihood calculation, and trellis visualizer of the optimal Viterbi decoded state path.
10. **Export Engine:** One-click export of captions and analysis in standard formats: JSON, SubRip (`.srt`), WebVTT (`.vtt`), CSV, and plain text (`.txt`).
11. **Academic Transparency:** Explicit UI labels and a Model Card distinguishing between the team's custom-trained CNN, syllabus-demonstration algorithms (DTW, HMM), and third-party foundation models (Whisper).

---

## 1. Course alignment and assessment

The MAI522-5 syllabus lists five units. The CIA III project submission explicitly assesses **Units 1, 2, 3, and 4**. SonicCaption bridges rigorous theoretical DSP/PR formulations with a modern full-stack web application.

| Unit & Syllabus Content | SonicCaption Implementation | Frontend & Backend Evidence |
|---|---|---|
| **U1 — Speech Fundamentals:** Speech production/perception; time/frequency domain; speech sounds; energy, ZCR, STFT | Audio decode; interactive waveform; short-time energy; zero-crossing rate; FFT magnitude; STFT log-spectrogram; voiced/unvoiced frames | React Signal Studio: WaveSurfer.js waveform, canvas-rendered FFT spectrum and 2D spectrogram with frequency cursor |
| **U2 — Approaches & Analysis:** Acoustic-phonetic, pattern recognition, AI; filter banks; LPC analysis; Vector Quantisation (VQ) | Linear and Mel filter-bank decomposition; 13–20 MFCC coefficient extraction; LPC autoregressive coefficients and spectral envelope; K-means VQ codebook clustering | React Feature Studio: MFCC heatmap visualizer, LPC formant overlay against FFT, interactive VQ codeword scatter plot |
| **U3 — Pattern Comparison:** Speech detection; distance metrics; time alignment; Dynamic Time Warping (DTW) | Silero VAD frame masking; frame-wise Euclidean/Cosine distortion; custom dynamic programming DTW recurrence; accumulated cost matrix calculation | React Pattern Studio: VAD speech-boundary overlays; dual-clip alignment player; 2D DTW cost matrix with red backtracking path |
| **U4 — HMM & Viterbi:** Markov processes; HMM elements ($\pi, A, B$); Evaluation (Forward), Decoding (Viterbi) | Discrete HMM implementation; forward variable calculation for sequence likelihood; Viterbi dynamic programming recursion and backpointers | React HMM Sandbox: Interactive state-transition graph; user-editable probability matrices; full dynamic trellis visualizer |
| **End-to-End System** | Production ASR + Environmental Sound Classifier + Timeline Fusion + Subtitle Exporters | Live Web Application (React SPA + FastAPI backend) |

### CIA III Evaluation Rubric

| Criterion | Marks | Evidence to Deliver |
|---|---:|---|
| **Execution of speech-processing tasks** | 10 | Seamless audio upload, real-time waveform display, feature extraction, VAD segmentation, Whisper ASR, sound event detection, and multi-format export |
| **Code/algorithm correctness** | 4 | Unit-tested implementations of DSP preprocessing, MFCC, LPC, VQ, VAD, DTW recurrence, and HMM Viterbi trellis; clean API separation |
| **Concept clarity (viva)** | 5 | All 3 members demonstrate fluency across DSP theory, ML training curves, HMM equations, and system architecture |
| **Presentation/neatness** | 3 | Commercial-grade React UI, responsive layout, dark/light theme, interactive charts, and comprehensive Swagger documentation |
| **Innovation/real-world application** | 3 | Integrated accessibility captioning (speech + ambient sound cues) useful for deaf/hard-of-hearing users or automated video indexing |

---

## 2. Feasibility study & $0.00 budget breakdown

### 2.1 Feasibility Verdict
**100% Feasible at $0.00 total expense.**  
Moving to a decoupled **React + FastAPI** architecture elevates the product aesthetics and performance while remaining completely within the zero-cost free tiers of modern developer platforms.

| Component | Technology | Free Tier Provider | Limits / Constraints | Zero-Cost Feasibility |
|---|---|---|---|---|
| **Frontend SPA** | React 18/19, Vite, Tailwind CSS | **Vercel** (Hobby) | 100 GB bandwidth/month, unlimited builds | **100% Free** |
| **Backend REST API** | FastAPI, Uvicorn, Python 3.10/3.11 | **Hugging Face Spaces** (Docker) or **Render** | HF: 16 GB RAM, 2 vCPU, persistent; Render: 750 free hrs/mo (spins down after 15m idle) | **100% Free** |
| **Model Training** | PyTorch, torchaudio, Librosa | **Google Colab** | Free T4 GPU (~12 hrs/session) | **100% Free** |
| **Model Storage** | Git LFS, Hugging Face Hub | **Hugging Face Model Hub** | Unlimited public repositories, automated CDN delivery | **100% Free** |
| **Speech ASR** | Whisper Large v3 / Turbo | **Groq Cloud API** | Free developer tier (~30 RPM, 25 MB max file) | **100% Free** (with local CPU fallback) |
| **Local ASR Fallback** | `faster-whisper` (CTranslate2) | Local Hardware / Backend CPU | CPU quantized `tiny` or `base` model (runs in ~1s) | **100% Free** |
| **Dataset** | ESC-50 (2,000 clips) | GitHub Official Repo | Public open-source dataset (CC BY-NC 3.0) | **100% Free** |
| **Version Control & CI** | Git, GitHub Actions | **GitHub** | Unlimited public repos, 2,000 free CI minutes/month | **100% Free** |

### 2.2 Model Trained by the Team
The team will train **one custom compact Convolutional Neural Network (CNN)** in Google Colab on environmental acoustic signals:
- **Dataset:** ESC-50 (Environmental Sound Classification, 2,000 5-second recordings across 50 classes).
- **Target Subset:** 8–10 distinct classes relevant to ambient captioning (e.g., `clapping`, `laughter`, `coughing`, `sneezing`, `footsteps`, `door_knock`, `glass_breaking`, `siren`, `dog`, `rain`).
- **Feature Pipeline:** 16 kHz mono resampling $\rightarrow$ 128-band log-Mel spectrogram $\rightarrow$ SpecAugment (time/frequency masking) $\rightarrow$ compact 4-layer CNN with BatchNorm, Dropout, and Softmax.
- **Evaluation:** Strict 5-fold cross-validation adhering to the official ESC-50 fold structure.
- **Artefacts Generated:** `model.pt`, `labels.json`, `config.json`, training loss curves, per-class F1 metrics, and a published Hugging Face repository.

### 2.3 What NOT to Train from Scratch
- **Whisper ASR:** Whisper-large comprises 1.5 billion parameters trained on 680,000 hours of multilingual audio. Building an ASR engine from scratch in a few days is impossible. We leverage Groq's high-speed Whisper API and document this as an integration.
- **Voice Activity Detection:** Pretrained Silero VAD provides state-of-the-art enterprise-grade voice detection (<1ms inference) without requiring custom training.
- **HMM & DTW:** Implemented algorithmically from mathematical first principles in NumPy to prove syllabus mastery.

### 2.4 Risk Analysis and Mitigation

| Potential Risk | Impact | Concrete Mitigation |
|---|---|---|
| **Free Backend Spin-down (Render)** | First API call takes 30–50s to cold start | Deploy on **Hugging Face Spaces (Docker)** which does not spin down, or provide a frontend visual ping/loader during cold start |
| **CORS / Cross-Origin Blocking** | Frontend fails to talk to backend | FastAPI `CORSMiddleware` configured with explicit origins (`http://localhost:5173`, `https://*.vercel.app`) |
| **Groq API Rate Limits / Network Outage** | Transcription fails during demo | Automatically fall back to local `faster-whisper` (`tiny.en` or `base`) running directly on the backend CPU; cache pre-computed sample clips |
| **Large Audio Upload Bottlenecks** | Upload timeouts or memory spikes | Enforce 25 MB max file size and 120-second clip duration limit in both React client and FastAPI request validator |
| **Colab GPU Disconnection** | Loss of training epoch progress | Mount Google Drive; save model checkpoints (`best_checkpoint.pth`) after every epoch |

---

## 3. System architecture & technology stack

### 3.1 Architecture Overview

```text
+-----------------------------------------------------------------------------------+
|                            CLIENT TIER: REACT SPA (Vite + TS)                     |
|                                                                                   |
|  [ Modern SaaS UI / Dark Studio Theme / Tailwind CSS / Lucide Icons / Axios ]     |
|                                                                                   |
|  +-------------------+  +--------------------+  +-------------------------------+  |
|  | Studio Workspace  |  | Signal Lab (U1-U2) |  | Pattern Lab (U3) & HMM (U4)   |  |
|  | - Audio Dropzone  |  | - Waveform & FFT   |  | - DTW Cost Heatmap & Path     |  |
|  | - WaveSurfer.js   |  | - STFT Spectrogram |  | - HMM Transition State Graph  |  |
|  | - Dual Timeline   |  | - MFCC / LPC / VQ  |  | - Viterbi Trellis Visualizer  |  |
|  +---------+---------+  +---------+----------+  +---------------+---------------+  |
+------------|----------------------|-----------------------------|-----------------+
             | REST / JSON / FormData                             |
             v                      v                             v
+-----------------------------------------------------------------------------------+
|                           BACKEND TIER: FASTAPI (Python 3.10+)                    |
|                                                                                   |
|  [ API Router | Request Validation (Pydantic v2) | CORS | Error Handling ]        |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Orchestrator Service: /api/v1/analyze                                       |  |
|  +-------+-------------------------+-------------------------+-----------------+  |
|          |                         |                         |                    |
|          v                         v                         v                    |
|  +---------------+         +---------------+         +-----------------+          |
|  | Audio Core    |         | ASR Service   |         | Acoustic Event  |          |
|  | - librosa     |         | - Groq Whisper|         | Classifier      |          |
|  | - soundfile   |         | - Local Fall- |         | - Team CNN      |          |
|  | - Silero VAD  |         |   back        |         | - Log-Mel Spect.|          |
|  +-------+-------+         +-------+-------+         +--------+--------+          |
|          |                         |                          |                   |
|          +-------------------------+--------------------------+                   |
|                                    |                                              |
|                                    v                                              |
|                    +-------------------------------+                              |
|                    | Caption & Timeline Fusion     |                              |
|                    | - Chronological Merge         |                              |
|                    | - Exporters: SRT/VTT/JSON/CSV |                              |
|                    +-------------------------------+                              |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Syllabus Lab Endpoints:                                                     |  |
|  | - POST /api/v1/labs/signal   --> Waveform, FFT, STFT, MFCC, LPC, VQ         |  |
|  | - POST /api/v1/labs/dtw      --> Frame Alignment, Cost Matrix, Warp Path    |  |
|  | - POST /api/v1/labs/viterbi  --> Forward Likelihood, Trellis, Best Path     |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### 3.2 Technology Stack

| Layer | Technology | Version | Purpose | Cost |
|---|---|---|---|---|
| **Frontend Framework** | React + Vite | 18.3+ / 5.0+ | Modern high-performance Single Page Application | Free |
| **Language (Client)** | TypeScript | 5.0+ | Type-safe state, API responses, and audio schemas | Free |
| **Styling & UI Components**| Tailwind CSS + Lucide React | 3.4+ | Dark studio UI, responsive layout, modal drawers | Free |
| **Audio Waveform UI** | WaveSurfer.js | 7.0+ | Interactive audio scrubbing, zoom, region selection | Free |
| **Visualisation (Client)**| Canvas API / Plotly.js / Chart.js | Latest | Interactive 2D spectrograms, DTW heatmaps, HMM trellis | Free |
| **Backend Framework** | FastAPI | 0.110+ | Async REST API, OpenAPI docs, background tasks | Free |
| **Language (Server)** | Python | 3.10 / 3.11 | High performance signal processing & machine learning | Free |
| **Audio Processing** | Librosa, SoundFile, SciPy, NumPy | Latest | Resampling, FFT, STFT, MFCC, LPC, digital filters | Free |
| **VAD Engine** | Silero VAD | v4 / v5 | Deep learning voice activity detection (<1ms latency) | Free |
| **ASR Provider** | Groq Python SDK (`whisper-large-v3-turbo`) | Latest | High-speed cloud speech-to-text with word timestamps | Free tier |
| **Local ASR Fallback**| `faster-whisper` | Latest | CPU-optimised local CTranslate2 Whisper fallback | Free |
| **Neural Network** | PyTorch + TorchAudio | 2.2+ | Custom 4-layer CNN for ESC-50 event classification | Free |
| **Model Registry** | Hugging Face Hub (`huggingface_hub`) | Latest | Checkpoint storage, automated download, model card | Free |
| **Testing** | Vitest (Frontend) + Pytest (Backend) | Latest | Automated unit and integration testing | Free |
| **CI / Deployment** | GitHub Actions + Vercel + HF Spaces | Latest | Continuous Integration and Automated Zero-Cost Hosting | Free |

---

## 4. API specification & service contracts

The FastAPI backend exposes fully documented endpoints adhering to OpenAPI 3.1 standards (accessible via `/docs`).

### 4.1 REST API Routes

| Method | Endpoint | Description | Request Payload | Response Schema |
|---|---|---|---|---|
| `GET` | `/api/v1/health` | Service health and active models check | None | `{ status: "ok", groq: bool, cnn_loaded: bool }` |
| `POST` | `/api/v1/audio/analyze` | Full end-to-end pipeline (VAD + ASR + Event CNN + Timeline) | `multipart/form-data` (file: Audio) | `AnalysisResponse` |
| `POST` | `/api/v1/labs/signal` | Unit 1 & 2: Extract DSP features (FFT, STFT, MFCC, LPC, VQ) | `multipart/form-data` (file: Audio, params: JSON) | `SignalLabResponse` |
| `POST` | `/api/v1/labs/dtw` | Unit 3: Dynamic Time Warping alignment between two clips | `multipart/form-data` (file_a, file_b, metric) | `DTWResponse` |
| `POST` | `/api/v1/labs/hmm/viterbi` | Unit 4: Compute forward likelihood and Viterbi state trellis | `application/json` (pi, A, B, observations) | `HMMResponse` |
| `POST` | `/api/v1/export/{format}` | Export timeline as `.srt`, `.vtt`, `.json`, `.csv`, or `.txt` | `application/json` (timeline items) | File Stream (`Content-Disposition: attachment`) |

### 4.2 Data Schemas (Pydantic v2 & TypeScript)

#### Backend (Pydantic)
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float

class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: Optional[List[WordTimestamp]] = []

class AcousticEvent(BaseModel):
    start: float
    end: float
    label: str
    confidence: float
    model_source: str = "soniccaption-esc50-cnn"

class TimelineItem(BaseModel):
    id: str
    start: float
    end: float
    category: Literal["speech", "sound_event"]
    label: str
    display_text: str
    confidence: Optional[float] = None

class AnalysisResponse(BaseModel):
    filename: str
    duration: float
    sample_rate: int
    speech_segments: List[TranscriptSegment]
    sound_events: List[AcousticEvent]
    timeline: List[TimelineItem]
    metrics: dict
```

#### Frontend (TypeScript Interface)
```typescript
export interface WordTimestamp {
  word: string;
  start: number;
  end: number;
}

export interface TranscriptSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  words?: WordTimestamp[];
}

export interface AcousticEvent {
  start: number;
  end: number;
  label: string;
  confidence: number;
  model_source: string;
}

export interface TimelineItem {
  id: string;
  start: number;
  end: number;
  category: "speech" | "sound_event";
  label: string;
  display_text: string;
  confidence?: number;
}

export interface AnalysisResult {
  filename: string;
  duration: number;
  sample_rate: number;
  speech_segments: TranscriptSegment[];
  sound_events: AcousticEvent[];
  timeline: TimelineItem[];
  metrics: {
    vad_speech_ratio: number;
    transcription_time_ms: number;
    cnn_inference_time_ms: number;
  };
}
```

---

## 5. Google Colab model training (ESC-50 CNN)

### Objective
Train a lightweight, robust 2D Convolutional Neural Network on environmental audio clips using the standard ESC-50 dataset. This model serves as the team's primary machine learning deliverable.

### Training Protocol
1. **Dataset Ingestion:** Download ESC-50 via Git in Colab (`git clone https://github.com/karolpiczak/ESC-50.git`).
2. **Subset Extraction:** Select 8–10 distinct classes representing indoor/human/warning sounds:
   - `clapping`, `laughter`, `coughing`, `sneezing`, `footsteps`, `door_knock`, `glass_breaking`, `siren`, `dog`, `rain`.
3. **Preprocessing Pipeline (16 kHz mono):**
   - Resample audio from 44.1 kHz to 16,000 Hz.
   - Compute short-time Fourier transform (STFT): `n_fft = 1024`, `hop_length = 512`, `win_length = 1024`.
   - Apply Mel filter-bank: `n_mels = 128` (output dimension: $128 \times 157$ frames for a 5-second clip).
   - Convert to log-power scale (dB).
4. **Data Augmentation:** SpecAugment (random time masking with max width 16, frequency masking with max width 16) and random additive Gaussian noise.
5. **Network Architecture (`SonicAcousticCNN`):**
   - Conv Block 1: Conv2D(1, 32, 3x3) + BatchNorm + ReLU + MaxPool2D(2, 2) + Dropout(0.2)
   - Conv Block 2: Conv2D(32, 64, 3x3) + BatchNorm + ReLU + MaxPool2D(2, 2) + Dropout(0.2)
   - Conv Block 3: Conv2D(64, 128, 3x3) + BatchNorm + ReLU + MaxPool2D(2, 2) + Dropout(0.3)
   - Dense Head: AdaptiveAvgPool2d((1, 1)) $\rightarrow$ Flatten $\rightarrow$ Linear(128, 64) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.4) $\rightarrow$ Linear(64, NumClasses).
   - Total parameters: < 150,000 (ultra-fast inference, <15ms on CPU).
6. **Validation Strategy:** Official 5-fold cross-validation. Evaluate Folds 1–4 for training and Fold 5 as the strict held-out test fold.
7. **Publishing to Hugging Face Hub:**
   - Commit `model.pt`, `config.json`, `labels.json`, confusion matrix plot, and a structured `README.md` (Model Card) to `https://huggingface.co/<team-org>/soniccaption-esc50-cnn`.

---

## 6. Syllabus algorithm implementation details

All syllabus implementations are housed under `backend/src/soniccaption/algorithms/` and exposed via dedicated `/labs` endpoints to enable interactive visual learning in the React frontend.

### Unit 1: Speech Signal Fundamentals
- **Waveform & Energy:** Compute instantaneous short-time energy $E_n = \sum_{m} [x(m) w(n-m)]^2$ and root-mean-square (RMS) with sliding 25ms Hamming windows.
- **Zero-Crossing Rate (ZCR):** $Z_n = \frac{1}{2} \sum_{m} |\text{sgn}[x(m)] - \text{sgn}[x(m-1)]| w(n-m)$ for distinguishing voiced from unvoiced speech.
- **Spectral Representation:** Compute Discrete Fourier Transform (DFT) via FFT algorithm; STFT computed with adjustable window sizes (128 to 2048 samples) and hop sizes.
- **Voiced / Unvoiced / Silence Classification:** Energy-ZCR thresholding matrix identifying unvoiced fricatives (high ZCR, low energy) vs voiced vowels (low ZCR, high energy).

### Unit 2: Approaches & Feature Extraction
- **Filter-Bank Analysis:** Triangular Mel-scale filter-bank spanning 0 Hz to $F_s/2$ (Nyquist frequency), computing sub-band energy integration.
- **MFCC:** Discrete Cosine Transform (DCT-II) applied to log-filterbank energies, retaining the first 13 static cepstral coefficients ($c_0$ to $c_{12}$), with dynamic delta ($\Delta$) and delta-delta ($\Delta^2$) derivatives.
- **Linear Predictive Coding (LPC):** Autoregressive all-pole vocal tract filter estimation using Levinson-Durbin recursion on the auto-correlation sequence $R(k)$; roots of the predictor polynomial provide formant frequencies and bandwidths.
- **Vector Quantisation (VQ):** Linde-Buzo-Gray (LBG) / K-means clustering algorithm generating codebooks of sizes $K \in \{16, 32, 64, 128\}$ codewords; quantisation error and codeword histograms visualised in the UI.

### Unit 3: Pattern Comparison & Dynamic Time Warping (DTW)
- **Voice Activity Detection:** Energy/VAD frame masking to isolate active speech intervals.
- **Local Distortion Metrics:** Compute Euclidean distance $d(x_i, y_j) = ||x_i - y_j||_2$ and Cosine distance $1 - \frac{x_i \cdot y_j}{||x_i|| ||y_j||}$ between feature vectors of two recordings.
- **DTW Dynamic Programming:**
  $$D(i, j) = d(i, j) + \min \big( D(i-1, j), D(i, j-1), D(i-1, j-1) \big)$$
- **Backtracking:** Trace optimal path $W = \{w_1, w_2, \dots, w_K\}$ from $(N, M)$ to $(1, 1)$, displaying normalized path cost $\frac{D(N, M)}{K}$.

### Unit 4: Discrete Hidden Markov Models & Viterbi
- **HMM Parameterisation:** Model defined by $\lambda = (A, B, \pi)$:
  - State space $S = \{s_1, s_2, \dots, s_N\}$
  - Transition matrix $A = [a_{ij}]$ where $a_{ij} = P(q_{t+1} = s_j \mid q_t = s_i)$
  - Emission matrix $B = [b_j(k)]$ where $b_j(k) = P(O_t = v_k \mid q_t = s_j)$
  - Initial probabilities $\pi = [\pi_i]$ where $\pi_i = P(q_1 = s_i)$
- **Forward Algorithm (Likelihood Evaluation):** Computes $P(O \mid \lambda)$ via forward recursion variable $\alpha_t(i)$ with log-scale scaling to prevent floating-point underflow.
- **Viterbi Algorithm (Decoding):** Dynamic programming recursion computing maximum likelihood state path:
  $$v_t(j) = \max_{1 \le i \le N} \big[ v_{t-1}(i) a_{ij} \big] \cdot b_j(O_t)$$
  $$\psi_t(j) = \arg\max_{1 \le i \le N} \big[ v_{t-1}(i) a_{ij} \big]$$
  Backtracking via $\psi_t$ produces optimal state trajectory $Q^* = (q_1^*, q_2^*, \dots, q_T^*)$.

---

## 7. Product UX & application interfaces

SonicCaption is designed as a contemporary, dark-themed audio intelligence workstation reminiscent of professional creative and analysis suites (e.g., Descript, Ableton, Adobe Audition).

```text
+---------------------------------------------------------------------------------------------------------+
| [O] SONICCAPTION    |  [Studio Workspace]  [Signal Lab]  [Pattern Matcher]  [HMM Trellis]  [Model Card] |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ AUDIO WORKSPACE & SYNCHRONIZED PLAYER ]                                                             |
|  +---------------------------------------------------------------------------------------------------+  |
|  | [Play/Pause] 00:03.45 / 00:15.00  | Zoom: [ - | + ] | Speed: 1.0x | Vol: [========]               |  |
|  |                                                                                                   |  |
|  |  WaveSurfer Waveform:                                                                             |  |
|  |  |/|/\/\/|/\|/\/\/\/\/\/|/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/|     |  |
|  |  [==== Speech Segment 1 ====]     [== Clapping ==]     [====== Speech Segment 2 ======]           |  |
|  +---------------------------------------------------------------------------------------------------+  |
|                                                                                                         |
|  +-------------------------------------------------------+  +----------------------------------------+  |
|  | DUAL-TRACK CAPTION TIMELINE                           |  | REAL-TIME INSPECTOR & METRICS          |  |
|  |                                                       |  |                                        |  |
|  | 00:00.50 - 00:02.80                                   |  | Audio Info: 16 kHz Mono | 15.2s WAV   |  |
|  | [SPEECH] "Welcome to the presentation."               |  | VAD Active Speech: 68.4%               |  |
|  |                                                       |  | Dominant Event: Clapping (92.4%)       |  |
|  | 00:03.10 - 00:04.20                                   |  | ASR Engine: Groq Whisper Large v3      |  |
|  | [SOUND] [Applause / Clapping] (Conf: 94.2%)           |  |                                        |  |
|  |                                                       |  | [ Export Captions v ]                  |  |
|  | 00:04.50 - 00:08.20                                   |  |  * SubRip (.srt)                       |  |
|  | [SPEECH] "Today we are demonstrating SonicCaption..." |  |  * WebVTT (.vtt)                       |  |
|  |                                                       |  |  * JSON Timeline (.json)               |  |
|  +-------------------------------------------------------+  +----------------------------------------+  |
+---------------------------------------------------------------------------------------------------------+
```

### 7.1 Studio Screens & Pages
1. **Studio Workspace (Main Dashboard):**
   - Drag-and-drop audio ingestion zone with built-in sample clips (clean speech, noisy speech, applause + speech, coughing).
   - Audio waveform player with millisecond scrubber cursor and synced playback.
   - Dual-track caption timeline displaying speech segments and acoustic sound tags with confidence chips.
   - Click-to-seek: clicking any word or event jumps the audio player to that exact millisecond.
   - Export modal supporting instant downloads of `.srt`, `.vtt`, `.json`, `.csv`, and `.txt`.

2. **Signal Lab (Units 1 & 2):**
   - High-definition STFT spectrogram visualizer with customizable colormaps (Viridis, Inferno, Magma).
   - FFT spectrum analyzer with peak frequency detection.
   - Energy and Zero-Crossing Rate time-series plots.
   - MFCC 13-coefficient heatmap with delta-derivative toggle.
   - LPC spectrum envelope overlay against FFT magnitude showing formant peaks ($F_1, F_2, F_3$).
   - Vector Quantisation (VQ) codebook distribution scatter chart.

3. **Pattern Matcher (Unit 3):**
   - Dual audio loader to compare two spoken utterances or sound clips.
   - VAD boundary indicator comparing voiced vs silence regions.
   - Interactive 2D Dynamic Time Warping (DTW) accumulated cost heatmap with highlighted optimal warp path.
   - Real-time display of alignment distance, path length, and time-normalization factor.

4. **HMM Trellis Visualizer (Unit 4):**
   - Interactive Discrete HMM sandbox with presets (e.g., Fair/Biased Coin Toss, Weather Rainy/Sunny, Urn/Ball model).
   - Real-time editable transition matrix $A$, emission matrix $B$, and initial vector $\pi$.
   - Observation sequence input builder (e.g., `H, H, T, T, H`).
   - Dynamic Viterbi Trellis Diagram: interactive node graph showing state probabilities per time step, backpointers, and the winning optimal state path highlighted in green.

5. **Model Card & Architecture:**
   - Detailed model provenance card detailing dataset (ESC-50), training splits, hyperparameter config, confusion matrix, and accuracy/F1 metrics.
   - Clear distinction between student-trained ML (ESC-50 CNN) and external APIs (Whisper).

---

## 8. GitHub repository & monorepo structure

A clean, decoupled monorepo facilitates parallel development among the three team members without code collision.

```text
soniccaption/
├── .github/
│   └── workflows/
│       ├── frontend-ci.yml          # Vitest, ESLint, TypeScript check
│       └── backend-ci.yml           # Pytest, flake8, syntax check
├── backend/
│   ├── Dockerfile                   # Deployment on Hugging Face Spaces / Render
│   ├── requirements.txt             # Pinned backend dependencies
│   ├── main.py                      # FastAPI app entry point & CORS configuration
│   ├── configs/
│   │   ├── default_config.yaml      # Audio & feature extraction hyperparameters
│   │   └── esc50_classes.json       # Target classification labels
│   ├── src/
│   │   └── soniccaption/
│   │       ├── api/                 # FastAPI routers & route handlers
│   │       │   ├── routes_audio.py  # Analysis and timeline endpoints
│   │       │   ├── routes_labs.py   # Signal, DTW, and HMM lab endpoints
│   │       │   └── routes_export.py # SRT, VTT, JSON exporters
│   │       ├── core/                # Core configurations, schemas, and logging
│   │       │   ├── config.py
│   │       │   └── schemas.py
│   │       ├── audio/               # Audio I/O, validation, resampling
│   │       │   ├── io.py
│   │       │   ├── preprocessing.py
│   │       │   └── vad.py
│   │       ├── algorithms/          # Syllabus mathematical implementations
│   │       │   ├── spectral.py      # FFT, STFT, ZCR, Energy
│   │       │   ├── lpc.py           # Levinson-Durbin LPC & formants
│   │       │   ├── vq.py            # K-means / LBG vector quantisation
│   │       │   ├── dtw.py           # Custom DTW recurrence & backtracking
│   │       │   └── hmm.py           # Forward likelihood & Viterbi trellis
│   │       ├── services/            # External integrations & models
│   │       │   ├── asr_service.py   # Groq Whisper API + faster-whisper fallback
│   │       │   └── sound_cnn.py     # Custom ESC-50 PyTorch model loader
│   │       └── timeline/            # Fusion logic and subtitle generators
│   │           ├── fusion.py
│   │           └── exporters.py
│   └── tests/
│       ├── test_spectral.py
│       ├── test_lpc_vq.py
│       ├── test_dtw.py
│       ├── test_hmm_viterbi.py
│       └── test_api_endpoints.py
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/                     # Axios / Fetch client calling FastAPI
│       │   └── apiClient.ts
│       ├── types/                   # TypeScript interfaces (matching Pydantic schemas)
│       │   └── index.ts
│       ├── components/              # Reusable UI widgets
│       │   ├── Navbar.tsx
│       │   ├── AudioPlayer.tsx      # WaveSurfer.js wrapper
│       │   ├── TimelineView.tsx     # Synchronized caption & event track
│       │   ├── ExportModal.tsx      # SRT/VTT/JSON export dialogue
│       │   └── MetricsBadge.tsx
│       ├── pages/                   # Application pages
│       │   ├── WorkspacePage.tsx    # Main studio captioning page
│       │   ├── SignalLabPage.tsx    # U1 & U2 DSP visualizer
│       │   ├── PatternLabPage.tsx   # U3 DTW comparison studio
│       │   ├── HMMPage.tsx          # U4 Interactive Trellis visualizer
│       │   └── ModelCardPage.tsx    # Academic documentation & metrics
│       └── utils/                   # Subtitle parsers, formatters, color mappers
│           └── formatters.ts
├── notebooks/
│   ├── 01_esc50_dataset_exploration.ipynb
│   ├── 02_train_sonic_acoustic_cnn.ipynb
│   └── 03_model_evaluation_and_export.ipynb
├── data/
│   └── samples/                     # Test WAV audio clips for instant testing
│       ├── speech_clean.wav
│       ├── speech_with_applause.wav
│       └── environmental_door_knock.wav
├── docs/
│   ├── architecture.md
│   ├── model_card.md
│   └── viva_guide.md
├── .env.example
├── README.md
└── LICENSE
```

### 8.1 Product Engineering Pillars & Group Ownership

#### Architectural Philosophy: Notebook-First Research Core & Lean Service Adapters
In a real-world machine learning audio product, production `.py` files should not obscure scientific and mathematical derivations:
1. **Interactive Notebooks (`notebooks/`) Hold the Core Algorithmic & Modeling Logic:**  
   All signal processing equations, psychoacoustic transforms, deep neural network training loops, hyperparameter tuning, metric evaluations, and visual interpretations live directly inside reproducible Jupyter Notebooks (`01_*`, `02_*`, `03_*`).
2. **Backend Python Modules (`backend/src/`) Act as Clean Service Adapters:**  
   The Python files under `backend/` are lightweight, modular application wrappers and FastAPI endpoints that package the notebook-proven algorithms into production services for the React web application.

| Team Member | Product Role | Core Research Notebooks (The Logic) | Backend Service Scope & Deliverables |
|---|---|---|---|
| **Member 1** | **Audio Signal Processing & Acoustic Intelligence Lead** | • `notebooks/01_audio_feature_extraction_pipeline.ipynb` (from-scratch DSP, framing, STFT, Mel, MFCCs, Levinson-Durbin LPC formants, VQ, DTW)<br>• `notebooks/02_environmental_sound_classifier_training.ipynb` (ESC-50 ResNet-18 model training, SpecAugment, 88.75% Val / 80.00% held-out Fold-5 Test accuracy, 0.787 macro F1, model export) | `backend/src/soniccaption/audio/`, `algorithms/` (spectral, LPC, VQ, DTW), and automated test suite (`79/79 passing tests`). Report Pages 2 & 3. |
| **Member 2** | **Speech Recognition, Sequence Modeling & Timeline Fusion Lead** | • `notebooks/03_speech_recognition_and_sequence_fusion.ipynb` (Silero VAD interval segmentation, Whisper ASR evaluation, Discrete HMM formulation, Viterbi dynamic programming trellis decoding, and chronological timeline fusion) | `backend/src/soniccaption/services/` (Whisper ASR, ResNet sliding-window loader), `audio/vad.py`, `algorithms/hmm.py`, `timeline/fusion.py`. Report Page 4. |
| **Member 3** | **Full-Stack Product, UI/UX & Deployment Lead** | • Integrates and demonstrates all notebook-proven capabilities in the commercial web product | `frontend/` (React 18, Vite, WaveSurfer.js, Tailwind), `backend/main.py`, `backend/api/`, subtitle exporters (`.srt`, `.vtt`, `.json`), Dockerfile, and Vercel/Spaces deployment. Report Pages 1 & 5 + Screenshots PDF. |

---

## 9. Day-by-day execution sprint (5-day timeline)

### Day 0: Setup & Zero-Cost Infrastructure Verification (Hours 1–4)
- **Repo & Monorepo Initialization:** Create GitHub repository; set up `frontend/` (Vite + React + TS + Tailwind) and `backend/` (FastAPI).
- **Accounts & Keys:** Obtain free Groq API key (`GROQ_API_KEY`); create Hugging Face account and public model repository.
- **Verification Gates:**
  1. Frontend builds cleanly (`npm run build`).
  2. Backend starts with CORS enabled (`uvicorn main:app`).
  3. Single test audio file transcribes via Groq Whisper API.
  4. Google Colab GPU environment confirmed.

### Day 1: Audio Core & Model Training Initiation
- **Member 1:** Implement audio decoding, resampling to 16 kHz, mono conversion, short-time energy, ZCR, FFT, and STFT. Add Pytest coverage for finite values and silence inputs.
- **Member 2:** Set up Colab notebook; mount Google Drive; download ESC-50; filter target 10 classes; build PyTorch Dataset and log-Mel feature extractor; start baseline CNN training.
- **Member 3:** Build React layout shell (Navbar, tab router, dark theme); integrate WaveSurfer.js audio player with play/pause/timecode controls.

### Day 2: Advanced Features, ASR & Custom CNN
- **Member 1:** Implement linear filter banks, MFCC calculation (13 coefficients + deltas), LPC reflection coefficients, and K-means VQ clustering.
- **Member 2:** Complete Colab CNN training; compute confusion matrix, macro-F1 score; export PyTorch weights to Hugging Face Hub; implement backend model loading service and Groq ASR service.
- **Member 3:** Build main Studio Workspace: drag-and-drop audio dropzone, upload progress indicator, and dual-track caption timeline component with click-to-seek audio integration.

### Day 3: Syllabus Algorithms & Interactive Labs
- **Member 1:** Implement custom Dynamic Time Warping (DTW) with Euclidean and Cosine distances; output accumulated cost matrix and backtracking coordinates.
- **Member 2:** Implement Discrete HMM, Forward evaluation algorithm, and Viterbi decoding with state backpointers in NumPy.
- **Member 3:** Build React visualizers for the labs:
  - Signal Lab: Waveform, Canvas STFT Spectrogram, and MFCC Heatmap.
  - Pattern Lab: Dual-waveform viewer with interactive DTW cost matrix heatmap.
  - HMM Lab: Dynamic Viterbi Trellis graph with highlighted optimal path.

### Day 4: Timeline Fusion, Exporters & Deployment
- **All Members:** Integrate end-to-end pipeline in FastAPI (`/api/v1/audio/analyze`): Audio Ingest $\rightarrow$ VAD Mask $\rightarrow$ Groq Whisper $\rightarrow$ ESC-50 CNN $\rightarrow$ Chronological Timeline Fusion.
- **Member 3:** Implement one-click caption export endpoints for `.srt`, `.vtt`, `.json`, `.csv`, and `.txt`.
- **Zero-Cost Deployment:**
  - Deploy React SPA to **Vercel** (connect GitHub repo, configure auto-deploy).
  - Deploy FastAPI backend to **Hugging Face Spaces** (Docker environment, 16 GB RAM free tier) or **Render**.
  - Configure backend CORS for the Vercel domain.

### Day 5: Validation, Evaluation & Viva Preparation
- **Evaluation Suite:** Run test suite across 8 benchmark clips (clean speech, noisy speech, clapping, silence). Document WER on speech samples and macro-F1 on sound events.
- **Documentation:** Finalize `README.md`, `model_card.md`, and API documentation.
- **Rehearsal:** Execute 5–7 minute synchronized product demo. Ensure every team member can explain all math from Units 1–4 and live system architecture.

---

## 10. Verification, testing, and evaluation metrics

### 10.1 Automated Test Suite (Pytest & Vitest)
1. **Audio Validation Tests:** Verify graceful rejection of corrupt files, non-audio formats, and clips exceeding 25 MB with appropriate HTTP 422/400 errors.
2. **DSP Numerical Stability:** Test that silence vectors produce zero energy and do not trigger divide-by-zero errors in LPC Levinson-Durbin or log-Mel spectrograms.
3. **DTW Mathematical Properties:**
   - Identity: $DTW(X, X) = 0$.
   - Symmetry: $DTW(X, Y) = DTW(Y, X)$.
   - Boundary Condition: Alignment path begins strictly at $(1, 1)$ and ends at $(N, M)$.
4. **HMM Viterbi Correctness:** Test known textbook coin-toss and weather-model state transitions against hand-calculated analytical probabilities.
5. **Caption Timeline Integrity:** Assert that timeline items are strictly sorted in monotonically increasing order of start timestamps with no negative intervals.

### 10.2 Empirical Metrics to Report
- **ASR Evaluation:** Word Error Rate (WER) and Character Error Rate (CER) evaluated against 5 manually transcribed audio clips:
  $$\text{WER} = \frac{S + D + I}{N}$$
- **Sound Event Evaluation:** Accuracy, Precision, Recall, and Macro-averaged F1-Score computed on the official ESC-50 held-out fold:
  $$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
- **Latency Benchmarks:** Report wall-clock processing time for a 10-second audio clip:
  - Audio decode & resampling: ~50ms
  - Silero VAD segmentation: ~15ms
  - Groq Whisper ASR: ~400–800ms
  - ESC-50 CNN inference (10 sliding windows): ~80ms
  - Timeline fusion & JSON serialization: ~5ms
  - **Total Pipeline Latency:** < 1.5 seconds (near real-time feel).

---

## 11. Deployment guide ($0.00 zero-cost stack)

### 11.1 Frontend Deployment on Vercel
1. Push codebase to GitHub.
2. Log in to [vercel.com](https://vercel.com) using GitHub account.
3. Select **Add New Project** $\rightarrow$ Import your `soniccaption` repository.
4. Set **Root Directory** to `frontend`.
5. Framework Preset: **Vite**.
6. Environment Variables:
   - `VITE_API_BASE_URL`: `https://<your-backend-host>.hf.space` (or Render URL).
7. Click **Deploy**. Vercel provisions global CDN hosting with automatic HTTPS in ~60 seconds.

### 11.2 Backend Deployment on Hugging Face Spaces (Recommended for ML)
Hugging Face Spaces offers a **100% Free Docker/FastAPI environment with 16 GB RAM and 2 vCPUs**, eliminating memory crashes and spin-down latency.
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Space SDK: **Docker**.
3. Space Hardware: **CPU Basic (Free - 2 vCPU, 16 GB RAM)**.
4. Set Repository Secrets:
   - `GROQ_API_KEY`: Your free Groq developer API key.
   - `ALLOWED_ORIGINS`: `https://<your-app>.vercel.app,http://localhost:5173`.
5. Point to `backend/Dockerfile` or configure GitHub Actions to automatically sync `backend/` to Hugging Face Spaces Git.

*(Alternative: Render Free Web Service with Docker/Python; note that free instances sleep after 15 minutes of inactivity).*

---

## 12. Antigravity execution strategy & prompts

When delegating tasks to Antigravity agents, utilize specialized persona prompts to maintain clean separation of concerns:

### Prompt 1: Full-Stack Architect
```text
You are the Lead Systems Architect for SonicCaption. Inspect the workspace and build
a decoupled full-stack application using React (Vite + TS + Tailwind) in frontend/ and
FastAPI in backend/. Enforce the agreed OpenAPI contracts and Pydantic schemas.
Configure CORS, basic health check routes, and Dockerfile for free deployment.
Do not introduce mock data placeholders or skip mathematical implementations.
Verify setup using npm build and pytest.
```

### Prompt 2: Audio Signal Processing & Acoustic Intelligence Lead (Member 1)
```text
You are Member 1 (Audio Signal Processing & Acoustic Intelligence Lead) on the SonicCaption team.
Your objective is to own the foundational audio research, from-scratch DSP transforms, and the acoustic neural classifier.

Your assigned scope:
1. notebooks/01_audio_feature_extraction_pipeline.ipynb: Complete from-scratch DSP pipeline (resampling, framing, Hamming window, FFT, STFT, 40-channel Mel filterbanks, 13 MFCCs + Deltas, Levinson-Durbin LPC formants, VQ, DTW).
2. notebooks/02_environmental_sound_classifier_training.ipynb: ESC-50 dataset ingestion, DSP feature bridge, SpecAugment, ResNet-18 transfer learning achieving 88.75% Fold-4 validation accuracy and 80.00% strictly held-out Fold-5 test accuracy (0.787 macro F1), model weights export.
3. backend/src/soniccaption/audio/ and algorithms/: Modular service implementations of all audio DSP transforms.
4. backend/tests/: Pytest test suite validating numerical stability, silence handling, and algorithm accuracy (17/17 passing tests).

Follow the data contracts in core/schemas.py. Do not edit UI or ASR service files.
```

### Prompt 3: Speech Recognition, Sequence Modeling & Timeline Fusion Lead (Member 2)
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

### Prompt 4: Frontend & Product Engineer (Member 3)
```text
You are the Product and UI Engineer for SonicCaption. Own frontend/ (React 18, Vite,
TypeScript, Tailwind CSS, WaveSurfer.js). Build a modern dark-mode audio intelligence
studio with synchronized waveform scrubbing, dual-track caption timeline, click-to-seek,
interactive STFT spectrogram, DTW cost matrix heatmap, and interactive HMM Viterbi
trellis visualizer. Ensure responsive layout, error alerts, and one-click SRT/JSON exports.
```

---

## 13. Rehearsed 5–7 minute live demonstration script

| Time | Segment | Visual Actions on Screen | Key Viva Script / Technical Narration |
|---|---|---|---|
| **0:00–0:45** | **The Problem & Value** | Open live Vercel web app. Show clean Studio UI. | *"Standard ASR only transcribes speech, ignoring critical acoustic context like applause, coughing, or sirens. SonicCaption is an intelligent audio understanding system that fuses speech transcription with custom-trained environmental sound classification."* |
| **0:45–1:45** | **Live Processing & Studio** | Upload `speech_with_applause.wav`. Click **Analyze Audio**. | Show audio decoding $\rightarrow$ VAD $\rightarrow$ Whisper ASR $\rightarrow$ ESC-50 CNN. WaveSurfer renders waveform. Dual-track timeline populates with speech text and `[Applause]` badge. Click `[Applause]` chip to immediately scrub audio to that timestamp. |
| **1:45–2:45** | **Signal Lab (Units 1 & 2)** | Switch to **Signal Lab** tab. | Show real-time FFT spectrum, STFT 2D spectrogram, and short-time energy/ZCR. Highlight MFCC 13-coefficient heatmap. Show LPC spectral envelope formant peaks tracking vocal tract resonances. Explain K-means VQ codebook clusters. |
| **2:45–3:45** | **Pattern Matching (Unit 3)** | Switch to **Pattern Matcher** tab. Load two utterances. | Run DTW alignment. Display the 2D accumulated cost matrix with the red dynamic programming path. Explain: *"The diagonal trace shows non-linear time-warping aligning speaking rates between the two utterances."* |
| **3:45–4:30** | **HMM & Viterbi (Unit 4)** | Switch to **HMM Sandbox** tab. Select Fair/Biased Coin or Weather model. | Click **Run Viterbi**. Dynamic SVG Trellis highlights state transition path. Point out forward variable log-likelihood and optimal decoded state sequence. |
| **4:30–5:15** | **Model Training & Proof** | Switch to **Model Card** tab. Show Colab training history and confusion matrix. | *"We trained our custom ResNet-18 on ESC-50 in Google Colab using log-Mel spectrograms with SpecAugment under a strict leak-free fold split, achieving 88.75% Fold-4 validation accuracy and 80.00% strictly held-out Fold-5 test accuracy (0.787 macro F1) across all 10 balanced target classes, hosted locally and on Hugging Face Hub."* |
| **5:15–5:45** | **Export & Accessibility** | Click **Export**, select `.srt` and `.vtt`. Open downloaded file. | Demonstrate timestamped subtitles with embedded sound cues ready for VLC, YouTube, or video editing software. |
| **5:45–6:00** | **Conclusion & Q&A** | Show FastAPI Swagger docs (`/docs`) and GitHub commit history. | Summarize full syllabus coverage (Units 1–4), modular architecture, and invite faculty questions. |

---

## 14. Viva readiness checklist

Each group member must be prepared to defend the theoretical and practical foundations of the project:

### Fundamental Speech Processing (Unit 1 & 2)
- [ ] Explain why speech is non-stationary and why short-time analysis (20–30ms frames) is required.
- [ ] Contrast time-domain features (Energy, ZCR) with frequency-domain features (FFT, Spectrogram).
- [ ] Explain the physiological rationale of the Mel scale ($m = 2595 \log_{10}(1 + f/700)$) reflecting human auditory perception.
- [ ] Derive why DCT is used in MFCCs to decorrelate filter-bank energies into cepstral coefficients.
- [ ] Explain the all-pole source-filter model of speech production in LPC and how Levinson-Durbin solves the Toeplitz equation system.
- [ ] Explain the role of Vector Quantisation (VQ) in reducing continuous feature vectors to discrete acoustic codewords.

### Pattern Matching & Dynamic Time Warping (Unit 3)
- [ ] Explain why Euclidean distance between feature matrices of different lengths fails without dynamic alignment.
- [ ] Formulate the DTW boundary conditions, monotonicity condition, and continuity constraints.
- [ ] Walk through the dynamic programming equation $D(i, j) = d(i, j) + \min(D_{i-1, j}, D_{i, j-1}, D_{i-1, j-1})$.

### Hidden Markov Models (Unit 4)
- [ ] Define the 5 components of a discrete HMM: $N$ states, $M$ observation symbols, $A$ transitions, $B$ emissions, and $\pi$ initial distribution.
- [ ] State the Three Basic Problems of HMMs:
  1. Evaluation: $P(O \mid \lambda)$ via Forward-Backward algorithm.
  2. Decoding: Optimal state sequence $Q^*$ via Viterbi algorithm.
  3. Learning: Parameter estimation via Baum-Welch (EM) algorithm.
  - [ ] Explain how Viterbi avoids $O(N^T)$ exponential complexity by using dynamic programming to achieve $O(N^2 T)$ complexity.

### System & Machine Learning Engineering
- [ ] Clearly delineate: Whisper (pretrained cloud ASR) vs custom ESC-50 CNN (student-trained ML) vs HMM Lab (algorithmic syllabus demo).
- [ ] Justify why decoupling React and FastAPI makes the system resilient, scalable, and testable compared to monolithic scripts.
- [ ] Explain how the timeline fusion algorithm handles overlapping speech and sound events.
- [ ] Defend why the system incurs $0.00 infrastructure cost via Vercel, Hugging Face Spaces, and Groq Cloud.

---

## 15. Definition of done (SonicCaption v1.0 Release)

- [ ] Modern React Single Page Application runs locally and is deployed live on **Vercel**.
- [ ] FastAPI backend service is deployed on **Hugging Face Spaces** (or Render) with live Swagger `/docs`.
- [ ] Interactive audio waveform displays scrubbing and playback via **WaveSurfer.js**.
- [ ] End-to-end pipeline processes uploaded WAV/MP3 files and returns a unified caption timeline in < 2 seconds.
- [ ] Custom ESC-50 CNN is trained in Colab, achieves >80% accuracy, and weights load seamlessly from Hugging Face Hub.
- [ ] Interactive **Signal Lab** displays live FFT spectrum, STFT spectrogram, MFCC heatmap, and LPC formants.
- [ ] Interactive **Pattern Lab** visualizes real DTW cost matrix heatmap and optimal warp path.
- [ ] Interactive **HMM Sandbox** calculates forward likelihood and renders the dynamic Viterbi state trellis.
- [ ] Subtitle exports (`.srt`, `.vtt`, `.json`, `.csv`, `.txt`) download cleanly with accurate timestamps.
- [ ] Pytest suite achieves >85% code coverage across DSP algorithms and API endpoints.
- [ ] GitHub repository contains clean branch commit histories, GitHub Actions CI workflows, and no hardcoded API secrets.
- [ ] All 3 group members have rehearsed the live demonstration and are prepared for viva questioning.
