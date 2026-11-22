# SonicCaption — Updated Complete Product Implementation Plan
**Speech Processing and Recognition | MSc AI & ML | Group of 3**  
**Version 2.0 | Model training, feasibility, APIs, GitHub, deployment and evaluation**

---

## 0. Executive decision

SonicCaption is an intelligent audio understanding application. A user uploads or records an audio clip; the system analyses the signal, transcribes speech, detects selected environmental sounds, and combines the results into a timestamped caption timeline. Users can inspect the signal-processing work and explore Unit 1–4 algorithms through interactive labs.

This is a complete-product plan, not a set of disconnected lab exercises. The team will use Antigravity for implementation, Google Colab for model training, GitHub for source control, Hugging Face Hub for trained model artefacts, and Streamlit for the application.

### Product requirements

1. Upload WAV, MP3, M4A and FLAC audio; optional browser recording if stable.
2. Validate, decode, convert to mono and resample where required.
3. Display waveform, duration, sample rate, channels, RMS/energy, zero-crossing rate, FFT spectrum and STFT spectrogram.
4. Extract and visualise filter-bank energies, MFCCs, LPC coefficients and vector-quantisation codebooks.
5. Detect speech regions using VAD.
6. Transcribe speech with timestamps.
7. Detect selected non-speech events using a model trained by the team in Colab.
8. Merge speech and sound events into one chronological caption timeline.
9. Compare speech feature sequences using distortion measures and DTW.
10. Demonstrate a discrete HMM and Viterbi decoder on a reproducible observation sequence.
11. Export transcript/captions as JSON, SRT and TXT/CSV.
12. Include tests, model card, documentation, limitations and a rehearsed live demo.

**Be precise in the report:** the environmental sound classifier is the team's trained model. Whisper is a pretrained ASR model/API, not trained by the team. The HMM/Viterbi component is an educational implementation unless a separate speech HMM is genuinely trained; it is not the decoder powering Whisper.

---

## 1. Course alignment and assessment

The MAI522-5 syllabus lists five units. The CIA III project-specific submission portion explicitly lists **Units 1, 2, 3 and 4**. The course-plan CIA III summary also contains a line mentioning Units 3, 4 and half of Unit 5. Confirm this discrepancy with faculty. This plan prioritises the explicit project submission portion (Units 1–4); the end-to-end ASR application also demonstrates system-building concepts.

| Unit and listed content | SonicCaption implementation | Evidence |
|---|---|---|
| **U1 — Fundamentals:** speech paradigm/history; speech production/perception; time/frequency representation; speech sounds/features | Audio upload; waveform; FFT; STFT/spectrogram; energy; ZCR; voiced/unvoiced analysis | Signal Lab, plots and sample clips |
| **U2 — Approaches and analysis:** acoustic-phonetic, pattern recognition, AI; spectral analysis; filter-bank front end; LPC; VQ | Explanations of approaches; filter-bank/MFCC; LPC; K-means VQ codebook | Feature plots, LPC coefficients and codeword assignments |
| **U3 — Pattern comparison:** speech detection; distortion; perceptual/spectral measures; dynamic features; time alignment and normalisation | VAD; feature distance; MFCC sequence comparison; DTW cost matrix and path | Compare two recordings and display aligned sequences |
| **U4 — HMM:** Markov processes; HMM; coin/urn examples; HMM elements; three basic problems; Viterbi | Discrete HMM; initial/transition/emission probabilities; forward likelihood; Viterbi best path | Interactive HMM panel, trellis and unit tests |
| **Working application** | ASR + environmental event detection + timeline + exports | Live end-to-end demonstration |

### CIA III rubric — build for the marks

| Criterion | Marks | Evidence to prepare |
|---|---:|---|
| Execution of speech-processing tasks | 10 | Upload real clip, preprocess, run recognition/analysis, display results and export |
| Code/algorithm correctness | 4 | Correct preprocessing, feature extraction, VAD, DTW and HMM/Viterbi; tests |
| Concept clarity (viva) | 5 | All members explain the pipeline and algorithms |
| Presentation/neatness | 3 | Clean UI, readable plots, architecture and documentation |
| Innovation/real-world application | 3 | Unified speech + environmental captioning with practical use |

The course plan says groups contain three members and also mentions individual submission. Each person should retain evidence of their contribution and be able to explain the whole application.

---

## 2. Feasibility study

### 2.1 Verdict

**Feasible as a complete academic product in a few intensive days** if the team uses pretrained ASR, trains one compact environmental sound model, and implements the syllabus algorithms directly. It is not realistic to train a general-purpose speech recogniser from scratch, build production-grade diarisation, or guarantee recognition of every sound in the same period.

| Feature | Proposed implementation | Feasibility | Main risk |
|---|---|---|---|
| Upload/playback/audio validation | Streamlit + soundfile/librosa/FFmpeg | High | Codec and file-size edge cases |
| Waveform, FFT, STFT, RMS, ZCR | NumPy/SciPy/librosa + Plotly | High | Correct axis/scaling |
| Filter banks and MFCC | librosa | High | Consistent parameters |
| LPC | librosa/custom stable implementation | Medium–High | Numerical stability |
| VQ | K-means codebook on feature vectors | High | Explain codebook training |
| Speech detection | Silero VAD or WebRTC VAD | High | Threshold tuning |
| Speech-to-text | Groq Whisper API; faster-whisper fallback | High for demo | Network, quota, API key |
| Custom sound model | Compact CNN on log-Mel spectrograms, trained in Colab | Medium–High | Small dataset and domain shift |
| Sound-event timestamps | Sliding windows + overlap + smoothing | Medium | Window labels are not exact event boundaries |
| DTW/distortion | Custom NumPy implementation | High | Correct path/backtracking |
| HMM/Viterbi | Custom NumPy implementation | High | Correct probability handling |
| Timeline/export | Python data model + Streamlit | High | Overlapping event handling |
| Public deployment | Streamlit Community Cloud | Medium | Memory/CPU limits |

### 2.2 Model that the team will train

Train **one compact environmental sound classifier in Google Colab**.

- Dataset: ESC-50.
- Architecture: small CNN operating on log-Mel spectrograms.
- Initial target: 8–10 relevant classes, for example clapping, laughing, coughing, sneezing, footsteps, door knock, glass breaking, siren, dog and rain. Verify exact category names in `meta/esc50.csv` before finalising.
- Split: use official dataset folds; do not randomly split if fold information is available.
- Outputs: checkpoint, label mapping, preprocessing config, training history, confusion matrix, metrics and model card.
- Publish the model to Hugging Face Hub; pin the model revision in the app.

ESC-50 contains 2,000 labelled five-second environmental recordings across 50 classes, with 40 recordings per class. It is appropriate for a compact academic experiment but small for broad real-world sound understanding. A restricted classifier with a confidence threshold is more defensible than claiming universal event detection. Check dataset licence/attribution before redistributing clips or weights; the full ESC-50 and ESC-10 subset do not necessarily have identical terms.

### 2.3 What not to train from scratch

- **ASR:** Do not train Whisper or another general speech recogniser from scratch; data and compute requirements are far beyond a few-day project.
- **VAD:** Use pretrained Silero VAD or WebRTC VAD.
- **HMM:** Implement a small discrete HMM and Viterbi yourself for syllabus correctness. Do not claim it replaces Whisper's decoder.
- **DTW:** Implement the dynamic-programming recurrence yourself using extracted features.
- **Speaker recognition/emotion:** Optional extensions only after the full core workflow works.

### 2.4 Runtime/deployment reality

Use hosted Groq transcription for the public demo and keep local faster-whisper as a fallback. This avoids loading a large ASR model into a small free app host. Keep the custom CNN small enough for CPU inference and cache it after loading.

Colab free GPU access, runtime duration and resource availability are dynamic and not guaranteed. Save checkpoints and logs to Drive after each epoch. Streamlit Community Cloud is convenient but has resource limits; keep uploads small, cache models and use hosted ASR. If deployment resources are insufficient, use a Hugging Face Space or another suitable host rather than removing core features.

### 2.5 Risks and mitigation

| Risk | Mitigation |
|---|---|
| Colab GPU unavailable/disconnects | Compact model, checkpoint to Drive each epoch, CPU fallback |
| API key/quota/network failure | Local faster-whisper fallback, cached sample results, clear error |
| App memory limit | Hosted ASR, small CNN, caching, upload/duration limits |
| Small-data overfitting | Official fold split, augmentation, macro-F1 and per-class metrics |
| False event labels | Confidence threshold, smoothing, uncertain/unknown state |
| Integration conflicts | Module contracts, branch ownership, daily merges |
| Fake agent-reported success | Run commands yourselves; require test output and real metrics |

---

## 3. Architecture

```text
USER
  |
  v
STREAMLIT FRONTEND
  |-- Upload / optional recording
  |-- Audio player and waveform
  |-- Signal Lab (U1/U2)
  |-- Pattern Lab (U3)
  |-- HMM Lab (U4)
  |-- Caption timeline / exports
  |
  v
APPLICATION ORCHESTRATOR
  |
  +--> AUDIO I/O + PREPROCESSING
  |      decode -> validate -> mono -> resample -> duration checks
  |
  +--> SIGNAL ANALYSIS
  |      waveform / FFT / STFT / RMS / ZCR / filterbank / MFCC / LPC / VQ
  |
  +--> VAD -> speech segments
  |
  +--> ASR ADAPTER
  |      Groq Whisper API (deployed) OR faster-whisper (local fallback)
  |      -> transcript + timestamps
  |
  +--> CUSTOM SOUND CNN (trained in Colab)
  |      -> sound label + confidence + time window
  |
  +--> PATTERN COMPARISON: distortion + DTW
  |
  +--> HMM LAB: discrete HMM + forward likelihood + Viterbi
  |
  v
TIMELINE FUSION -> JSON / SRT / TXT / CSV
```

**Architecture decision:** use one Streamlit application with modular Python backend modules. Do not split into React + FastAPI unless there is spare time. A separate server adds deployment, CORS and networking work without improving the core result. Keep algorithms outside `app.py`.

### Recommended stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10/3.11 | Local and Colab consistency |
| UI | Streamlit | Product screens and interactive labs |
| Audio I/O | soundfile, FFmpeg; pydub if required | Decode WAV/compressed formats |
| Signal processing | NumPy, SciPy, librosa | FFT, STFT, MFCC, LPC, resampling |
| Visualisation | Plotly | Interactive plots and matrices |
| VAD | Silero VAD or WebRTC VAD | Speech-region detection |
| ASR | Groq Whisper API + faster-whisper fallback | Speech transcription |
| Training | PyTorch + torchaudio/librosa | CNN training in Colab |
| Dataset | ESC-50 | Environmental sound events |
| Model hosting | Hugging Face Hub | Store weights/config/model card |
| Algorithms | NumPy + custom implementations | DTW, HMM and Viterbi |
| Tests | pytest | Unit/integration tests |
| Version control | Git + GitHub | Branches, PRs, issues |
| CI | GitHub Actions | Automated test checks |
| Deployment | Streamlit Community Cloud | Public demo |

Pin compatible dependency versions after a clean environment test.

---

## 4. APIs, services and model integration

### 4.1 Services

| Service | Role | Required? | Credentials |
|---|---|---|---|
| Groq Speech-to-Text | Whisper transcription and timestamps | Recommended for deployed ASR | `GROQ_API_KEY` |
| Hugging Face Hub | Store/download custom CNN | Recommended | Public model needs no token; private/write access does |
| Google Colab | Train/evaluate model | Yes | Google account |
| Google Drive | Persist Colab checkpoints | Strongly recommended | Google account |
| GitHub | Source control and CI | Yes | GitHub account |
| Silero/WebRTC VAD | Local speech detection | Yes | None |
| faster-whisper | Local ASR fallback | Recommended | None |
| Streamlit Community Cloud | Deploy app | Recommended | Streamlit account linked to GitHub |

### 4.2 Groq transcription API

- Endpoint: `https://api.groq.com/openai/v1/audio/transcriptions`
- Model: `whisper-large-v3-turbo` for speed; `whisper-large-v3` if accuracy is the priority.
- Request `verbose_json` and segment/word timestamp granularity supported by the current SDK.
- Current docs list WAV, MP3, M4A, FLAC, OGG and WEBM.
- Current docs describe 25 MB max file size for free tier and 100 MB for developer tier; check the live account limit before the demo.
- Chunk longer recordings and merge overlapping timestamps.
- Do not assume free quota or uninterrupted availability.

### 4.3 ASR adapter contract

Create `src/soniccaption/services/asr_service.py`:

```python
def transcribe_audio(audio_path: str, language: str | None = None) -> dict:
    # {
    #   "text": str,
    #   "segments": [
    #     {"start": float, "end": float, "text": str,
    #      "words": [{"start": float, "end": float, "word": str}]}
    #   ],
    #   "provider": str,
    #   "language": str | None
    # }
```

Implement `GroqASRProvider` and `FasterWhisperProvider` behind the same contract. The UI/timeline must not depend on which provider ran.

### 4.4 Sound-model loading contract

Hugging Face model repository should contain:
- `model.pt` or `model.safetensors`
- `config.json` (sample rate, window length, Mel bins, architecture, normalisation)
- `labels.json`
- `README.md` model card
- `metrics.json`
- optional training curves/confusion matrix

Create `src/soniccaption/services/sound_classifier.py`:

```python
def load_sound_model():
    """Load weights and preprocessing config once; cache result."""

def classify_audio_window(audio, sample_rate: int) -> list[dict]:
    """Return ranked labels: [{"label": str, "score": float}, ...]."""
```

The training and inference pipelines must use identical sample rate, window length, Mel-bin count and normalisation.

### 4.5 Optional LLM refinement

Do not make an LLM necessary for core operation. Timeline generation should be deterministic. An optional summary can describe structured events, but must never invent events. Always display the raw transcript and event list.

---

## 5. Google Colab model training

### Objective

Train a compact environmental sound classifier from labelled audio. This is the team's own trained ML artefact.

### Dataset and classes

Use ESC-50 and verify the selected labels in `meta/esc50.csv`. Candidate classes: clapping, laughing, coughing, sneezing, footsteps, door knock, glass breaking, siren, dog and rain. ESC-50 has only 40 examples per class; a 10-class model remains a small academic model and may not generalise to arbitrary recordings.

### Training workflow

1. Create `notebooks/02_train_sound_classifier.ipynb`.
2. Select GPU runtime if available.
3. Mount Google Drive and create a persistent experiment folder.
4. Download ESC-50 from the official repository; retain metadata/licence.
5. Inspect metadata and filter the chosen classes.
6. Split using official folds; document exact fold assignments.
7. Decode audio and resample consistently to e.g. 16 kHz mono.
8. Convert each clip to log-Mel spectrogram with fixed parameters.
9. Apply training-only augmentation: random gain, small time shifts, background noise and time/frequency masking where suitable.
10. Train a compact CNN.
11. Track train/validation loss; use early stopping or a fixed short epoch budget.
12. Evaluate on held-out official fold; rotate folds if time permits.
13. Save best checkpoint, labels, config, metrics, confusion matrix and training history.
14. Upload artefacts to Hugging Face Hub.
15. Download the published model in the app and verify predictions.

### Notebook layout

```text
notebooks/
  01_dataset_inspection.ipynb
  02_train_sound_classifier.ipynb
  03_evaluate_sound_classifier.ipynb
  04_inference_smoke_test.ipynb
```

### Training deliverables

- Reproducible Colab notebook.
- Dataset source, licence and attribution.
- Fold/split protocol.
- Training/validation loss plot.
- Confusion matrix.
- Accuracy, macro precision/recall/F1 and per-class scores.
- Saved weights, `labels.json` and preprocessing config.
- Inference test on clips not used for training.
- Hugging Face repository and model card.

**Never invent a metric.** Report the real score and discuss weak classes and domain shift.

---

## 6. Syllabus implementation details

### U1 — speech signal fundamentals

Implement file metadata, waveform vs time, short-time RMS/energy, ZCR, FFT magnitude spectrum, STFT/log spectrogram, and approximate voiced/unvoiced analysis. Use a Signal Lab tab with time-range selection. If pitch is estimated, label the method and limitations.

### U2 — recognition approaches and feature extraction

Explain acoustic-phonetic, pattern-recognition and AI-based approaches. Implement windowing/pre-emphasis if used, filter-bank energies, MFCC, LPC coefficients and VQ using a K-means codebook. Visualise feature matrices and codeword assignments.

Do not claim Whisper uses your custom LPC/VQ implementation. These features demonstrate the syllabus and support your own analysis.

### U3 — speech detection, distortion and DTW

Implement VAD, MFCC sequence extraction for two clips/segments, frame-level Euclidean or cosine distance, DTW accumulated-cost matrix and backtracking. Use:

`D(i,j) = d(i,j) + min(D(i-1,j), D(i,j-1), D(i-1,j-1))`

Display the optimal alignment path and normalised path cost. Explain that DTW aligns feature sequences; it is not semantic similarity.

### U4 — HMM and Viterbi

Implement hidden states, observation symbols, initial probabilities `pi`, transition matrix `A`, emission matrix `B`, observation sequence, forward likelihood, Viterbi recursion, backpointer matrix and decoded path. Include coin-toss or urn-and-ball analogy. Let the user edit a small preset HMM and view the trellis.

The HMM lab demonstrates Unit 4. It is not the Whisper decoder.

---

## 7. Product UX and screens

1. **Home/Audio workspace:** upload dropzone, sample clips, player, metadata and Analyse action.
2. **Analysis overview:** transcript, sound-event cards, confidence, timestamped timeline and export.
3. **Signal Lab (U1/U2):** waveform, spectrogram, FFT, RMS, ZCR, filter bank, MFCC, LPC and VQ.
4. **Pattern Lab (U3):** choose two clips/segments, feature distance, DTW matrix, path and normalised distance.
5. **HMM Lab (U4):** preset/editable HMM, matrices, forward likelihood, Viterbi path.
6. **About/Model:** architecture, dataset attribution, model card, metrics, limitations and privacy note.

Design requirements: consistent theme, readable laptop-sized plots, meaningful loading/error states, no fake progress, no fabricated confidence, file-size/duration checks, sample clips for reliable demo.

---

## 8. GitHub repository and version control

### Repository structure

```text
soniccaption/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── LICENSE
├── src/soniccaption/
│   ├── config.py
│   ├── schemas.py
│   ├── orchestrator.py
│   ├── audio/
│   │   ├── io.py
│   │   ├── preprocessing.py
│   │   ├── segmentation.py
│   │   └── features.py
│   ├── algorithms/
│   │   ├── spectral.py
│   │   ├── lpc.py
│   │   ├── vector_quantization.py
│   │   ├── distortion.py
│   │   ├── dtw.py
│   │   ├── hmm.py
│   │   └── viterbi.py
│   ├── services/
│   │   ├── asr_service.py
│   │   ├── groq_provider.py
│   │   ├── faster_whisper_provider.py
│   │   ├── vad_service.py
│   │   └── sound_classifier.py
│   ├── timeline/
│   │   ├── fusion.py
│   │   └── exporters.py
│   └── ui/
│       ├── pages.py
│       ├── charts.py
│       └── components.py
├── notebooks/
├── configs/
├── tests/
├── data/samples/
├── artifacts/README.md
├── docs/
└── .github/workflows/tests.yml
```

Do not commit the full ESC-50 dataset or large model weights to GitHub. Keep small demo audio only where redistribution is permitted. Store model weights on Hugging Face Hub.

### GitHub setup

1. One member creates a repository; keep it private during initial setup.
2. Add all three group members as collaborators.
3. Protect `main` from direct pushes if possible.
4. Create branches:
   - `feature/audio-signal-lab`
   - `feature/models-and-algorithms`
   - `feature/ui-integration`
5. Merge through pull requests; require another member to review.
6. Track work in GitHub Issues/Projects.
7. Tag milestones: `v0.1-audio-upload`, `v0.2-analysis`, `v0.3-models`, `v0.4-integrated`, `v1.0-demo`.

### Non-negotiable Git rules

- Commit small working changes every few hours.
- Pull `main` before starting and before opening PR.
- Do not have two people edit the same file simultaneously.
- Never commit `.env`, API keys, tokens, private recordings or full datasets.
- `.gitignore` should cover `.env`, `.streamlit/secrets.toml`, `data/raw/`, `*.pt`, `*.pth`, `*.ckpt` and generated outputs as appropriate.
- Store weights in Hugging Face, not Git history.
- Record setup/model download steps in README.
- Resolve conflicts with the file owner; never blindly accept all changes.

### Ownership

| Member | Primary ownership | Coordination |
|---|---|---|
| Member 1 — signal processing | `audio/`, spectral, LPC, VQ, Signal Lab data | Shared schemas/preprocessing |
| Member 2 — ML/recognition | Colab notebooks, services, DTW, HMM/Viterbi | ASR and event output contracts |
| Member 3 — product/integration | `app.py`, UI, orchestrator, timeline, deployment, CI | Integrates modules; does not rewrite algorithms |

### CI

GitHub Actions should install the supported Python version, install dependencies, run import/syntax checks and `pytest -q` on each pull request. CI must not require API keys, a GPU, a large model download or paid external API calls. Mock external API providers in tests.

---

## 9. Shared data contracts

### Audio object

```python
AudioData(
    original_path: str,
    samples: np.ndarray,  # mono float32
    sample_rate: int,
    duration_seconds: float,
    original_channels: int
)
```

### Transcript segment

```json
{
  "start": 1.20,
  "end": 2.85,
  "text": "Please close the door.",
  "words": []
}
```

### Sound event

```json
{
  "start": 3.10,
  "end": 3.80,
  "label": "door_knock",
  "confidence": 0.87,
  "model": "soniccaption-esc50-cnn"
}
```

### Timeline item

```json
{
  "start": 1.20,
  "end": 2.85,
  "type": "speech",
  "label": "Speech",
  "content": "Please close the door.",
  "confidence": null,
  "source": "groq-whisper"
}
```

Timeline fusion must convert all results to one schema, apply configurable confidence thresholds, merge adjacent same-class windows, sort by start time, preserve source/confidence and export without altering predictions.

---

## 10. Day-by-day execution plan

This schedule assumes three people can work in parallel for intensive days. If fewer days remain, run workstreams concurrently and prove the end-to-end path early.

### Day 0 — first 2–3 hours: setup and feasibility gates
- Confirm scope and clarify Unit 1–4 versus Unit 3–5 wording with faculty.
- Create GitHub repo, collaborators, branches and issue board.
- Create Python environment and run Streamlit skeleton.
- Create Colab notebook and verify GPU selection.
- Download/inspect ESC-50 metadata and confirm chosen labels.
- Create Groq API key if using hosted ASR.
- Create Hugging Face account/model repository.
- Test one audio file through ASR.
- Test one sample through model preprocessing/inference skeleton.
- Agree module contracts.

**Exit:** accounts exist, repo runs, one sample decodes, API and training paths are proven.

### Day 1 — audio foundation + training begins
**Member 1:** audio upload/decode/validation, mono/resample, waveform, FFT, STFT, RMS, ZCR, tests.  
**Member 2:** Colab data loader, fold split, log-Mel preprocessing, compact CNN, start training/checkpointing; implement DTW and HMM/Viterbi with synthetic tests.  
**Member 3:** Streamlit shell/navigation, upload page, player/metadata, schemas/orchestrator, GitHub Actions.

**Exit:** upload and core plots work; training has started; module contracts are stable.

### Day 2 — recognition and syllabus algorithms
**Member 1:** filter-bank, MFCC, LPC, VQ and plots.  
**Member 2:** finish/evaluate model, publish Hugging Face weights, integrate VAD, Groq ASR and faster-whisper fallback; finish DTW/HMM/Viterbi.  
**Member 3:** analysis overview, Signal Lab and timeline UI; integrate modules.

**Exit:** transcript works, custom classifier predicts, U1–U4 algorithms execute independently.

### Day 3 — end-to-end integration
- Merge branches.
- Run full pipeline on 5–8 recordings.
- Fix preprocessing inconsistencies.
- Add thresholds and event smoothing.
- Build chronological timeline.
- Add JSON/SRT/TXT exports.
- Complete Pattern Lab and HMM Lab.
- Add model information and limitations.

**Exit:** one uploaded clip completes the full pipeline and produces a useful timeline.

### Day 4 — testing and deployment
- Unit/integration tests.
- Test noisy speech, silence, background music, overlap and event classes.
- Measure WER on a manually transcribed set.
- Measure sound-model accuracy/macro-F1 on held-out data.
- Record DTW/HMM test evidence.
- Deploy to Streamlit Community Cloud.
- Add secrets securely.
- Verify deployed model download and API access.
- Prepare offline/local fallback and sample clips.

**Exit:** public URL works and metrics/limitations are documented.

### Day 5 — polish and evaluation readiness
- Fix UI spacing, labels, empty states and errors.
- Verify exports.
- Finish architecture, README, feasibility and model documentation.
- Rehearse 5–7 minute demo.
- Each member answers questions across all four units.
- Tag `v1.0-demo`.

**Exit:** clean clone → install → run → process sample → export works; demo rehearsed.

---

## 11. Testing and evaluation

### Test audio set

Prepare legally usable examples:
1. Clean single-speaker speech.
2. Speech with background noise.
3. Speech with clapping.
4. Speech with laughter.
5. Speech with door knock/impact.
6. Environmental sound without speech.
7. Silence/near-silence.
8. Longer clip with multiple events.

Use public or team-recorded audio with permission. Manually transcribe the speech evaluation samples.

### Metrics

**ASR:** WER = `(Substitutions + Deletions + Insertions) / Number of reference words`; optionally CER. State test-set size and language.

**Sound classifier:** accuracy, macro precision/recall/F1, per-class F1 and confusion matrix. State whether metrics are clip-level or window-level.

**VAD:** compare predicted and manually annotated speech intervals; report precision/recall or boundary error where enough annotations exist.

**DTW:** identical sequences should have near-zero distance; time-stretched matching signals should align sensibly; unrelated sequences should generally have higher distance.

**HMM/Viterbi:** use a small known HMM with hand-checkable probabilities; verify the decoded path and edge cases.

### Acceptance tests

- WAV and MP3 uploads work.
- Invalid audio produces a useful error.
- Mono conversion/resampling produces expected rate.
- Waveform, FFT and spectrogram render.
- MFCC/LPC/VQ produce finite values, including silence handling.
- VAD returns ordered segments.
- ASR returns transcript/timestamps or clear provider error.
- Published CNN loads and returns ranked predictions.
- Timeline sorts events chronologically.
- SRT timestamps are valid.
- DTW path begins/ends at valid sequence indices.
- Viterbi path length equals observation sequence length.
- Fresh clone runs using README.
- No API key appears in Git history/logs.

---

## 12. Antigravity execution strategy and prompts

### Rules for agents

1. Give Antigravity this plan and ask it to inspect the repo before editing.
2. Ask for file ownership and module contracts first.
3. Each member works on a separate branch.
4. Agents must not edit files outside ownership without approval.
5. Require changed-file list, commands, tests and actual output.
6. Never accept “all tests pass” without evidence.
7. Merge working modules daily and run the app after every merge.

### Prompt A — architect

```text
You are the lead architect for SonicCaption. Read the project plan and inspect
the repository before editing. Build a modular Python + Streamlit app. Keep
algorithms out of app.py. Create the agreed structure, schemas, config,
requirements, README and tests skeleton. First return architecture and file
ownership. Then implement only setup and audio-upload foundation.

Do not invent model weights, metrics, API responses or completed features.
Do not create fake placeholders that appear to work. Never commit secrets or
large datasets. Report exact commands and test output.
```

### Prompt B — Member 1: signal processing

```text
Own only the assigned audio and signal-processing modules. Implement robust
audio loading, validation, mono conversion, resampling, safe normalisation,
waveform, FFT, STFT, RMS, ZCR, filter banks, MFCC, LPC and VQ. Follow the
agreed contracts. Handle silence, short clips, stereo and invalid input.
Return arrays and metadata, not Streamlit UI. Add pytest tests with synthetic
signals and a real sample. Verify finite values and documented dimensions.
Do not claim unmeasured accuracy. Open a PR when tests pass.
```

### Prompt C — Member 2: training and recognition

```text
Own Colab training notebooks and recognition/algorithm modules. Create a
reproducible ESC-50 sound-classification workflow using verified labels and
official folds. Use fixed log-Mel preprocessing, validation, held-out
evaluation, macro-F1, per-class metrics and confusion matrix. Save checkpoints
and logs to Drive. Publish weights/config/labels/model card to Hugging Face.

Implement VAD, Groq Whisper provider, faster-whisper fallback, sound-model
inference, DTW and discrete HMM/Viterbi. Do not claim Whisper was trained by
the team or that HMM powers Whisper. Never hardcode tokens. Report actual
notebook outputs, model revision and real metrics.
```

### Prompt D — Member 3: UI and integration

```text
Own app.py, UI, orchestrator, timeline and deployment configuration. Build
Home/Upload, Analysis Overview, Signal Lab, Pattern Lab, HMM Lab and
About/Evaluation screens. Integrate only through agreed contracts. Merge
transcript and sound events into one chronological timeline. Add JSON, SRT,
TXT/CSV exports. Show errors, confidence and model provenance. Cache models.
Do not modify algorithm files owned by others. Ensure a fresh clone can run.
```

### Prompt E — independent QA auditor

```text
Act as a strict QA engineer. Run the application from documented setup.
Inspect upload, preprocessing, waveform, FFT, STFT, MFCC, LPC, VQ, VAD, ASR,
custom classifier, timeline, exports, DTW and HMM/Viterbi. Look for fake
outputs, fabricated metrics, timestamp errors, inconsistent sample rates,
missing errors, secret leaks and deployment failures.

Run tests and provide exact output. Produce Critical/High/Medium/Low defects
with file paths and reproduction steps. Do not call the project ready while a
critical end-to-end path is broken.
```

### Prompt F — integration fixer

```text
Inspect the merged repository for integration failures between audio, model,
algorithms, UI and timeline. Preserve agreed contracts. Fix the smallest
possible set of files, add regression tests and run the full suite. Do not
replace real implementations with mock data or remove features to pass tests.
Report every changed file and actual test results.
```

---

## 13. Deployment

### Local
- Python 3.10/3.11 and virtual environment.
- `pip install -r requirements.txt`.
- Copy `.env.example` to `.env`; add key locally.
- Run `streamlit run app.py`.

### Streamlit Community Cloud
1. Push stable code to GitHub.
2. Connect Community Cloud to the repository.
3. Select `main` and `app.py`.
4. Choose compatible Python version.
5. Add `GROQ_API_KEY` through Streamlit Secrets.
6. Add HF token only if model is private.
7. Deploy and inspect logs.
8. Test upload, ASR, CNN, labs and exports.
9. Test after restart.
10. Keep local fallback if API/network fails.

### Model deployment
- Hugging Face repo public or access configured.
- Pin model revision.
- Confirm weights/config download.
- Match inference preprocessing to training.
- Model card lists dataset, licence, labels, method, metrics and limitations.

### Security/privacy
- No secrets in source/notebooks/screenshots/GitHub.
- Do not retain personal audio longer than necessary.
- State when audio is sent to a third-party ASR provider.
- Do not log raw audio or tokens.
- Provide a privacy note.

---

## 14. Submission documentation

Repository should include:
1. `README.md` — overview, features, setup, run instructions and demo.
2. `docs/feasibility-study.md` — options, rationale, risks and mitigation.
3. `docs/architecture.md` — architecture and data flow.
4. `docs/model-card.md` — custom model, data, labels, training and evaluation.
5. `docs/evaluation.md` — test set, real metrics, results and limitations.
6. `docs/demo-script.md` — exact live demo sequence.
7. `docs/viva-preparation.md` — U1–U4 and each member's role.
8. Colab notebooks — training/evaluation.
9. GitHub release/tag — `v1.0-demo`.
10. Deployed URL and backup demo video.

---

## 15. Five-to-seven-minute live demonstration

1. **0:00–0:30 — Problem:** transcript alone misses environmental sounds.
2. **0:30–1:15 — Upload:** prepared clip with speech + sound event; show metadata and waveform.
3. **1:15–2:30 — Product:** run analysis; show transcript, detected event, combined timeline and playback.
4. **2:30–3:15 — U1/U2:** spectrogram, FFT, MFCC, LPC or VQ.
5. **3:15–4:00 — U3:** compare two clips and show DTW alignment.
6. **4:00–4:45 — U4:** HMM matrices and Viterbi best path.
7. **4:45–5:30 — Training evidence:** Colab notebook, held-out metrics and HF model card.
8. **5:30–6:00 — Export:** download SRT/JSON and show deployment.
9. **6:00–7:00 — Impact/limitations:** use cases, accuracy limits and privacy.

Prepare one reliable sample and one backup. Do not depend on live internet being perfect.

---

## 16. Viva readiness

Every member must know:
- Problem statement and complete architecture.
- Sampling, waveform, FFT and STFT.
- Why mono conversion/resampling is used.
- MFCC, LPC and VQ.
- VAD and ASR difference.
- Difference between ASR and environmental sound classification.
- WER and macro-F1.
- Distortion and DTW.
- HMM elements and Viterbi.
- Which model the team trained, dataset, split and metrics.
- API failure behaviour, deployment and privacy.

Suggested primary areas: Member 1 — signal processing/features; Member 2 — model training, ASR/VAD, DTW/HMM; Member 3 — architecture, UI, timeline, GitHub/deployment. All three must be able to explain the entire pipeline.

---

## 17. Definition of done — SonicCaption v1.0

- [ ] Fresh clone installs and runs using README.
- [ ] Upload and playback work.
- [ ] Audio metadata/preprocessing are correct.
- [ ] Waveform, FFT and spectrogram work.
- [ ] RMS, ZCR, filter-bank and MFCC work.
- [ ] LPC and VQ demonstrations work.
- [ ] VAD detects speech regions.
- [ ] ASR returns real transcript/timestamps or clear error.
- [ ] Team-trained classifier loads from Hugging Face and returns real predictions.
- [ ] Timeline merges events chronologically.
- [ ] JSON/SRT/TXT/CSV export works.
- [ ] DTW displays an actual alignment path.
- [ ] HMM/Viterbi pass known test cases.
- [ ] Unit/integration tests pass.
- [ ] Colab notebook and metrics saved.
- [ ] GitHub has meaningful commits and release tag.
- [ ] Deployment URL works.
- [ ] Secrets are not in Git.
- [ ] Model/data licences and attributions are documented.
- [ ] Every member can explain the complete product.
- [ ] Backup demo and sample clips are ready.

---

## 18. Final recommendation

Build the real end-to-end workflow first while model training and syllabus labs proceed in parallel.

The defensible technical story:
- The team implements signal-processing features and U1–U4 algorithms.
- The team trains a compact environmental sound classifier in Colab.
- The team uses a documented pretrained Whisper-based service for ASR.
- The application fuses transcript and detected sound events into a timestamped caption timeline.
- GitHub tracks source code and collaboration; Hugging Face hosts trained weights; Streamlit hosts the product.

This gives the group a trained model, working application, syllabus coverage, measurable results, reproducibility and a realistic live demonstration.

---

## 19. Official references

**Course sources supplied with this project**
- `SPR_2026_CoursePlan_15092026.pdf`
- `SPEECH PROCESSING AND RECOGNITION_MAI522-5_TRIMESTER 2 and 5_2026_2026-08-19T11_37_11.954392312 (1).pdf`

**Technical documentation**
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- Hugging Face audio classification fine-tuning: https://huggingface.co/docs/transformers/tasks/audio_classification
- Hugging Face model upload/versioning: https://huggingface.co/docs/hub/models-uploading
- ESC-50 official dataset repository: https://github.com/karolpiczak/ESC-50
- Groq speech-to-text API: https://console.groq.com/docs/speech-to-text
- Streamlit Community Cloud: https://docs.streamlit.io/deploy/streamlit-community-cloud
- Streamlit secrets management: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- GitHub Actions Python workflow: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python
