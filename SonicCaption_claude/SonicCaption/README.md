# SonicCaption

Record or upload a conversation and get one timeline that says, for every moment, whether it was **speech** (and which speaker), a **sound effect** (clap, rain, dog, siren…) or **background noise**, plus what each person talked about and **who has to do what**.

* **Live recording tab**: camera + microphone, results appear on the timeline about 1.5 s behind real time. When you stop, the whole take is re-analysed for the final result.
* **Upload audio tab**: same analysis on a file (wav, mp3, m4a, ogg, flac, webm, mp4).
* **Result sub-tabs**: Speakers (what each person said), Tasks (assigned to speakers), Sounds & noise, MFCC & spectrogram, HMM, Voice patterns.
* Theme: dark blue and light yellow on white; green = speech, red = sound effect / alert, amber = background noise.

## Run it (Antigravity, VS Code, or any terminal)

Needs Python 3.10–3.13. The web UI is pre-built, so Node is not required.

```bash
./setup.sh      # Windows: setup.bat     (creates .venv, installs requirements)
./run.sh        # Windows: run.bat       -> open http://localhost:8000
```

In Antigravity: open this folder, then *Terminal → Run Task…* → **SonicCaption: setup**, then **SonicCaption: run**. `AGENTS.md` tells the agent how the project is laid out. Allow camera/microphone when the browser asks (works on `localhost`).

### Speech-to-text (read this)
Transcript, scenario, topics and tasks all need ASR. Choose one:

1. **Groq (fast, free tier)**: put `GROQ_API_KEY=...` in `.env` (copy `.env.example`).
2. **Local**: nothing to configure; `faster-whisper` downloads a ~150 MB model on first use (`python backend/scripts/download_models.py` to do it up front).

If neither works, the app says so in a red banner and still delivers speakers, sound effects and noise. It never invents text.

## How the three members' work fits together

| Stage | Component | Origin |
|---|---|---|
| Denoising, STFT, MFCC, LPC/formants, pitch | `algorithms/spectral.py, lpc.py, enhancement.py` | Member 1 |
| ESC-50 ResNet-18 sound classifier (frozen backbone) | `services/weights/soniccaption_esc50_resnet.pt` | Member 1 |
| Silero VAD, Whisper ASR, HMM/Viterbi, timeline fusion/exports | `audio/vad.py, services/asr_service.py, algorithms/hmm.py, timeline/` | Member 2 |
| React UI, FastAPI app, SQLite history | `frontend/`, `backend/app/` | Member 3 |
| **New in this integration** | scene classifier, speaker diarization, NLU, live session, pipeline | below |

What was wrong and what changed:

* **Noise model "not working"**: the classifier only knew 10 ESC-50 classes, so any steady noise or any speech got forced into one of them. Now: VAD gates speech first; an open-set head on the frozen ResNet embeddings adds an 11th *background noise* class (`services/scene_classifier.py`); a new 4-state HMM (`scene_state` preset) smooths the decisions with Viterbi; effect spans are trimmed to where the energy rises. Impulsive labels (dog, knock, cough…) are rejected on windows with no burst.
* **VAD** lost quiet speakers and was slow; rewritten (`audio/vad.py`), shared by live and batch paths.
* **Speakers**: previously a pitch heuristic limited to 2 people. Now a neural voice embedding (vendored Resemblyzer GE2E, Apache-2.0, `licenses/`) + clustering, any number of speakers, online tracking for the live tab.
* **Understanding** (`services/nlu.py`): scenario (meeting, call, lecture, interview…), per-speaker topics/key points/decisions, and tasks with owners ("I'll make the changes to the repository by Friday" said by Speaker 2 → task for Speaker 2, due Friday; "Rahul, can you update the docs?" + "Sure" → Rahul). Rule-based, offline; if `GROQ_API_KEY` is set an LLM adds extra tasks/summaries on top.
* **App**: no silent mock fallback any more (errors are shown), job/progress API, WebSocket live endpoint, tasks stored with their owner, HMM horizon no longer capped at 30 s, single process serves API + UI.

## Measured accuracy (be aware of the limits)

Held-out tests on public data in the build environment. Reproduce with the scripts in `backend/scripts/`.

| Test | Result |
|---|---|
| Scene hops (10 composed scenes, ESC-50 fold 5 never used for training) | speech recall 96% (93% over a noise bed); background-noise recall 92–96%; silence 71%; overall hop accuracy ~85% |
| Sound effects | found 56–63% of effect hops; **correct label only ~45%**. The 10-class backbone is ~77–80% accurate on isolated clips and applause resembles rain. Treat effect names as suggestions. |
| Speaker count / frame accuracy (4 distinct voices, neural) | 32/32 counts right, 92% clean / 88% noisy frame accuracy |
| Speaker separation without torch (DSP fallback) | unreliable (~coin flip); the UI warns when this fallback is in use |
| NLU | rule-based; unit-tested on scripted conversations. Real transcripts are messier, so expect missed or extra tasks. |

**Not tested here:** real Whisper transcription (the build sandbox had no internet). The ASR code and every ASR-dependent step are covered with an injected fake ASR in tests; first real run needs your Groq key or a model download. Camera video is only previewed/recorded locally; analysis uses the audio.

## Development

```bash
.venv/bin/python -m pytest backend/tests -q        # 120 tests, ~20 s, no network needed
cd frontend && npm install && npm run dev           # hot reload on :5173 (proxies to :8000); npm run build refreshes dist/
```

Key files: `backend/app/main.py` (API), `services/analysis_pipeline.py` (upload + final pass), `services/live_session.py` (live), `services/scene_classifier.py`, `algorithms/diarization.py`, `services/nlu.py`. Retrain the scene head with `backend/scripts/train_scene_head.py` (needs ESC-50).

Data lives in `backend/data/soniccaption.db`. Sound-class data: ESC-50 (CC BY-NC 4.0). Speaker encoder: Resemblyzer (Apache-2.0).
