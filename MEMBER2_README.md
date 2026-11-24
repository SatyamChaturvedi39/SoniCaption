# Member 2 — Speech Recognition, Sequence Modelling & Timeline Fusion

Copy these folders into the repo root. Paths already match the repo layout and nothing here overwrites a Member 1 file.

| File | What it does |
|---|---|
| `backend/src/soniccaption/audio/vad.py` | Silero VAD → `[{"start", "end"}]` speech segments |
| `backend/src/soniccaption/services/asr_service.py` | Whisper via Groq (`GROQ_API_KEY`), local faster-whisper fallback |
| `backend/src/soniccaption/services/sound_cnn.py` | Sliding-window sound events with Member 1's ResNet (default) or the CNN backup |
| `backend/src/soniccaption/algorithms/hmm.py` | Discrete HMM, forward, backward, Viterbi, presets, `decode_for_ui()` |
| `backend/src/soniccaption/timeline/fusion.py` | Speech + sound events → one chronological caption timeline |
| `backend/tests/test_{vad,asr_service,model_inference,hmm,fusion}.py` | 62 tests; ASR tests mock the API, so no key or network is needed |
| `notebooks/03_speech_recognition_and_sequence_fusion.ipynb` | Member 2 notebook: VAD, Whisper + WER/CER, sound events, HMM/Viterbi trellis, fusion, `.srt` |
| `notebooks/02b_sound_classifier_cnn_5fold_backup.ipynb` | Backup classifier: 10 classes, 5-fold CV, accuracy 0.843, macro-F1 0.842 |

## Run the tests
```
pip install silero-vad faster-whisper torchvision torchaudio librosa soundfile requests huggingface_hub pytest
cd backend && pytest -q
```

## For Member 3: the `/api/v1/audio/analyze` pipeline
```python
from soniccaption.audio.vad import detect_speech, speech_ratio
from soniccaption.services.asr_service import transcribe_audio
from soniccaption.services.sound_cnn import detect_sound_events
from soniccaption.timeline.fusion import fuse_timeline, FusionConfig

vad = detect_speech(samples, 16000)
asr = transcribe_audio(path)                                  # {"text", "segments", "provider", "language", "warnings"}
events = detect_sound_events(samples, 16000, threshold=0.0)   # raw; fusion applies the thresholds
fused = fuse_timeline(asr["segments"], events, vad_segments=vad,
                      config=FusionConfig(duration=len(samples) / 16000), speech_source=asr["provider"])
fused["timeline"]   # [{"id","start","end","category","label","display_text","confidence","source","overlaps_speech","words"}]
fused["dropped"]    # every filtered item with its reason
```
Use `decode_for_ui(model, observations)` from `algorithms/hmm.py` for the HMM Lab (`/labs/hmm/viterbi`).

## Notes for Member 1
- The class name `'laughter'` in notebook 02 is not an ESC-50 category (the dataset uses `'laughing'`). That class received no training clips, so the model cannot detect laughter, and macro-F1 0.837 is roughly 0.93 × 9/10. Changing it to `'laughing'` and retraining fixes this.
- Fold 5 was used both to choose the best epoch and to report the score. The 5-fold backup notebook keeps validation and test folds separate.
