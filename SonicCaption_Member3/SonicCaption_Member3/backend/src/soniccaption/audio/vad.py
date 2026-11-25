"""Voice activity detection (Unit 3: speech detection) using the pretrained Silero VAD model.

VAD answers *where* someone is speaking; ASR answers *what* they said.
Silero is a small neural network that scores every 32 ms frame (512 samples at 16 kHz) with a
speech probability; frames above `threshold` are grouped into segments, short gaps are bridged
and very short bursts are dropped.

Contract:
    detect_speech(audio, sample_rate) -> [{"start": float, "end": float}, ...]   (seconds, sorted)
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

VAD_SAMPLE_RATE = 16000          # Silero supports 8 kHz and 16 kHz; the project standard is 16 kHz
_MIN_SAMPLES = 512               # one Silero frame


@lru_cache(maxsize=1)
def _load_model():
    """Load Silero VAD once (the weights ship inside the `silero-vad` pip package, no download)."""
    from silero_vad import load_silero_vad
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return load_silero_vad()


def _prepare(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:                                   # (channels, samples) or (samples, channels)
        audio = audio.mean(axis=0 if audio.shape[0] < audio.shape[1] else 1)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio, got shape {audio.shape}.")
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate {sample_rate}.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite values.")
    if sample_rate != VAD_SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=VAD_SAMPLE_RATE)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:                                        # int16-style input -> [-1, 1]
        audio = audio / peak
    return audio.astype(np.float32)


def detect_speech(audio: np.ndarray, sample_rate: int, threshold: float = 0.5,
                  min_speech_ms: int = 250, min_silence_ms: int = 100, pad_ms: int = 30) -> list[dict]:
    """Return speech segments in seconds, sorted by start time.

    threshold       speech probability needed to count a frame as speech (higher = stricter)
    min_speech_ms   drop speech bursts shorter than this
    min_silence_ms  gaps shorter than this do not split a segment
    pad_ms          padding added to both ends of every segment
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    x = _prepare(audio, sample_rate)
    if x.size < _MIN_SAMPLES or np.max(np.abs(x)) < 1e-4:   # too short or digital silence
        return []

    import torch
    from silero_vad import get_speech_timestamps
    stamps = get_speech_timestamps(torch.from_numpy(x), _load_model(), threshold=threshold,
                                   sampling_rate=VAD_SAMPLE_RATE, min_speech_duration_ms=min_speech_ms,
                                   min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms,
                                   return_seconds=False)
    duration = x.size / VAD_SAMPLE_RATE
    segments = [{"start": round(max(0.0, s["start"] / VAD_SAMPLE_RATE), 3),
                 "end": round(min(duration, s["end"] / VAD_SAMPLE_RATE), 3)} for s in stamps]
    return sorted((s for s in segments if s["end"] > s["start"]), key=lambda s: s["start"])


def speech_ratio(segments: list[dict], duration: float) -> float:
    """Fraction of the clip that contains speech (the `vad_speech_ratio` metric)."""
    if duration <= 0:
        return 0.0
    return round(min(1.0, sum(s["end"] - s["start"] for s in segments) / duration), 3)
