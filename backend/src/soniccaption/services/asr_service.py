"""Speech-to-text (ASR) adapter.

Whisper is a *pretrained* model (OpenAI); the team does not train it. Two providers sit behind one contract:

    GroqASRProvider          hosted Whisper via the Groq API (fast; needs GROQ_API_KEY; audio leaves the machine)
    FasterWhisperProvider    local Whisper on CPU via faster-whisper (offline fallback; slower)

Contract (from the implementation plan):
    transcribe_audio(audio_path, language=None) -> {
        "text": str,
        "segments": [{"start": float, "end": float, "text": str,
                      "words": [{"start": float, "end": float, "word": str}]}],
        "provider": str,            # "groq-whisper" or "faster-whisper"
        "language": str | None,
        "warnings": [str],          # e.g. why the fallback was used
    }

The UI and timeline must not depend on which provider ran.
"""
from __future__ import annotations

import io
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ASR_SAMPLE_RATE = 16000
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
LOCAL_MODEL = os.environ.get("FASTER_WHISPER_MODEL", "base")
# 16 kHz mono 16-bit WAV is 32 kB/s, so 10-minute chunks (~19 MB) stay under Groq's 25 MB free-tier limit.
GROQ_CHUNK_SECONDS = 600


class ASRError(RuntimeError):
    """Raised when no provider could transcribe the audio; the message is safe to show in the UI."""


# ----------------------------------------------------------------------------- helpers
def _load_audio(audio_path: str | Path) -> np.ndarray:
    path = Path(audio_path)
    if not path.is_file():
        raise ASRError(f"Audio file not found: {path.name}")
    import librosa
    try:
        y, _ = librosa.load(str(path), sr=ASR_SAMPLE_RATE, mono=True)
    except Exception as e:  # noqa: BLE001 - decoding errors come from several libraries
        raise ASRError(f"Could not decode {path.name}: {e}") from e
    return y.astype(np.float32)


def _wav_bytes(y: np.ndarray) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, y, ASR_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _attach_words(segments: list[dict], words: list[dict]) -> None:
    """Groq returns words as one flat list; give each segment the words whose midpoint falls inside it."""
    for seg in segments:
        seg["words"] = [w for w in words if seg["start"] <= (w["start"] + w["end"]) / 2 < seg["end"] + 1e-6]


def _clean(result: dict) -> dict:
    """Round times, strip text, drop empty segments and sort - identical output for every provider."""
    segs = []
    for s in result["segments"]:
        text = s["text"].strip()
        if not text:
            continue
        segs.append({"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3), "text": text,
                     "words": [{"start": round(float(w["start"]), 3), "end": round(float(w["end"]), 3),
                                "word": w["word"].strip()} for w in s.get("words", []) if w["word"].strip()]})
    segs.sort(key=lambda s: s["start"])
    return {"text": " ".join(s["text"] for s in segs), "segments": segs, "provider": result["provider"],
            "language": result.get("language"), "warnings": result.get("warnings", [])}


# --------------------------------------------------------------------------- providers
class GroqASRProvider:
    name = "groq-whisper"

    def __init__(self, api_key: str | None = None, model: str = GROQ_MODEL, timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ASRError("GROQ_API_KEY is not set.")
        self.model, self.timeout = model, timeout

    def _post(self, wav: bytes, language: str | None) -> dict:
        import requests
        data = [("model", self.model), ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"), ("timestamp_granularities[]", "word"),
                ("temperature", "0")]
        if language:
            data.append(("language", language))
        for attempt in range(2):                                     # one retry on rate limit / server error
            r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {self.api_key}"},
                              files={"file": ("audio.wav", wav, "audio/wav")}, data=data, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503) and attempt == 0:
                time.sleep(2)
                continue
            msg = r.json().get("error", {}).get("message", r.text[:200]) if "json" in r.headers.get("content-type", "") else r.text[:200]
            raise ASRError(f"Groq API error {r.status_code}: {msg}")
        raise ASRError("Groq API did not respond successfully.")

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> dict:
        y = _load_audio(audio_path)
        chunk = GROQ_CHUNK_SECONDS * ASR_SAMPLE_RATE
        segments, detected = [], None
        for offset in range(0, max(len(y), 1), chunk):            # long files are sent in chunks
            resp = self._post(_wav_bytes(y[offset:offset + chunk]), language)
            t0 = offset / ASR_SAMPLE_RATE
            segs = [{"start": s["start"] + t0, "end": s["end"] + t0, "text": s.get("text", "")}
                    for s in resp.get("segments", [])]
            words = [{"start": w["start"] + t0, "end": w["end"] + t0, "word": w.get("word", "")}
                     for w in resp.get("words", []) or []]
            if not segs and resp.get("text", "").strip():          # no segments returned: one segment for the chunk
                segs = [{"start": t0, "end": t0 + min(chunk, len(y) - offset) / ASR_SAMPLE_RATE, "text": resp["text"]}]
            _attach_words(segs, words)
            segments += segs
            detected = detected or resp.get("language")
        return {"segments": segments, "provider": self.name, "language": language or detected}


class FasterWhisperProvider:
    name = "faster-whisper"

    def __init__(self, model_size: str = LOCAL_MODEL):
        self.model_size = model_size

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> dict:
        y = _load_audio(audio_path)
        model = _load_faster_whisper(self.model_size)
        seg_iter, info = model.transcribe(y, language=language, word_timestamps=True, vad_filter=False, beam_size=5)
        segments = [{"start": s.start, "end": s.end, "text": s.text,
                     "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]}
                    for s in seg_iter]
        return {"segments": segments, "provider": f"{self.name}-{self.model_size}",
                "language": language or getattr(info, "language", None)}


@lru_cache(maxsize=2)
def _load_faster_whisper(model_size: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ASRError("faster-whisper is not installed (pip install faster-whisper).") from e
    return WhisperModel(model_size, device="cpu", compute_type="int8")


# ------------------------------------------------------------------------ public API
def transcribe_audio(audio_path: str | Path, language: str | None = None, provider: str = "auto") -> dict:
    """Transcribe an audio file.

    provider="auto"   Groq if GROQ_API_KEY is set, otherwise local; falls back to local if Groq fails
    provider="groq"   Groq only
    provider="local"  faster-whisper only
    """
    if provider not in ("auto", "groq", "local"):
        raise ValueError("provider must be 'auto', 'groq' or 'local'.")
    warnings_: list[str] = []
    if provider in ("auto", "groq") and (provider == "groq" or os.environ.get("GROQ_API_KEY")):
        try:
            return _clean(GroqASRProvider().transcribe(audio_path, language))
        except ASRError as e:
            if provider == "groq" or "not found" in str(e) or "Could not decode" in str(e):
                raise
            warnings_.append(f"Groq transcription failed ({e}); used local faster-whisper instead.")
    elif provider == "auto":
        warnings_.append("GROQ_API_KEY not set; used local faster-whisper.")
    try:
        result = FasterWhisperProvider().transcribe(audio_path, language)
    except ASRError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ASRError(f"Local transcription failed: {e}") from e
    result["warnings"] = warnings_
    return _clean(result)
