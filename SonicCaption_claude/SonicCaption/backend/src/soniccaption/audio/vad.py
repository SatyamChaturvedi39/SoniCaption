"""Voice activity detection (Unit 3: speech detection) using the pretrained Silero VAD model.

VAD answers *where* someone is speaking; ASR answers *what* they said.
Silero scores every 32 ms frame (512 samples at 16 kHz) with a speech probability. Frames are grouped into
segments with hysteresis (a higher "on" than "off" threshold), short gaps are bridged, short bursts dropped.

Both the batch path (`detect_speech`, used for uploads) and the streaming path (`StreamingVAD`, used for the
live tab) share the same frame scorer and the same segmentation rules, so a file analysed live and the same file
uploaded give the same speech regions.

Integration fixes over the first version (Member 2):
  * quiet speakers are no longer discarded: audio is level-normalised before scoring instead of using a hard
    absolute RMS floor (a talker 12 dB below "normal" used to be invisible);
  * the high-pass filter is vectorised (`scipy.signal.lfilter`) instead of a per-sample Python loop;
  * each call owns its own Silero instance, so concurrent uploads and live sessions never share recurrent state.

Contract (unchanged):
    detect_speech(audio, sample_rate) -> [{"start": float, "end": float}, ...]   (seconds, sorted)
"""
from __future__ import annotations

import warnings

import numpy as np

VAD_SAMPLE_RATE = 16000          # Silero supports 8 kHz and 16 kHz; the project standard is 16 kHz
FRAME = 512                      # one Silero frame (32 ms)
_MIN_SAMPLES = FRAME
FRAME_S = FRAME / VAD_SAMPLE_RATE


def _new_model():
    """Silero weights ship inside the `silero-vad` pip package (no download); loading takes ~60 ms."""
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


def highpass_speech_band(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset and sub-70 Hz AC/fan motor rumble (first-order IIR, alpha 0.97 -> fc ~ 77 Hz)."""
    if audio.size < 4:
        return audio
    from scipy.signal import lfilter
    x = audio - float(np.mean(audio))
    return lfilter([1.0, -1.0], [1.0, -0.97], x).astype(np.float32)


_highpass_speech_band = highpass_speech_band          # backwards-compatible private alias


def level_gain(x: np.ndarray, target_rms: float = 0.05, max_gain: float = 10.0) -> float:
    """Gain (>= 1) that lifts quiet recordings so Silero sees a normal speech level.

    Based on the 90th percentile of 25 ms frame RMS, i.e. the level of the *loud* parts, so steady
    background noise is not what sets the gain. Never attenuates and never boosts more than +20 dB.
    """
    if x.size < 400:
        return 1.0
    n = (x.size // 400) * 400
    frame_rms = np.sqrt(np.mean(x[:n].reshape(-1, 400) ** 2, axis=1))
    loud = float(np.percentile(frame_rms, 90))
    if loud < 1e-5:
        return 1.0
    return float(np.clip(target_rms / loud, 1.0, max_gain))


def speech_probabilities(audio: np.ndarray, sample_rate: int = VAD_SAMPLE_RATE) -> np.ndarray:
    """Per-frame speech probability, one value per 32 ms. Returns an empty array for too-short/silent input."""
    import torch
    x = _prepare(audio, sample_rate)
    if x.size < _MIN_SAMPLES or np.max(np.abs(x)) < 1e-4:
        return np.zeros(0, dtype=np.float32)
    x = highpass_speech_band(x)
    x = np.clip(x * level_gain(x), -1.0, 1.0)
    model = _new_model()
    model.reset_states()
    n = x.size // FRAME
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            out[i] = float(model(torch.from_numpy(x[i * FRAME:(i + 1) * FRAME]), VAD_SAMPLE_RATE))
    return out


def segments_from_probs(probs: np.ndarray, threshold: float = 0.5, neg_threshold: float | None = None,
                        min_speech_ms: int = 250, min_silence_ms: int = 100, pad_ms: int = 30,
                        duration: float | None = None) -> list[dict]:
    """Hysteresis segmentation of a probability track into [{"start","end"}] (seconds)."""
    if probs.size == 0:
        return []
    off = neg_threshold if neg_threshold is not None else max(threshold - 0.15, 0.01)
    min_sp = max(1, int(round(min_speech_ms / 1000 / FRAME_S)))
    min_sil = max(1, int(round(min_silence_ms / 1000 / FRAME_S)))
    raw: list[list[int]] = []
    start, last_voiced, active = 0, 0, False
    for i, p in enumerate(probs):
        if p >= threshold:
            if not active:
                start, active = i, True
            last_voiced = i
        elif active and p < off and i - last_voiced >= min_sil:
            raw.append([start, last_voiced + 1])
            active = False
    if active:
        raw.append([start, last_voiced + 1])
    pad = int(round(pad_ms / 1000 / FRAME_S))
    out: list[dict] = []
    total = duration if duration is not None else probs.size * FRAME_S
    for a, b in raw:
        if b - a < min_sp:
            continue
        st = max(0.0, (a - pad) * FRAME_S)
        en = min(total, (b + pad) * FRAME_S)
        if out and st <= out[-1]["end"]:
            out[-1]["end"] = round(en, 3)
        else:
            out.append({"start": round(st, 3), "end": round(en, 3)})
    return out


def detect_speech(audio: np.ndarray, sample_rate: int, threshold: float = 0.5,
                  min_speech_ms: int = 250, min_silence_ms: int = 100, pad_ms: int = 30,
                  min_segment_rms: float = 0.002) -> list[dict]:
    """Return speech segments in seconds, sorted by start time.

    threshold       speech probability needed to count a frame as speech (higher = stricter)
    min_speech_ms   drop speech bursts shorter than this
    min_silence_ms  gaps shorter than this do not split a segment
    pad_ms          padding added to both ends of every segment
    min_segment_rms drop segments whose *original* RMS is below this (-54 dBFS): digital-silence guard only
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    x = _prepare(audio, sample_rate)
    if x.size < _MIN_SAMPLES or np.max(np.abs(x)) < 1e-4:   # too short or digital silence
        return []
    x_clean = highpass_speech_band(x)
    if float(np.sqrt(np.mean(x_clean ** 2))) < min_segment_rms * 0.65:
        return []
    probs = speech_probabilities(x, VAD_SAMPLE_RATE)
    duration = x.size / VAD_SAMPLE_RATE
    segs = segments_from_probs(probs, threshold=threshold, min_speech_ms=min_speech_ms,
                               min_silence_ms=min_silence_ms, pad_ms=pad_ms, duration=duration)
    keep = []
    for s in segs:
        i0, i1 = int(s["start"] * VAD_SAMPLE_RATE), int(s["end"] * VAD_SAMPLE_RATE)
        if i1 > i0 and float(np.sqrt(np.mean(x_clean[i0:i1] ** 2))) >= min_segment_rms:
            keep.append(s)
    return keep


def speech_ratio(segments: list[dict], duration: float) -> float:
    """Fraction of the clip that contains speech (the `vad_speech_ratio` metric)."""
    if duration <= 0:
        return 0.0
    return round(min(1.0, sum(s["end"] - s["start"] for s in segments) / duration), 3)


class StreamingVAD:
    """Incremental VAD for the live tab. Feed arbitrary-sized 16 kHz float chunks; read frame probabilities.

    Level normalisation in streaming mode uses a slowly adapting estimate of the loud-signal level (running
    peak-ish RMS with decay) because the future is unknown.
    """

    def __init__(self) -> None:
        self.model = _new_model()
        self.model.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)
        self._hp_state = np.zeros(1, dtype=np.float64)
        self._loud = 0.0
        self.probs: list[float] = []

    def feed(self, chunk: np.ndarray) -> list[float]:
        import torch
        from scipy.signal import lfilter
        c = np.asarray(chunk, dtype=np.float32)
        if c.size == 0:
            return []
        y, self._hp_state = lfilter([1.0, -1.0], [1.0, -0.97], c, zi=self._hp_state)
        self._buf = np.concatenate([self._buf, y.astype(np.float32)])
        new: list[float] = []
        with torch.no_grad():
            while self._buf.size >= FRAME:
                fr, self._buf = self._buf[:FRAME], self._buf[FRAME:]
                rms = float(np.sqrt(np.mean(fr ** 2)))
                self._loud = max(rms, self._loud * 0.9995)            # slow decay, instant attack
                gain = float(np.clip(0.05 / self._loud, 1.0, 10.0)) if self._loud > 1e-5 else 1.0
                p = float(self.model(torch.from_numpy(np.clip(fr * gain, -1, 1)), VAD_SAMPLE_RATE))
                new.append(p)
        self.probs.extend(new)
        return new
