"""Pretrained speaker (voice) encoder: 256-d d-vectors that are close for the same voice and far for different voices.

The network (3-layer LSTM + linear, GE2E-trained) and its weights come from the open-source Resemblyzer project
(https://github.com/resemble-ai/Resemblyzer, Apache License 2.0, see licenses/Resemblyzer-LICENSE-Apache-2.0.txt).
Only the inference path is vendored here so the project has no webrtcvad/C-compiler dependency; weights live in
weights/speaker_encoder.pt (inference weights only, 5.7 MB).

    embed_utterances([audio16k, ...]) -> (n, 256) float32, each row L2-normalised
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 16000
PARTIAL_FRAMES = 160                      # 1.6 s partial utterances, as in the original
WEIGHTS = Path(__file__).resolve().parent / "weights" / "speaker_encoder.pt"
MIN_SAMPLES = int(0.4 * SR)


def _build():
    import torch.nn as nn

    class VoiceEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(40, 256, 3, batch_first=True)
            self.linear = nn.Linear(256, 256)
            self.relu = nn.ReLU()

        def forward(self, mels):
            import torch
            _, (hidden, _) = self.lstm(mels)
            e = self.relu(self.linear(hidden[-1]))
            return e / torch.norm(e, dim=1, keepdim=True)

    return VoiceEncoder()


@lru_cache(maxsize=1)
def load_encoder():
    import torch
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Speaker encoder weights missing: {WEIGHTS}")
    net = _build()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return net.eval()


def available() -> bool:
    return WEIGHTS.exists()


def _mel(wav: np.ndarray) -> np.ndarray:
    import librosa
    return librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=400, hop_length=160, n_mels=40).astype(np.float32).T


def _normalise(wav: np.ndarray, target_dbfs: float = -30.0) -> np.ndarray:
    ms = float(np.mean(wav ** 2))
    if ms < 1e-12:
        return wav
    gain_db = target_dbfs - 10 * np.log10(ms)
    return wav * float(10 ** (max(gain_db, 0.0) / 20))        # only ever boosts (as the original does)


def embed_utterances(wavs: list[np.ndarray]) -> np.ndarray:
    import torch
    net = load_encoder()
    out = []
    for w in wavs:
        w = _normalise(np.asarray(w, dtype=np.float32))
        if w.size < MIN_SAMPLES:
            w = np.pad(w, (0, MIN_SAMPLES - w.size))
        m = _mel(w)
        if len(m) < PARTIAL_FRAMES:
            m = np.pad(m, ((0, PARTIAL_FRAMES - len(m)), (0, 0)))
        starts = list(range(0, len(m) - PARTIAL_FRAMES + 1, 80))
        if starts[-1] + PARTIAL_FRAMES < len(m) and len(m) - (starts[-1] + PARTIAL_FRAMES) > 40:
            starts.append(len(m) - PARTIAL_FRAMES)
        batch = torch.from_numpy(np.stack([m[s:s + PARTIAL_FRAMES] for s in starts]))
        with torch.no_grad():
            e = net(batch).numpy().mean(axis=0)
        out.append(e / (np.linalg.norm(e) + 1e-9))
    return np.stack(out).astype(np.float32) if out else np.zeros((0, 256), np.float32)
