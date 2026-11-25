"""Sound-event detection service: sliding-window inference with the team's environmental sound classifier.

Supports both trained models the team has, chosen by the `architecture` field in their config.json:

    SonicAcousticResNet18   Member 1's fine-tuned ResNet-18 (notebook 02, weights in services/weights/)
                            torchaudio log-Mel: 128 bands, n_fft 1024, hop 512, AmplitudeToDB, no normalisation
    SonicCNN                Compact 4-block CNN (Hugging Face: hudhud26/soniccaption-esc50-cnn), 5-fold CV backup
                            librosa log-Mel: 64 bands, n_fft 400, hop 160, power->dB, mean/std normalisation

Preprocessing always follows the loaded model's own config, so inference matches training.

Contract:
    load_sound_model(model_dir=None)                -> SoundModel (cached)
    classify_audio_window(audio, sample_rate)       -> [{"label": str, "score": float}, ...]   ranked
    detect_sound_events(audio, sample_rate)         -> [{"start", "end", "label", "confidence", "model"}, ...]
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "weights"
HF_CNN_REPO = "hudhud26/soniccaption-esc50-cnn"
HF_CNN_REVISION = "2d8879b91e2c5dbcaa8c3e9862bedb01dafccf2c"


# =============================================================== architectures
class SonicAcousticResNet(nn.Module):
    """ResNet-18 with a 1-channel first conv and a dropout MLP head (same layout as notebook 02)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        from torchvision import models
        base = models.resnet18(weights=None)            # weights come from the checkpoint, no download
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1, self.relu, self.maxpool = base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(base.fc.in_features, 128),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.classifier(torch.flatten(self.avgpool(x), 1))


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self, x):
        return self.net(x)


class SonicCNN(nn.Module):
    """Compact 4-block CNN (same layout as the Hugging Face backup model)."""

    def __init__(self, n_classes: int, channels=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers, cin = [], 1
        for c in channels:
            layers.append(_ConvBlock(cin, c))
            cin = c
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        return self.fc(self.dropout(self.features(x).mean(dim=(2, 3))))


# =============================================================== loading
@dataclass
class SoundModel:
    net: nn.Module
    kind: str                 # "resnet" or "cnn"
    labels: list[str]
    sample_rate: int
    name: str
    config: dict
    _mel: object = None       # torchaudio transform (resnet only)

    def features(self, audio: np.ndarray) -> torch.Tensor:
        """audio (mono float32 at self.sample_rate) -> model input (1, 1, n_mels, frames)."""
        if self.kind == "resnet":
            import torchaudio.transforms as T
            if self._mel is None:
                c = self.config
                self._mel = (T.MelSpectrogram(sample_rate=c["sample_rate"], n_fft=c["n_fft"],
                                              win_length=c.get("win_length", c["n_fft"]),
                                              hop_length=c["hop_length"], n_mels=c["n_mels"]),
                             T.AmplitudeToDB())
            mel, to_db = self._mel
            return to_db(mel(torch.from_numpy(audio)[None]))[None]          # (1, 1, n_mels, frames)
        import librosa
        c = self.config
        S = librosa.feature.melspectrogram(y=audio, sr=c["sample_rate"], n_fft=c["n_fft"], hop_length=c["hop_length"],
                                           n_mels=c["n_mels"], fmin=c["fmin"], fmax=c["fmax"], power=c.get("power", 2.0))
        S = librosa.power_to_db(S, ref=c.get("db_ref", 1.0), top_db=None)
        S = (S - c["normalization"]["mean"]) / c["normalization"]["std"]
        return torch.tensor(S, dtype=torch.float32)[None, None]

    @property
    def min_samples(self) -> int:
        # enough frames for the network's downsampling (ResNet /32, CNN /16)
        return self.config["hop_length"] * (32 if self.kind == "resnet" else 16)


def _read_labels(path: Path) -> list[str]:
    raw = json.loads(path.read_text())
    mapping = raw.get("id2label", raw)                    # {"id2label": {...}} or flat {"0": "clapping", ...}
    return [mapping[str(i)] for i in range(len(mapping))]


def _resolve_dir(model_dir: str | Path | None, model_source: str | None = None) -> Path:
    if model_dir:
        return Path(model_dir)
    if os.environ.get("SOUND_MODEL_DIR"):
        return Path(os.environ["SOUND_MODEL_DIR"])
    src = (model_source or os.environ.get("SOUND_MODEL_SOURCE", "")).strip().lower()
    if src == "hf-cnn":
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(HF_CNN_REPO, revision=HF_CNN_REVISION,
                                      allow_patterns=["model.pt", "config.json", "labels.json"]))
    return DEFAULT_MODEL_DIR


@lru_cache(maxsize=4)
def _load(model_dir: str) -> SoundModel:
    d = Path(model_dir)
    if not (d / "config.json").exists():
        raise FileNotFoundError(f"No config.json in {d}. Expected the exported sound model folder.")
    cfg = json.loads((d / "config.json").read_text())
    labels = _read_labels(d / "labels.json")
    arch = cfg["architecture"]
    if isinstance(arch, str) and "resnet" in arch.lower():
        net, kind = SonicAcousticResNet(len(labels)), "resnet"
        weights = next((d / f for f in ("soniccaption_esc50_resnet.pt", "soniccaption_esc50_cnn.pt", "model.pt")
                        if (d / f).exists()), None)
        name = "soniccaption-esc50-resnet18"
    else:
        a = arch if isinstance(arch, dict) else {}
        net = SonicCNN(a.get("n_classes", len(labels)), tuple(a.get("channels", (32, 64, 128, 256))), a.get("dropout", 0.3))
        kind, weights, name = "cnn", d / "model.pt", cfg.get("model_name", "soniccaption-esc50-cnn")
    if weights is None or not Path(weights).exists():
        raise FileNotFoundError(f"No weight file found in {d}.")
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()
    return SoundModel(net=net, kind=kind, labels=labels, sample_rate=int(cfg["sample_rate"]), name=name, config=cfg)


def load_sound_model(model_dir: str | Path | None = None, model_source: str | None = None) -> SoundModel:
    """Load (and cache) the sound model. Default: Member 1's ResNet-18 in services/weights/; or 'hf-cnn' for Member 2's CNN."""
    return _load(str(_resolve_dir(model_dir, model_source=model_source).resolve()))


# =============================================================== inference
def _mono_resampled(audio: np.ndarray, sample_rate: int, target_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0 if audio.shape[0] < audio.shape[1] else 1)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio, got shape {audio.shape}.")
    if sample_rate != target_sr and audio.size:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    return np.ascontiguousarray(audio, dtype=np.float32)


def classify_audio_window(audio: np.ndarray, sample_rate: int, model: SoundModel | None = None) -> list[dict]:
    """Classify one window; returns every label with its softmax probability, highest first."""
    m = model or load_sound_model()
    x = _mono_resampled(audio, sample_rate, m.sample_rate)
    if x.size < m.min_samples:
        x = np.pad(x, (0, m.min_samples - x.size))
    with torch.no_grad():
        p = F.softmax(m.net(m.features(x)), dim=1)[0].numpy()
    return [{"label": m.labels[i], "score": round(float(p[i]), 4)} for i in np.argsort(-p)]


def _window_acoustics(seg: np.ndarray) -> tuple[float, float, float, float]:
    """Compute (rms, p95_frame_rms, dynamic_range_db, spectral_flatness) for ambient/fan filtering."""
    rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    if seg.size < 400:
        return rms, rms, 0.0, 1.0
    frame_len, hop = 400, 160
    n_frames = 1 + (seg.size - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + (np.arange(n_frames) * hop)[:, None]
    frames = seg[idx]
    f_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    p15 = float(np.percentile(f_rms, 15))
    p95 = float(np.percentile(f_rms, 95))
    dyn_db = float(20.0 * np.log10((p95 + 1e-9) / (p15 + 1e-9)))
    win = np.hanning(frame_len).astype(np.float32)
    spec_pow = np.abs(np.fft.rfft(frames * win[None, :], n=512, axis=1)) ** 2 + 1e-12
    spec_pow = spec_pow[:, 3:]  # ignore sub-90 Hz fan motor/AC rumble when measuring flatness
    geo_mean = np.exp(np.mean(np.log(spec_pow), axis=1))
    arith_mean = np.mean(spec_pow, axis=1)
    sfm = float(np.median(geo_mean / np.maximum(arith_mean, 1e-12)))
    return rms, p95, dyn_db, sfm


def _is_stationary_ambient_prediction(
    label: str,
    score: float,
    rms: float,
    p95_rms: float,
    dyn_db: float,
    sfm: float,
    clip_floor_rms: float,
    baseline_rms: float = 0.0,
) -> bool:
    """Reject closed-set CNN guesses triggered by quiet room noise or steady fan airflow."""
    rms_floor = max(0.022, 1.8 * baseline_rms)   # ~ -33.1 dB default; adapts upward to loud calibrated fans
    peak_floor = max(0.036, 2.5 * baseline_rms)  # ~ -28.9 dB foreground peak requirement
    if rms < rms_floor and p95_rms < peak_floor:
        return True
    eff_clip_floor = max(clip_floor_rms, baseline_rms, 1e-6)
    prom_db = float(20.0 * np.log10((p95_rms + 1e-9) / eff_clip_floor))
    if label == "rain":
        # Real rain is uniformly loud and steady (low dyn_db); reject speech-modulated
        # pause windows (dyn_db >= 5.5) or soft indoor fan/HVAC airflow.
        is_real_downpour = (rms >= max(0.070, 2.5 * baseline_rms)) and (score >= 0.85) and (dyn_db < 5.5)
        return not is_real_downpour
    if label == "siren":
        # Genuine sirens are strongly tonal (low spectral flatness), unlike broadband fan noise
        return sfm > 0.22 or p95_rms < max(0.028, 2.0 * baseline_rms)
    # Transient / modulated classes (laughing, coughing, clapping, glass_breaking, dog, etc.)
    is_transient = (dyn_db >= 5.2 or prom_db >= 5.2) and p95_rms >= max(0.030, 2.0 * baseline_rms)
    is_clear_tonal_burst = sfm < 0.14 and p95_rms >= max(0.032, 1.6 * eff_clip_floor)
    return not (is_transient or is_clear_tonal_burst)


def detect_sound_events(audio: np.ndarray, sample_rate: int, window_s: float = 2.0, hop_s: float = 0.5,
                        threshold: float = 0.5, silence_rms: float = 1e-3,
                        model: SoundModel | None = None, ignore_ambient: bool = False,
                        baseline_rms: float = 0.0, model_source: str | None = None) -> list[dict]:
    """Slide a window over the clip and return sound events (raw: merging/fusion happens in timeline.fusion).

    The default 2 s window (0.5 s hop) was chosen on the demo clip: 1 s windows gave too little context for a model
    trained on 5 s clips, while 3-5 s windows smeared events into their neighbours.

    Windows quieter than `silence_rms` are skipped because the model has no "silence" class, and predictions
    below `threshold` are dropped because a closed-set model gives *some* label to every input.
    When `ignore_ambient=True`, stationary room/fan noise windows are also filtered before labeling.
    Consecutive windows with the same label are joined. Timestamps are window-level approximations.
    """
    m = model or load_sound_model(model_source=model_source)
    sr = m.sample_rate
    x = _mono_resampled(audio, sample_rate, sr)
    if x.size == 0:
        return []
    W, H = int(window_s * sr), int(hop_s * sr)
    starts = list(range(0, max(x.size - W, 0) + 1, H))
    if x.size > W and starts[-1] + W < x.size:
        starts.append(x.size - W)                          # cover the tail

    acoustics = [_window_acoustics(x[s:s + W]) for s in starts]
    clip_floor_rms = float(np.percentile([a[1] for a in acoustics], 15)) if acoustics else 0.0
    eff_silence = max(silence_rms, baseline_rms * 1.15) if ignore_ambient and baseline_rms > 0 else silence_rms

    events: list[dict] = []
    for s, (rms, p95_rms, dyn_db, sfm) in zip(starts, acoustics):
        seg = x[s:s + W]
        if rms < eff_silence:
            continue
        top = classify_audio_window(seg, sr, m)[0]
        if top["score"] < threshold:
            continue
        if ignore_ambient and _is_stationary_ambient_prediction(
            top["label"], float(top["score"]), rms, p95_rms, dyn_db, sfm, clip_floor_rms, baseline_rms=baseline_rms
        ):
            continue
        st, en = round(s / sr, 2), round(min(s + W, x.size) / sr, 2)
        if events and events[-1]["label"] == top["label"] and st <= events[-1]["end"]:
            events[-1]["end"] = en
            events[-1]["confidence"] = max(events[-1]["confidence"], round(top["score"], 3))
        else:
            events.append({"start": st, "end": en, "label": top["label"],
                           "confidence": round(top["score"], 3), "model": m.name})
    return events