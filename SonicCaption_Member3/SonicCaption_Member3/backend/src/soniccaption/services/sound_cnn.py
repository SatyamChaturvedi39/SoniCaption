"""
SonicCaption Acoustic Event Classifier Service.
Loads Member 1's trained SonicAcousticResNet-18 weights on ESC-50
and performs windowed inference on audio signals.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchaudio.transforms as T


class SonicAcousticResNet(nn.Module):
    """Environmental sound classifier leveraging pretrained ResNet-18 adapted for 1-channel log-Mel spectrograms."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = models.resnet18(weights=None)

        orig_conv1 = base.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(base.fc.in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AcousticClassifierService:
    """Production service for sliding-window environmental sound inference."""

    def __init__(self, weights_dir: Optional[str] = None):
        if weights_dir is None:
            # Locate weights directory relative to this file
            current_dir = Path(__file__).resolve().parent
            weights_dir = current_dir / "weights"

        self.weights_dir = Path(weights_dir)
        self.model_path = self.weights_dir / "soniccaption_esc50_resnet.pt"
        if not self.model_path.exists():
            self.model_path = self.weights_dir / "soniccaption_esc50_cnn.pt"

        self.labels_path = self.weights_dir / "labels.json"
        self.config_path = self.weights_dir / "config.json"

        # Load labels
        if self.labels_path.exists():
            with open(self.labels_path, "r", encoding="utf-8") as f:
                labels_raw = json.load(f)
                self.idx_to_label = {int(k): v for k, v in labels_raw.items()}
        else:
            self.idx_to_label = {
                0: "clapping", 1: "coughing", 2: "dog", 3: "door_wood_knock",
                4: "footsteps", 5: "glass_breaking", 6: "laughter", 7: "rain",
                8: "siren", 9: "sneezing"
            }

        # Friendly display names
        self.display_names = {
            "clapping": "Clapping / Applause",
            "coughing": "Coughing",
            "dog": "Dog Barking",
            "door_wood_knock": "Door Knock",
            "footsteps": "Footsteps",
            "glass_breaking": "Glass Breaking",
            "laughter": "Laughter",
            "rain": "Rain",
            "siren": "Emergency Siren",
            "sneezing": "Sneezing"
        }

        # Initialize model
        self.device = torch.device("cpu")
        self.model = SonicAcousticResNet(num_classes=len(self.idx_to_label))
        if self.model_path.exists():
            state = torch.load(str(self.model_path), map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

        # Spectrogram transforms matching Notebook 02
        self.sample_rate = 16000
        self.clip_samples = self.sample_rate * 5  # 5.0 seconds = 80,000 samples
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=128
        )
        self.amp_to_db = T.AmplitudeToDB()

    def classify_window(self, samples: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Classifies a single 5.0s audio window."""
        target_len = self.clip_samples
        if len(samples) < target_len:
            # Pad with zeros
            padded = np.pad(samples, (0, target_len - len(samples)), mode="constant")
        else:
            padded = samples[:target_len]

        wav_tensor = torch.from_numpy(padded).float().unsqueeze(0)  # Shape: (1, 80000)
        with torch.no_grad():
            mel = self.amp_to_db(self.mel_transform(wav_tensor))  # Shape: (1, 128, 157)
            mel = mel.unsqueeze(0)  # Shape: (1, 1, 128, 157)
            logits = self.model(mel)
            probs = torch.softmax(logits, dim=-1)[0]

        pred_idx = int(torch.argmax(probs).item())
        pred_label = self.idx_to_label.get(pred_idx, "unknown")
        confidence = float(probs[pred_idx].item())

        all_probs = {self.idx_to_label[i]: float(probs[i].item()) for i in range(len(self.idx_to_label))}
        return pred_label, confidence, all_probs

    def analyze_audio(
        self,
        samples: np.ndarray,
        sr: int = 16000,
        min_confidence: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Runs sliding-window inference across the audio signal.
        Emits environmental sound events with timestamps.
        """
        if len(samples) == 0:
            return []

        duration = len(samples) / float(sr)
        events: List[Dict[str, Any]] = []

        window_sec = 5.0
        hop_sec = 2.5
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)

        if duration <= window_sec:
            label, conf, _ = self.classify_window(samples)
            if conf >= min_confidence:
                events.append({
                    "start": 0.0,
                    "end": round(duration, 2),
                    "label": self.display_names.get(label, label.title()),
                    "raw_class": label,
                    "confidence": round(conf, 3),
                    "model_source": "SonicAcousticResNet-18"
                })
            return events

        # Sliding window for longer files
        t_start = 0.0
        idx = 0
        while idx < len(samples):
            w_samples = samples[idx: idx + window_samples]
            if len(w_samples) < int(0.5 * sr):
                break  # Skip tiny trailing stub
            t_end = min(duration, t_start + len(w_samples) / float(sr))

            label, conf, _ = self.classify_window(w_samples)
            if conf >= min_confidence:
                events.append({
                    "start": round(t_start, 2),
                    "end": round(t_end, 2),
                    "label": self.display_names.get(label, label.title()),
                    "raw_class": label,
                    "confidence": round(conf, 3),
                    "model_source": "SonicAcousticResNet-18"
                })

            idx += hop_samples
            t_start += hop_sec

        # Deduplicate / merge adjacent windows predicting the same sound class
        merged: List[Dict[str, Any]] = []
        for ev in events:
            if merged and merged[-1]["raw_class"] == ev["raw_class"] and ev["start"] <= merged[-1]["end"] + 1.0:
                merged[-1]["end"] = max(merged[-1]["end"], ev["end"])
                merged[-1]["confidence"] = max(merged[-1]["confidence"], ev["confidence"])
            else:
                merged.append(ev)

        return merged


# Singleton instance
_service: Optional[AcousticClassifierService] = None

def get_classifier() -> AcousticClassifierService:
    global _service
    if _service is None:
        _service = AcousticClassifierService()
    return _service
