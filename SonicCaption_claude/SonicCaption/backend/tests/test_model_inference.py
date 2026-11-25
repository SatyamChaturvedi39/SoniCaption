"""Sound-model inference tests.

Small randomly initialised models are saved in both export formats (Member 1's ResNet and the CNN backup)
so loading, preprocessing and the event contract are checked without large downloads. If the team ResNet
weights are present in services/weights/, a real-weights smoke test also runs.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from soniccaption.services import sound_cnn as sc

LABELS = ["clapping", "coughing", "dog", "door_wood_knock", "footsteps",
          "glass_breaking", "laughing", "rain", "siren", "sneezing"]


@pytest.fixture(scope="module")
def resnet_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("resnet")
    torch.manual_seed(0)
    torch.save(sc.SonicAcousticResNet(10).state_dict(), d / "soniccaption_esc50_resnet.pt")
    (d / "config.json").write_text(json.dumps({"architecture": "SonicAcousticResNet18", "sample_rate": 16000,
                                               "n_fft": 1024, "hop_length": 512, "win_length": 1024, "n_mels": 128}))
    (d / "labels.json").write_text(json.dumps({str(i): c for i, c in enumerate(LABELS)}))
    return d


@pytest.fixture(scope="module")
def cnn_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cnn")
    torch.manual_seed(0)
    torch.save(sc.SonicCNN(10).state_dict(), d / "model.pt")
    (d / "config.json").write_text(json.dumps({
        "model_name": "soniccaption-esc50-cnn",
        "architecture": {"type": "SonicCNN", "channels": [32, 64, 128, 256], "dropout": 0.3, "n_classes": 10},
        "sample_rate": 16000, "n_fft": 400, "hop_length": 160, "n_mels": 64, "fmin": 20, "fmax": 8000,
        "power": 2.0, "db_ref": 1.0, "normalization": {"mean": -50.0, "std": 30.0}}))
    (d / "labels.json").write_text(json.dumps({"id2label": {str(i): c for i, c in enumerate(LABELS)}}))
    return d


def noise(seconds, sr=16000, amp=0.1):
    return (amp * np.random.default_rng(1).standard_normal(int(seconds * sr))).astype(np.float32)


@pytest.mark.parametrize("which", ["resnet_dir", "cnn_dir"])
def test_both_export_formats_load_and_classify(which, request):
    m = sc.load_sound_model(request.getfixturevalue(which))
    assert m.kind == ("resnet" if which == "resnet_dir" else "cnn") and m.labels == LABELS
    out = sc.classify_audio_window(noise(1.0), 16000, m)
    scores = [o["score"] for o in out]
    assert sorted(o["label"] for o in out) == sorted(LABELS)
    assert scores == sorted(scores, reverse=True) and sum(scores) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("which", ["resnet_dir", "cnn_dir"])
def test_short_long_stereo_and_resampled_windows(which, request):
    m = sc.load_sound_model(request.getfixturevalue(which))
    for audio, sr in [(noise(0.05), 16000), (noise(5.0), 16000), (noise(1.0, 44100), 44100),
                      (np.stack([noise(1.0), noise(1.0)]), 16000)]:
        assert len(sc.classify_audio_window(audio, sr, m)) == 10


def test_events_contract(resnet_dir):
    m = sc.load_sound_model(resnet_dir)
    events = sc.detect_sound_events(noise(6.0), 16000, threshold=0.0, model=m)
    assert events
    for e in events:
        assert set(e) == {"start", "end", "label", "confidence", "model"}
        assert e["label"] in LABELS and 0 <= e["start"] < e["end"] <= 6.0 and 0 <= e["confidence"] <= 1
    assert [e["start"] for e in events] == sorted(e["start"] for e in events)


def test_silence_threshold_and_empty(resnet_dir):
    m = sc.load_sound_model(resnet_dir)
    assert sc.detect_sound_events(np.zeros(16000 * 3, np.float32), 16000, model=m) == []
    assert sc.detect_sound_events(np.array([], np.float32), 16000, model=m) == []
    assert sc.detect_sound_events(noise(3.0), 16000, threshold=1.01, model=m) == []


def test_missing_model_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        sc.load_sound_model(tmp_path)


TEAM_WEIGHTS = sc.DEFAULT_MODEL_DIR / "soniccaption_esc50_resnet.pt"


@pytest.mark.skipif(not TEAM_WEIGHTS.exists(), reason="team ResNet weights not present")
def test_team_resnet_weights_load():
    m = sc.load_sound_model()
    assert m.kind == "resnet" and len(m.labels) == 10
    assert len(sc.classify_audio_window(noise(2.0), 16000, m)) == 10
