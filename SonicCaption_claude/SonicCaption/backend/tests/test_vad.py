from pathlib import Path

import librosa
import numpy as np
import pytest

from soniccaption.audio.vad import detect_speech, speech_ratio

SPEECH = Path(__file__).parent / "data" / "speech_sample.ogg"   # LibriSpeech excerpt (CC BY 4.0), 14.8 s


@pytest.fixture(scope="module")
def speech():
    y, sr = librosa.load(SPEECH, sr=16000, mono=True)
    return y, sr


def test_detects_speech_in_real_recording(speech):
    y, sr = speech
    segs = detect_speech(y, sr)
    assert len(segs) >= 1
    assert speech_ratio(segs, len(y) / sr) > 0.7                 # read speech is mostly voiced


def test_segments_are_sorted_non_overlapping_and_in_range(speech):
    y, sr = speech
    segs = detect_speech(y, sr)
    dur = len(y) / sr
    for s in segs:
        assert 0 <= s["start"] < s["end"] <= dur + 1e-6
    for a, b in zip(segs, segs[1:]):
        assert a["end"] <= b["start"]


def test_silence_inserted_between_speech_is_found(speech):
    y, sr = speech
    gap = 3.0
    x = np.concatenate([y[:4 * sr], np.zeros(int(gap * sr), np.float32), y[4 * sr:8 * sr]])
    segs = detect_speech(x, sr)
    assert not any(s["start"] < 5.0 and s["end"] > 6.5 for s in segs)   # nothing spans the middle of the gap
    assert any(s["start"] >= 6.5 for s in segs)                           # speech after the gap is found


def test_pure_silence_and_tiny_input_return_empty():
    assert detect_speech(np.zeros(16000 * 3, np.float32), 16000) == []
    assert detect_speech(np.zeros(100, np.float32), 16000) == []
    assert detect_speech(np.array([], np.float32), 16000) == []


def test_tone_is_not_speech():
    t = np.arange(16000 * 3) / 16000
    assert speech_ratio(detect_speech(0.3 * np.sin(2 * np.pi * 440 * t), 16000), 3.0) < 0.2


def test_other_sample_rates_and_stereo(speech):
    y, sr = speech
    y44 = librosa.resample(y, orig_sr=sr, target_sr=44100)
    stereo = np.stack([y44, y44])
    a, b = detect_speech(y, sr), detect_speech(stereo, 44100)
    assert len(b) >= 1
    assert abs(speech_ratio(a, len(y) / sr) - speech_ratio(b, len(y) / sr)) < 0.1


def test_int16_scaled_input(speech):
    y, sr = speech
    assert len(detect_speech(y * 32767, sr)) >= 1


def test_invalid_input():
    with pytest.raises(ValueError):
        detect_speech(np.array([np.nan] * 1000), 16000)
    with pytest.raises(ValueError):
        detect_speech(np.zeros(1000), 16000, threshold=1.5)


def test_speech_ratio():
    assert speech_ratio([{"start": 0, "end": 2}, {"start": 3, "end": 4}], 10) == 0.3
    assert speech_ratio([], 0) == 0.0
