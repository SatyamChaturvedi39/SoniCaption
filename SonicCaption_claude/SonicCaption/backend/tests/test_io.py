import numpy as np
import pytest
from soniccaption.audio.io import generate_synthetic_signal, load_audio, save_audio
from soniccaption.audio.preprocessing import pre_emphasis, de_emphasis, frame_signal, get_window


def test_synthetic_signals():
    """Verify synthetic benchmark signal generation."""
    for sig_type in ["sine", "chirp", "fricative", "vowel_ah", "vowel_ee", "silence"]:
        audio = generate_synthetic_signal(sig_type, duration=0.5, sr=16000)
        assert audio.sample_rate == 16000
        assert len(audio.samples) == 8000
        assert audio.duration == pytest.approx(0.5, abs=1e-3)
        assert np.all(np.isfinite(audio.samples))
        assert np.max(np.abs(audio.samples)) <= 1.0


def test_silence_handling():
    """Verify silence handling produces all zeros without error."""
    audio = generate_synthetic_signal("silence", duration=0.2, sr=16000)
    assert np.all(audio.samples == 0.0)


def test_pre_emphasis_and_de_emphasis():
    """Verify pre-emphasis and its inverse de-emphasis reconstruct signal."""
    audio = generate_synthetic_signal("sine", duration=0.1, sr=16000)
    orig = audio.samples

    emp = pre_emphasis(orig, alpha=0.97)
    assert len(emp) == len(orig)
    assert np.all(np.isfinite(emp))

    rec = de_emphasis(emp, alpha=0.97)
    # Reconstructed signal should match original very closely
    np.testing.assert_allclose(rec, orig, rtol=1e-4, atol=1e-4)


def test_frame_signal():
    """Verify frame segmentation and windowing properties."""
    audio = generate_synthetic_signal("sine", duration=0.1, sr=16000)
    frames, window = frame_signal(audio.samples, frame_length=400, hop_length=160, window_type="hamming")

    assert frames.ndim == 2
    assert frames.shape[1] == 400
    assert len(window) == 400
    assert np.all(np.isfinite(frames))
