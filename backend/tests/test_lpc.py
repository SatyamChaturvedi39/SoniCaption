import numpy as np
import pytest
from soniccaption.audio.io import generate_synthetic_signal
from soniccaption.audio.preprocessing import frame_signal
from soniccaption.algorithms.lpc import (
    autocorrelate,
    levinson_durbin,
    extract_formants,
    analyze_frame_lpc
)


def test_levinson_durbin_order():
    """Verify Levinson-Durbin recursion output shapes and reflection coefficient limits."""
    audio = generate_synthetic_signal("vowel_ah", duration=0.2, sr=16000)
    frames, _ = frame_signal(audio.samples, 400, 160)
    frame = frames[frames.shape[0] // 2]

    order = 14
    r = autocorrelate(frame, order)
    a, k, err = levinson_durbin(r, order)

    assert len(a) == order + 1
    assert a[0] == 1.0
    assert len(k) == order
    # Stability condition: reflection coefficients |k_i| < 1.0 for stable AR filter
    assert np.all(np.abs(k) <= 1.0)
    assert err >= 0.0


def test_formant_extraction_vowel():
    """Verify formant frequencies are detected for synthetic /a/ vowel."""
    # Synthetic /a/ was generated with formants near 730 Hz, 1090 Hz, 2440 Hz
    audio = generate_synthetic_signal("vowel_ah", duration=0.4, sr=16000)
    frames, _ = frame_signal(audio.samples, 400, 160)
    frame = frames[frames.shape[0] // 2]

    res = analyze_frame_lpc(frame, sr=16000, order=16)

    assert len(res.formant_frequencies) >= 2
    # First formant F1 should be in 600-900 Hz range
    f1 = res.formant_frequencies[0]
    assert 500.0 <= f1 <= 1000.0


def test_lpc_silence():
    """Verify LPC analysis gracefully handles silence without throwing NaNs."""
    silence_frame = np.zeros(400, dtype=np.float32)
    res = analyze_frame_lpc(silence_frame, sr=16000, order=12)

    assert res.lpc_coefficients[0] == 1.0
    assert np.all(np.isfinite(res.spectral_envelope))
