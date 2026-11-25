"""Unit tests for Member 1's enhancement, pitch tracking, and VQ speaker diarization."""

import numpy as np

from soniccaption.algorithms.enhancement import (
    diarize_speakers,
    estimate_pitch_contour,
    spectral_subtraction_denoise,
)
from soniccaption.audio.io import generate_synthetic_signal


def test_spectral_subtraction_improves_snr_and_preserves_length():
    clean = generate_synthetic_signal("vowel_ah", duration=1.0, sr=16000, f0=130.0).samples
    # Create audio with initial noise-only segment + speech+noise so noise floor drops noticeably
    rng = np.random.default_rng(7)
    noise = (0.04 * rng.standard_normal(len(clean))).astype(np.float32)
    noisy = clean.copy()
    noisy[:4000] = 0.0
    noisy = (noisy + noise).astype(np.float32)

    res = spectral_subtraction_denoise(noisy, sr=16000)
    assert res["denoised"].shape == noisy.shape
    assert res["denoised"].dtype == np.float32
    assert res["snr_after_db"] > res["snr_before_db"]
    assert res["snr_gain_db"] > 0.0


def test_pitch_contour_tracks_synthetic_vowel_f0():
    audio = generate_synthetic_signal("vowel_ah", duration=0.8, sr=16000, f0=140.0)
    res = estimate_pitch_contour(audio.samples, sr=16000)
    assert res["voiced_ratio"] > 0.5
    assert 125.0 <= res["mean_f0_hz"] <= 155.0


def test_diarize_speakers_separates_distinct_pitches():
    sr = 16000
    s1 = generate_synthetic_signal("vowel_ah", duration=0.6, sr=sr, f0=115.0).samples
    s2 = generate_synthetic_signal("vowel_ee", duration=0.6, sr=sr, f0=235.0).samples
    gap = np.zeros(int(0.2 * sr), dtype=np.float32)
    clip = np.concatenate([s1, gap, s2, gap, s1])
    segs = [
        {"start": 0.0, "end": 0.6, "text": "Hello one"},
        {"start": 0.8, "end": 1.4, "text": "Hello two"},
        {"start": 1.6, "end": 2.2, "text": "Back to one"},
    ]
    out = diarize_speakers(clip, segs, sr=sr, max_speakers=2)
    assert len(out) == 3
    assert out[0]["speaker"] == "Speaker 1"
    assert out[1]["speaker"] == "Speaker 2"
    assert out[2]["speaker"] == "Speaker 1"


def test_empty_signal_guards():
    empty = np.array([], dtype=np.float32)
    assert spectral_subtraction_denoise(empty)["snr_gain_db"] == 0.0
    assert estimate_pitch_contour(empty)["mean_f0_hz"] == 0.0
    assert diarize_speakers(empty, []) == []
