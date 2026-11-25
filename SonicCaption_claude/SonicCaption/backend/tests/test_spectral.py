import numpy as np
import pytest
from soniccaption.audio.io import generate_synthetic_signal
from soniccaption.audio.preprocessing import frame_signal
from soniccaption.algorithms.spectral import (
    compute_short_time_energy,
    compute_rms,
    compute_zcr,
    classify_voiced_unvoiced,
    compute_fft_spectrum,
    compute_stft,
    compute_mel_spectrogram,
    compute_mfcc,
    extract_spectral_features
)


def test_energy_and_rms():
    """Verify energy and RMS metric calculation."""
    audio_vowel = generate_synthetic_signal("vowel_ah", duration=0.2, sr=16000)
    audio_silence = generate_synthetic_signal("silence", duration=0.2, sr=16000)

    frames_vowel, _ = frame_signal(audio_vowel.samples, 400, 160)
    frames_silence, _ = frame_signal(audio_silence.samples, 400, 160)

    energy_vowel = compute_short_time_energy(frames_vowel)
    rms_vowel = compute_rms(frames_vowel)

    energy_silence = compute_short_time_energy(frames_silence)
    rms_silence = compute_rms(frames_silence)

    assert np.all(energy_vowel > 0)
    assert np.all(rms_vowel > 0)
    assert np.all(energy_silence == 0)
    assert np.all(rms_silence <= 1e-5)


def test_zcr_voiced_vs_unvoiced():
    """Voiced vowels should have lower ZCR than unvoiced fricatives."""
    audio_vowel = generate_synthetic_signal("vowel_ah", duration=0.3, sr=16000)
    audio_fricative = generate_synthetic_signal("fricative", duration=0.3, sr=16000)

    frames_vowel, _ = frame_signal(audio_vowel.samples, 400, 160)
    frames_fricative, _ = frame_signal(audio_fricative.samples, 400, 160)

    zcr_vowel = compute_zcr(frames_vowel)
    zcr_fricative = compute_zcr(frames_fricative)

    # Average ZCR for fricative /s/ must be significantly higher than vowel /a/
    assert np.mean(zcr_fricative) > np.mean(zcr_vowel)


def test_fft_spectrum_peak():
    """440 Hz sine tone should exhibit maximum FFT magnitude at 440 Hz."""
    audio_sine = generate_synthetic_signal("sine", duration=0.5, sr=16000)
    freqs, mag_db = compute_fft_spectrum(audio_sine.samples, sr=16000, n_fft=2048)

    peak_freq = freqs[np.argmax(mag_db)]
    # Peak frequency should be close to 440 Hz
    assert abs(peak_freq - 440.0) < 15.0


def test_stft_dimensions():
    """Verify STFT spectrogram dimensions and non-NaN values."""
    audio = generate_synthetic_signal("chirp", duration=0.5, sr=16000)
    freqs, times, spec = compute_stft(audio.samples, sr=16000, frame_length=400, hop_length=160, n_fft=512)

    assert spec.shape[0] == 257  # n_fft // 2 + 1
    assert spec.shape[1] == len(times)
    assert np.all(np.isfinite(spec))


def test_mfcc_dimensions_and_deltas():
    """Verify 13 MFCCs, delta, and delta-delta dimensions."""
    audio = generate_synthetic_signal("vowel_ah", duration=0.5, sr=16000)
    mfcc, delta1, delta2 = compute_mfcc(audio.samples, sr=16000, n_mfcc=13)

    assert mfcc.shape[0] == 13
    assert delta1.shape[0] == 13
    assert delta2.shape[0] == 13
    assert mfcc.shape[1] == delta1.shape[1] == delta2.shape[1]
    assert np.all(np.isfinite(mfcc))
    assert np.all(np.isfinite(delta1))
    assert np.all(np.isfinite(delta2))
