"""
Spectral Analysis, Time-Domain Metrics, and Acoustic Feature Extraction.
Fulfills Unit 1 (Fundamentals) and Unit 2 (Spectral Analysis & Filter-banks) requirements:
- Short-time Energy and RMS
- Zero-Crossing Rate (ZCR)
- Voiced / Unvoiced / Silence classification
- Discrete Fourier Transform (FFT) Magnitude Spectrum
- Short-Time Fourier Transform (STFT) Spectrogram
- Mel-scale Filter-Bank Energies
- Mel-Frequency Cepstral Coefficients (MFCCs) with Delta and Delta-Delta
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from scipy import fft as sp_fft

from soniccaption.core.schemas import AudioData, SpectralFeatures
from soniccaption.audio.preprocessing import frame_signal, pre_emphasis


def compute_short_time_energy(frames: np.ndarray) -> np.ndarray:
    """
    Computes Short-Time Energy for each frame.
    Formula: E_i = sum_{m=0}^{N-1} (x_i[m])^2
    """
    if len(frames) == 0:
        return np.array([], dtype=np.float32)
    return np.sum(frames ** 2, axis=1).astype(np.float32)


def compute_rms(frames: np.ndarray) -> np.ndarray:
    """
    Computes Root-Mean-Square (RMS) amplitude per frame.
    Formula: RMS_i = sqrt( (1/N) * sum_{m=0}^{N-1} (x_i[m])^2 )
    """
    if len(frames) == 0:
        return np.array([], dtype=np.float32)
    mean_sq = np.mean(frames ** 2, axis=1)
    return np.sqrt(np.maximum(mean_sq, 1e-12)).astype(np.float32)


def compute_zcr(frames: np.ndarray) -> np.ndarray:
    """
    Computes Zero-Crossing Rate (ZCR) per frame.
    Formula: ZCR_i = (1 / (2*N)) * sum_{m=1}^{N-1} |sgn(x[m]) - sgn(x[m-1])|

    Voiced speech typically has low ZCR due to vocal fold periodicity.
    Unvoiced fricatives (/s/, /sh/) exhibit high ZCR due to turbulent noise.
    """
    if len(frames) == 0:
        return np.array([], dtype=np.float32)

    # sgn with 0 mapped to 1 to prevent false transitions
    signs = np.sign(frames)
    signs[signs == 0] = 1.0

    # Count sign changes between consecutive samples
    diffs = np.abs(np.diff(signs, axis=1))
    zcr = np.sum(diffs, axis=1) / (2.0 * frames.shape[1])
    return zcr.astype(np.float32)


def classify_voiced_unvoiced(
    energy: np.ndarray,
    zcr: np.ndarray,
    energy_threshold_ratio: float = 0.08,
    zcr_threshold: float = 0.12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classifies each frame into Voiced, Unvoiced, or Silence.

    Classification Rule:
    - Silence: Energy < silence_threshold (fraction of max energy).
    - Voiced: Energy >= silence_threshold AND ZCR <= zcr_threshold.
    - Unvoiced: Energy >= silence_threshold AND ZCR > zcr_threshold.

    Returns:
        Tuple of (labels, voiced_boolean_mask):
            labels: array of strings ('voiced', 'unvoiced', 'silence').
            voiced_boolean_mask: boolean array (True where frame is voiced).
    """
    n_frames = len(energy)
    if n_frames == 0:
        return np.array([]), np.array([], dtype=bool)

    max_energy = np.max(energy) if np.max(energy) > 1e-9 else 1.0
    silence_threshold = max_energy * energy_threshold_ratio

    labels = np.empty(n_frames, dtype=object)
    voiced_mask = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        if energy[i] < silence_threshold:
            labels[i] = "silence"
        elif zcr[i] > zcr_threshold:
            labels[i] = "unvoiced"
        else:
            labels[i] = "voiced"
            voiced_mask[i] = True

    return labels, voiced_mask


def compute_fft_spectrum(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the one-sided Discrete Fourier Transform (FFT) magnitude spectrum.

    Parameters:
        signal: 1D audio samples.
        sr: Sample rate in Hz.
        n_fft: FFT length (zero-padded if len(signal) < n_fft).

    Returns:
        Tuple of (frequencies, magnitude_db):
            frequencies: 1D array of frequency bins from 0 to Nyquist (sr/2).
            magnitude_db: 1D array of magnitude values in decibels (dB).
    """
    if len(signal) == 0:
        return np.array([]), np.array([])

    # Compute FFT
    fft_vals = sp_fft.rfft(signal, n=n_fft)
    freqs = sp_fft.rfftfreq(n_fft, d=1.0 / sr)

    mag = np.abs(fft_vals)
    # Convert to dB relative to maximum
    ref = np.max(mag) if np.max(mag) > 1e-9 else 1.0
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-9) / ref)

    return freqs.astype(np.float32), mag_db.astype(np.float32)


def compute_stft(
    signal: np.ndarray,
    sr: int,
    frame_length: int = 400,
    hop_length: int = 160,
    n_fft: int = 512,
    window_type: str = "hamming"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Short-Time Fourier Transform (STFT) magnitude spectrogram.

    Parameters:
        signal: 1D audio samples.
        sr: Sample rate in Hz.
        frame_length: Window size in samples.
        hop_length: Hop/shift size in samples.
        n_fft: FFT length per frame.
        window_type: Window function ('hamming', 'hanning').

    Returns:
        Tuple of (frequencies, times, spectrogram_db):
            frequencies: 1D array of frequency bins (0 to sr/2).
            times: 1D array of frame center timestamps in seconds.
            spectrogram_db: 2D array of shape (num_freq_bins, num_frames) in dB.
    """
    windowed_frames, _ = frame_signal(signal, frame_length, hop_length, window_type)
    num_frames = windowed_frames.shape[0]

    # Compute rfft along each frame
    fft_complex = sp_fft.rfft(windowed_frames, n=n_fft, axis=1)  # (num_frames, num_freq_bins)
    mag_spectrogram = np.abs(fft_complex).T  # Transpose to (num_freq_bins, num_frames)

    freqs = sp_fft.rfftfreq(n_fft, d=1.0 / sr)
    times = (np.arange(num_frames) * hop_length + frame_length / 2.0) / float(sr)

    # Convert to log-power scale (dB)
    ref = np.max(mag_spectrogram) if np.max(mag_spectrogram) > 1e-9 else 1.0
    spectrogram_db = 20.0 * np.log10(np.maximum(mag_spectrogram, 1e-9) / ref)

    return freqs.astype(np.float32), times.astype(np.float32), spectrogram_db.astype(np.float32)


def create_mel_filterbank(
    sr: int,
    n_fft: int = 512,
    n_mels: int = 40,
    f_min: float = 0.0,
    f_max: Optional[float] = None
) -> np.ndarray:
    """
    Constructs triangular Mel-scale filter-bank matrix.
    Mel conversion: m = 2595 * log10(1 + f / 700)
    Inverse: f = 700 * (10^(m / 2595) - 1)

    Returns:
        Filter-bank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    if f_max is None:
        f_max = float(sr // 2)

    # Convert Hertz bounds to Mel
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)

    # Linearly spaced points in Mel scale
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    # Convert back to Hertz
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # Map to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    num_bins = n_fft // 2 + 1
    fbank = np.zeros((n_mels, num_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        # Up-slope
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / float(f_m - f_m_minus)
        # Down-slope
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                fbank[m - 1, k] = (f_m_plus - k) / float(f_m_plus - f_m)

    return fbank


def compute_mel_spectrogram(
    signal: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 512,
    hop_length: int = 160,
    frame_length: int = 400
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes log-Mel power spectrogram.

    Returns:
        Tuple of (mel_frequencies, times, log_mel_spectrogram):
            mel_frequencies: (n_mels,) center frequencies.
            times: (num_frames,) timestamps.
            log_mel_spectrogram: (n_mels, num_frames) in dB.
    """
    windowed_frames, _ = frame_signal(signal, frame_length, hop_length, "hamming")
    num_frames = windowed_frames.shape[0]

    # Power spectrum |FFT|^2
    fft_complex = sp_fft.rfft(windowed_frames, n=n_fft, axis=1)
    power_spectrum = (np.abs(fft_complex) ** 2) / float(n_fft)  # (num_frames, num_freq_bins)

    fbank = create_mel_filterbank(sr, n_fft=n_fft, n_mels=n_mels)
    # Matrix multiplication: (n_mels, num_bins) x (num_bins, num_frames)
    mel_power = np.dot(fbank, power_spectrum.T)  # (n_mels, num_frames)

    log_mel = 10.0 * np.log10(np.maximum(mel_power, 1e-10))

    times = (np.arange(num_frames) * hop_length + frame_length / 2.0) / float(sr)
    mel_freqs = np.linspace(0.0, float(sr // 2), n_mels)

    return mel_freqs.astype(np.float32), times.astype(np.float32), log_mel.astype(np.float32)


def compute_deltas(features: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Computes dynamic delta (velocity) regression features:
    d_t = sum_{n=1}^N n * (c_{t+n} - c_{t-n}) / (2 * sum_{n=1}^N n^2)
    with N = order.
    """
    num_feats, num_frames = features.shape
    deltas = np.zeros_like(features, dtype=np.float32)
    denom = 2.0 * sum(n ** 2 for n in range(1, order + 1))

    # Pad edges symmetrically
    padded = np.pad(features, ((0, 0), (order, order)), mode="edge")

    for t in range(num_frames):
        d_val = np.zeros(num_feats, dtype=np.float32)
        for n in range(1, order + 1):
            d_val += n * (padded[:, t + order + n] - padded[:, t + order - n])
        deltas[:, t] = d_val / denom

    return deltas


def compute_mfcc(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
    frame_length: int = 400
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCCs) along with
    delta (velocity) and delta-delta (acceleration) dynamic coefficients.

    Algorithm:
    1. Pre-emphasis.
    2. Framing and Hamming windowing.
    3. FFT power spectrum.
    4. Mel filter-bank integration.
    5. Logarithm of filter-bank energies.
    6. Discrete Cosine Transform (DCT-II).
    7. Cepstral mean subtraction (CMS) for channel normalization.
    8. Dynamic delta and delta-delta computation.

    Returns:
        Tuple of (static_mfcc, delta_mfcc, delta2_mfcc):
            Each of shape (n_mfcc, num_frames).
    """
    # 1. Pre-emphasis
    emp_signal = pre_emphasis(signal, alpha=0.97)

    # 2-5. Log-Mel spectrogram
    _, _, log_mel = compute_mel_spectrogram(
        emp_signal, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, frame_length=frame_length
    )

    # 6. Discrete Cosine Transform (DCT-II)
    # c_n = sum_{m=0}^{M-1} s_m * cos(pi * n * (m + 0.5) / M)
    M, T = log_mel.shape
    mfcc = np.zeros((n_mfcc, T), dtype=np.float32)

    for n in range(n_mfcc):
        m_idx = np.arange(M)
        basis = np.cos(np.pi * n * (m_idx + 0.5) / float(M))
        mfcc[n, :] = np.dot(basis, log_mel)

    # Orthogonal scaling factor
    mfcc[0, :] *= np.sqrt(1.0 / float(M))
    mfcc[1:, :] *= np.sqrt(2.0 / float(M))

    # 7. Cepstral Mean Normalization (CMS)
    mfcc -= np.mean(mfcc, axis=1, keepdims=True)

    # 8. Compute Delta and Delta-Delta
    delta1 = compute_deltas(mfcc, order=2)
    delta2 = compute_deltas(delta1, order=2)

    return mfcc, delta1, delta2


def extract_spectral_features(
    audio: AudioData,
    frame_length: int = 400,
    hop_length: int = 160,
    n_fft: int = 512,
    n_mels: int = 40,
    n_mfcc: int = 13
) -> SpectralFeatures:
    """
    High-level extractor aggregating all Unit 1 & 2 representations.
    """
    signal = audio.samples
    sr = audio.sample_rate

    # Frame signal
    frames, _ = frame_signal(signal, frame_length=frame_length, hop_length=hop_length, window_type="hamming")
    num_frames = frames.shape[0]
    times = (np.arange(num_frames) * hop_length + frame_length / 2.0) / float(sr)

    # Energy, RMS, ZCR
    energy = compute_short_time_energy(frames)
    rms = compute_rms(frames)
    zcr = compute_zcr(frames)
    _, voiced_mask = classify_voiced_unvoiced(energy, zcr)

    # Global FFT spectrum
    fft_freqs, fft_mag = compute_fft_spectrum(signal, sr, n_fft=2048)

    # STFT Spectrogram
    stft_freqs, _, stft_spec = compute_stft(signal, sr, frame_length, hop_length, n_fft)

    # Mel spectrogram & MFCC
    _, _, log_mel = compute_mel_spectrogram(signal, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, frame_length=frame_length)
    mfcc, delta1, delta2 = compute_mfcc(signal, sr, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, frame_length=frame_length)

    return SpectralFeatures(
        times=times,
        energy=energy,
        rms=rms,
        zcr=zcr,
        frequencies=fft_freqs,
        fft_magnitude=fft_mag,
        stft_spectrogram=stft_spec,
        stft_frequencies=stft_freqs,
        mel_spectrogram=log_mel,
        mfcc=mfcc,
        delta_mfcc=delta1,
        delta2_mfcc=delta2,
        voiced_flags=voiced_mask
    )
