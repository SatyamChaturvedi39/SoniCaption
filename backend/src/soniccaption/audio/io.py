"""
Audio Ingestion, Conditioning, and Synthetic Signal Generation Module.
Provides robust file decoding, channel normalization, 16 kHz resampling,
and deterministic benchmark signal generation for audio pipeline validation.
"""

from __future__ import annotations
import io
import math
import os
from typing import BinaryIO, Optional, Union
import numpy as np
from scipy import signal as sp_signal

from soniccaption.core.schemas import AudioData


def load_audio(
    source: Union[str, bytes, BinaryIO],
    target_sr: int = 16000,
    to_mono: bool = True,
    normalize_peak: bool = True
) -> AudioData:
    """
    Decodes audio from a file path, raw bytes, or file-like stream.
    Converts channels to mono and resamples to target_sr (default 16 kHz).

    Parameters:
        source: File path (str), raw audio bytes, or open BinaryIO stream.
        target_sr: Desired output sample rate in Hz (default 16000 Hz).
        to_mono: If True, multi-channel audio is averaged to single-channel mono.
        normalize_peak: If True, scales amplitude to peak 0.95 to prevent clipping.

    Returns:
        AudioData dataclass containing normalized 1D float32 samples.
    """
    file_path = source if isinstance(source, str) else None

    # Load audio using soundfile if available, otherwise fallback to scipy.io.wavfile or wave
    try:
        import soundfile as sf
        if isinstance(source, (bytes, bytearray)):
            stream = io.BytesIO(source)
            samples, orig_sr = sf.read(stream, dtype="float32")
        elif hasattr(source, "read"):
            samples, orig_sr = sf.read(source, dtype="float32")
        else:
            samples, orig_sr = sf.read(source, dtype="float32")
    except Exception:
        # Fallback to scipy.io.wavfile for WAV sources
        from scipy.io import wavfile
        if isinstance(source, (bytes, bytearray)):
            stream = io.BytesIO(source)
            orig_sr, samples = wavfile.read(stream)
        elif hasattr(source, "read"):
            orig_sr, samples = wavfile.read(source)
        else:
            orig_sr, samples = wavfile.read(source)

        # Convert integer PCM to float32 normalized [-1.0, 1.0]
        if samples.dtype == np.int16:
            samples = (samples / 32768.0).astype(np.float32)
        elif samples.dtype == np.int32:
            samples = (samples / 2147483648.0).astype(np.float32)
        elif samples.dtype == np.uint8:
            samples = ((samples - 128) / 128.0).astype(np.float32)

    # Inspect channels
    orig_channels = 1
    if samples.ndim > 1:
        orig_channels = samples.shape[1]
        if to_mono:
            samples = np.mean(samples, axis=1)

    samples = samples.astype(np.float32)

    # Resample if original sample rate does not match target
    if orig_sr != target_sr and len(samples) > 0:
        # Compute GCD for efficient polyphase resampling
        gcd = math.gcd(int(target_sr), int(orig_sr))
        up = int(target_sr // gcd)
        down = int(orig_sr // gcd)
        samples = sp_signal.resample_poly(samples, up, down).astype(np.float32)

    # Peak normalization to prevent clipping and ensure numerical consistency
    if normalize_peak and len(samples) > 0:
        peak = np.max(np.abs(samples))
        if peak > 1e-6:
            samples = (samples / peak) * 0.95

    return AudioData(
        samples=samples,
        sample_rate=target_sr,
        channels=1 if to_mono else orig_channels,
        original_path=file_path
    )


def save_audio(audio: AudioData, target_path: str, subtype: str = "PCM_16") -> None:
    """Saves AudioData samples to a WAV file."""
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    sf.write(target_path, audio.samples, audio.sample_rate, subtype=subtype)


def generate_synthetic_signal(
    signal_type: str = "vowel_ah",
    duration: float = 1.0,
    sr: int = 16000,
    f0: float = 130.0
) -> AudioData:
    """
    Generates synthetic deterministic acoustic benchmarks for audio pipeline verification.

    Signal Types:
      - 'vowel_ah': Synthetic /a/ vowel with vocal tract resonances (F1=730Hz, F2=1090Hz, F3=2440Hz).
      - 'vowel_ee': Synthetic /i/ vowel with vocal tract resonances (F1=270Hz, F2=2290Hz, F3=3010Hz).
      - 'sine': Clean sinusoidal tone at 440 Hz (Concert A).
      - 'chirp': Linear frequency sweep from 100 Hz to 4000 Hz.
      - 'fricative': Bandpass-filtered white noise modeling unvoiced /s/.
      - 'silence': Zero signal.
    """
    num_samples = int(duration * sr)
    t = np.linspace(0.0, duration, num_samples, endpoint=False, dtype=np.float32)

    if signal_type == "sine":
        samples = 0.8 * np.sin(2.0 * np.pi * 440.0 * t)

    elif signal_type == "chirp":
        samples = 0.8 * sp_signal.chirp(t, f0=100.0, t1=duration, f1=4000.0, method="linear")

    elif signal_type == "fricative":
        # White noise passed through 3000-7000 Hz bandpass filter to model /s/
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0.0, 0.5, num_samples).astype(np.float32)
        sos = sp_signal.butter(4, [3000.0, 7000.0], btype="bandpass", fs=sr, output="sos")
        samples = sp_signal.sosfilt(sos, noise).astype(np.float32)

    elif signal_type in ("vowel_ah", "vowel_ee"):
        # Source-filter speech production model:
        # Glottal pulse train excitation (F0) passed through 3 formant resonators
        glottal_pulse = np.zeros(num_samples, dtype=np.float32)
        period_samples = int(sr / f0)
        glottal_pulse[::period_samples] = 1.0

        if signal_type == "vowel_ah":
            formants = [730.0, 1090.0, 2440.0]
            bandwidths = [80.0, 100.0, 120.0]
        else:
            formants = [270.0, 2290.0, 3010.0]
            bandwidths = [60.0, 110.0, 140.0]

        filtered = np.zeros(num_samples, dtype=np.float32)
        for F, B in zip(formants, bandwidths):
            # 2nd-order resonator (biquad filter)
            r = np.exp(-np.pi * B / sr)
            theta = 2.0 * np.pi * F / sr
            b = [1.0 - r, 0.0, 0.0]
            a = [1.0, -2.0 * r * np.cos(theta), r * r]
            filtered += sp_signal.lfilter(b, a, glottal_pulse).astype(np.float32)
        samples = filtered

    elif signal_type == "silence":
        samples = np.zeros(num_samples, dtype=np.float32)

    else:
        raise ValueError(f"Unknown synthetic signal type: '{signal_type}'")

    # Normalize peak
    peak = np.max(np.abs(samples))
    if peak > 1e-6:
        samples = (samples / peak) * 0.90

    return AudioData(
        samples=samples.astype(np.float32),
        sample_rate=sr,
        duration=duration,
        channels=1,
        original_path=f"synthetic://{signal_type}"
    )
