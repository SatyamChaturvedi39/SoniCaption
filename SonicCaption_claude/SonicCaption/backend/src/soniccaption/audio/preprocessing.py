"""
Speech Preprocessing and Framing Module.
Implements pre-emphasis filtering, short-time windowing, and frame blocking
as formulated in Units 1 and 2 of the syllabus.
"""

from __future__ import annotations
from typing import Tuple, Literal
import numpy as np


def pre_emphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    Applies high-pass pre-emphasis filter to boost high-frequency components.
    Formula: y[n] = x[n] - alpha * x[n-1], with y[0] = x[0].

    High frequencies are naturally attenuated in speech production (~-6 dB/octave
    lip radiation effect). Pre-emphasis flattens the spectral tilt.
    """
    if len(signal) == 0:
        return np.array([], dtype=np.float32)
    emphasized = np.empty_like(signal, dtype=np.float32)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - alpha * signal[:-1]
    return emphasized


def de_emphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    Applies inverse de-emphasis filter to reconstruct original spectral balance.
    Formula: x[n] = y[n] + alpha * x[n-1].
    """
    if len(signal) == 0:
        return np.array([], dtype=np.float32)
    de_emphasized = np.empty_like(signal, dtype=np.float32)
    de_emphasized[0] = signal[0]
    for n in range(1, len(signal)):
        de_emphasized[n] = signal[n] + alpha * de_emphasized[n - 1]
    return de_emphasized


def get_window(window_type: str, length: int) -> np.ndarray:
    """
    Computes a symmetric analytical analysis window.
    Supports 'hamming', 'hanning', 'blackman', and 'rectangular'.
    """
    if length <= 0:
        return np.array([], dtype=np.float32)
    if length == 1:
        return np.ones(1, dtype=np.float32)

    n = np.arange(length, dtype=np.float32)
    w_type = window_type.lower()
    if w_type == "hamming":
        # w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
        return (0.54 - 0.46 * np.cos(2.0 * np.pi * n / (length - 1))).astype(np.float32)
    elif w_type in ("hanning", "hann"):
        # w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
        return (0.5 * (1.0 - np.cos(2.0 * np.pi * n / (length - 1)))).astype(np.float32)
    elif w_type == "blackman":
        return (0.42 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
                + 0.08 * np.cos(4.0 * np.pi * n / (length - 1))).astype(np.float32)
    elif w_type in ("rectangular", "rect"):
        return np.ones(length, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported window type: '{window_type}'. Use 'hamming', 'hanning', or 'rectangular'.")


def frame_signal(
    signal: np.ndarray,
    frame_length: int = 400,
    hop_length: int = 160,
    window_type: str = "hamming"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segments a 1D audio signal into overlapping 2D frames and applies a window.

    Parameters:
        signal: 1D numpy array of audio samples.
        frame_length: Frame size in samples (default 400 samples = 25 ms at 16 kHz).
        hop_length: Frame step/shift in samples (default 160 samples = 10 ms at 16 kHz).
        window_type: Type of analysis window ('hamming', 'hanning', 'rectangular').

    Returns:
        Tuple of (frames, window):
            frames: 2D array of shape (num_frames, frame_length).
            window: 1D array of shape (frame_length,).
    """
    signal_len = len(signal)
    if signal_len < frame_length:
        # Zero-pad short signal to reach at least one full frame
        pad_amount = frame_length - signal_len
        signal = np.pad(signal, (0, pad_amount), mode="constant")
        signal_len = len(signal)

    # Compute total number of frames with boundary padding
    num_frames = 1 + int(np.ceil((signal_len - frame_length) / float(hop_length)))
    total_pad_len = (num_frames - 1) * hop_length + frame_length
    pad_needed = total_pad_len - signal_len
    if pad_needed > 0:
        padded_signal = np.pad(signal, (0, pad_needed), mode="constant")
    else:
        padded_signal = signal

    # Create index matrix for strided vectorization
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * hop_length, hop_length), (frame_length, 1)).T

    frames = padded_signal[indices].astype(np.float32)
    window = get_window(window_type, frame_length)

    # Multiply each frame element-wise by the analysis window
    windowed_frames = frames * window

    return windowed_frames, window
