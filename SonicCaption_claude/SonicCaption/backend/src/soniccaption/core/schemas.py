"""
Core Data Schemas and Type Definitions for SonicCaption.
Adheres strictly to the shared contracts for speech signal processing,
feature extraction, pattern matching, and captioning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Literal
import numpy as np


@dataclass
class AudioData:
    """Represents a validated, conditioned audio signal."""
    samples: np.ndarray  # 1D float32 normalized between [-1.0, 1.0]
    sample_rate: int = 16000
    duration: float = 0.0
    channels: int = 1
    original_path: Optional[str] = None

    def __post_init__(self):
        if self.duration == 0.0 and len(self.samples) > 0 and self.sample_rate > 0:
            self.duration = float(len(self.samples)) / float(self.sample_rate)


@dataclass
class SpectralFeatures:
    """Acoustic and spectral representation of an audio signal (Units 1 & 2)."""
    times: np.ndarray  # (T,) time centroids in seconds
    energy: np.ndarray  # (T,) short-time energy
    rms: np.ndarray  # (T,) root-mean-square energy
    zcr: np.ndarray  # (T,) zero-crossing rate
    frequencies: np.ndarray  # (F,) frequency bins in Hz for FFT spectrum
    fft_magnitude: np.ndarray  # (F,) global or frame FFT magnitude
    stft_spectrogram: np.ndarray  # (F, T) 2D STFT magnitude spectrogram
    stft_frequencies: np.ndarray  # (F,) frequency axis for STFT
    mel_spectrogram: Optional[np.ndarray] = None  # (M, T) log-mel spectrogram
    mfcc: Optional[np.ndarray] = None  # (13, T) static MFCC coefficients
    delta_mfcc: Optional[np.ndarray] = None  # (13, T) first-order delta
    delta2_mfcc: Optional[np.ndarray] = None  # (13, T) second-order delta-delta
    voiced_flags: Optional[np.ndarray] = None  # (T,) boolean voiced/unvoiced flags


@dataclass
class LPCResult:
    """Linear Predictive Coding analysis results (Unit 2)."""
    order: int
    lpc_coefficients: np.ndarray  # (order + 1,) a[0] = 1.0, a[1..p]
    reflection_coefficients: np.ndarray  # (order,) PARCOR k_1 .. k_p
    prediction_error: float
    formant_frequencies: List[float]  # Detected vocal tract resonances in Hz
    formant_bandwidths: List[float]  # Corresponding bandwidths in Hz
    spectral_envelope: np.ndarray  # Frequency response magnitude H(e^jw)
    envelope_frequencies: np.ndarray  # Frequency axis for spectral envelope


@dataclass
class VQResult:
    """Vector Quantisation codebook and quantization metrics (Unit 2)."""
    codebook: np.ndarray  # (K, D) cluster centroids / codewords
    quantized_indices: np.ndarray  # (N,) index of nearest codeword for each vector
    distortion: float  # Mean squared quantization error
    distortion_history: List[float] = field(default_factory=list)
    cluster_counts: List[int] = field(default_factory=list)


@dataclass
class DTWResult:
    """Dynamic Time Warping sequence alignment result (Unit 3)."""
    distance: float  # Total accumulated alignment cost D(N-1, M-1)
    normalized_distance: float  # Total cost divided by alignment path length
    path_length: int
    path: List[Tuple[int, int]]  # Sequence of aligned frame pairs [(i_0, j_0), ..., (i_K, j_K)]
    cost_matrix: np.ndarray  # (N, M) 2D accumulated cost matrix
    local_distance_matrix: np.ndarray  # (N, M) 2D frame-wise distance matrix
    metric: str = "euclidean"
