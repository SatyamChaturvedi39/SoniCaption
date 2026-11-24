"""
Linear Predictive Coding (LPC) and Vocal Tract Formant Modeling.
Fulfills Unit 2 (LPC Analysis & Acoustic Modeling) requirements:
- Autocorrelation method
- Levinson-Durbin recursion for AR predictor coefficients
- PARCOR / reflection coefficients
- LPC spectral envelope computation
- Root-solving formant frequency and bandwidth extraction
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
from scipy import fft as sp_fft

from soniccaption.core.schemas import LPCResult
from soniccaption.audio.preprocessing import pre_emphasis, frame_signal


def autocorrelate(x: np.ndarray, order: int) -> np.ndarray:
    """
    Computes sample autocorrelation values R[0], R[1], ..., R[order].
    Formula: R[k] = sum_{n=0}^{N-1-k} x[n] * x[n+k]
    """
    N = len(x)
    if N == 0:
        return np.zeros(order + 1, dtype=np.float32)

    r = np.zeros(order + 1, dtype=np.float64)
    for k in range(order + 1):
        if k < N:
            r[k] = np.sum(x[:N - k].astype(np.float64) * x[k:].astype(np.float64))
    return r.astype(np.float32)


def levinson_durbin(r: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the Toeplitz Yule-Walker equations using Levinson-Durbin recursion.

    Solves R * a = -r for all-pole AR model coefficients a_1, ..., a_p.

    Parameters:
        r: Autocorrelation array [R[0], R[1], ..., R[order]].
        order: Model order p (typically 12 to 16 for 16 kHz speech).

    Returns:
        Tuple of (a, k, e):
            a: Full LPC predictor polynomial [1.0, a_1, a_2, ..., a_p].
            k: Reflection / PARCOR coefficients [k_1, ..., k_p].
            e: Residual prediction error energy E.
    """
    # Guard against zero/silence signal
    if len(r) == 0 or r[0] <= 1e-12:
        a_dummy = np.zeros(order + 1, dtype=np.float32)
        a_dummy[0] = 1.0
        return a_dummy, np.zeros(order, dtype=np.float32), 0.0

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    k_coeffs = np.zeros(order, dtype=np.float64)

    # Initial step i = 0
    e = float(r[0])

    for i in range(1, order + 1):
        # Calculate reflection coefficient k_i
        # lambda_i = R[i] + sum_{j=1}^{i-1} a_j * R[i - j]
        acc = float(r[i])
        for j in range(1, i):
            acc += a[j] * float(r[i - j])

        k_i = -acc / (e if abs(e) > 1e-12 else 1e-12)
        k_coeffs[i - 1] = k_i

        # Update LPC coefficients
        a_prev = a.copy()
        a[i] = k_i
        for j in range(1, i):
            a[j] = a_prev[j] + k_i * a_prev[i - j]

        # Update prediction error energy E_i = E_{i-1} * (1 - k_i^2)
        e *= (1.0 - k_i * k_i)
        if e <= 0:
            e = 1e-12

    return a.astype(np.float32), k_coeffs.astype(np.float32), float(e)


def lpc_spectral_envelope(
    lpc_coeffs: np.ndarray,
    gain: float = 1.0,
    sr: int = 16000,
    n_fft: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the vocal tract all-pole filter frequency response magnitude.
    H(e^jw) = Gain / A(e^jw) = Gain / (1 + sum_{k=1}^p a_k e^{-j*w*k})

    Returns:
        Tuple of (frequencies_hz, magnitude_db).
    """
    freqs = sp_fft.rfftfreq(n_fft, d=1.0 / sr)
    # Frequency response of denominator polynomial A(z)
    A_fft = sp_fft.rfft(lpc_coeffs, n=n_fft)
    # H(w) = G / A(w)
    H_mag = gain / np.maximum(np.abs(A_fft), 1e-9)
    # Convert to dB scale
    H_db = 20.0 * np.log10(np.maximum(H_mag, 1e-9))
    return freqs.astype(np.float32), H_db.astype(np.float32)


def extract_formants(
    lpc_coeffs: np.ndarray,
    sr: int = 16000,
    min_freq: float = 200.0,
    max_freq: float = 4000.0,
    max_bandwidth: float = 500.0
) -> Tuple[List[float], List[float]]:
    """
    Extracts vocal tract formant resonances by finding roots of the LPC polynomial.

    Roots z_i are complex conjugate pairs inside the unit circle:
    Resonance frequency: F = (angle(z) * sr) / (2 * pi)
    Bandwidth: B = - (sr / pi) * ln(|z|)

    Returns:
        Tuple of (sorted_formant_frequencies_hz, corresponding_bandwidths_hz).
    """
    if len(lpc_coeffs) <= 1:
        return [], []

    # Find roots of A(z) = a[0]*z^p + a[1]*z^(p-1) + ... + a[p]
    roots = np.roots(lpc_coeffs)

    formants: List[Tuple[float, float]] = []

    for r in roots:
        # Consider only roots with positive imaginary part (upper half-plane)
        if np.imag(r) > 1e-6:
            r_mag = np.abs(r)
            if 0.0 < r_mag < 1.0:
                # Calculate resonance frequency in Hz
                angle = np.angle(r)
                freq = float((angle * sr) / (2.0 * np.pi))
                # Calculate formant bandwidth in Hz
                bw = float(- (sr / np.pi) * np.log(r_mag))

                # Filter realistic vocal tract constraints
                if min_freq <= freq <= max_freq and bw <= max_bandwidth:
                    formants.append((freq, bw))

    # Sort formants by frequency in ascending order (F1, F2, F3...)
    formants.sort(key=lambda x: x[0])

    freqs = [round(f[0], 1) for f in formants]
    bandwidths = [round(f[1], 1) for f in formants]

    return freqs, bandwidths


def analyze_frame_lpc(
    frame: np.ndarray,
    sr: int = 16000,
    order: int = 16,
    n_fft: int = 1024
) -> LPCResult:
    """
    Performs full LPC analysis on a single windowed speech frame.
    """
    # 1. Autocorrelation
    r = autocorrelate(frame, order)

    # 2. Levinson-Durbin recursion
    a, k, err = levinson_durbin(r, order)
    gain = np.sqrt(max(err, 1e-9))

    # 3. Spectral envelope
    freqs, envelope_db = lpc_spectral_envelope(a, gain=gain, sr=sr, n_fft=n_fft)

    # 4. Formant extraction
    formants, bandwidths = extract_formants(a, sr=sr)

    return LPCResult(
        order=order,
        lpc_coefficients=a,
        reflection_coefficients=k,
        prediction_error=err,
        formant_frequencies=formants,
        formant_bandwidths=bandwidths,
        spectral_envelope=envelope_db,
        envelope_frequencies=freqs
    )
