"""
AI Synthetic Voice & Deepfake Authenticator Module.

Forensic acoustic analysis to detect synthetic, text-to-speech (TTS),
and neural vocoder generated speech (e.g., ElevenLabs, HiFi-GAN, VITS, StyleTTS)
versus natural human vocal tract acoustics.

Key Biometric Cues:
1. Micro-Pitch Jitter (Relative Average Perturbation of F0)
2. Amplitude Shimmer (Cycle-to-Cycle Vocal Energy Perturbation)
3. LPC Inverse-Filter Prediction Residual Entropy & Kurtosis
4. High-Frequency Spectral Cutoff & Vocoder Phase Dispersion
5. Harmonics-to-Noise Naturalness Ratio (HNR)
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from scipy import signal as sp_signal

from soniccaption.algorithms.enhancement import estimate_pitch_contour
from soniccaption.algorithms.lpc import analyze_frame_lpc
from soniccaption.algorithms.spectral import compute_short_time_energy


def compute_pitch_jitter(f0_series: List[float]) -> tuple[float, str]:
    """
    Compute micro-pitch perturbation (Relative Average Perturbation / Local Jitter).
    Natural human vocal cords exhibit natural involuntary neuromuscular jitter: 0.5% - 2.2%.
    Neural vocoders typically show unnaturally smooth pitch (< 0.35%) or discrete stepping.
    """
    voiced = [f for f in f0_series if f > 60.0]
    if len(voiced) < 8:
        return 0.0, "Insufficient Voiced Frames"

    v = np.array(voiced, dtype=np.float32)
    mean_f0 = float(np.mean(v))
    if mean_f0 < 1e-4:
        return 0.0, "Zero Pitch"

    diffs = np.abs(np.diff(v))
    jitter_ratio = float(np.mean(diffs) / mean_f0) * 100.0  # as percentage

    if 0.45 <= jitter_ratio <= 2.8:
        status = "Natural Human Micro-Tremor"
    elif jitter_ratio < 0.45:
        status = "Oversmoothed (AI Vocoder Signature)"
    else:
        status = "High Variance / Irregular"

    return round(jitter_ratio, 2), status


def compute_amplitude_shimmer(samples: np.ndarray, sr: int = 16000) -> tuple[float, str]:
    """
    Compute short-term cycle-to-cycle amplitude perturbation (Shimmer in dB).
    Natural human glottal pulses vary frame-to-frame by 0.6 dB to 2.8 dB.
    Synthetic TTS vocoders produce mathematically uniform cycle amplitudes (< 0.45 dB).
    """
    frame_len = int(sr * 0.025)  # 25 ms
    hop_len = int(sr * 0.010)    # 10 ms
    if len(samples) < frame_len * 3:
        return 0.0, "Clip Too Short"

    # Frame energy
    n_frames = 1 + (len(samples) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + (np.arange(n_frames) * hop_len)[:, None]
    frames = samples[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    # Consider active frames above noise floor
    peak_rms = np.max(rms) if rms.size else 0.0
    active = rms[rms >= 0.15 * peak_rms]
    if len(active) < 6:
        return 0.0, "Insufficient Energy"

    # Frame-to-frame amplitude ratio in dB
    ratios = np.abs(20.0 * np.log10((active[1:] + 1e-9) / (active[:-1] + 1e-9)))
    shimmer_db = float(np.mean(ratios))

    if 0.55 <= shimmer_db <= 3.2:
        status = "Natural Amplitude Dynamic"
    elif shimmer_db < 0.55:
        status = "Unnaturally Uniform (Synthetic Vocoder)"
    else:
        status = "High Dynamic Modulation"

    return round(shimmer_db, 2), status


def compute_lpc_residual_profile(samples: np.ndarray, sr: int = 16000) -> tuple[float, float, str]:
    """
    Compute LPC prediction residual statistics.
    Natural human speech filtered through inverse vocal tract filter A(z) leaves
    quasi-periodic glottal pulses with high impulse kurtosis.
    Neural vocoders leave oversmoothed or diffused residuals.
    """
    if len(samples) < sr * 0.2:
        return 0.0, 0.0, "Too Short"

    # Use first 1.5s active segment
    x = samples[: min(len(samples), int(sr * 1.5))]
    lpc_res = analyze_frame_lpc(x, sr=sr, order=14)
    if lpc_res is None or len(lpc_res.lpc_coefficients) < 15:
        return 0.0, 0.0, "LPC Failed"

    # Inverse filter: e[n] = x[n] + sum_{k=1}^p a_k x[n-k]
    a = lpc_res.lpc_coefficients
    residual = sp_signal.lfilter(a, [1.0], x)

    # Residual spectral flatness
    res_pow = np.abs(np.fft.rfft(residual[: min(len(residual), 2048)])) ** 2 + 1e-12
    geo = np.exp(np.mean(np.log(res_pow)))
    arith = np.mean(res_pow)
    flatness = float(geo / max(arith, 1e-12))

    # Kurtosis of excitation residual
    std_res = float(np.std(residual))
    if std_res > 1e-6:
        norm_res = (residual - float(np.mean(residual))) / std_res
        kurtosis = float(np.mean(norm_res ** 4)) - 3.0
    else:
        kurtosis = 0.0

    if kurtosis >= 3.5:
        status = "Natural Glottal Impulses"
    elif kurtosis < 1.8:
        status = "Diffusion / Vocoder Excitation"
    else:
        status = "Moderate Glottal Structure"

    return round(flatness, 3), round(kurtosis, 2), status


def compute_spectral_cutoff(samples: np.ndarray, sr: int = 16000) -> tuple[float, str]:
    """
    Check for high-frequency vocoder cutoff or phase smearing above 7.0 kHz.
    Many fast neural vocoders (22 kHz upsampled or 16 kHz bandlimited) show
    steep drop-offs or harmonic discontinuities in top bands.
    """
    if len(samples) < 512:
        return 0.0, "Too Short"

    fft_mag = np.abs(np.fft.rfft(samples[: min(len(samples), 4096)])) ** 2
    freqs = np.fft.rfftfreq(min(len(samples), 4096), 1.0 / sr)

    low_band = np.sum(fft_mag[(freqs >= 300) & (freqs < 3500)])
    high_band = np.sum(fft_mag[freqs >= 7000])

    ratio = float(high_band / max(low_band, 1e-9))

    if ratio < 0.001:
        status = "Steep HF Cutoff (Low-Bitrate / Fast Vocoder)"
    elif ratio > 0.35:
        status = "Elevated HF Dispersion"
    else:
        status = "Natural Harmonic Roll-off"

    return round(ratio, 4), status


def detect_synthetic_voice(
    samples: np.ndarray,
    sr: int = 16000,
) -> Dict[str, Any]:
    """
    Perform forensic acoustic authentication to evaluate if an audio recording
    is produced by a natural human vocal tract or an AI synthesis system.
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.size < sr * 0.35:
        return {
            "verdict": "Audio Sample Too Short",
            "authenticity_score": 50,
            "is_synthetic": False,
            "confidence": 0.5,
            "risk_level": "unknown",
            "summary": "Audio recording must be at least 0.5 seconds for acoustic forensic analysis.",
            "metrics": {},
        }

    # Normalize audio
    peak = float(np.max(np.abs(x)))
    if peak > 1e-4:
        x = x / peak
    else:
        return {
            "verdict": "Silence Detected",
            "authenticity_score": 0,
            "is_synthetic": False,
            "confidence": 0.0,
            "risk_level": "unknown",
            "summary": "No speech energy present in audio recording.",
            "metrics": {},
        }

    # 1. Pitch contour & Jitter
    pitch_data = estimate_pitch_contour(x, sr=sr)
    f0_series = pitch_data.get("f0_hz", [])
    jitter_val, jitter_status = compute_pitch_jitter(f0_series)

    # 2. Shimmer
    shimmer_val, shimmer_status = compute_amplitude_shimmer(x, sr=sr)

    # 3. LPC Residual
    flatness_val, kurtosis_val, lpc_status = compute_lpc_residual_profile(x, sr=sr)

    # 4. HF Cutoff
    hf_ratio, hf_status = compute_spectral_cutoff(x, sr=sr)

    # Composite Human Authenticity Score Calculation (0 to 100)
    score = 50.0

    # Jitter scoring
    if 0.50 <= jitter_val <= 2.2:
        score += 18.0  # highly indicative of human vocal fold dynamics
    elif jitter_val < 0.40 and len(f0_series) >= 10:
        score -= 22.0  # characteristic of neural vocoder interpolation
    elif jitter_val > 3.0:
        score -= 8.0

    # Shimmer scoring
    if 0.70 <= shimmer_val <= 2.8:
        score += 16.0  # natural cycle amplitude variability
    elif shimmer_val < 0.50 and shimmer_val > 0.0:
        score -= 18.0  # unnaturally uniform amplitude
    elif shimmer_val > 3.5:
        score -= 6.0

    # LPC residual scoring
    if kurtosis_val >= 3.0:
        score += 14.0  # natural glottal impulse spikes
    elif kurtosis_val < 1.6 and kurtosis_val != 0.0:
        score -= 14.0  # diffuse vocoder excitation

    # Spectral roll-off scoring
    if "Natural" in hf_status:
        score += 8.0
    elif "Steep HF Cutoff" in hf_status:
        score -= 12.0

    # Clamp score
    auth_score = int(np.clip(score, 5.0, 98.0))

    if auth_score >= 68:
        verdict = "Authentic Human Voice"
        is_synthetic = False
        risk_level = "low"
        summary = (
            f"Vocal biomarkers show natural human dynamics: micro-pitch jitter ({jitter_val}%), "
            f"natural amplitude shimmer ({shimmer_val} dB), and characteristic glottal impulse residuals."
        )
    elif auth_score <= 42:
        verdict = "Likely AI-Generated / Synthetic"
        is_synthetic = True
        risk_level = "high"
        summary = (
            f"Detected synthetic vocoder artifacts: oversmoothed pitch trajectory (jitter: {jitter_val}%), "
            f"unnaturally uniform frame energy ({shimmer_val} dB), or diffuse LPC excitation residual."
        )
    else:
        verdict = "Inconclusive / Mixed Biomarkers"
        is_synthetic = False
        risk_level = "medium"
        summary = (
            f"Acoustic biomarkers are borderline (authenticity {auth_score}%). "
            f"Pitch jitter is {jitter_val}% and shimmer is {shimmer_val} dB."
        )

    return {
        "verdict": verdict,
        "authenticity_score": auth_score,
        "is_synthetic": is_synthetic,
        "confidence": round(abs(auth_score - 50) / 50.0, 2),
        "risk_level": risk_level,
        "summary": summary,
        "metrics": {
            "jitter_percent": jitter_val,
            "jitter_status": jitter_status,
            "shimmer_db": shimmer_val,
            "shimmer_status": shimmer_status,
            "lpc_residual_kurtosis": kurtosis_val,
            "lpc_residual_flatness": flatness_val,
            "lpc_status": lpc_status,
            "hf_ratio": hf_ratio,
            "hf_status": hf_status,
            "mean_f0_hz": pitch_data.get("mean_f0_hz", 0.0),
        },
    }
