"""
Acoustic Enhancement, Fundamental Pitch (F0) Estimation & VQ Speaker Diarization.

Member 1 Unit 1-3 DSP Product Extensions:
1. Boll's Spectral Subtraction Denoising via Overlap-Add (OLA)
2. Normalized Autocorrelation Pitch (F0) Contour Tracking
3. MFCC + F0 + Vector Quantisation (VQ) Speaker Turn Diarization
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from scipy import fft as sp_fft

from soniccaption.audio.preprocessing import frame_signal
from soniccaption.algorithms.spectral import (
    classify_voiced_unvoiced,
    compute_mel_spectrogram,
    compute_short_time_energy,
    compute_zcr,
)
from soniccaption.algorithms.vq import kmeans_vq


def _estimate_snr_db(signal: np.ndarray, noise_rms: float) -> float:
    """Estimate segmental signal-to-noise ratio (dB) from active signal RMS vs noise floor."""
    if signal.size == 0 or noise_rms <= 1e-9:
        return 0.0
    sig_rms = float(np.sqrt(np.mean(signal ** 2)))
    ratio = max(sig_rms, 1e-9) / max(noise_rms, 1e-9)
    return round(float(20.0 * np.log10(max(ratio, 1.0))), 2)


def spectral_subtraction_denoise(
    samples: np.ndarray,
    sr: int = 16000,
    frame_length: int = 400,
    hop_length: int = 160,
    n_fft: int = 512,
    over_subtraction: float = 1.6,
    spectral_floor: float = 0.08,
) -> Dict[str, Any]:
    """
    Denoise 1D audio via magnitude spectral subtraction and weighted overlap-add.

    Estimates the stationary background noise profile from the quietest 20% of STFT
    frames, subtracts scaled noise magnitude, enforces a spectral floor to prevent
    musical noise artifacts, and reconstructs the clean time-domain signal.
    """
    x = np.asarray(samples, dtype=np.float32)
    orig_len = len(x)
    if orig_len == 0:
        return {
            "denoised": x,
            "noise_floor_rms": 0.0,
            "snr_before_db": 0.0,
            "snr_after_db": 0.0,
            "snr_gain_db": 0.0,
        }

    windowed_frames, window = frame_signal(x, frame_length, hop_length, window_type="hanning")
    n_frames = windowed_frames.shape[0]
    spec = sp_fft.rfft(windowed_frames, n=n_fft, axis=1)
    mag = np.abs(spec)
    phase = np.angle(spec)

    frame_energies = np.mean(windowed_frames ** 2, axis=1)
    n_noise_frames = max(1, min(n_frames, max(3, int(np.ceil(0.20 * n_frames)))))
    quiet_idx = np.argsort(frame_energies)[:n_noise_frames]
    noise_mag = np.mean(mag[quiet_idx], axis=0, keepdims=True)
    noise_rms_before = float(np.sqrt(np.mean(frame_energies[quiet_idx]) + 1e-12))

    # Boll's magnitude subtraction with spectral floor
    enhanced_mag = np.maximum(mag - over_subtraction * noise_mag, spectral_floor * mag)
    enhanced_spec = enhanced_mag * np.exp(1j * phase)
    recon_frames = sp_fft.irfft(enhanced_spec, n=n_fft, axis=1)[:, :frame_length] * window[None, :]

    out_len = (n_frames - 1) * hop_length + frame_length
    y_out = np.zeros(out_len, dtype=np.float64)
    w_norm = np.zeros(out_len, dtype=np.float64)
    w_sq = (window ** 2).astype(np.float64)

    for i in range(n_frames):
        s = i * hop_length
        y_out[s : s + frame_length] += recon_frames[i]
        w_norm[s : s + frame_length] += w_sq

    valid = w_norm > 1e-8
    y_out[valid] /= w_norm[valid]
    denoised = y_out[:orig_len].astype(np.float32)

    # Preserve peak safety <= 0.98 if gain shifts slightly
    peak = float(np.max(np.abs(denoised))) if denoised.size else 0.0
    if peak > 0.98:
        denoised = (denoised / peak * 0.98).astype(np.float32)

    d_frames, _ = frame_signal(denoised, frame_length, hop_length, window_type="hanning")
    d_energies = np.mean(d_frames ** 2, axis=1)
    noise_rms_after = float(np.sqrt(np.mean(d_energies[quiet_idx]) + 1e-12))

    snr_before = _estimate_snr_db(x, noise_rms_before)
    snr_after = _estimate_snr_db(denoised, noise_rms_after)

    return {
        "denoised": denoised,
        "noise_floor_rms": round(noise_rms_before, 5),
        "snr_before_db": snr_before,
        "snr_after_db": snr_after,
        "snr_gain_db": round(snr_after - snr_before, 2),
    }


def _frame_f0_autocorr(frame: np.ndarray, sr: int, lag_min: int, lag_max: int) -> float:
    """Estimate single-frame fundamental frequency (Hz) via normalized autocorrelation."""
    centered = frame - np.mean(frame)
    energy = float(np.dot(centered, centered))
    if energy <= 1e-9 or len(centered) <= lag_max:
        return 0.0

    # Normalized autocorrelation over candidate pitch lags
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    best_lag = -1
    best_norm = 0.0
    for lag in range(lag_min, min(lag_max + 1, len(corr))):
        denom = float(np.sqrt(np.sum(centered[:-lag] ** 2) * np.sum(centered[lag:] ** 2))) + 1e-12
        norm_r = float(corr[lag]) / denom
        if norm_r > best_norm:
            best_norm = norm_r
            best_lag = lag

    if best_norm < 0.35 or best_lag <= 0:
        return 0.0
    return float(sr) / float(best_lag)


def estimate_pitch_contour(
    samples: np.ndarray,
    sr: int = 16000,
    frame_length: int = 480,
    hop_length: int = 160,
    fmin: float = 75.0,
    fmax: float = 380.0,
) -> Dict[str, Any]:
    """Track fundamental pitch (F0 in Hz) across voiced frames using Unit 1 & 2 cues."""
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return {
            "times": [],
            "f0_hz": [],
            "mean_f0_hz": 0.0,
            "min_f0_hz": 0.0,
            "max_f0_hz": 0.0,
            "pitch_std_hz": 0.0,
            "voiced_ratio": 0.0,
        }

    frames, _ = frame_signal(x, frame_length, hop_length, window_type="hamming")
    energy = compute_short_time_energy(frames)
    zcr = compute_zcr(frames)
    _, voiced_mask = classify_voiced_unvoiced(energy, zcr, energy_threshold_ratio=0.05, zcr_threshold=0.18)

    lag_min = max(2, int(sr / fmax))
    lag_max = min(frame_length - 4, int(sr / fmin))

    f0_vals = np.zeros(len(frames), dtype=np.float32)
    for idx, is_voiced in enumerate(voiced_mask):
        if is_voiced:
            f0_vals[idx] = _frame_f0_autocorr(frames[idx], sr, lag_min, lag_max)

    active_f0 = f0_vals[f0_vals > 0.0]
    times = ((np.arange(len(frames)) * hop_length + frame_length / 2.0) / float(sr)).tolist()

    return {
        "times": [round(float(t), 3) for t in times],
        "f0_hz": [round(float(v), 1) for v in f0_vals],
        "mean_f0_hz": round(float(np.mean(active_f0)), 1) if active_f0.size else 0.0,
        "min_f0_hz": round(float(np.min(active_f0)), 1) if active_f0.size else 0.0,
        "max_f0_hz": round(float(np.max(active_f0)), 1) if active_f0.size else 0.0,
        "pitch_std_hz": round(float(np.std(active_f0)), 1) if active_f0.size else 0.0,
        "voiced_ratio": round(float(np.mean(voiced_mask)), 3) if len(voiced_mask) else 0.0,
    }


def diarize_speakers(
    samples: np.ndarray,
    speech_segments: List[Dict[str, Any]],
    sr: int = 16000,
    max_speakers: int = 2,
) -> List[Dict[str, Any]]:
    """
    Assign speaker labels ('Speaker 1', 'Speaker 2', ...) to speech segments
    using Member 1's MFCC + F0 pitch embeddings clustered via K-Means VQ.
    """
    if not speech_segments:
        return []
    if len(speech_segments) == 1 or max_speakers <= 1:
        return [{**seg, "speaker": "Speaker 1"} for seg in speech_segments]

    x = np.asarray(samples, dtype=np.float32)
    embeddings: List[np.ndarray] = []
    valid_indices: List[int] = []

    for idx, seg in enumerate(speech_segments):
        s_idx = max(0, int(round(float(seg["start"]) * sr)))
        e_idx = min(len(x), int(round(float(seg["end"]) * sr)))
        chunk = x[s_idx:e_idx]
        if chunk.size < int(0.15 * sr):
            continue
        # Use un-CMS-zeroed log-Mel bands (summarized into 8 coarse spectral bands) + log F0 pitch
        _, _, mel = compute_mel_spectrogram(chunk, sr=sr, n_mels=24, n_fft=512, hop_length=160, frame_length=400)
        mel_mean = np.mean(mel, axis=1)
        mel_norm = (mel_mean - np.mean(mel_mean)) / (np.std(mel_mean) + 1e-6)
        bands = np.mean(mel_norm.reshape(8, 3), axis=1)
        pitch = estimate_pitch_contour(chunk, sr=sr)
        f0_feat = np.log1p(pitch["mean_f0_hz"]) * 3.0
        emb = np.concatenate([bands, [f0_feat]]).astype(np.float32)
        embeddings.append(emb)
        valid_indices.append(idx)

    if len(embeddings) < 2:
        return [{**seg, "speaker": "Speaker 1"} for seg in speech_segments]

    feat_mat = np.vstack(embeddings)
    vq_res = kmeans_vq(feat_mat, k=min(max_speakers, len(embeddings)), max_iter=30, seed=42)

    # Guard against over-splitting single-speaker audio
    if vq_res.codebook.shape[0] >= 2:
        centroid_dist = float(np.linalg.norm(vq_res.codebook[0] - vq_res.codebook[1]))
    else:
        centroid_dist = 0.0

    assigned = ["Speaker 1"] * len(speech_segments)
    if centroid_dist >= 1.5:
        # Chronological label mapping so first speaker encountered is always Speaker 1
        cluster_map: Dict[int, str] = {}
        for pos, orig_idx in enumerate(valid_indices):
            cid = int(vq_res.quantized_indices[pos])
            if cid not in cluster_map:
                cluster_map[cid] = f"Speaker {len(cluster_map) + 1}"
            assigned[orig_idx] = cluster_map[cid]

    return [{**seg, "speaker": assigned[idx]} for idx, seg in enumerate(speech_segments)]
