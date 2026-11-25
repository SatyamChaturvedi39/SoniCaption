"""
Acoustic Enhancement, Fundamental Pitch (F0) Estimation & VQ Speaker Diarization.

Acoustic DSP & Feature Architecture:
1. Boll's Spectral Subtraction Denoising via Overlap-Add (OLA)
2. Normalized Autocorrelation Pitch (F0) Contour Tracking
3. MFCC + F0 + Vector Quantisation (VQ) Speaker Turn Diarization
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from scipy import fft as sp_fft

from soniccaption.audio.preprocessing import frame_signal
from soniccaption.algorithms.dtw import dtw
from soniccaption.algorithms.spectral import (
    classify_voiced_unvoiced,
    compute_mel_spectrogram,
    compute_mfcc,
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

    # Exclude trailing partial/zero-padded frame so end padding does not skew noise estimation
    n_full = max(1, min(n_frames, (orig_len - frame_length) // hop_length + 1)) if orig_len >= frame_length else n_frames
    frame_energies = np.mean(windowed_frames ** 2, axis=1)
    valid_energies = frame_energies[:n_full]
    non_silent = np.where(valid_energies > 1e-12)[0]
    pool = non_silent if non_silent.size >= 3 else np.arange(n_full)
    n_noise_frames = max(1, min(len(pool), max(3, int(np.ceil(0.30 * len(pool))))))
    quiet_idx = pool[np.argsort(valid_energies[pool])[:n_noise_frames]]

    # Robust smoothed noise floor estimate (median reduces transient leakage)
    noise_mag = np.median(mag[quiet_idx], axis=0, keepdims=True)
    if noise_mag.shape[1] >= 3:
        k_freq = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        noise_mag = np.apply_along_axis(lambda r: np.convolve(r, k_freq, mode="same"), 1, noise_mag)
    noise_rms_before = float(np.sqrt(np.mean(frame_energies[quiet_idx]) + 1e-12))

    # Time-smoothed magnitude before gain estimation suppresses isolated musical-noise chirps
    if n_frames >= 3:
        k_time = np.array([0.22, 0.56, 0.22], dtype=np.float32)
        smooth_mag = np.apply_along_axis(lambda c: np.convolve(c, k_time, mode="same"), 0, mag)
    else:
        smooth_mag = mag

    # Smoothed Boll/Berouti spectral gain mask (bounded by spectral_floor)
    ratio = noise_mag / np.maximum(smooth_mag, 1e-9)
    gain = np.clip(1.0 - over_subtraction * ratio, spectral_floor, 1.0)
    if n_frames >= 3:
        gain = np.apply_along_axis(lambda c: np.convolve(c, np.array([0.2, 0.6, 0.2], dtype=np.float32), mode="same"), 0, gain)
    if gain.shape[1] >= 3:
        gain = np.apply_along_axis(lambda r: np.convolve(r, np.array([0.2, 0.6, 0.2], dtype=np.float32), mode="same"), 1, gain)

    # Gently attenuate sub-65 Hz AC mains / ceiling-fan motor rumble while preserving speech (>= 75 Hz)
    freqs = sp_fft.rfftfreq(n_fft, 1.0 / sr)
    rumble_mask = np.where(freqs < 65.0, 0.25, 1.0).astype(np.float32)[None, :]
    gain = np.clip(gain * rumble_mask, spectral_floor * 0.5, 1.0)

    enhanced_mag = np.maximum(gain * mag, spectral_floor * 0.5 * mag)
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
    """Track fundamental pitch (F0 in Hz) across voiced frames using autocorrelation & spectral cues."""
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


def extract_clip_signature(samples: np.ndarray, sr: int = 16000, max_frames: int = 24) -> List[List[float]]:
    """Build a compact time-sampled 13-MFCC descriptor matrix for DTW similarity search."""
    x = np.asarray(samples, dtype=np.float32)[: sr * 12]
    if x.size < 400:
        return []
    mfcc, _, _ = compute_mfcc(x, sr=sr, n_mfcc=13, n_mels=40)
    frames_t = mfcc.T  # (T, 13)
    if frames_t.shape[0] > max_frames:
        idx = np.linspace(0, frames_t.shape[0] - 1, max_frames, dtype=int)
        frames_t = frames_t[idx]
    return [[round(float(v), 3) for v in row] for row in frames_t]


def find_similar_sessions_dtw(
    query_signature: List[List[float]],
    sessions: List[Dict[str, Any]],
    current_session_id: int | None = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Rank past saved recordings using Dynamic Time Warping over MFCC signatures."""
    if not query_signature:
        return []
    q_arr = np.asarray(query_signature, dtype=np.float32)
    matches: List[Dict[str, Any]] = []
    for sess in sessions:
        if current_session_id is not None and sess.get("id") == current_session_id:
            continue
        sig = (sess.get("metrics") or {}).get("mfcc_signature")
        if not sig or len(sig) < 2:
            continue
        c_arr = np.asarray(sig, dtype=np.float32)
        res = dtw(q_arr, c_arr, metric="euclidean")
        norm_dist = round(float(res.normalized_distance), 3)
        similarity = round(float(100.0 * np.exp(-norm_dist / 40.0)), 1)
        matches.append(
            {
                "session_id": sess["id"],
                "filename": sess["filename"],
                "created_at": sess.get("created_at", ""),
                "normalized_distance": norm_dist,
                "similarity_pct": similarity,
                "summary": sess.get("summary", ""),
            }
        )
    matches.sort(key=lambda m: m["normalized_distance"])
    return matches[:top_k]


def analyze_segment_prosody(
    samples: np.ndarray,
    speech_segments: List[Dict[str, Any]],
    sr: int = 16000,
) -> List[Dict[str, Any]]:
    """Compute vocal delivery cues (F0 slope, RMS stress ratio, speech rate) per segment."""
    if not speech_segments:
        return []
    x = np.asarray(samples, dtype=np.float32)
    seg_rms_list: List[float] = []
    raw_chunks: List[np.ndarray] = []
    for seg in speech_segments:
        i0 = max(0, int(float(seg.get("start", 0.0)) * sr))
        i1 = min(len(x), int(float(seg.get("end", 0.0)) * sr))
        c = x[i0:i1] if i1 > i0 else np.array([], dtype=np.float32)
        raw_chunks.append(c)
        seg_rms_list.append(float(np.sqrt(np.mean(c ** 2))) if c.size else 0.0)

    positive_rms = [r for r in seg_rms_list if r > 1e-6]
    ref_rms = float(np.median(positive_rms)) if positive_rms else 0.02

    enriched: List[Dict[str, Any]] = []
    for seg, chunk, s_rms in zip(speech_segments, raw_chunks, seg_rms_list):
        dur = max(0.05, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        pitch = estimate_pitch_contour(chunk, sr=sr) if chunk.size >= 480 else {"f0_hz": [], "mean_f0_hz": 0.0, "pitch_std_hz": 0.0}
        voiced = [f for f in pitch.get("f0_hz", []) if f > 0.0]
        slope_hz = 0.0
        if len(voiced) >= 6:
            split = max(2, int(len(voiced) * 0.65))
            slope_hz = round(float(np.mean(voiced[split:]) - np.mean(voiced[:split])), 1)

        energy_ratio = round(s_rms / max(ref_rms, 1e-6), 2) if ref_rms > 0 else 1.0
        words_cnt = len(str(seg.get("text", "")).split())
        rate_wps = round(words_cnt / dur, 2) if words_cnt else 0.0
        text_str = str(seg.get("text", "")).strip()

        if text_str.endswith("?") or slope_hz >= 14.0:
            style = "Question / Rising"
        elif energy_ratio >= 1.25 or float(pitch.get("pitch_std_hz", 0.0)) >= 28.0:
            style = "Emphasized / Key Point"
        elif 0.0 < energy_ratio <= 0.72:
            style = "Soft / Understated"
        else:
            style = "Steady Statement"

        enriched.append({
            **seg,
            "delivery_style": style,
            "mean_f0_hz": round(float(pitch.get("mean_f0_hz", 0.0)), 1),
            "pitch_slope_hz": slope_hz,
            "energy_ratio": energy_ratio,
            "speaking_rate_wps": rate_wps,
        })
    return enriched


def _z_norm_frames(mat: np.ndarray) -> np.ndarray:
    mean = np.mean(mat, axis=0, keepdims=True)
    std = np.std(mat, axis=0, keepdims=True) + 1e-5
    return ((mat - mean) / std).astype(np.float32)


def subsequence_dtw_search(
    query_samples: np.ndarray,
    target_samples: np.ndarray,
    sr: int = 16000,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Query-by-Example: locate occurrences of a spoken word/sound query inside target audio via DTW over MFCCs."""
    q = np.asarray(query_samples, dtype=np.float32)[: sr * 6]
    t = np.asarray(target_samples, dtype=np.float32)[: sr * 60]
    if q.size < 800 or t.size < 800:
        return []

    # Trim dead silence around query edges so template focuses on active speech/sound
    q_abs = np.abs(q)
    q_peak = float(np.max(q_abs)) if q_abs.size else 0.0
    if q_peak <= 1e-5:
        return []
    active_idx = np.where(q_abs >= 0.08 * q_peak)[0]
    if active_idx.size >= 800:
        q = q[active_idx[0] : active_idx[-1] + 1]

    hop = 160  # 10 ms MFCC frame step
    q_mfcc, _, _ = compute_mfcc(q, sr=sr, n_mfcc=13, n_mels=40)
    t_mfcc, _, _ = compute_mfcc(t, sr=sr, n_mfcc=13, n_mels=40)

    # Subsample 2x (20 ms steps) for responsive real-time subsequence search
    q_seq = q_mfcc.T[::2]
    t_seq = t_mfcc.T[::2]
    step_sec = (hop * 2) / float(sr)

    if q_seq.shape[0] < 3 or t_seq.shape[0] < 3:
        return []

    q_len = min(q_seq.shape[0], t_seq.shape[0])
    q_norm = _z_norm_frames(q_seq[:q_len])
    stride = max(2, q_len // 5)

    # Compute per-frame target RMS magnitude aligned roughly to t_seq steps to ignore silence regions
    target_rms = float(np.sqrt(np.mean(t ** 2)))
    min_win_rms = max(0.005, 0.12 * target_rms)

    candidates: List[Dict[str, Any]] = []
    win_lengths = sorted({max(3, int(round(q_len * s))) for s in (0.85, 1.0, 1.15) if 3 <= int(round(q_len * s)) <= t_seq.shape[0]})
    if not win_lengths:
        win_lengths = [t_seq.shape[0]]

    for w_len in win_lengths:
        for start_f in range(0, max(1, t_seq.shape[0] - w_len + 1), stride):
            t0_samp = int(start_f * step_sec * sr)
            t1_samp = min(len(t), int((start_f + w_len) * step_sec * sr))
            if t1_samp > t0_samp:
                w_rms = float(np.sqrt(np.mean(t[t0_samp:t1_samp] ** 2)))
                if w_rms < min_win_rms:
                    continue
            c_norm = _z_norm_frames(t_seq[start_f : start_f + w_len])
            res = dtw(q_norm, c_norm, metric="euclidean")
            norm_d = round(float(res.normalized_distance), 3)
            sim = round(float(100.0 * np.exp(-max(0.0, norm_d - 1.5) / 3.5)), 1)
            candidates.append(
                {
                    "start": round(start_f * step_sec, 2),
                    "end": round(min(len(t) / float(sr), (start_f + w_len) * step_sec), 2),
                    "normalized_distance": norm_d,
                    "similarity_pct": sim,
                }
            )

    candidates.sort(key=lambda item: item["normalized_distance"])
    selected: List[Dict[str, Any]] = []
    for cand in candidates:
        if any(min(cand["end"], s["end"]) - max(cand["start"], s["start"]) > 0.35 * max(0.1, cand["end"] - cand["start"]) for s in selected):
            continue
        selected.append(cand)
        if len(selected) >= top_k:
            break
    return selected
