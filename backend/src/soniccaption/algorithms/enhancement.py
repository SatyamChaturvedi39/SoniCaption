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
    compute_rms,
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
    """Estimate single-frame fundamental frequency (Hz) via normalized autocorrelation with subharmonic octave guard."""
    centered = frame - np.mean(frame)
    energy = float(np.dot(centered, centered))
    if energy <= 1e-9 or len(centered) <= lag_max:
        return 0.0

    # Normalized autocorrelation over candidate pitch lags
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    best_lag = -1
    best_norm = 0.0
    norms = np.zeros(min(lag_max + 2, len(corr)), dtype=np.float32)
    for lag in range(lag_min, min(lag_max + 1, len(corr))):
        denom = float(np.sqrt(np.sum(centered[:-lag] ** 2) * np.sum(centered[lag:] ** 2))) + 1e-12
        norm_r = float(corr[lag]) / denom
        norms[lag] = norm_r
        if norm_r > best_norm:
            best_norm = norm_r
            best_lag = lag

    if best_norm < 0.35 or best_lag <= 0:
        return 0.0

    # Octave-doubling protection: prefer fundamental lag near 2 * best_lag if correlation is comparable
    sub_center = best_lag * 2
    if sub_center - 2 >= lag_min and sub_center + 2 <= lag_max and sub_center + 2 < len(norms):
        window = norms[sub_center - 2 : sub_center + 3]
        sub_offset = int(np.argmax(window)) - 2
        if float(window[sub_offset + 2]) >= 0.78 * best_norm:
            best_lag = sub_center + sub_offset

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


def split_sentences_with_words(speech_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split multi-sentence Whisper segments into clean individual sentence turns using
    word timestamps when available, and snap segment bounds to actual spoken words.
    """
    if not speech_segments:
        return []
    out: List[Dict[str, Any]] = []
    for seg in speech_segments:
        words = seg.get("words") or []
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if len(words) < 2:
            out.append(seg)
            continue

        cur_words: List[Dict[str, Any]] = []
        for w in words:
            w_txt = str(w.get("word", "")).strip()
            if not w_txt:
                continue
            cur_words.append(w)
            if w_txt[-1] in ".?!" and len(cur_words) >= 2:
                st = round(float(cur_words[0].get("start", seg["start"])), 3)
                en = round(float(cur_words[-1].get("end", seg["end"])), 3)
                sent = " ".join(str(x.get("word", "")).strip() for x in cur_words).strip()
                if en > st and sent:
                    out.append({**seg, "start": st, "end": en, "text": sent, "words": list(cur_words)})
                cur_words = []

        if cur_words:
            st = round(float(cur_words[0].get("start", seg["start"])), 3)
            en = round(float(cur_words[-1].get("end", seg["end"])), 3)
            sent = " ".join(str(x.get("word", "")).strip() for x in cur_words).strip()
            if en > st and sent:
                out.append({**seg, "start": st, "end": en, "text": sent, "words": list(cur_words)})
    return out if out else speech_segments


def _segment_active_timbre_and_pitch(
    chunk: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, float, bool]:
    """Extract L2-normalized DCT vocal-tract timbre + median voiced pitch from core active speech frames."""
    frames, _ = frame_signal(chunk, frame_length=400, hop_length=160, window_type="hamming")
    if len(frames) == 0:
        return np.zeros(12, dtype=np.float32), 0.0, False

    rms_f = compute_rms(frames)
    peak_rms = float(np.max(rms_f)) if len(rms_f) else 0.0
    active_mask = rms_f >= max(0.0025, 0.28 * peak_rms)
    active_frames = frames[active_mask]
    if len(active_frames) < 2:
        active_frames = frames

    active_sig = active_frames.reshape(-1)
    _, _, mel = compute_mel_spectrogram(active_sig, sr=sr, n_mels=20, n_fft=512, hop_length=160, frame_length=400)
    mel_mean = np.mean(mel, axis=1)

    M = len(mel_mean)
    timbre = np.zeros(12, dtype=np.float32)
    m_idx = np.arange(M)
    for k in range(1, 13):
        timbre[k - 1] = float(np.dot(np.cos(np.pi * k * (m_idx + 0.5) / float(M)), mel_mean)) * np.sqrt(2.0 / float(M))

    # L2-normalize vocal-tract spectral vector so volume changes do not shift speaker identity
    t_norm = float(np.linalg.norm(timbre))
    if t_norm > 1e-6:
        timbre = (timbre / t_norm).astype(np.float32)

    pitch = estimate_pitch_contour(active_sig, sr=sr)
    voiced_f0 = [f for f in pitch.get("f0_hz", []) if f > 60.0]
    med_f0 = float(np.median(voiced_f0)) if voiced_f0 else 0.0
    return timbre, med_f0, med_f0 > 60.0


def _kmeans_farthest_init(feat_mat: np.ndarray, k: int, max_iter: int = 40) -> np.ndarray:
    """Deterministic K-Means using farthest-point initialization so extreme voices anchor speakers."""
    n = feat_mat.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int32)

    # Pick initial pair maximizing pairwise squared distance
    d_mat = np.sum((feat_mat[:, None, :] - feat_mat[None, :, :]) ** 2, axis=2)
    i0, i1 = np.unravel_index(int(np.argmax(d_mat)), d_mat.shape)
    chosen = [int(i0), int(i1)]
    while len(chosen) < k:
        min_d = np.min(d_mat[:, chosen], axis=1)
        chosen.append(int(np.argmax(min_d)))
    centroids = feat_mat[chosen].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dists = np.sum((feat_mat[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            mem = feat_mat[labels == c]
            if len(mem) > 0:
                centroids[c] = np.mean(mem, axis=0)
    return labels


def diarize_speakers(
    samples: np.ndarray,
    speech_segments: List[Dict[str, Any]],
    sr: int = 16000,
    max_speakers: int = 3,
) -> List[Dict[str, Any]]:
    """
    Assign speaker labels ('Speaker 1', 'Speaker 2', ...) to speech segments using
    active-frame L2 vocal-tract timbre + semitone F0 with deterministic farthest-point VQ.
    """
    if not speech_segments:
        return []
    if len(speech_segments) == 1 or max_speakers <= 1:
        return [{**seg, "speaker": "Speaker 1"} for seg in speech_segments]

    x = np.asarray(samples, dtype=np.float32)
    timbres: List[np.ndarray] = []
    f0s: List[float] = []
    has_pitch: List[bool] = []
    valid_indices: List[int] = []

    for idx, seg in enumerate(speech_segments):
        s_idx = max(0, int(round(float(seg["start"]) * sr)))
        e_idx = min(len(x), int(round(float(seg["end"]) * sr)))
        chunk = x[s_idx:e_idx]
        if chunk.size < int(0.15 * sr):
            continue
        timbre, med_f0, pitched = _segment_active_timbre_and_pitch(chunk, sr=sr)
        timbres.append(timbre)
        f0s.append(med_f0)
        has_pitch.append(pitched)
        valid_indices.append(idx)

    if len(timbres) < 2:
        return [{**seg, "speaker": "Speaker 1"} for seg in speech_segments]

    valid_f0s = [f for f, ok in zip(f0s, has_pitch) if ok]
    fallback_f0 = float(np.median(valid_f0s)) if valid_f0s else 140.0
    semitones = np.array(
        [12.0 * np.log2(max(f if ok else fallback_f0, 50.0) / 100.0) for f, ok in zip(f0s, has_pitch)],
        dtype=np.float32,
    )

    # Balanced embedding: L2-normalized vocal-tract shape (4.0x weight) + semitone pitch (1/3.6 weight)
    # so emphatic pitch inflection by Speaker 1 does not override vocal-tract identity.
    timbre_mat = np.vstack(timbres)
    pitch_feat = ((semitones - float(np.median(semitones))) / 3.6)[:, None]
    feat_mat = np.hstack([timbre_mat * 4.0, pitch_feat]).astype(np.float32)

    def _cosine_dist(u: np.ndarray, v: np.ndarray) -> float:
        nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if nu < 1e-6 or nv < 1e-6:
            return 0.0
        return float(1.0 - np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))

    best_labels = np.zeros(len(timbres), dtype=np.int32)
    max_k = min(max_speakers, len(timbres))
    for k in range(max_k, 1, -1):
        labels = _kmeans_farthest_init(feat_mat, k=k, max_iter=40)
        used = [c for c in range(k) if np.sum(labels == c) > 0]
        if len(used) < 2:
            continue
        all_distinct = True
        for i_a in range(len(used)):
            for i_b in range(i_a + 1, len(used)):
                ca, cb = used[i_a], used[i_b]
                idx_a = np.where(labels == ca)[0]
                idx_b = np.where(labels == cb)[0]
                semi_diff = abs(float(np.median(semitones[idx_a])) - float(np.median(semitones[idx_b])))
                cos_diff = _cosine_dist(np.mean(timbre_mat[idx_a], axis=0), np.mean(timbre_mat[idx_b], axis=0))
                # Distinct speakers require genuine timbre shift, major pitch shift (>= 5.2 st), or both combined
                is_distinct = (cos_diff >= 0.05) or (semi_diff >= 5.2) or (semi_diff >= 2.2 and cos_diff >= 0.015)
                if not is_distinct:
                    all_distinct = False
                    break
            if not all_distinct:
                break
        if all_distinct:
            best_labels = labels
            break

    assigned = ["Speaker 1"] * len(speech_segments)
    cluster_map: Dict[int, str] = {}
    for pos, orig_idx in enumerate(valid_indices):
        cid = int(best_labels[pos])
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

        pitch_std = float(pitch.get("pitch_std_hz", 0.0))
        vocal_rising = slope_hz >= 10.0
        has_q_mark = text_str.endswith("?")
        is_question = vocal_rising or has_q_mark

        if vocal_rising and has_q_mark:
            q_reason = f"Rising inflection (+{slope_hz:.1f} Hz)"
        elif vocal_rising:
            q_reason = f"Vocal rising inflection (+{slope_hz:.1f} Hz)"
        elif has_q_mark:
            q_reason = "Question phrasing"
        else:
            q_reason = "Statement contour"

        # Strictly one-word acoustic emphasis tag per sentence
        if energy_ratio >= 1.35 and rate_wps >= 2.7:
            emphasis_tag = "Urgent"
        elif energy_ratio >= 1.22 or pitch_std >= 28.0:
            emphasis_tag = "Emphatic"
        elif energy_ratio >= 1.08 or pitch_std >= 16.0:
            emphasis_tag = "Assertive"
        elif 0.0 < energy_ratio <= 0.72 and rate_wps < 2.0:
            emphasis_tag = "Hesitant"
        elif 0.0 < energy_ratio <= 0.78:
            emphasis_tag = "Soft"
        else:
            emphasis_tag = "Neutral"

        if is_question:
            style = "Question / Rising"
        elif emphasis_tag in ("Urgent", "Emphatic"):
            style = "Emphasized / Key Point"
        elif emphasis_tag in ("Soft", "Hesitant"):
            style = "Soft / Understated"
        else:
            style = "Steady Statement"

        enriched.append({
            **seg,
            "delivery_style": style,
            "emphasis_tag": emphasis_tag,
            "is_question": is_question,
            "question_reason": q_reason,
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
