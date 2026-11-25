"""
AI Synthetic Voice & Deepfake Authenticator Module.

Forensic acoustic analysis to detect synthetic, text-to-speech (TTS),
and neural vocoder generated speech (e.g., ElevenLabs, HiFi-GAN, VITS, StyleTTS)
versus natural human vocal tract acoustics.

Key Biometric Cues (computed strictly on VAD-active human speech segments):
1. Sub-sample Parabolic Micro-Pitch Jitter (Relative Average Perturbation of F0)
2. Non-Overlapping Sub-Frame Amplitude Shimmer (Cycle Energy Perturbation)
3. Active-Frame LPC Inverse-Filter Glottal Residual Kurtosis & Flatness
4. High-Frequency Spectral Cutoff & Harmonics Roll-off
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import signal as sp_signal

from soniccaption.algorithms.lpc import analyze_frame_lpc
from soniccaption.audio.preprocessing import frame_signal
from soniccaption.audio.vad import detect_speech


def _parabolic_f0(frame: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 380.0) -> Tuple[float, float]:
    """Sub-sample F0 (Hz) and peak normalized autocorrelation via parabolic interpolation."""
    centered = frame - np.mean(frame)
    energy = float(np.dot(centered, centered))
    lag_min = max(3, int(sr / fmax))
    lag_max = min(len(centered) - 3, int(sr / fmin))
    if energy <= 1e-8 or lag_max <= lag_min + 2:
        return 0.0, 0.0

    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    best_lag = -1
    best_norm = 0.0
    norms = np.zeros(lag_max + 2, dtype=np.float32)

    for lag in range(lag_min - 1, lag_max + 2):
        if lag < 1 or lag >= len(corr):
            continue
        denom = float(np.sqrt(np.sum(centered[:-lag] ** 2) * np.sum(centered[lag:] ** 2))) + 1e-12
        norms[lag] = float(corr[lag]) / denom
        if lag_min <= lag <= lag_max and norms[lag] > best_norm:
            best_norm = float(norms[lag])
            best_lag = lag

    if best_norm < 0.40 or best_lag <= lag_min:
        return 0.0, best_norm

    y_l, y_c, y_r = float(norms[best_lag - 1]), float(norms[best_lag]), float(norms[best_lag + 1])
    denom = 2.0 * (y_l - 2.0 * y_c + y_r)
    delta = (y_l - y_r) / denom if abs(denom) > 1e-9 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    refined_lag = float(best_lag) + delta
    return float(sr) / max(refined_lag, 1.0), best_norm


def _extract_voiced_metrics(speech_segments: List[np.ndarray], sr: int) -> Tuple[float, float, float, float, str, str]:
    """
    Analyze contiguous voiced runs across VAD speech segments.
    Returns (jitter_pct, shimmer_db, mean_f0_hz, pitch_std_hz, jitter_status, shimmer_status).
    """
    frame_len = int(sr * 0.025)  # 25 ms
    hop_len = int(sr * 0.008)    # 8 ms
    shimmer_hop = int(sr * 0.010)  # 10 ms non-overlapping chunks for unsmoothed amplitude delta

    all_f0: List[float] = []
    jitter_rel_diffs: List[float] = []
    shimmer_diffs_db: List[float] = []

    for seg in speech_segments:
        if len(seg) < frame_len * 2:
            continue

        # 1. Track sub-sample F0 within contiguous voiced runs
        frames, _ = frame_signal(seg, frame_len, hop_len, window_type="hanning")
        rms_f = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        rms_thresh = max(0.003, 0.18 * float(np.max(rms_f)))

        prev_f0 = 0.0
        for fr, r_val in zip(frames, rms_f):
            if r_val < rms_thresh:
                prev_f0 = 0.0
                continue
            f0, strength = _parabolic_f0(fr, sr)
            if f0 > 65.0 and strength >= 0.42:
                all_f0.append(f0)
                if prev_f0 > 65.0 and abs(f0 - prev_f0) / prev_f0 < 0.15:  # exclude octave jumps
                    jitter_rel_diffs.append(abs(f0 - prev_f0) / prev_f0)
                prev_f0 = f0
            else:
                prev_f0 = 0.0

        # 2. Non-overlapping 10 ms sub-frame amplitude shimmer (preserves real glottal variability)
        n_sub = len(seg) // shimmer_hop
        if n_sub >= 4:
            sub = seg[: n_sub * shimmer_hop].reshape(n_sub, shimmer_hop)
            sub_rms = np.sqrt(np.mean(sub ** 2, axis=1) + 1e-12)
            sub_thresh = max(0.004, 0.22 * float(np.max(sub_rms)))
            for i in range(1, n_sub):
                if sub_rms[i] >= sub_thresh and sub_rms[i - 1] >= sub_thresh:
                    db_diff = abs(20.0 * np.log10((sub_rms[i] + 1e-9) / (sub_rms[i - 1] + 1e-9)))
                    shimmer_diffs_db.append(float(db_diff))

    mean_f0 = round(float(np.mean(all_f0)), 1) if all_f0 else 0.0
    pitch_std = round(float(np.std(all_f0)), 1) if all_f0 else 0.0

    if len(jitter_rel_diffs) < 5:
        jitter_pct = 0.0
        jitter_status = "Insufficient Voiced Speech"
    else:
        jitter_pct = round(float(np.median(jitter_rel_diffs)) * 100.0, 2)
        if jitter_pct >= 0.25 and jitter_pct <= 4.0:
            jitter_status = "Natural Human Micro-Tremor"
        elif jitter_pct < 0.25:
            jitter_status = "Oversmoothed (Synthetic Pitch Track)"
        else:
            jitter_status = "High Pitch Perturbation"

    if len(shimmer_diffs_db) < 6:
        shimmer_db = 0.0
        shimmer_status = "Insufficient Voiced Speech"
    else:
        shimmer_db = round(float(np.median(shimmer_diffs_db)), 2)
        if 0.35 <= shimmer_db <= 4.5:
            shimmer_status = "Natural Amplitude Dynamics"
        elif shimmer_db < 0.35:
            shimmer_status = "Unnaturally Uniform (Synthetic Envelope)"
        else:
            shimmer_status = "High Dynamic Modulation"

    return jitter_pct, shimmer_db, mean_f0, pitch_std, jitter_status, shimmer_status


def _extract_active_lpc_and_spectral(speech_segments: List[np.ndarray], sr: int) -> Tuple[float, float, float, str, str]:
    """Compute LPC glottal residual kurtosis/flatness and HF spectral cutoff on loudest speech frames."""
    speech_cat = np.concatenate(speech_segments) if speech_segments else np.array([], dtype=np.float32)
    if len(speech_cat) < 400:
        return 0.0, 0.0, 0.0, "Too Short", "Too Short"

    frames, _ = frame_signal(speech_cat, frame_length=400, hop_length=160, window_type="hanning")
    rms_f = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    top_idx = np.argsort(rms_f)[-min(12, len(frames)) :]

    kurt_list: List[float] = []
    flat_list: List[float] = []
    hf_ratios: List[float] = []

    freqs = np.fft.rfftfreq(512, 1.0 / sr)
    low_mask = (freqs >= 300.0) & (freqs < 3500.0)
    high_mask = freqs >= 7000.0

    for idx in top_idx:
        fr = frames[idx]
        if rms_f[idx] < 1e-4:
            continue

        # 16th-order LPC inverse filter residual
        lpc = analyze_frame_lpc(fr, sr=sr, order=16, n_fft=512)
        if len(lpc.lpc_coefficients) >= 16:
            res = sp_signal.lfilter(lpc.lpc_coefficients, [1.0], fr)
            std_r = float(np.std(res))
            if std_r > 1e-7:
                norm_r = (res - float(np.mean(res))) / std_r
                kurt_list.append(float(np.mean(norm_r ** 4)) - 3.0)
            res_pow = np.abs(np.fft.rfft(res, n=512)) ** 2 + 1e-12
            geo = float(np.exp(np.mean(np.log(res_pow))))
            arith = float(np.mean(res_pow))
            flat_list.append(geo / max(arith, 1e-12))

        # High-band spectral cutoff ratio
        spec_pow = np.abs(np.fft.rfft(fr, n=512)) ** 2
        low_e = float(np.sum(spec_pow[low_mask]))
        high_e = float(np.sum(spec_pow[high_mask]))
        if low_e > 1e-9:
            hf_ratios.append(high_e / low_e)

    kurtosis = round(float(np.median(kurt_list)), 2) if kurt_list else 0.0
    flatness = round(float(np.median(flat_list)), 3) if flat_list else 0.0
    hf_ratio = round(float(np.median(hf_ratios)), 4) if hf_ratios else 0.0

    if not kurt_list:
        lpc_status = "Insufficient Voiced Frames"
    elif kurtosis >= 1.8:
        lpc_status = "Natural Glottal Excitation"
    elif kurtosis < 0.8:
        lpc_status = "Diffuse / Synthetic Residual"
    else:
        lpc_status = "Moderate Glottal Structure"

    if not hf_ratios:
        hf_status = "Insufficient Spectrum"
    elif hf_ratio <= 0.0008:
        hf_status = "Steep HF Cutoff (Bandlimited Vocoder)"
    else:
        hf_status = "Natural Harmonic Roll-off"

    return flatness, kurtosis, hf_ratio, lpc_status, hf_status


def detect_synthetic_voice(
    samples: np.ndarray,
    sr: int = 16000,
    speech_intervals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Perform forensic acoustic authentication on spoken human speech segments
    to evaluate natural human vocal tract dynamics vs synthetic/TTS artifacts.
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.size < sr * 0.4:
        return {
            "verdict": "Audio Sample Too Short",
            "authenticity_score": 0,
            "is_synthetic": False,
            "confidence": 0.0,
            "risk_level": "unknown",
            "summary": "Record or upload at least 1.0 second of spoken speech for forensic analysis.",
            "metrics": {},
        }

    peak = float(np.max(np.abs(x)))
    if peak <= 1e-4:
        return {
            "verdict": "Silence Detected",
            "authenticity_score": 0,
            "is_synthetic": False,
            "confidence": 0.0,
            "risk_level": "unknown",
            "summary": "No acoustic signal energy present.",
            "metrics": {},
        }

    x_norm = x / peak
    vad_segs = speech_intervals if speech_intervals is not None else detect_speech(x_norm, sr, min_segment_rms=0.002)

    speech_chunks: List[np.ndarray] = []
    for seg in vad_segs:
        i0 = max(0, int(float(seg["start"]) * sr))
        i1 = min(len(x_norm), int(float(seg["end"]) * sr))
        if i1 - i0 >= int(0.18 * sr):
            speech_chunks.append(x_norm[i0:i1])

    total_speech_s = sum(len(c) for c in speech_chunks) / float(sr)
    if total_speech_s < 0.30:
        return {
            "verdict": "No Human Speech Detected",
            "authenticity_score": 0,
            "is_synthetic": False,
            "confidence": 0.0,
            "risk_level": "unknown",
            "summary": "No spoken speech detected above ambient noise floor. Speak clearly into the microphone or upload voice audio.",
            "metrics": {},
        }

    jitter_pct, shimmer_db, mean_f0, pitch_std, jitter_status, shimmer_status = _extract_voiced_metrics(speech_chunks, sr)
    flatness, kurtosis, hf_ratio, lpc_status, hf_status = _extract_active_lpc_and_spectral(speech_chunks, sr)

    if jitter_pct == 0.0 and shimmer_db == 0.0:
        return {
            "verdict": "Insufficient Voiced Phonemes",
            "authenticity_score": 50,
            "is_synthetic": False,
            "confidence": 0.2,
            "risk_level": "unknown",
            "summary": "Captured speech was too brief or unvoiced to reliably measure glottal cycle perturbations.",
            "metrics": {
                "jitter_percent": jitter_pct,
                "jitter_status": jitter_status,
                "shimmer_db": shimmer_db,
                "shimmer_status": shimmer_status,
                "lpc_residual_kurtosis": kurtosis,
                "lpc_residual_flatness": flatness,
                "lpc_status": lpc_status,
                "hf_ratio": hf_ratio,
                "hf_status": hf_status,
                "mean_f0_hz": mean_f0,
            },
        }

    # Conservative forensic calibration: preserve human authenticity unless clear synthetic patterns co-occur
    score = 72.0

    # Pitch micro-jitter
    if 0.22 <= jitter_pct <= 4.0:
        score += 12.0
    elif jitter_pct < 0.12:
        score -= 28.0
    elif jitter_pct < 0.22:
        score -= 18.0
    elif jitter_pct > 5.0:
        score -= 6.0

    # Amplitude shimmer
    if 0.35 <= shimmer_db <= 4.5:
        score += 10.0
    elif shimmer_db < 0.30:
        score -= 22.0

    # LPC excitation kurtosis
    if kurtosis >= 1.8:
        score += 6.0
    elif kurtosis < 0.7:
        score -= 14.0

    # High-band vocoder rolloff
    if hf_ratio <= 0.0008:
        score -= 12.0
    else:
        score += 4.0

    # Prosodic phrase inflection check
    if pitch_std < 3.0 and mean_f0 > 0:
        score -= 14.0  # monotone robotic pitch

    auth_score = int(np.clip(round(score), 5, 98))

    if auth_score >= 65:
        verdict = "Authentic Human Voice"
        is_synthetic = False
        risk_level = "low"
        summary = (
            f"Vocal biomarkers align with a natural human vocal tract: natural micro-pitch tremor "
            f"(jitter {jitter_pct}%), organic energy shimmer ({shimmer_db} dB), and impulsive LPC glottal structure."
        )
    elif auth_score <= 44:
        verdict = "Likely AI-Generated / Synthetic"
        is_synthetic = True
        risk_level = "high"
        summary = (
            f"Detected synthetic speech characteristics: oversmoothed pitch contour (jitter {jitter_pct}%), "
            f"uniform cycle energy ({shimmer_db} dB), or bandlimited neural vocoder spectrum."
        )
    else:
        verdict = "Borderline / Mixed Biomarkers"
        is_synthetic = False
        risk_level = "medium"
        summary = (
            f"Acoustic biomarkers are inconclusive ({auth_score}% human similarity; jitter {jitter_pct}%, shimmer {shimmer_db} dB). "
            f"Heavy audio compression or noise filtering can mask natural glottal micro-tremor."
        )

    return {
        "verdict": verdict,
        "authenticity_score": auth_score,
        "is_synthetic": is_synthetic,
        "confidence": round(max(0.15, abs(auth_score - 50) / 50.0), 2),
        "risk_level": risk_level,
        "summary": summary,
        "metrics": {
            "jitter_percent": jitter_pct,
            "jitter_status": jitter_status,
            "shimmer_db": shimmer_db,
            "shimmer_status": shimmer_status,
            "lpc_residual_kurtosis": kurtosis,
            "lpc_residual_flatness": flatness,
            "lpc_status": lpc_status,
            "hf_ratio": hf_ratio,
            "hf_status": hf_status,
            "mean_f0_hz": mean_f0,
        },
    }
