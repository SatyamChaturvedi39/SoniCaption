"""Unit tests for Member 1's enhancement, pitch tracking, and VQ speaker diarization."""

import numpy as np

from soniccaption.algorithms.enhancement import (
    diarize_speakers,
    estimate_pitch_contour,
    spectral_subtraction_denoise,
)
from soniccaption.audio.io import generate_synthetic_signal


def test_spectral_subtraction_improves_snr_and_preserves_length():
    clean = generate_synthetic_signal("vowel_ah", duration=1.0, sr=16000, f0=130.0).samples
    # Create audio with initial noise-only segment + speech+noise so noise floor drops noticeably
    rng = np.random.default_rng(7)
    noise = (0.04 * rng.standard_normal(len(clean))).astype(np.float32)
    noisy = clean.copy()
    noisy[:4000] = 0.0
    noisy = (noisy + noise).astype(np.float32)

    res = spectral_subtraction_denoise(noisy, sr=16000)
    assert res["denoised"].shape == noisy.shape
    assert res["denoised"].dtype == np.float32
    assert res["snr_after_db"] > res["snr_before_db"]
    assert res["snr_gain_db"] > 0.0


def test_pitch_contour_tracks_synthetic_vowel_f0():
    audio = generate_synthetic_signal("vowel_ah", duration=0.8, sr=16000, f0=140.0)
    res = estimate_pitch_contour(audio.samples, sr=16000)
    assert res["voiced_ratio"] > 0.5
    assert 125.0 <= res["mean_f0_hz"] <= 155.0


def test_diarize_speakers_separates_distinct_pitches():
    sr = 16000
    s1 = generate_synthetic_signal("vowel_ah", duration=0.6, sr=sr, f0=115.0).samples
    s2 = generate_synthetic_signal("vowel_ee", duration=0.6, sr=sr, f0=235.0).samples
    gap = np.zeros(int(0.2 * sr), dtype=np.float32)
    clip = np.concatenate([s1, gap, s2, gap, s1])
    segs = [
        {"start": 0.0, "end": 0.6, "text": "Hello one"},
        {"start": 0.8, "end": 1.4, "text": "Hello two"},
        {"start": 1.6, "end": 2.2, "text": "Back to one"},
    ]
    out = diarize_speakers(clip, segs, sr=sr, max_speakers=2)
    assert len(out) == 3
    assert out[0]["speaker"] == "Speaker 1"
    assert out[1]["speaker"] == "Speaker 2"
    assert out[2]["speaker"] == "Speaker 1"


def test_empty_signal_guards():
    empty = np.array([], dtype=np.float32)
    assert spectral_subtraction_denoise(empty)["snr_gain_db"] == 0.0
    assert estimate_pitch_contour(empty)["mean_f0_hz"] == 0.0
    assert diarize_speakers(empty, []) == []


def test_acoustic_scene_hmm_decoder():
    from soniccaption.algorithms.hmm import decode_acoustic_scene_hmm

    res = decode_acoustic_scene_hmm(
        duration=6.0,
        vad_segments=[{"start": 1.0, "end": 3.0}],
        sound_events=[{"label": "siren", "start": 4.0, "end": 5.5, "confidence": 0.92}],
        step_s=1.0,
    )
    assert res["step_s"] == 1.0
    assert len(res["observations"]) == 6
    assert len(res["best_path"]) == 6
    assert set(res["best_path"]).issubset({"Ambient", "Speech", "SoundEvent"})
    assert "Speech" in res["best_path"]
    assert "SoundEvent" in res["best_path"]

    empty_res = decode_acoustic_scene_hmm(0.0, [], [], step_s=1.0)
    assert isinstance(empty_res["best_path"], list)


def test_dtw_clip_signature_and_session_similarity():
    from soniccaption.algorithms.enhancement import extract_clip_signature, find_similar_sessions_dtw

    sr = 16000
    t = np.linspace(0.0, 1.2, int(sr * 1.2), endpoint=False, dtype=np.float32)
    tone_low = (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    tone_low_near = (0.4 * np.sin(2.0 * np.pi * 450.0 * t)).astype(np.float32)
    tone_high = (0.4 * np.sin(2.0 * np.pi * 3400.0 * t)).astype(np.float32)

    assert extract_clip_signature(np.array([], dtype=np.float32)) == []

    sig_q = extract_clip_signature(tone_low, sr=sr, max_frames=24)
    sig_near = extract_clip_signature(tone_low_near, sr=sr, max_frames=24)
    sig_far = extract_clip_signature(tone_high, sr=sr, max_frames=24)

    assert 0 < len(sig_q) <= 24
    assert len(sig_q[0]) > 0

    sessions = [
        {"id": 10, "filename": "self.wav", "metrics": {"mfcc_signature": sig_q}},
        {"id": 11, "filename": "far.wav", "metrics": {"mfcc_signature": sig_far}},
        {"id": 12, "filename": "near.wav", "metrics": {"mfcc_signature": sig_near}},
        {"id": 13, "filename": "empty.wav", "metrics": {"mfcc_signature": []}},
    ]
    matches = find_similar_sessions_dtw(sig_q, sessions, current_session_id=10, top_k=3)
    matched_ids = [m["session_id"] for m in matches]
    assert len(matches) == 2
    assert 10 not in matched_ids
    assert 13 not in matched_ids
    assert matched_ids[0] == 12


def test_stationary_fan_noise_suppressed_while_tonal_siren_kept():
    from soniccaption.audio.vad import detect_speech
    from soniccaption.services.sound_cnn import detect_sound_events

    sr = 16000
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 2.5, int(sr * 2.5), endpoint=False, dtype=np.float32)
    fan = (0.006 * np.sin(2.0 * np.pi * 50.0 * t) + 0.003 * np.sin(2.0 * np.pi * 100.0 * t) + 0.010 * rng.normal(0.0, 1.0, t.size)).astype(np.float32)
    fan_den = spectral_subtraction_denoise(fan, sr=sr)["denoised"]

    for sig in (fan, fan_den):
        assert detect_speech(sig, sr) == []
        assert detect_sound_events(sig, sr, threshold=0.5, silence_rms=0.012, ignore_ambient=True) == []

    # Loud tonal siren sweeping above room fan must still survive ambient rejection
    siren = (fan + 0.25 * np.sin(2.0 * np.pi * np.cumsum(700.0 + 250.0 * np.sin(2.0 * np.pi * 1.5 * t)) / sr)).astype(np.float32)
    events = detect_sound_events(siren, sr, threshold=0.30, silence_rms=0.012, ignore_ambient=True)
    assert any(e["label"] == "siren" for e in events)


def test_prosody_analysis_prosody_hmm_and_subsequence_dtw_spotter():
    from soniccaption.algorithms.enhancement import analyze_segment_prosody, subsequence_dtw_search
    from soniccaption.algorithms.hmm import decode_prosody_hmm

    sr = 16000
    s_soft = (0.35 * generate_synthetic_signal("vowel_ah", duration=0.8, sr=sr, f0=120.0).samples).astype(np.float32)
    s_loud = (0.95 * generate_synthetic_signal("vowel_ee", duration=0.8, sr=sr, f0=210.0).samples).astype(np.float32)
    gap = np.zeros(int(0.8 * sr), dtype=np.float32)
    clip = np.concatenate([gap[:sr // 2], s_soft, gap, s_loud, gap[:sr // 2]])

    segs = [
        {"start": 0.5, "end": 1.3, "speaker": "Speaker 1", "text": "Introduces the concept steadily."},
        {"start": 2.1, "end": 2.9, "speaker": "Speaker 1", "text": "Why does Viterbi beam search matter?"},
    ]
    prosody = analyze_segment_prosody(clip, segs, sr=sr)
    assert len(prosody) == 2
    assert prosody[1]["energy_ratio"] > prosody[0]["energy_ratio"]
    assert prosody[1]["delivery_style"] in ("Question / Rising", "Emphasized / Key Point")

    phmm = decode_prosody_hmm(prosody, duration=len(clip) / float(sr), step_s=1.0)
    assert len(phmm["best_path"]) == len(phmm["observations"]) > 0
    assert set(phmm["best_path"]).issubset({"TurnPause", "Explanation", "KeyEmphasis", "QuestionPrompt"})

    # Query the exact vowel_ee motif and confirm Subsequence DTW localizes it near [2.1, 2.9]s
    matches = subsequence_dtw_search(s_loud, clip, sr=sr, top_k=2)
    assert len(matches) >= 1
    top = matches[0]
    assert abs(top["start"] - 2.1) <= 0.45
    assert top["similarity_pct"] > 50.0


def test_synthetic_voice_detector():
    from soniccaption.algorithms.authenticator import detect_synthetic_voice
    sr = 16000
    # Natural human speech sample
    vowel = generate_synthetic_signal("vowel_ah", duration=1.2, sr=sr, f0=130.0).samples
    res = detect_synthetic_voice(vowel, sr=sr)
    assert "verdict" in res
    assert "authenticity_score" in res
    assert "metrics" in res
    assert 0 <= res["authenticity_score"] <= 100
    assert "jitter_percent" in res["metrics"]
    assert "shimmer_db" in res["metrics"]

    # Short silence returns appropriate verdict
    silence = np.zeros(int(sr * 0.8), dtype=np.float32)
    sil_res = detect_synthetic_voice(silence, sr=sr)
    assert sil_res["verdict"] == "Silence Detected"
    assert sil_res["authenticity_score"] == 0
