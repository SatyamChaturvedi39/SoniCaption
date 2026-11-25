"""End-to-end analysis used by the upload tab and by the final pass of the live tab.

    load -> denoise (copy, for diarization / spectral views) -> VAD -> scene classification (speech / sound effect /
    background noise / silence, on the RAW audio so noise is not removed before it is classified) -> speaker diarization
    -> ASR -> speaker-attributed transcript -> NLU (scenario, per-speaker topics, per-speaker tasks) -> pattern views
    (MFCC, spectrogram, pitch, HMM) -> one JSON-serialisable result.

Nothing here fabricates content: when ASR is unavailable the transcript is empty and `asr.status` says why, so the UI can
show it instead of pretending that speech was understood.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Callable

import numpy as np

SR = 16000
SPECTRAL_CAP_S = 180.0

Progress = Callable[[str, float, str], None]


# ----------------------------------------------------------------------------------------------- small helpers
def _ds1(arr, n: int) -> list[float]:
    a = np.asarray(arr, dtype=np.float32).ravel()
    if a.size == 0:
        return []
    if a.size > n:
        a = a[np.linspace(0, a.size - 1, n, dtype=int)]
    return [round(float(v), 4) for v in a]


def _ds2(mat, max_rows: int, max_cols: int) -> list[list[float]]:
    m = np.asarray(mat, dtype=np.float32)
    if m.ndim != 2 or m.size == 0:
        return []
    r = np.linspace(0, m.shape[0] - 1, min(m.shape[0], max_rows), dtype=int)
    c = np.linspace(0, m.shape[1] - 1, min(m.shape[1], max_cols), dtype=int)
    return [[round(float(v), 2) for v in row] for row in m[np.ix_(r, c)]]


def _noop(stage: str, frac: float, msg: str) -> None:  # noqa: ARG001
    return None


STATUS = {"speech": "ok", "sound_effect": "alert", "background_noise": "warn", "silence": "idle"}


# ----------------------------------------------------------------------------------------------- ASR
def run_asr(path: str, vad_segments: list[dict], asr_fn: Callable | None, language: str | None = None) -> dict:
    """Returns {"segments", "status": ok|unavailable|skipped, "provider", "language", "warnings", "error"}."""
    if not vad_segments:
        return {"segments": [], "status": "skipped", "provider": None, "language": None,
                "warnings": ["No speech was detected, so no transcription was attempted."], "error": None}
    try:
        if asr_fn is None:
            from soniccaption.services.asr_service import transcribe_audio
            res = transcribe_audio(path, language=language)
        else:
            res = asr_fn(path)
        return {"segments": list(res.get("segments", [])), "status": "ok", "provider": res.get("provider", "asr"),
                "language": res.get("language"), "warnings": list(res.get("warnings", [])), "error": None}
    except Exception as exc:  # noqa: BLE001 - ASR is the one component that needs network/model downloads; report, don't crash
        return {"segments": [], "status": "unavailable", "provider": None, "language": None,
                "warnings": ["Speech-to-text did not run, so there is no transcript. Speakers, noises and sound effects are still analysed."],
                "error": str(exc)[:400]}


# ----------------------------------------------------------------------------------------------- pattern views
def _speaker_patterns(x: np.ndarray, turns: list[dict], n_speakers: int) -> list[dict]:
    from soniccaption.algorithms.enhancement import estimate_pitch_contour
    from soniccaption.algorithms.spectral import compute_mfcc
    out = []
    for sid in range(1, n_speakers + 1):
        chunks, total = [], 0.0
        for t in turns:
            if t["speaker_id"] != sid or total > 20.0:
                continue
            a, b = int(t["start"] * SR), int(t["end"] * SR)
            c = x[a:b]
            if c.size > 0:
                chunks.append(c)
                total += c.size / SR
        if not chunks:
            continue
        y = np.concatenate(chunks)
        if y.size < 1600:
            continue
        pitch = estimate_pitch_contour(y, sr=SR)
        mf, _, _ = compute_mfcc(y, sr=SR, n_mfcc=13, n_mels=40)
        voiced = [f for f in pitch["f0_hz"] if f > 0]
        out.append({"speaker_id": sid, "speaker": f"Speaker {sid}", "seconds_analysed": round(y.size / SR, 1),
                    "mean_f0_hz": round(float(np.mean(voiced)), 1) if voiced else 0.0,
                    "f0_std_hz": round(float(np.std(voiced)), 1) if voiced else 0.0,
                    "voiced_ratio": round(float(pitch.get("voiced_ratio", 0.0)), 3),
                    "mfcc_mean": [round(float(v), 3) for v in mf.mean(axis=1)],
                    "mfcc_std": [round(float(v), 3) for v in mf.std(axis=1)]})
    return out


def _spectral_views(raw: np.ndarray, den: np.ndarray, den_info: dict, duration: float) -> dict:
    from soniccaption.algorithms.enhancement import estimate_pitch_contour
    from soniccaption.algorithms.lpc import analyze_frame_lpc
    from soniccaption.algorithms.spectral import compute_mfcc, compute_rms, compute_stft, compute_zcr
    from soniccaption.audio.preprocessing import frame_signal
    n = int(min(raw.size, SPECTRAL_CAP_S * SR))
    r, d = raw[:n], den[:n]
    if d.size < 800:
        return {"covered_seconds": 0.0}
    frames, _ = frame_signal(d, 400, 160)
    rms = compute_rms(frames)
    zcr = compute_zcr(frames)
    _, _, stft_db = compute_stft(d, sr=SR, frame_length=400, hop_length=160, n_fft=512)
    mfcc, _, _ = compute_mfcc(d, sr=SR, n_mfcc=13, n_mels=40)
    pitch = estimate_pitch_contour(d, sr=SR)
    best = int(np.argmax(rms)) if len(rms) else 0
    lpc = analyze_frame_lpc(frames[best], sr=SR, order=16, n_fft=512) if len(frames) else None
    return {
        "covered_seconds": round(n / SR, 1),
        "waveform": _ds1(r, 600), "waveform_denoised": _ds1(d, 600), "rms": _ds1(rms, 240), "zcr": _ds1(zcr, 240),
        "stft_db": _ds2(stft_db, 64, 240), "stft_freq_max_hz": SR // 2,
        "mfcc": _ds2(mfcc, 13, 240),
        "f0_contour_hz": _ds1(np.array(pitch["f0_hz"], dtype=np.float32), 240), "f0_mean_hz": round(float(pitch["mean_f0_hz"]), 1),
        "voiced_ratio": round(float(pitch.get("voiced_ratio", 0.0)), 3),
        "lpc_formants_hz": [round(float(f), 0) for f in (lpc.formant_frequencies[:4] if lpc else [])],
        "lpc_envelope_db": _ds1(lpc.spectral_envelope, 64) if lpc else [],
        "snr_before_db": den_info.get("snr_before_db"), "snr_after_db": den_info.get("snr_after_db"),
        "snr_gain_db": den_info.get("snr_gain_db"), "noise_floor_rms": den_info.get("noise_floor_rms"),
    }


def _hmm_views(duration: float, vad: list[dict], scene: dict, utterances: list[dict]) -> dict:
    from soniccaption.algorithms.hmm import decode_prosody_hmm
    hm = scene["hmm"]
    step = max(1.0, duration / 120.0)
    prosody = decode_prosody_hmm(utterances, duration, step_s=step, max_steps=120) if utterances else None
    hops = scene["hops"]
    return {"scene": {"hop_s": round(hops[0]["t1"] - hops[0]["t0"], 2) if hops else 0.5, "observations": hm.get("observations", []),
                      "path": scene.get("hop_states", []), "log_likelihood": hm.get("log_likelihood"),
                      "best_path_log_prob": hm.get("best_path_log_prob"), "model": hm.get("model"),
                      "trellis_preview": hm.get("trellis_preview")},
            "prosody": prosody}


# ----------------------------------------------------------------------------------------------- timeline
def build_timeline(scene_segments: list[dict], utterances: list[dict]) -> list[dict]:
    """One chronological list: speech (with words, speaker, text) + sound effects + background noise + silence."""
    items: list[dict] = []
    speech_segs = [s for s in scene_segments if s["state"] == "speech"]
    for u in utterances:
        items.append({"start": round(float(u["start"]), 2), "end": round(float(u["end"]), 2), "category": "speech", "label": "Speech",
                      "text": u.get("text", ""), "speaker": u.get("speaker"), "speaker_id": u.get("speaker_id"),
                      "confidence": u.get("confidence"), "status": "ok", "words": u.get("words", []),
                      "delivery_style": u.get("delivery_style")})
    covered = [(i["start"], i["end"]) for i in items]
    for s in speech_segs:                                       # speech the ASR produced no text for stays visible as speech
        mid = (s["start"] + s["end"]) / 2
        if not any(a - 0.05 <= mid <= b + 0.05 for a, b in covered):
            items.append({"start": s["start"], "end": s["end"], "category": "speech", "label": "Speech", "text": "",
                          "speaker": s.get("speaker"), "speaker_id": s.get("speaker_id"), "confidence": s.get("confidence"),
                          "status": "ok", "words": []})
    for s in scene_segments:
        if s["state"] == "speech":
            continue
        items.append({"start": s["start"], "end": s["end"], "category": s["state"], "label": s["label"], "key": s.get("key"),
                      "text": "", "speaker": None, "speaker_id": None, "confidence": s.get("confidence"),
                      "status": STATUS.get(s["state"], "idle"), "words": []})
    items.sort(key=lambda i: (i["start"], i["end"]))
    for n, it in enumerate(items, 1):
        it["id"] = f"t{n}"
    return items


def _tag_speech_segments(scene_segments: list[dict], turns: list[dict]) -> None:
    from soniccaption.algorithms.diarization import speaker_at
    for s in scene_segments:
        if s["state"] == "speech":
            sid = speaker_at(turns, (s["start"] + s["end"]) / 2, 0.5)
            if sid:
                s["speaker_id"], s["speaker"] = sid, f"Speaker {sid}"


# ----------------------------------------------------------------------------------------------- main entry
def analyse_audio(path: str, filename: str | None = None, *, denoise: bool = True, speaker_mode: str = "auto",
                  asr_fn: Callable | None = None, language: str | None = None, progress: Progress | None = None,
                  use_llm: bool = True, samples: np.ndarray | None = None) -> dict:
    """Run the whole pipeline on an audio file. `asr_fn(path) -> {"segments": [...], "language", "provider"}` can be injected
    (tests, offline demos); by default Groq / faster-whisper are used. Returns a JSON-serialisable dict."""
    from soniccaption.algorithms.diarization import assign_speakers, diarize
    from soniccaption.algorithms.enhancement import analyze_segment_prosody, spectral_subtraction_denoise
    from soniccaption.audio.io import load_audio
    from soniccaption.audio.vad import detect_speech, speech_ratio
    from soniccaption.services.nlu import analyse_conversation
    from soniccaption.services.scene_classifier import analyze_scene

    prog = progress or _noop
    t_all = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(name: str, t0: float) -> None:
        timings[name] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    prog("load", 0.02, "Reading audio")
    if samples is None:
        audio = load_audio(path, target_sr=SR, to_mono=True, normalize_peak=False)
        raw = np.asarray(audio.samples, dtype=np.float32)
    else:
        raw = np.asarray(samples, dtype=np.float32)
    duration = raw.size / SR
    if duration < 0.3:
        raise ValueError("Audio is too short to analyse (under 0.3 s).")
    lap("load", t0)

    t0 = time.perf_counter()
    prog("denoise", 0.08, "Estimating background noise")
    den_info = spectral_subtraction_denoise(raw, sr=SR)
    den = np.asarray(den_info["denoised"], dtype=np.float32) if denoise else raw
    lap("denoise", t0)

    t0 = time.perf_counter()
    prog("vad", 0.15, "Finding speech")
    vad = detect_speech(raw, SR)
    lap("vad", t0)

    t0 = time.perf_counter()
    prog("scene", 0.22, "Classifying speech, noise and sound effects")
    scene = analyze_scene(raw, vad)
    lap("scene", t0)

    t0 = time.perf_counter()
    prog("diarize", 0.50, "Telling speakers apart")
    dia = diarize(den, vad, speaker_mode=speaker_mode)
    turns = dia["turns"]
    lap("diarize", t0)

    t0 = time.perf_counter()
    prog("asr", 0.62, "Transcribing speech")
    asr = run_asr(path, vad, asr_fn, language)
    lap("asr", t0)

    prog("assign", 0.85, "Attributing words to speakers")
    utterances = assign_speakers(asr["segments"], turns) if asr["segments"] else []
    # Whisper hallucinates on non-speech: drop text that has almost no VAD speech under it
    kept = []
    for u in utterances:
        a, b = float(u["start"]), float(u["end"])
        cov = sum(max(0.0, min(b, v["end"]) - max(a, v["start"])) for v in vad)
        if str(u.get("text", "")).strip() and (b <= a or cov / max(b - a, 1e-6) >= 0.3):
            kept.append(u)
    dropped = len(utterances) - len(kept)
    utterances = analyze_segment_prosody(den, kept, sr=SR) if kept else []
    if dropped:
        asr["warnings"].append(f"{dropped} transcript segment(s) fell on audio with no speech and were discarded as likely ASR hallucination.")

    _tag_speech_segments(scene["segments"], turns)
    timeline = build_timeline(scene["segments"], utterances)

    t0 = time.perf_counter()
    prog("nlu", 0.90, "Understanding the conversation")
    sound_events = [s for s in scene["segments"] if s["state"] != "speech"]
    nlu = analyse_conversation(utterances, sound_events, duration, use_llm=use_llm)
    if asr["status"] != "ok" and not utterances:
        nlu["warnings"] = list(nlu.get("warnings", [])) + ["Scenario, topics and tasks need a transcript; ASR is not available."]
    lap("nlu", t0)

    t0 = time.perf_counter()
    prog("patterns", 0.95, "Computing MFCC, spectrogram and HMM views")
    patterns = {"spectral": _spectral_views(raw, den, den_info, duration),
                "speakers": _speaker_patterns(den, turns, dia["n_speakers"]),
                "hmm": _hmm_views(duration, vad, scene, utterances)}
    lap("patterns", t0)

    speakers = []
    names = {int(k): v for k, v in nlu.get("names", {}).items()}
    profiles = {p["speaker_id"]: p for p in nlu["speakers"]}
    cents = {c["speaker_id"]: c for c in dia["centroids"]}
    for sid in range(1, dia["n_speakers"] + 1):
        p = profiles.get(sid, {})
        speakers.append({"speaker_id": sid, "speaker": f"Speaker {sid}", "name": names.get(sid),
                         "label": p.get("label", f"Speaker {sid}"),
                         "seconds": cents.get(sid, {}).get("seconds", 0.0), "share": p.get("share"),
                         "topics": p.get("topics", []), "summary": p.get("summary", ""), "turns": p.get("turns", 0)})
    prog("done", 1.0, "Done")
    warnings: list[str] = []
    if dia["n_speakers"] and dia["backend"] != "neural":
        warnings.append("The neural speaker model is not available, so a simple signal-processing voice fingerprint was used. "
                        "Speaker separation is much less reliable (about coin-flip accuracy in our tests). Install torch and re-run.")
    return {
        "warnings": warnings,
        "filename": filename or os.path.basename(path), "duration": round(duration, 2), "sample_rate": SR,
        "denoised": bool(denoise), "vad_speech_ratio": round(speech_ratio(vad, duration), 3) if vad else 0.0,
        "asr": {k: asr[k] for k in ("status", "provider", "language", "warnings", "error")},
        "scene": {"segments": scene["segments"], "stats": scene["stats"]},
        "diarization": {"n_speakers": dia["n_speakers"], "backend": dia["backend"], "turns": turns,
                        "projection": [{**q, "speaker_id": q.get("speaker")} for q in dia.get("projection", [])], "threshold": dia.get("threshold")},
        "speakers": speakers, "utterances": [{k: v for k, v in u.items() if k != "words"} | {"n_words": len(u.get("words", []))}
                                             for u in utterances],
        "timeline": timeline, "nlu": nlu, "patterns": patterns,
        "timings_ms": {**timings, "total": round((time.perf_counter() - t_all) * 1000, 1)},
    }


def analyse_bytes(data: bytes, filename: str, **kw) -> dict:
    """Write an upload to a temp WAV (decoding via soundfile/ffmpeg) and analyse it."""
    from soniccaption.audio.io import load_audio, save_audio
    ext = os.path.splitext(filename or "")[1].lower() or ".wav"
    fd, src = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    wav = None
    try:
        with open(src, "wb") as fh:
            fh.write(data)
        audio = load_audio(src, target_sr=SR, to_mono=True, normalize_peak=False)
        fd2, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd2)
        save_audio(audio, wav)
        return analyse_audio(wav, filename, samples=audio.samples, **kw)
    finally:
        for p in (src, wav):
            if p and os.path.exists(p):
                os.remove(p)
