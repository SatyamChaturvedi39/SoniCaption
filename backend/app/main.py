"""SonicCaption FastAPI Backend: Live Multimodal Audio Pipeline, Labs, Agent & Persistence."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
for candidate in (BASE_DIR / "src", ROOT_DIR / "backend" / "src", BASE_DIR / "app"):
    cand_str = str(candidate)
    if candidate.exists() and cand_str not in sys.path:
        sys.path.insert(0, cand_str)

from .mock_data import DTW as MOCK_DTW, HMM as MOCK_HMM, analysis as mock_analysis

try:
    from soniccaption.audio.io import generate_synthetic_signal, load_audio
    from soniccaption.audio.preprocessing import frame_signal
    from soniccaption.audio.vad import detect_speech, speech_ratio as compute_speech_ratio
    from soniccaption.algorithms.dtw import dtw
    from soniccaption.algorithms.enhancement import (
        diarize_speakers,
        estimate_pitch_contour,
        spectral_subtraction_denoise,
    )
    from soniccaption.algorithms.hmm import DiscreteHMM, decode_for_ui, load_preset
    from soniccaption.algorithms.lpc import analyze_frame_lpc
    from soniccaption.algorithms.spectral import (
        compute_fft_spectrum,
        compute_mfcc,
        compute_rms,
        compute_stft,
        compute_zcr,
    )
    from soniccaption.services import agent_service, session_store
    from soniccaption.services.asr_service import ASRError, transcribe_audio
    from soniccaption.services.sound_cnn import detect_sound_events
    from soniccaption.timeline.fusion import FusionConfig, fuse_timeline

    HAS_LIVE_MODEL = True
    LIVE_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    print(f"[SonicCaption Warning] Could not import live models: {exc}. Falling back to mock mode.")
    HAS_LIVE_MODEL = False
    LIVE_IMPORT_ERROR = str(exc)


ALLOWED_AUDIO_EXTS = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac"}
LAST_AUDIO_CONTEXT: Dict[str, Any] = {}

app = FastAPI(
    title="SonicCaption API",
    version="0.4.0",
    description="Acoustic-Aware Multimodal Audio Studio: Spectral Enhancement, VAD, Whisper ASR, 10-Class ResNet-18, Pitch Diarization, Labs & AI Copilot.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- helpers
async def _read_upload_bytes(file: UploadFile) -> bytes:
    if hasattr(file, "read"):
        raw = file.read()
        return await raw if asyncio.iscoroutine(raw) else raw
    if hasattr(file, "file") and hasattr(file.file, "read"):
        raw = file.file.read()
        return await raw if asyncio.iscoroutine(raw) else raw
    raise HTTPException(status_code=400, detail="Unable to read uploaded file.")


def _validate_audio_ext(filename: Optional[str]) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format '{ext or 'none'}'. Supported: {', '.join(sorted(ALLOWED_AUDIO_EXTS))}.",
        )
    return ext


def _write_temp_wav(data: bytes, ext: str) -> tuple[str, List[str]]:
    """Persist bytes as a 16 kHz mono WAV temp file, transcoding via ffmpeg when non-WAV."""
    fd_in, raw_path = tempfile.mkstemp(suffix=ext)
    os.close(fd_in)
    Path(raw_path).write_bytes(data)
    cleanup_paths = [raw_path]

    if ext == ".wav":
        return raw_path, cleanup_paths

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise HTTPException(
            status_code=415,
            detail=f"ffmpeg is required on server PATH to decode {ext} audio files.",
        )
    fd_out, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd_out)
    cleanup_paths.append(wav_path)
    proc = subprocess.run(
        [ffmpeg_bin, "-y", "-i", raw_path, "-ac", "1", "-ar", "16000", wav_path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not Path(wav_path).exists():
        raise HTTPException(status_code=400, detail=f"Could not decode {ext} audio via ffmpeg.")
    return wav_path, cleanup_paths


def _cleanup_files(paths: List[str]) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass


def _wav_duration(data: bytes) -> tuple[float, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
            return frames / float(rate), rate
    except Exception:
        return 2.0, 16000


def _downsample_1d(arr: np.ndarray, target_len: int) -> List[float]:
    if arr.size == 0:
        return []
    if arr.size <= target_len:
        return [round(float(v), 4) for v in arr]
    idx = np.linspace(0, arr.size - 1, target_len, dtype=int)
    return [round(float(arr[i]), 4) for i in idx]


def _downsample_2d(mat: np.ndarray, max_rows: int = 36, max_cols: int = 48) -> List[List[float]]:
    if mat.size == 0:
        return []
    r_idx = np.linspace(0, mat.shape[0] - 1, min(mat.shape[0], max_rows), dtype=int)
    c_idx = np.linspace(0, mat.shape[1] - 1, min(mat.shape[1], max_cols), dtype=int)
    sub = mat[np.ix_(r_idx, c_idx)]
    return [[round(float(v), 2) for v in row] for row in sub]


def _transcribe_with_vad_fallback(wav_path: str, vad_segments: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str, Optional[str], List[str]]:
    warnings: List[str] = []
    try:
        asr = transcribe_audio(wav_path)
        return asr["segments"], asr["provider"], asr.get("language"), list(asr.get("warnings", []))
    except ASRError as exc:
        warnings.append(f"Whisper ASR unavailable ({exc}); showing Silero VAD speech intervals.")
        segs = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": f"[Speech detected ({seg['start']:.2f}s - {seg['end']:.2f}s) — configure GROQ_API_KEY in Settings for full transcription]",
                "words": [],
            }
            for seg in vad_segments
        ]
        return segs, "silero-vad-only", None, warnings


def _run_multimodal_core(wav_path: str, filename: str, denoise: bool = True) -> Dict[str, Any]:
    """Execute end-to-end Member 1 DSP + Member 2 VAD/ASR/ResNet/Fusion pipeline."""
    audio_data = load_audio(wav_path, target_sr=16000, to_mono=True, normalize_peak=False)
    duration = float(audio_data.duration)
    sample_rate = int(audio_data.sample_rate)
    raw_samples = audio_data.samples

    denoise_info = spectral_subtraction_denoise(raw_samples, sr=sample_rate)
    active_samples = denoise_info["denoised"] if denoise else raw_samples

    vad_segments = detect_speech(active_samples, sample_rate)
    vad_ratio = compute_speech_ratio(vad_segments, duration)

    t_asr = time.perf_counter()
    raw_speech, speech_source, detected_lang, asr_warnings = _transcribe_with_vad_fallback(wav_path, vad_segments)
    asr_time_ms = round((time.perf_counter() - t_asr) * 1000, 1)

    pitch_stats = estimate_pitch_contour(active_samples, sr=sample_rate)
    diarized_speech = diarize_speakers(active_samples, raw_speech, sr=sample_rate, max_speakers=2)

    t_cnn = time.perf_counter()
    raw_events = detect_sound_events(active_samples, sample_rate, threshold=0.0)
    cnn_time_ms = round((time.perf_counter() - t_cnn) * 1000, 1)

    fused = fuse_timeline(
        diarized_speech,
        raw_events,
        vad_segments=vad_segments,
        config=FusionConfig(duration=duration),
        speech_source=speech_source,
    )
    timeline = fused["timeline"]

    # Map speaker identity back onto chronological speech items
    spk_idx = 0
    for item in timeline:
        if item["category"] == "speech":
            speaker = diarized_speech[spk_idx].get("speaker", "Speaker 1") if spk_idx < len(diarized_speech) else "Speaker 1"
            item["speaker"] = speaker
            item["label"] = speaker
            spk_idx += 1

    speech_segments = [
        {
            "id": idx,
            "start": item["start"],
            "end": item["end"],
            "speaker": item.get("speaker", "Speaker 1"),
            "text": item["display_text"],
            "words": item.get("words", []),
        }
        for idx, item in enumerate((x for x in timeline if x["category"] == "speech"), start=1)
    ]

    sound_events = [
        {
            "start": item["start"],
            "end": item["end"],
            "label": item["label"],
            "confidence": item["confidence"],
            "model_source": item.get("source", "soniccaption-esc50-resnet18"),
        }
        for item in timeline
        if item["category"] == "sound_event"
    ]

    LAST_AUDIO_CONTEXT.clear()
    LAST_AUDIO_CONTEXT.update(
        {
            "filename": filename,
            "raw_samples": raw_samples,
            "denoised_samples": denoise_info["denoised"],
            "sr": sample_rate,
            "denoise_info": {k: v for k, v in denoise_info.items() if k != "denoised"},
        }
    )

    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "raw_samples": raw_samples,
        "active_samples": active_samples,
        "denoise_info": denoise_info,
        "vad_ratio": vad_ratio,
        "speech_source": speech_source,
        "language": detected_lang,
        "asr_warnings": asr_warnings,
        "asr_time_ms": asr_time_ms,
        "cnn_time_ms": cnn_time_ms,
        "pitch_stats": pitch_stats,
        "speech_segments": speech_segments,
        "sound_events": sound_events,
        "timeline": timeline,
        "fused": fused,
    }


# --------------------------------------------------------------------------- health & config
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": "live" if HAS_LIVE_MODEL else "mock",
        "model": os.environ.get("SOUND_MODEL_SOURCE", "soniccaption-esc50-resnet18"),
        "groq_configured": bool(os.environ.get("GROQ_API_KEY")),
        "ffmpeg_available": bool(shutil.which("ffmpeg")),
        "import_error": LIVE_IMPORT_ERROR,
        "message": "SonicCaption live product pipeline ready." if HAS_LIVE_MODEL else "Running in mock fallback mode.",
    }


@app.post("/api/v1/settings/groq-key")
def set_groq_key(payload: Dict[str, str]):
    key = (payload.get("api_key") or "").strip()
    if key:
        os.environ["GROQ_API_KEY"] = key
    else:
        os.environ.pop("GROQ_API_KEY", None)
    return {"groq_configured": bool(os.environ.get("GROQ_API_KEY"))}


# --------------------------------------------------------------------------- audio analysis
@app.post("/api/v1/audio/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    denoise: bool = Form(True),
    source_type: str = Form("upload"),
    save_log: bool = Form(True),
):
    ext = _validate_audio_ext(file.filename)
    data = await _read_upload_bytes(file)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if not HAS_LIVE_MODEL:
        duration, rate = _wav_duration(data)
        result = mock_analysis(file.filename or "audio.wav", max(duration, 1.0))
        result["sample_rate"] = rate
        return result

    cleanup_paths: List[str] = []
    try:
        wav_path, cleanup_paths = _write_temp_wav(data, ext)
        res = _run_multimodal_core(wav_path, file.filename or "audio.wav", denoise=denoise)

        insights = agent_service.extract_session_insights(
            res["speech_segments"],
            res["sound_events"],
            pitch_stats=res["pitch_stats"],
        )

        metrics = {
            "model_architecture": "SonicAcousticResNet-18",
            "dataset": "ESC-50 (10 balanced classes incl. laughing)",
            "classes_monitored": 10,
            "best_val_accuracy": 88.75,
            "test_accuracy": 80.0,
            "macro_f1": 0.787,
            "speech_source": res["speech_source"],
            "language": res["language"],
            "vad_speech_ratio": res["vad_ratio"],
            "denoised": denoise,
            "snr_before_db": res["denoise_info"]["snr_before_db"],
            "snr_after_db": res["denoise_info"]["snr_after_db"],
            "snr_gain_db": res["denoise_info"]["snr_gain_db"],
            "mean_f0_hz": res["pitch_stats"]["mean_f0_hz"],
            "voiced_ratio": res["pitch_stats"]["voiced_ratio"],
            "asr_inference_time_ms": res["asr_time_ms"],
            "cnn_inference_time_ms": res["cnn_time_ms"],
            "fusion_stats": res["fused"]["stats"],
            "dropped_events": res["fused"]["dropped"],
            "warnings": res["asr_warnings"],
            "summary": insights["summary"],
            "insights_provider": insights["provider"],
        }

        session_id = None
        saved_reminders: List[Dict[str, Any]] = []
        if save_log:
            session_id = session_store.save_session(
                filename=file.filename or "recording.wav",
                source_type=source_type,
                duration=round(res["duration"], 2),
                speech_source=res["speech_source"],
                vad_speech_ratio=res["vad_ratio"],
                snr_before_db=res["denoise_info"]["snr_before_db"],
                snr_after_db=res["denoise_info"]["snr_after_db"],
                denoised=denoise,
                summary=insights["summary"],
                timeline=res["timeline"],
                metrics=metrics,
            )
            for rem in insights["reminders"]:
                saved_reminders.append(
                    session_store.add_reminder(
                        title=rem["title"],
                        due_time=rem["due_time"],
                        priority=rem["priority"],
                        trigger_source=rem["trigger_source"],
                        session_id=session_id,
                    )
                )
            for ev in res["sound_events"]:
                key = agent_service.canonical_sound_key(ev["label"])
                if key in agent_service.URGENT_SOUNDS and float(ev.get("confidence", 0)) >= 0.60:
                    sev, desc = agent_service.URGENT_SOUNDS[key]
                    session_store.log_alert(
                        event_label=ev["label"],
                        confidence=float(ev["confidence"]),
                        severity=sev,
                        snippet_text=f"{desc} inside {file.filename or 'recording'}",
                    )

        return {
            "session_id": session_id,
            "filename": file.filename,
            "duration": round(res["duration"], 2),
            "sample_rate": res["sample_rate"],
            "mode": "live",
            "summary": insights["summary"],
            "reminders": saved_reminders if saved_reminders else insights["reminders"],
            "speech_segments": res["speech_segments"],
            "sound_events": res["sound_events"],
            "timeline": res["timeline"],
            "metrics": metrics,
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[SonicCaption Error] Live analysis error: {exc}. Falling back to baseline.")
        duration, rate = _wav_duration(data)
        fallback = mock_analysis(file.filename or "audio.wav", max(duration, 1.0))
        fallback["sample_rate"] = rate
        fallback["metrics"]["fallback_error"] = str(exc)
        return fallback
    finally:
        _cleanup_files(cleanup_paths)


# --------------------------------------------------------------------------- Voice AI Agent & Watch Mode
@app.post("/api/v1/agent/turn")
async def voice_agent_turn(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    denoise: bool = Form(True),
    history_json: str = Form("[]"),
):
    if not HAS_LIVE_MODEL:
        raise HTTPException(status_code=503, detail="Live ML pipeline unavailable.")

    try:
        history = json.loads(history_json) if history_json else []
    except json.JSONDecodeError:
        history = []

    cleanup_paths: List[str] = []
    try:
        user_text = (text or "").strip()
        sound_events: List[Dict[str, Any]] = []
        pitch_stats: Dict[str, Any] = {}
        warnings: List[str] = []
        speech_source = "text-chat"
        duration = 0.0
        snr_before, snr_after, vad_ratio = 0.0, 0.0, 0.0
        timeline: List[Dict[str, Any]] = []
        filename = "voice_agent_turn"

        if file is not None:
            ext = _validate_audio_ext(file.filename)
            data = await _read_upload_bytes(file)
            if not data:
                raise HTTPException(status_code=400, detail="Audio turn is empty.")
            filename = file.filename or "voice_turn.wav"
            wav_path, cleanup_paths = _write_temp_wav(data, ext)
            res = _run_multimodal_core(wav_path, filename, denoise=denoise)

            spoken_lines = [
                s["text"] for s in res["speech_segments"] if not str(s["text"]).startswith("[Speech detected")
            ]
            if spoken_lines:
                user_text = " ".join(spoken_lines)
            elif not user_text and res["speech_segments"]:
                user_text = res["speech_segments"][0]["text"]

            sound_events = res["sound_events"]
            pitch_stats = res["pitch_stats"]
            warnings = res["asr_warnings"]
            speech_source = res["speech_source"]
            duration = res["duration"]
            snr_before = res["denoise_info"]["snr_before_db"]
            snr_after = res["denoise_info"]["snr_after_db"]
            vad_ratio = res["vad_ratio"]
            timeline = res["timeline"]

        pending = session_store.list_reminders(status="pending")
        turn = agent_service.run_voice_agent_turn(
            user_text=user_text,
            sound_events=sound_events,
            pitch_stats=pitch_stats,
            pending_reminders=pending,
            history=history,
        )

        sid = session_store.save_session(
            filename=filename,
            source_type="voice_agent",
            duration=round(duration, 2),
            speech_source=speech_source,
            vad_speech_ratio=vad_ratio,
            snr_before_db=snr_before,
            snr_after_db=snr_after,
            denoised=denoise,
            summary=user_text or "(ambient-only turn)",
            agent_reply=turn["reply"],
            timeline=timeline,
            metrics={"sound_events": sound_events, "pitch_stats": pitch_stats, "provider": turn["provider"]},
        )

        saved_reminders = [
            session_store.add_reminder(
                title=r["title"],
                due_time=r["due_time"],
                priority=r["priority"],
                trigger_source=r.get("trigger_source", "speech_action"),
                session_id=sid,
            )
            for r in turn["new_reminders"]
        ]

        return {
            "session_id": sid,
            "transcript": user_text,
            "sound_events": sound_events,
            "pitch_stats": pitch_stats,
            "reply": turn["reply"],
            "provider": turn["provider"],
            "new_reminders": saved_reminders,
            "warnings": warnings,
        }
    finally:
        _cleanup_files(cleanup_paths)


@app.post("/api/v1/watch/chunk")
async def watch_ambient_chunk(file: UploadFile = File(...), denoise: bool = Form(True)):
    """Low-latency ambient monitoring chunk handler for Live Safety Watch Mode."""
    if not HAS_LIVE_MODEL:
        raise HTTPException(status_code=503, detail="Live ML pipeline unavailable.")

    ext = _validate_audio_ext(file.filename)
    data = await _read_upload_bytes(file)
    if not data:
        raise HTTPException(status_code=400, detail="Chunk is empty.")

    cleanup_paths: List[str] = []
    try:
        wav_path, cleanup_paths = _write_temp_wav(data, ext)
        audio_data = load_audio(wav_path, target_sr=16000, to_mono=True, normalize_peak=False)
        samples = audio_data.samples
        den_info = spectral_subtraction_denoise(samples, sr=16000) if denoise else {"denoised": samples, "snr_gain_db": 0.0}
        active = den_info["denoised"]

        rms = float(np.sqrt(np.mean(active ** 2))) if active.size else 0.0
        rms_db = round(float(20.0 * np.log10(max(rms, 1e-6))), 1)
        vad_segs = detect_speech(active, 16000)
        events = detect_sound_events(active, 16000, window_s=1.5, hop_s=0.5, threshold=0.35)

        urgent_alert = None
        for ev in events:
            key = agent_service.canonical_sound_key(ev["label"])
            if key in agent_service.URGENT_SOUNDS and float(ev["confidence"]) >= 0.50:
                sev, desc = agent_service.URGENT_SOUNDS[key]
                urgent_alert = session_store.log_alert(
                    event_label=ev["label"],
                    confidence=float(ev["confidence"]),
                    severity=sev,
                    snippet_text=f"{desc} (Live Watch Mode)",
                )
                break

        return {
            "rms_db": rms_db,
            "has_speech": len(vad_segs) > 0,
            "events": events,
            "urgent_alert": urgent_alert,
            "snr_gain_db": den_info.get("snr_gain_db", 0.0),
        }
    finally:
        _cleanup_files(cleanup_paths)


# --------------------------------------------------------------------------- History, Reminders & Alerts CRUD
@app.get("/api/v1/sessions")
def get_sessions(search: str = Query(""), limit: int = Query(50, ge=1, le=200)):
    return {"sessions": session_store.list_sessions(search=search, limit=limit)}


@app.delete("/api/v1/sessions/{session_id}")
def remove_session(session_id: int):
    if not session_store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "id": session_id}


@app.get("/api/v1/reminders")
def get_reminders(status: Optional[str] = Query(None)):
    return {"reminders": session_store.list_reminders(status=status)}


@app.post("/api/v1/reminders")
def create_reminder(payload: Dict[str, Any]):
    title = str(payload.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Reminder title is required")
    return session_store.add_reminder(
        title=title,
        due_time=str(payload.get("due_time") or "Today"),
        priority=str(payload.get("priority") or "medium"),
        trigger_source=str(payload.get("trigger_source") or "manual"),
    )


@app.patch("/api/v1/reminders/{reminder_id}")
def patch_reminder(reminder_id: int, payload: Dict[str, str]):
    status = str(payload.get("status") or "").strip()
    if status not in ("pending", "completed"):
        raise HTTPException(status_code=400, detail="status must be 'pending' or 'completed'")
    updated = session_store.update_reminder_status(reminder_id, status)
    if not updated:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return updated


@app.delete("/api/v1/reminders/{reminder_id}")
def remove_reminder(reminder_id: int):
    if not session_store.delete_reminder(reminder_id):
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"deleted": True, "id": reminder_id}


@app.get("/api/v1/alerts")
def get_alerts(limit: int = Query(40, ge=1, le=200)):
    return {"alerts": session_store.list_alerts(limit=limit)}


# --------------------------------------------------------------------------- Interactive Labs (Real Audio + Benchmark)
@app.post("/api/v1/labs/signal")
async def signal_lab(file: Optional[UploadFile] = File(None), use_last: bool = Query(True)):
    """Real-Audio & Benchmark Signal Lab powered by Member 1's spectral, LPC, enhancement & F0 DSP."""
    if not HAS_LIVE_MODEL:
        return {
            "mode": "mock",
            "label": "Illustrative signal data",
            "waveform": [0.0, 0.25, 0.6, 0.2, -0.3, -0.7, -0.1, 0.5, 0.2, -0.2],
            "rms": [0.12, 0.34, 0.51, 0.28, 0.17],
            "zcr": [0.04, 0.08, 0.12, 0.06, 0.03],
            "mfcc": [[0.2, 0.4, 0.6, 0.3]],
        }

    cleanup_paths: List[str] = []
    try:
        sr = 16000
        if file is not None:
            ext = _validate_audio_ext(file.filename)
            data = await _read_upload_bytes(file)
            wav_path, cleanup_paths = _write_temp_wav(data, ext)
            audio = load_audio(wav_path, target_sr=sr, to_mono=True, normalize_peak=False)
            raw = audio.samples
            label = f"Uploaded Clip: {file.filename}"
            den_res = spectral_subtraction_denoise(raw, sr=sr)
            denoised = den_res["denoised"]
        elif use_last and LAST_AUDIO_CONTEXT.get("raw_samples") is not None:
            raw = LAST_AUDIO_CONTEXT["raw_samples"]
            denoised = LAST_AUDIO_CONTEXT["denoised_samples"]
            sr = int(LAST_AUDIO_CONTEXT.get("sr", 16000))
            den_res = LAST_AUDIO_CONTEXT["denoise_info"]
            label = f"Live Audio Session: {LAST_AUDIO_CONTEXT['filename']}"
        else:
            raw = generate_synthetic_signal("vowel_ah", duration=0.8, sr=sr, f0=130.0).samples
            # Inject controlled noise prefix to visualize Member 1 denoising benchmark cleanly
            rng = np.random.default_rng(7)
            raw = (raw + 0.025 * rng.standard_normal(len(raw))).astype(np.float32)
            den_res = spectral_subtraction_denoise(raw, sr=sr)
            denoised = den_res["denoised"]
            label = "DSP Benchmark (Noisy /a/ Vowel + Spectral Subtraction)"

        # Truncate analysis window to max 5s for responsive canvas rendering
        raw_seg = raw[: sr * 5]
        den_seg = denoised[: sr * 5]

        frames, _ = frame_signal(den_seg, 400, 160)
        rms = compute_rms(frames)
        zcrs = compute_zcr(frames)
        fft_freqs, fft_mag = compute_fft_spectrum(den_seg[: min(len(den_seg), 8192)], sr=sr, n_fft=512)
        _, _, stft_db = compute_stft(den_seg, sr=sr, frame_length=400, hop_length=160, n_fft=512)
        mfccs, _, _ = compute_mfcc(den_seg, sr=sr, n_mfcc=13, n_mels=40)
        pitch = estimate_pitch_contour(den_seg, sr=sr)

        # Pick highest-energy frame for LPC vocal-tract formant envelope
        best_frame_idx = int(np.argmax(rms)) if len(rms) else 0
        lpc = analyze_frame_lpc(frames[best_frame_idx], sr=sr, order=16, n_fft=512)

        return {
            "mode": "live",
            "label": label,
            "sample_rate": sr,
            "duration": round(len(raw) / float(sr), 2),
            "snr_before_db": den_res["snr_before_db"],
            "snr_after_db": den_res["snr_after_db"],
            "snr_gain_db": den_res["snr_gain_db"],
            "waveform": _downsample_1d(raw_seg, 90),
            "waveform_denoised": _downsample_1d(den_seg, 90),
            "rms": _downsample_1d(rms, 48),
            "zcr": _downsample_1d(zcrs, 48),
            "fft_freqs": _downsample_1d(fft_freqs, 64),
            "fft_mag_db": _downsample_1d(fft_mag, 64),
            "stft_matrix": _downsample_2d(stft_db, max_rows=32, max_cols=56),
            "mfcc": _downsample_2d(mfccs, max_rows=13, max_cols=40),
            "lpc_formants_hz": lpc.formant_frequencies[:4],
            "lpc_envelope_db": _downsample_1d(lpc.spectral_envelope, 64),
            "f0_mean_hz": pitch["mean_f0_hz"],
            "f0_contour_hz": _downsample_1d(np.array(pitch["f0_hz"], dtype=np.float32), 48),
        }
    except Exception as exc:
        print(f"[Signal Lab fallback] {exc}")
        return {
            "mode": "mock",
            "label": "Illustrative signal data",
            "waveform": [0.0, 0.25, 0.6, 0.2, -0.3, -0.7, -0.1, 0.5, 0.2, -0.2],
            "rms": [0.12, 0.34, 0.51, 0.28, 0.17],
            "zcr": [0.04, 0.08, 0.12, 0.06, 0.03],
            "mfcc": [[0.2, 0.4, 0.6, 0.3]],
        }
    finally:
        _cleanup_files(cleanup_paths)


@app.post("/api/v1/labs/dtw")
async def dtw_lab(file_a: Optional[UploadFile] = File(None), file_b: Optional[UploadFile] = File(None)):
    """Live Pattern Matching Lab powered by Member 1's MFCC + DTW dynamic programming."""
    if not HAS_LIVE_MODEL:
        return {"mode": "mock", **MOCK_DTW}

    cleanup_paths: List[str] = []
    try:
        sr = 16000
        if file_a is not None and file_b is not None:
            pa, clean_a = _write_temp_wav(await _read_upload_bytes(file_a), _validate_audio_ext(file_a.filename))
            pb, clean_b = _write_temp_wav(await _read_upload_bytes(file_b), _validate_audio_ext(file_b.filename))
            cleanup_paths.extend(clean_a + clean_b)
            sig1 = load_audio(pa, target_sr=sr, to_mono=True, normalize_peak=True).samples[: sr * 4]
            sig2 = load_audio(pb, target_sr=sr, to_mono=True, normalize_peak=True).samples[: sr * 4]
            label = f"Pairwise Clip Alignment: {file_a.filename} vs {file_b.filename}"
        else:
            sig1 = generate_synthetic_signal("vowel_ah", duration=0.22, sr=sr, f0=125.0).samples
            sig2 = generate_synthetic_signal("vowel_ah", duration=0.30, sr=sr, f0=140.0).samples
            label = "Synthetic Speaking-Rate Time Warp (0.22s vs 0.30s /a/)"

        mfcc1, _, _ = compute_mfcc(sig1, sr=sr, n_mfcc=13, n_mels=40)
        mfcc2, _, _ = compute_mfcc(sig2, sr=sr, n_mfcc=13, n_mels=40)
        res = dtw(mfcc1.T, mfcc2.T, metric="euclidean")

        cm = res.cost_matrix
        rows, cols = cm.shape
        r_idx = np.linspace(0, rows - 1, min(rows, 20), dtype=int)
        c_idx = np.linspace(0, cols - 1, min(cols, 20), dtype=int)
        sub = cm[np.ix_(r_idx, c_idx)]
        sub_rounded = [[round(float(v), 2) for v in row] for row in sub]

        # Scale warp path into downsampled grid coordinates
        path_scaled = [
            [int(round(p[0] * (len(r_idx) - 1) / max(rows - 1, 1))), int(round(p[1] * (len(c_idx) - 1) / max(cols - 1, 1)))]
            for p in res.path
        ]

        return {
            "mode": "live",
            "label": label,
            "distance": round(float(res.distance), 2),
            "normalized_distance": round(float(res.normalized_distance), 3),
            "path_length": int(res.path_length),
            "frames_a": int(rows),
            "frames_b": int(cols),
            "path": path_scaled,
            "cost_matrix_sample": sub_rounded,
        }
    except Exception as exc:
        print(f"[DTW Lab fallback] {exc}")
        return {"mode": "mock", **MOCK_DTW}
    finally:
        _cleanup_files(cleanup_paths)


@app.post("/api/v1/labs/hmm/viterbi")
def hmm_lab(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """Live HMM Forward + Viterbi Trellis endpoint powered by Member 2's hmm.py."""
    if not HAS_LIVE_MODEL:
        return {"mode": "mock", **MOCK_HMM}
    try:
        req = payload or {}
        preset_name = str(req.get("preset") or "weather")
        model, default_obs = load_preset(preset_name)
        obs = req.get("observations") or default_obs
        decoded = decode_for_ui(model, obs)
        return {"mode": "live", **decoded}
    except Exception as exc:
        print(f"[HMM Lab fallback] {exc}")
        return {"mode": "mock", **MOCK_HMM}


# --------------------------------------------------------------------------- Exporters
def timeline_from(payload: dict) -> List[dict]:
    return sorted(payload.get("timeline", []), key=lambda x: x.get("start", 0))


@app.post("/api/v1/export/{fmt}")
async def export_data(fmt: str, payload: dict):
    fmt = fmt.lower()
    if fmt not in {"srt", "vtt", "json", "csv", "txt"}:
        raise HTTPException(status_code=400, detail="Supported formats: srt, vtt, json, csv, txt")
    items = timeline_from(payload)
    if fmt == "json":
        body, mime = json.dumps(payload, indent=2), "application/json"
    elif fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["start", "end", "category", "label", "text", "confidence"])
        for x in items:
            writer.writerow(
                [x.get("start"), x.get("end"), x.get("category"), x.get("label"), x.get("display_text"), x.get("confidence")]
            )
        body, mime = buf.getvalue(), "text/csv"
    elif fmt == "txt":
        body = "\n".join(f'[{x["start"]:.2f}-{x["end"]:.2f}] ({x.get("label", "")}) {x["display_text"]}' for x in items)
        mime = "text/plain"
    else:

        def stamp(t: float, comma: str) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = t % 60
            return f"{h:02}:{m:02}:{s:06.3f}".replace(".", comma)

        lines = ["WEBVTT", ""] if fmt == "vtt" else []
        for i, x in enumerate(items, 1):
            start_s = float(x["start"])
            end_s = float(x["end"])
            line_txt = f"{x.get('speaker')}: {x['display_text']}" if x.get("speaker") and x.get("category") == "speech" else x["display_text"]
            if fmt == "srt":
                lines += [str(i), f"{stamp(start_s, ',')} --> {stamp(end_s, ',')}", line_txt, ""]
            else:
                lines += [f"{stamp(start_s, '.')} --> {stamp(end_s, '.')}", line_txt, ""]
        body = "\n".join(lines)
        mime = "text/vtt" if fmt == "vtt" else "application/x-subrip"

    return Response(
        content=body,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="soniccaption.{fmt}"'},
    )
