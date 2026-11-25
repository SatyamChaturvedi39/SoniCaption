"""SonicCaption API.

    POST /api/v1/analyze            upload a file -> {job_id}; poll GET /api/v1/jobs/{id} for progress and the result
    POST /api/v1/audio/analyze      same analysis, synchronous (scripts, tests)
    WS   /ws/live                   live recording: PCM in, timeline / transcript / insights events out
    GET  /api/health                what is installed and what is not (ASR, ffmpeg, models) - the UI shows this honestly
    sessions / tasks / alerts       SQLite history; tasks belong to a speaker
    POST /api/v1/export/{fmt}       srt | vtt | json | csv | txt
    POST /api/v1/labs/hmm/viterbi   interactive HMM trellis used by the HMM sub-tab

If frontend/dist exists it is served at "/", so a single `uvicorn` process is the whole product.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
for candidate in (BASE_DIR / "src",):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _load_dotenv_files() -> None:
    """Populate os.environ from a root or backend .env (never overrides real environment variables)."""
    for env_path in (ROOT_DIR / ".env", BASE_DIR / ".env"):
        if not env_path.is_file():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val


_load_dotenv_files()

from soniccaption.algorithms.hmm import decode_for_ui, load_preset  # noqa: E402
from soniccaption.services import agent_service, analysis_pipeline, session_store  # noqa: E402

ALLOWED_AUDIO_EXTS = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac", ".mp4", ".mov", ".mkv", ".aac"}
MAX_UPLOAD_MB = int(os.environ.get("SONICCAPTION_MAX_UPLOAD_MB", "200"))
SR = analysis_pipeline.SR

app = FastAPI(title="SonicCaption API", version="1.0.0",
              description="Speech, speakers, background noise and sound effects in one timeline, with per-speaker tasks.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

_POOL = ThreadPoolExecutor(max_workers=1)        # one analysis at a time: models are large and torch is not re-entrant friendly
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


# --------------------------------------------------------------------------- helpers
def _validate_audio_ext(filename: Optional[str]) -> str:
    ext = Path(filename or "").suffix.lower() or ".wav"
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(ALLOWED_AUDIO_EXTS))}")
    return ext


async def _read_limited(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return data


def _persist(result: Dict[str, Any], source_type: str, save: bool = True) -> Dict[str, Any]:
    """Save the session, its tasks (each owned by a speaker) and urgent-sound alerts. Returns ids."""
    if not save:
        return {"session_id": None, "tasks": []}
    nlu = result["nlu"]
    sc = nlu["scenario"]
    bits = [sc["label"]] if sc.get("label") else []
    bits.append(f"{result['diarization']['n_speakers']} speaker(s)")
    if nlu["tasks"]:
        bits.append(f"{len(nlu['tasks'])} task(s)")
    summary = " · ".join(bits)
    sid = session_store.save_session(
        filename=result["filename"], source_type=source_type, duration=result["duration"],
        speech_source=result["asr"].get("provider") or result["asr"]["status"], vad_speech_ratio=result["vad_speech_ratio"],
        snr_before_db=float(result["patterns"]["spectral"].get("snr_before_db") or 0.0),
        snr_after_db=float(result["patterns"]["spectral"].get("snr_after_db") or 0.0),
        denoised=result["denoised"], summary=summary, timeline=result["timeline"], metrics={"analysis": result})
    saved = []
    for t in nlu["tasks"]:
        saved.append(session_store.add_reminder(title=t["task"], due_time=t.get("due") or "Soon", priority=t.get("priority", "medium"),
                                                trigger_source="speech_action", session_id=sid, owner=t.get("owner", ""),
                                                owner_id=t.get("owner_id")))
    for seg in result["scene"]["segments"]:
        key = agent_service.canonical_sound_key(seg.get("key") or seg["label"])
        if seg["state"] == "sound_effect" and key in agent_service.URGENT_SOUNDS and float(seg.get("confidence", 0)) >= 0.6:
            sev, desc = agent_service.URGENT_SOUNDS[key]
            session_store.log_alert(event_label=seg["label"], confidence=float(seg["confidence"]), severity=sev,
                                    snippet_text=f"{desc} at {seg['start']:.1f}s in {result['filename']}")
    return {"session_id": sid, "tasks": saved}


def _run_job(job_id: str, data: bytes, filename: str, opts: Dict[str, Any]) -> None:
    def progress(stage: str, frac: float, msg: str) -> None:
        with _JOBS_LOCK:
            _JOBS[job_id].update(stage=stage, progress=round(frac, 3), message=msg)

    try:
        result = analysis_pipeline.analyse_bytes(data, filename, denoise=opts["denoise"], speaker_mode=opts["speaker_mode"],
                                                 progress=progress, language=opts.get("language"))
        stored = _persist(result, "upload", opts["save"])
        result["session_id"] = stored["session_id"]
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="done", progress=1.0, stage="done", message="Done", result=result)
    except Exception as exc:  # noqa: BLE001 - surface every failure to the UI instead of a silent mock result
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}"[:600], message="Analysis failed")


def _trim_jobs() -> None:
    with _JOBS_LOCK:
        for k in sorted(_JOBS, key=lambda k: _JOBS[k]["created"])[:-20]:
            _JOBS.pop(k, None)


# --------------------------------------------------------------------------- health
def _asr_status() -> Dict[str, Any]:
    from soniccaption.services.live_session import asr_possible
    ok, kind = asr_possible()
    return {"available": ok, "provider": kind,
            "hint": None if ok else "Set GROQ_API_KEY in .env (free key at console.groq.com) or `pip install faster-whisper`."}


@app.get("/api/health")
def health():
    head_meta: Dict[str, Any] = {}
    meta = BASE_DIR / "src" / "soniccaption" / "services" / "weights" / "scene_head.json"
    if meta.is_file():
        try:
            head_meta = json.loads(meta.read_text(encoding="utf-8")).get("held_out_metrics", {})
        except Exception:  # noqa: BLE001
            head_meta = {}
    from soniccaption.services import speaker_encoder
    return {"status": "ok", "version": app.version, "asr": _asr_status(), "groq_configured": bool(os.environ.get("GROQ_API_KEY")),
            "ffmpeg_available": bool(shutil.which("ffmpeg")), "speaker_encoder": "neural" if speaker_encoder.available() else "dsp-fallback",
            "scene_head_metrics": head_meta}


# --------------------------------------------------------------------------- upload analysis
@app.post("/api/v1/analyze")
async def start_analysis(file: UploadFile = File(...), denoise: bool = Form(True), speaker_mode: str = Form("auto"),
                         save_log: bool = Form(True), language: Optional[str] = Form(None)):
    _validate_audio_ext(file.filename)
    data = await _read_limited(file)
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {"id": job_id, "status": "running", "stage": "queued", "progress": 0.0, "message": "Queued",
                         "created": time.time(), "filename": file.filename}
    _trim_jobs()
    _POOL.submit(_run_job, job_id, data, file.filename or "audio.wav",
                 {"denoise": denoise, "speaker_mode": speaker_mode, "save": save_log, "language": language or None})
    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job")
        return {k: v for k, v in job.items() if k != "created"}


@app.post("/api/v1/audio/analyze")
async def analyze_sync(file: UploadFile = File(...), denoise: bool = Form(True), speaker_mode: str = Form("auto"),
                       save_log: bool = Form(True), language: Optional[str] = Form(None)):
    _validate_audio_ext(file.filename)
    data = await _read_limited(file)
    loop = asyncio.get_running_loop()

    def work():
        r = analysis_pipeline.analyse_bytes(data, file.filename or "audio.wav", denoise=denoise, speaker_mode=speaker_mode, language=language)
        r["session_id"] = _persist(r, "upload", save_log)["session_id"]
        return r

    try:
        return await loop.run_in_executor(_POOL, work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------- live recording
@app.websocket("/ws/live")
async def live_ws(ws: WebSocket):
    """Protocol: text {"type":"start","sample_rate":16000,"denoise":true,"asr":true} -> binary little-endian int16 mono PCM chunks
    -> text {"type":"stop"}. Server events: ready | segment | transcript | insights | status | progress | final | error."""
    from soniccaption.services.live_session import LiveSession
    await ws.accept()
    loop = asyncio.get_running_loop()
    send_lock = asyncio.Lock()
    asr_lock = asyncio.Lock()
    tasks: set[asyncio.Task] = set()

    async def send(obj: Dict[str, Any]) -> None:
        async with send_lock:
            await ws.send_text(json.dumps(obj))

    async def transcribe(sess: LiveSession, seg: Dict[str, Any]) -> None:
        async with asr_lock:
            events = await loop.run_in_executor(None, sess.transcribe, seg)
        for e in events:
            await send(e)

    session: Optional[LiveSession] = None
    opts = {"denoise": True, "sr": SR}
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                cmd = json.loads(msg["text"])
                kind = cmd.get("type")
                if kind == "start":
                    opts.update(denoise=bool(cmd.get("denoise", True)), sr=int(cmd.get("sample_rate", SR)))
                    session = await loop.run_in_executor(None, lambda: LiveSession(asr_enabled=bool(cmd.get("asr", True))))
                    await send({"type": "ready", "sample_rate": SR, "asr": _asr_status()})
                elif kind == "stop" and session is not None:
                    for e in await loop.run_in_executor(None, session.finish_stream):
                        await send(e)
                    for seg in session.pending_speech():
                        tasks.add(asyncio.create_task(transcribe(session, seg)))
                    if tasks:
                        await asyncio.gather(*list(tasks), return_exceptions=True)
                    rec = session.recording()
                    if rec.size < int(0.5 * SR):
                        await send({"type": "error", "message": "Recording was too short to analyse."})
                        break
                    await _finalize(ws, send, loop, rec, opts, session)
                    break
            elif msg.get("bytes") is not None and session is not None:
                pcm = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                if opts["sr"] != SR and pcm.size:
                    from scipy.signal import resample_poly
                    from math import gcd
                    g = gcd(SR, opts["sr"])
                    pcm = resample_poly(pcm, SR // g, opts["sr"] // g).astype(np.float32)
                for e in await loop.run_in_executor(None, session.feed, pcm):
                    await send(e)
                for seg in session.pending_speech():
                    t = asyncio.create_task(transcribe(session, seg))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"[:300]})
        except Exception:  # noqa: BLE001
            pass
    finally:
        for t in list(tasks):
            t.cancel()
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


async def _finalize(ws: WebSocket, send, loop, rec: np.ndarray, opts: Dict[str, Any], session) -> None:
    import soundfile as sf
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, rec, SR, subtype="PCM_16")

    def progress(stage: str, frac: float, msg: str) -> None:
        asyncio.run_coroutine_threadsafe(send({"type": "progress", "stage": stage, "progress": round(frac, 3), "message": msg}), loop)

    def work():
        r = analysis_pipeline.analyse_audio(path, "live-recording.wav", denoise=opts["denoise"], samples=rec, progress=progress,
                                            asr_fn=session.asr_fn if session.asr_fn else None)
        r["session_id"] = _persist(r, "live", True)["session_id"]
        return r

    try:
        result = await loop.run_in_executor(_POOL, work)
        await send({"type": "final", "result": result})
    except Exception as exc:  # noqa: BLE001
        await send({"type": "error", "message": f"Final analysis failed: {type(exc).__name__}: {exc}"[:400]})
    finally:
        if os.path.exists(path):
            os.remove(path)


# --------------------------------------------------------------------------- history, tasks, alerts
@app.get("/api/v1/sessions")
def get_sessions(search: str = Query(""), limit: int = Query(30, ge=1, le=200)):
    rows = session_store.list_sessions(search=search, limit=limit)
    return {"sessions": [{k: r[k] for k in ("id", "created_at", "filename", "source_type", "duration", "summary")} for r in rows]}


@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: int):
    s = session_store.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    analysis = (s.get("metrics") or {}).get("analysis")
    return {"id": s["id"], "created_at": s["created_at"], "analysis": analysis}


@app.delete("/api/v1/sessions/{session_id}")
def remove_session(session_id: int):
    if not session_store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "id": session_id}


@app.get("/api/v1/reminders")
def get_reminders(status: Optional[str] = Query(None), session_id: Optional[int] = Query(None)):
    rows = session_store.list_reminders(status=status)
    if session_id is not None:
        rows = [r for r in rows if r.get("session_id") == session_id]
    return {"reminders": rows}


@app.post("/api/v1/reminders")
def create_reminder(payload: Dict[str, Any]):
    title = str(payload.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Task title is required")
    return session_store.add_reminder(title=title, due_time=str(payload.get("due_time") or "Soon"), priority=str(payload.get("priority") or "medium"),
                                      trigger_source=str(payload.get("trigger_source") or "manual"), owner=str(payload.get("owner") or ""),
                                      owner_id=payload.get("owner_id"))


@app.patch("/api/v1/reminders/{reminder_id}")
def patch_reminder(reminder_id: int, payload: Dict[str, str]):
    status = str(payload.get("status") or "").strip()
    if status not in ("pending", "completed"):
        raise HTTPException(status_code=400, detail="status must be 'pending' or 'completed'")
    updated = session_store.update_reminder_status(reminder_id, status)
    if not updated:
        raise HTTPException(status_code=404, detail="Task not found")
    return updated


@app.delete("/api/v1/reminders/{reminder_id}")
def remove_reminder(reminder_id: int):
    if not session_store.delete_reminder(reminder_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"deleted": True, "id": reminder_id}


@app.get("/api/v1/alerts")
def get_alerts(limit: int = Query(40, ge=1, le=200)):
    return {"alerts": session_store.list_alerts(limit=limit)}


# --------------------------------------------------------------------------- HMM lab
@app.post("/api/v1/labs/hmm/viterbi")
def hmm_lab(payload: Optional[Dict[str, Any]] = Body(default=None)):
    req = payload or {}
    try:
        model, default_obs = load_preset(str(req.get("preset") or "scene_state"))
        return {"mode": "live", **decode_for_ui(model, req.get("observations") or default_obs)}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# --------------------------------------------------------------------------- export
def _stamp(t: float, sep: str) -> str:
    return f"{int(t // 3600):02}:{int((t % 3600) // 60):02}:{t % 60:06.3f}".replace(".", sep)


def _line(x: Dict[str, Any]) -> str:
    text = x.get("text") or x.get("display_text") or ""
    if x.get("category") == "speech":
        return f"{x.get('speaker') or 'Speaker'}: {text}" if text else f"{x.get('speaker') or 'Speaker'}: [speech]"
    return f"[{x.get('label', '')}]"


@app.post("/api/v1/export/{fmt}")
async def export_data(fmt: str, payload: dict):
    fmt = fmt.lower()
    if fmt not in {"srt", "vtt", "json", "csv", "txt"}:
        raise HTTPException(status_code=400, detail="Supported formats: srt, vtt, json, csv, txt")
    items = sorted(payload.get("timeline", []), key=lambda x: x.get("start", 0))
    if fmt in ("srt", "vtt", "txt"):
        items = [x for x in items if x.get("category") != "silence"]
    if fmt == "json":
        body, mime = json.dumps(payload, indent=2), "application/json"
    elif fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["start", "end", "category", "label", "speaker", "text", "confidence"])
        for x in items:
            w.writerow([x.get("start"), x.get("end"), x.get("category"), x.get("label"), x.get("speaker") or "", x.get("text") or "", x.get("confidence")])
        body, mime = buf.getvalue(), "text/csv"
    elif fmt == "txt":
        body, mime = "\n".join(f'[{x["start"]:.2f}-{x["end"]:.2f}] {_line(x)}' for x in items), "text/plain"
    else:
        lines = ["WEBVTT", ""] if fmt == "vtt" else []
        sep = "." if fmt == "vtt" else ","
        for i, x in enumerate(items, 1):
            cue = f"{_stamp(float(x['start']), sep)} --> {_stamp(float(x['end']), sep)}"
            lines += ([cue, _line(x), ""] if fmt == "vtt" else [str(i), cue, _line(x), ""])
        body, mime = "\n".join(lines), ("text/vtt" if fmt == "vtt" else "application/x-subrip")
    return Response(content=body, media_type=mime, headers={"Content-Disposition": f'attachment; filename="soniccaption.{fmt}"'})


# --------------------------------------------------------------------------- frontend (single-process deployment)
_DIST = ROOT_DIR / "frontend" / "dist"
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith(("api/", "ws/")):
            raise HTTPException(status_code=404)
        f = (_DIST / path).resolve()
        if path and f.is_file() and _DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(_DIST / "index.html")
