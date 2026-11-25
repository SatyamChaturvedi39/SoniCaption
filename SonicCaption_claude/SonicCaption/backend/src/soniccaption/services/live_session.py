"""Live recording session: audio chunks in -> timeline events out, with ~1.5 s latency.

The browser streams 16 kHz mono PCM. `LiveSession.feed()` returns events the UI paints immediately:

    {"type": "segment",    ...}   a speech / sound_effect / background_noise / silence span (open spans keep growing)
    {"type": "transcript", ...}   text for a closed speech span, attributed to a speaker (needs ASR)
    {"type": "insights",   ...}   scenario + per-speaker tasks recomputed from the transcript so far
    {"type": "status",     ...}   e.g. ASR unavailable (so the UI never silently shows nothing)

When the recording stops, `finalize()` re-runs the whole take through `analyse_audio` for the authoritative result
(better diarization, smoothed scene labels, spectral and HMM views). Live labels are provisional by design.
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable

import numpy as np

from soniccaption.audio.vad import FRAME, FRAME_S, segments_from_probs
from soniccaption.services.analysis_pipeline import SR, STATUS

MIN_SPEAKER_AUDIO_S = 0.8
MIN_ASR_AUDIO_S = 0.4
PAUSE_S = 0.35            # a pause this long inside speech ends an utterance (speaker turn / sentence boundary)


def asr_possible() -> tuple[bool, str]:
    """Cheap check (no model load, no network) of whether any ASR backend could run."""
    if os.environ.get("GROQ_API_KEY"):
        return True, "groq"
    try:
        import faster_whisper  # noqa: F401
        return True, "local"
    except Exception:  # noqa: BLE001
        return False, "none"


class LiveSession:
    def __init__(self, asr_fn: Callable | None = None, classifier=None, tracker=None, asr_enabled: bool | None = None,
                 use_llm: bool = True):
        from soniccaption.algorithms.diarization import OnlineSpeakerTracker
        from soniccaption.services.scene_classifier import LiveSceneClassifier
        self.clf = classifier or LiveSceneClassifier()
        self.tracker = tracker or OnlineSpeakerTracker()
        self.asr_fn = asr_fn
        ok, kind = (True, "injected") if asr_fn else asr_possible()
        self.asr_enabled = ok if asr_enabled is None else (asr_enabled and ok)
        self.asr_kind = kind
        self.use_llm = use_llm
        self.cur: dict | None = None
        self.segments: list[dict] = []
        self.utterances: list[dict] = []
        self._n = 0
        self._u = 0
        self._done_until = 0.0        # speech before this time has already been emitted as utterances
        self._announced = False
        self._last_speaker = 0
        self.chunks: list[np.ndarray] = []

    # ------------------------------------------------------------------ audio in
    @property
    def seconds(self) -> float:
        return self.clf.seconds

    def feed(self, pcm: np.ndarray) -> list[dict]:
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.size == 0:
            return []
        self.chunks.append(pcm)
        events: list[dict] = []
        if not self._announced:
            self._announced = True
            if not self.asr_enabled:
                events.append({"type": "status", "asr": "unavailable",
                               "message": "Speech-to-text is not configured (set GROQ_API_KEY or install faster-whisper). "
                                          "Noise, sound effects and speakers are still analysed live."})
            else:
                events.append({"type": "status", "asr": "ready", "message": f"Speech-to-text: {self.asr_kind}"})
        for hop in self.clf.feed(pcm):
            events.extend(self._on_hop(hop))
        return events

    def _on_hop(self, h: dict) -> list[dict]:
        ev: list[dict] = []
        same = (self.cur is not None and self.cur["state"] == h["state"]
                and (h["state"] != "sound_effect" or self.cur["label"] == h["label"]))
        if same:
            c = self.cur
            c["end"] = round(h["t1"], 2)
            c["confidence"] = round(max(c["confidence"], h["confidence"]) if h["state"] == "sound_effect" else
                                    (c["confidence"] * c["_n"] + h["confidence"]) / (c["_n"] + 1), 3)
            c["_n"] += 1
            ev.extend(self._speech_utterances(c, final=False) if c["state"] == "speech" else [self._seg_event(c, closed=False)])
            return ev
        if self.cur is not None:
            ev.extend(self._close(self.cur))
        self._n += 1
        self.cur = {"id": f"L{self._n}", "state": h["state"], "label": h["label"], "key": h.get("key"), "start": round(h["t0"], 2),
                    "end": round(h["t1"], 2), "confidence": h["confidence"], "_n": 1}
        if h["state"] == "speech":
            ev.extend(self._speech_utterances(self.cur, final=False))
        else:
            ev.append(self._seg_event(self.cur, closed=False))
        return ev

    def _speech_utterances(self, c: dict, final: bool) -> list[dict]:
        """Split the running speech region into utterances at pauses; finished ones get a speaker and go to ASR."""
        ev: list[dict] = []
        f0 = int(max(c["start"], self._done_until) / FRAME_S)
        probs = np.asarray(self.clf.vad.probs[f0:], dtype=np.float32)
        vad_end = (f0 + probs.size) * FRAME_S
        segs = segments_from_probs(probs, min_speech_ms=200, min_silence_ms=int(PAUSE_S * 1000), pad_ms=30)
        segs = [{"start": round(f0 * FRAME_S + g["start"], 2), "end": round(f0 * FRAME_S + g["end"], 2)} for g in segs]
        if not segs and final and c["end"] - max(c["start"], self._done_until) >= 0.3:      # classifier said speech, VAD fragments too short
            segs = [{"start": max(c["start"], self._done_until), "end": c["end"]}]
        for i, g in enumerate(segs):
            finished = final or g["end"] < vad_end - PAUSE_S
            if not finished:
                ev.append({"type": "segment", "id": f"U{self._u + 1}", "category": "speech", "label": "Speech", "key": "speech",
                           "start": g["start"], "end": g["end"], "confidence": c["confidence"], "status": "ok", "open": True})
                continue
            self._u += 1
            utt = {"id": f"U{self._u}", "state": "speech", "label": "Speech", "key": "speech", "start": g["start"], "end": g["end"],
                   "confidence": c["confidence"]}
            audio = self._audio(utt["start"], utt["end"])
            if audio.size >= MIN_SPEAKER_AUDIO_S * SR:
                try:
                    sid, _ = self.tracker.assign(audio)
                except Exception:  # noqa: BLE001 - a speaker-embedding failure must not break the live stream
                    sid = 0
                if sid:
                    self._last_speaker = sid
            if self._last_speaker:
                utt["speaker_id"] = self._last_speaker
            self._done_until = max(self._done_until, g["end"])
            if utt["end"] - utt["start"] >= MIN_ASR_AUDIO_S:
                self.segments.append(utt)
            ev.append(self._seg_event(utt, closed=True))
        return ev

    def _seg_event(self, c: dict, closed: bool) -> dict:
        d = {"type": "segment", "id": c["id"], "category": c["state"], "label": c["label"], "key": c.get("key"),
             "start": c["start"], "end": c["end"], "confidence": c["confidence"], "status": STATUS.get(c["state"], "idle"),
             "open": not closed}
        if c.get("speaker_id"):
            d["speaker_id"], d["speaker"] = c["speaker_id"], f"Speaker {c['speaker_id']}"
        return d

    def _audio(self, a: float, b: float) -> np.ndarray:
        x = self.clf.x
        return x[int(a * SR):int(b * SR)]

    def _close(self, c: dict) -> list[dict]:
        if c["state"] == "speech":
            return self._speech_utterances(c, final=True)
        return [self._seg_event(c, closed=True)]

    # ------------------------------------------------------------------ transcription (called off the event loop)
    def pending_speech(self) -> list[dict]:
        out = [s for s in self.segments if not s.get("asr_done")]
        for s in out:
            s["asr_done"] = True
        return out

    def transcribe(self, seg: dict) -> list[dict]:
        """Transcribe one closed speech segment and refresh insights. Safe to run in a worker thread."""
        if not self.asr_enabled:
            return []
        import soundfile as sf
        a, b = max(0.0, seg["start"] - 0.15), seg["end"] + 0.15
        audio = self._audio(a, b)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, audio, SR, subtype="PCM_16")
            if self.asr_fn:
                res = self.asr_fn(path)
            else:
                from soniccaption.services.asr_service import transcribe_audio
                res = transcribe_audio(path)
        except Exception as exc:  # noqa: BLE001
            return [{"type": "status", "asr": "error", "message": f"Transcription failed for {seg['start']:.1f}s: {str(exc)[:160]}"}]
        finally:
            if os.path.exists(path):
                os.remove(path)
        text = " ".join(str(s.get("text", "")).strip() for s in res.get("segments", [])).strip()
        if not text:
            return []
        sid = seg.get("speaker_id") or 1
        u = {"start": seg["start"], "end": seg["end"], "text": text, "speaker_id": sid, "speaker": f"Speaker {sid}"}
        self.utterances.append(u)
        events = [{"type": "transcript", "id": seg["id"], **{k: u[k] for k in ("start", "end", "text", "speaker_id", "speaker")}}]
        events.append(self.insights())
        return events

    def insights(self) -> dict:
        from soniccaption.services.nlu import analyse_conversation
        r = analyse_conversation(self.utterances, None, self.seconds, use_llm=False)
        return {"type": "insights", "scenario": r["scenario"], "tasks": r["tasks"], "speakers": r["speakers"],
                "decisions": r["decisions"]}

    # ------------------------------------------------------------------ end of recording
    def finish_stream(self) -> list[dict]:
        ev: list[dict] = []
        for hop in self.clf.finish():
            ev.extend(self._on_hop(hop))
        if self.cur is not None:
            ev.extend(self._close(self.cur))
            self.cur = None
        return ev

    def recording(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
