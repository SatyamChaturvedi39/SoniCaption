"""Whole pipeline on a real file. Speech is injected (the sample has none) so no network or ASR model is needed."""
import json

import numpy as np
import pytest

from soniccaption.services import analysis_pipeline, live_session

SAMPLE = "sample_test.wav"
SR = 16000
TEXTS = ["Hi, this is Priya. Let's review the release.", "I'll update the repository by Friday."]


def fake_asr(_path):
    return {"segments": [{"start": 0.3, "end": 2.0, "text": TEXTS[0]}, {"start": 2.4, "end": 4.6, "text": TEXTS[1]}],
            "provider": "fake", "language": "en", "warnings": []}


@pytest.fixture
def with_speech(monkeypatch):
    """Make the VAD report two speech segments so utterances survive the hallucination filter."""
    segs = [{"start": 0.3, "end": 2.0}, {"start": 2.4, "end": 4.6}]
    monkeypatch.setattr("soniccaption.audio.vad.detect_speech", lambda *a, **k: [dict(s) for s in segs])


def test_full_analysis_schema_and_tasks(with_speech):
    r = analysis_pipeline.analyse_audio(SAMPLE, asr_fn=fake_asr, use_llm=False)
    json.dumps(r)
    assert r["asr"]["status"] == "ok" and r["duration"] == pytest.approx(5.0, abs=0.05)
    assert r["diarization"]["n_speakers"] >= 1
    cats = {t["category"] for t in r["timeline"]}
    assert "speech" in cats
    assert any(t["category"] == "speech" and t["text"] == TEXTS[0] for t in r["timeline"])
    assert all(a["start"] <= a["end"] for a in r["timeline"])
    assert any("repository" in t["task"].lower() and t["due"] == "Friday" for t in r["nlu"]["tasks"])
    sp = r["patterns"]["spectral"]
    assert len(sp["mfcc"]) == 13 and len(sp["stft_db"]) > 0
    assert r["patterns"]["hmm"]["scene"]["path"] and r["patterns"]["hmm"]["scene"]["model"]["states"]


def test_asr_failure_is_reported_not_faked(with_speech):
    def broken(_p):
        raise RuntimeError("no network")
    r = analysis_pipeline.analyse_audio(SAMPLE, asr_fn=broken, use_llm=False)
    assert r["asr"]["status"] == "unavailable" and "no network" in r["asr"]["error"]
    assert r["nlu"]["tasks"] == [] and r["utterances"] == []
    assert not any(t["text"] for t in r["timeline"])                       # no invented text
    assert r["diarization"]["n_speakers"] >= 1                             # everything that does not need ASR still ran


def test_hallucinated_text_on_non_speech_is_dropped():
    r = analysis_pipeline.analyse_audio(SAMPLE, asr_fn=fake_asr, use_llm=False)   # real VAD: the sample has no speech
    assert r["asr"]["status"] == "skipped" and r["utterances"] == []


def test_too_short_audio_is_rejected():
    with pytest.raises(ValueError):
        analysis_pipeline.analyse_audio(SAMPLE, samples=np.zeros(1000, np.float32))


def test_live_session_streams_events_and_finishes(with_speech):
    s = live_session.LiveSession(asr_fn=fake_asr, use_llm=False)
    import soundfile as sf
    x, _ = sf.read(SAMPLE, dtype="float32")
    events = []
    for i in range(0, len(x), 4096):
        events += s.feed(x[i:i + 4096])
    events += s.finish_stream()
    kinds = {e["type"] for e in events}
    assert {"status", "segment"} <= kinds
    segs = [e for e in events if e["type"] == "segment" and not e["open"]]
    assert segs and all(e["status"] in ("ok", "alert", "warn", "idle") for e in segs)
    assert s.seconds == pytest.approx(5.0, abs=0.05)


def test_live_session_without_asr_says_so():
    s = live_session.LiveSession(asr_enabled=False)
    ev = s.feed(np.zeros(8000, np.float32))
    assert ev[0]["type"] == "status" and ev[0]["asr"] == "unavailable"
