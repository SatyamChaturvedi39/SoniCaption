"""ASR tests use mocked providers: CI never needs an API key, network access or a model download."""
import types

import numpy as np
import pytest
import soundfile as sf

from soniccaption.services import asr_service as asr


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "clip.wav"
    sf.write(p, (0.1 * np.random.default_rng(0).standard_normal(16000 * 3)).astype(np.float32), 16000)
    return p


GROQ_JSON = {
    "text": " Please close the door. Thank you.",
    "language": "English",
    "segments": [{"start": 0.0, "end": 1.6, "text": " Please close the door."},
                 {"start": 1.8, "end": 2.6, "text": " Thank you."}],
    "words": [{"word": "Please", "start": 0.0, "end": 0.4}, {"word": "close", "start": 0.4, "end": 0.8},
              {"word": "the", "start": 0.8, "end": 1.0}, {"word": "door.", "start": 1.0, "end": 1.6},
              {"word": "Thank", "start": 1.8, "end": 2.1}, {"word": "you.", "start": 2.1, "end": 2.6}],
}


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload
        self.headers = {"content-type": "application/json"}
        self.text = str(payload)

    def json(self):
        return self._p


def check_contract(out):
    assert set(out) == {"text", "segments", "provider", "language", "warnings"}
    for s in out["segments"]:
        assert set(s) == {"start", "end", "text", "words"} and s["start"] <= s["end"]
        for w in s["words"]:
            assert set(w) == {"start", "end", "word"}
    starts = [s["start"] for s in out["segments"]]
    assert starts == sorted(starts)


def test_groq_provider_contract_and_word_assignment(wav, monkeypatch):
    calls = []
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", lambda url, **kw: calls.append(kw) or FakeResponse(200, GROQ_JSON))
    out = asr.transcribe_audio(wav)
    check_contract(out)
    assert out["provider"] == "groq-whisper"
    assert out["text"] == "Please close the door. Thank you."
    assert [w["word"] for w in out["segments"][0]["words"]] == ["Please", "close", "the", "door."]
    assert [w["word"] for w in out["segments"][1]["words"]] == ["Thank", "you."]
    sent = dict(calls[0]["data"])
    assert sent["model"] == asr.GROQ_MODEL and sent["response_format"] == "verbose_json"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_groq_long_audio_is_chunked_with_time_offsets(tmp_path, monkeypatch):
    p = tmp_path / "long.wav"
    sf.write(p, np.zeros(16000 * 25, np.float32), 16000)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(asr, "GROQ_CHUNK_SECONDS", 10)
    monkeypatch.setattr("requests.post", lambda url, **kw: FakeResponse(200, GROQ_JSON))
    out = asr.transcribe_audio(p, provider="groq")
    assert len(out["segments"]) == 6                                  # 3 chunks x 2 segments
    assert [s["start"] for s in out["segments"]] == [0.0, 1.8, 10.0, 11.8, 20.0, 21.8]
    assert out["segments"][2]["words"][0]["start"] == 10.0


def test_groq_retries_once_then_succeeds(wav, monkeypatch):
    replies = iter([FakeResponse(429, {"error": {"message": "rate limited"}}), FakeResponse(200, GROQ_JSON)])
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(asr.time, "sleep", lambda s: None)
    monkeypatch.setattr("requests.post", lambda url, **kw: next(replies))
    assert asr.transcribe_audio(wav, provider="groq")["provider"] == "groq-whisper"


def test_groq_only_mode_raises_clear_error(wav, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "bad")
    monkeypatch.setattr("requests.post", lambda url, **kw: FakeResponse(401, {"error": {"message": "Invalid API Key"}}))
    with pytest.raises(asr.ASRError, match="401.*Invalid API Key"):
        asr.transcribe_audio(wav, provider="groq")


class FakeWhisperModel:
    def transcribe(self, audio, **kw):
        W = lambda w, s, e: types.SimpleNamespace(word=w, start=s, end=e)
        segs = [types.SimpleNamespace(start=0.0, end=1.2, text=" Hello there.", words=[W(" Hello", 0.0, 0.5), W(" there.", 0.5, 1.2)]),
                types.SimpleNamespace(start=1.5, end=1.6, text="  ", words=[])]          # empty -> dropped
        return iter(segs), types.SimpleNamespace(language="en")


def test_auto_falls_back_to_local_when_groq_fails(wav, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr("requests.post", lambda url, **kw: FakeResponse(503, {"error": {"message": "down"}}))
    monkeypatch.setattr(asr.time, "sleep", lambda s: None)
    monkeypatch.setattr(asr, "_load_faster_whisper", lambda size: FakeWhisperModel())
    out = asr.transcribe_audio(wav)
    check_contract(out)
    assert out["provider"].startswith("faster-whisper")
    assert out["text"] == "Hello there." and len(out["segments"]) == 1
    assert "Groq transcription failed" in out["warnings"][0]


def test_auto_uses_local_without_key(wav, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(asr, "_load_faster_whisper", lambda size: FakeWhisperModel())
    out = asr.transcribe_audio(wav)
    assert out["provider"].startswith("faster-whisper") and out["language"] == "en"
    assert "not set" in out["warnings"][0]


def test_missing_or_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(asr.ASRError, match="not found"):
        asr.transcribe_audio(tmp_path / "nope.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(asr.ASRError, match="Could not decode"):
        asr.transcribe_audio(bad)


def test_invalid_provider_name(wav):
    with pytest.raises(ValueError):
        asr.transcribe_audio(wav, provider="openai")
