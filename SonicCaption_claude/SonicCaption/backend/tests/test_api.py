"""HTTP + WebSocket API."""
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICCAPTION_DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return TestClient(app)


def _no_asr(monkeypatch):
    import soniccaption.services.asr_service as asr

    def boom(*a, **k):
        raise asr.ASRError("offline in tests")
    monkeypatch.setattr(asr, "transcribe_audio", boom)


def test_health_reports_asr_and_models(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "available" in h["asr"] and h["speaker_encoder"] in ("neural", "dsp-fallback")


def test_invalid_upload_rejected(client):
    assert client.post("/api/v1/analyze", files={"file": ("bad.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/api/v1/audio/analyze", files={"file": ("x.wav", b"", "audio/wav")}).status_code == 400


def test_job_flow_history_export_and_honest_asr_failure(client, monkeypatch):
    _no_asr(monkeypatch)
    with open("sample_test.wav", "rb") as f:
        jid = client.post("/api/v1/analyze", files={"file": ("sample_test.wav", f, "audio/wav")}).json()["job_id"]
    for _ in range(300):
        job = client.get(f"/api/v1/jobs/{jid}").json()
        if job["status"] != "running":
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    res = job["result"]
    assert res["session_id"] and res["asr"]["status"] in ("skipped", "unavailable")
    assert client.get("/api/v1/jobs/nope").status_code == 404

    stored = client.get(f"/api/v1/sessions/{res['session_id']}").json()["analysis"]
    assert stored["duration"] == res["duration"]
    assert client.get("/api/v1/sessions").json()["sessions"][0]["id"] == res["session_id"]

    srt = client.post("/api/v1/export/srt", json={"timeline": [
        {"start": 0, "end": 1, "category": "speech", "speaker": "Speaker 1", "text": "hi"},
        {"start": 1, "end": 2, "category": "silence", "label": "Silence"},
        {"start": 2, "end": 3, "category": "sound_effect", "label": "Dog bark"}]}).text
    assert "Speaker 1: hi" in srt and "[Dog bark]" in srt and "Silence" not in srt
    assert client.post("/api/v1/export/xml", json={}).status_code == 400
    assert client.delete(f"/api/v1/sessions/{res['session_id']}").json()["deleted"]


def test_tasks_belong_to_speakers(client):
    t = client.post("/api/v1/reminders", json={"title": "Update docs", "owner": "Speaker 2 (Rahul)", "owner_id": 2}).json()
    assert t["owner"] == "Speaker 2 (Rahul)" and t["owner_id"] == 2
    assert client.patch(f"/api/v1/reminders/{t['id']}", json={"status": "completed"}).json()["status"] == "completed"
    assert client.get("/api/v1/reminders", params={"status": "completed"}).json()["reminders"][0]["owner_id"] == 2
    assert client.patch(f"/api/v1/reminders/{t['id']}", json={"status": "bogus"}).status_code == 400
    assert client.delete(f"/api/v1/reminders/{t['id']}").json()["deleted"]


def test_hmm_lab(client):
    r = client.post("/api/v1/labs/hmm/viterbi", json={"preset": "scene_state"}).json()
    assert r["best_path"] and r["model"]["states"][0] == "Silence"
    assert client.post("/api/v1/labs/hmm/viterbi", json={"preset": "nope"}).status_code == 422


def test_live_websocket_end_to_end(client, monkeypatch):
    _no_asr(monkeypatch)
    import soundfile as sf
    x, _ = sf.read("sample_test.wav", dtype="float32")
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    seen = set()
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "start", "sample_rate": 16000, "denoise": True, "asr": True}))
        assert ws.receive_json()["type"] == "ready"
        for i in range(0, len(pcm), 4096):
            ws.send_bytes(pcm[i:i + 4096].tobytes())
        ws.send_text(json.dumps({"type": "stop"}))
        while True:
            m = ws.receive_json()
            seen.add(m["type"])
            if m["type"] in ("final", "error"):
                break
    assert m["type"] == "final", m
    assert "segment" in seen and m["result"]["duration"] == pytest.approx(5.0, abs=0.1)
