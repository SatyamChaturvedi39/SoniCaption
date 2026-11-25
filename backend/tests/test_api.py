"""End-to-end API integration tests for SonicCaption live endpoints, agent, watch mode, and DB store."""

from fastapi.testclient import TestClient
from app.main import app


def test_health_and_labs():
    c = TestClient(app)
    h = c.get("/api/health")
    assert h.status_code == 200 and h.json()["mode"] == "live"

    sig = c.post("/api/v1/labs/signal")
    assert sig.status_code == 200 and sig.json()["mode"] == "live"
    assert len(sig.json()["stft_matrix"]) > 0
    assert "f0_mean_hz" in sig.json()

    dtw = c.post("/api/v1/labs/dtw")
    assert dtw.status_code == 200 and dtw.json()["mode"] == "live"
    assert len(dtw.json()["cost_matrix_sample"]) > 0

    for preset in ("weather", "coin_toss", "urn_ball"):
        hmm = c.post("/api/v1/labs/hmm/viterbi", json={"preset": preset})
        assert hmm.status_code == 200 and hmm.json()["mode"] == "live"
        assert len(hmm.json()["trellis_nodes"]) > 0


def test_invalid_upload_rejected():
    c = TestClient(app)
    r = c.post("/api/v1/audio/analyze", files={"file": ("bad.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_live_analyze_agent_watch_and_crud(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICCAPTION_DB_PATH", str(tmp_path / "api_test.db"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    c = TestClient(app)

    with open("sample_test.wav", "rb") as f:
        r = c.post(
            "/api/v1/audio/analyze",
            files={"file": ("sample_test.wav", f, "audio/wav")},
            data={"denoise": "true", "source_type": "upload", "save_log": "true"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "live" and body["session_id"] is not None
    assert "snr_gain_db" in body["metrics"]

    # Sessions CRUD
    sess_list = c.get("/api/v1/sessions").json()["sessions"]
    assert len(sess_list) == 1 and sess_list[0]["id"] == body["session_id"]

    # Watch chunk
    with open("sample_test.wav", "rb") as f:
        w = c.post("/api/v1/watch/chunk", files={"file": ("chunk.wav", f, "audio/wav")}, data={"denoise": "true"})
    assert w.status_code == 200 and "rms_db" in w.json()

    # Agent text+voice reminder extraction turn
    ag = c.post("/api/v1/agent/turn", data={"text": "Remind me to present SonicCaption tomorrow", "history_json": "[]"})
    assert ag.status_code == 200
    ag_json = ag.json()
    assert len(ag_json["new_reminders"]) == 1
    rem_id = ag_json["new_reminders"][0]["id"]

    # Reminder toggle & delete
    patched = c.patch(f"/api/v1/reminders/{rem_id}", json={"status": "completed"})
    assert patched.status_code == 200 and patched.json()["status"] == "completed"
    assert c.delete(f"/api/v1/reminders/{rem_id}").status_code == 200
    assert c.delete(f"/api/v1/sessions/{body['session_id']}").status_code == 200

    # Export check
    srt = c.post("/api/v1/export/srt", json=body)
    assert srt.status_code == 200 and "-->" in srt.text
