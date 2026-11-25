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

    scene_hmm = body["metrics"].get("scene_hmm", {})
    assert len(scene_hmm.get("best_path", [])) > 0
    assert len(scene_hmm["best_path"]) == len(scene_hmm.get("observations", []))

    diag = body["metrics"].get("diagnostics", {})
    assert len(diag.get("stft_matrix", [])) > 0
    assert len(diag.get("lpc_envelope_db", [])) > 0
    assert isinstance(diag.get("lpc_formants_hz"), list)
    assert body.get("similar_sessions") == []

    # Second upload triggers DTW similarity match against first stored session
    with open("sample_test.wav", "rb") as f:
        r2 = c.post(
            "/api/v1/audio/analyze",
            files={"file": ("sample_test_2.wav", f, "audio/wav")},
            data={"denoise": "true", "source_type": "upload", "save_log": "true"},
        )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["session_id"] is not None and body2["session_id"] != body["session_id"]
    assert len(body2.get("similar_sessions", [])) >= 1

    sim_resp = c.get(f"/api/v1/sessions/{body2['session_id']}/similar")
    assert sim_resp.status_code == 200
    sim_payload = sim_resp.json()
    sim_items = sim_payload.get("similar_sessions") if "similar_sessions" in sim_payload else sim_payload.get("matches", [])
    assert len(sim_items) >= 1

    # Sessions CRUD
    sess_list = c.get("/api/v1/sessions").json()["sessions"]
    assert len(sess_list) == 2

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
    assert c.delete(f"/api/v1/sessions/{body2['session_id']}").status_code == 200
    remaining_ids = {s["id"] for s in c.get("/api/v1/sessions").json()["sessions"]}
    assert body["session_id"] not in remaining_ids and body2["session_id"] not in remaining_ids

    # Export check
    export_payload = body if body.get("timeline") else {
        **body,
        "timeline": [{"start": 0.5, "end": 2.0, "category": "speech", "speaker": "Speaker 1", "display_text": "Test caption"}],
    }
    srt = c.post("/api/v1/export/srt", json=export_payload)
    assert srt.status_code == 200 and "-->" in srt.text


def test_stationary_fan_noise_produces_no_false_events_or_alerts(tmp_path, monkeypatch):
    import io
    import numpy as np
    import soundfile as sf

    monkeypatch.setenv("SONICCAPTION_DB_PATH", str(tmp_path / "fan_test.db"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    c = TestClient(app)

    sr = 16000
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 2.5, int(sr * 2.5), endpoint=False, dtype=np.float32)
    fan = (0.005 * np.sin(2.0 * np.pi * 50.0 * t) + 0.003 * np.sin(2.0 * np.pi * 100.0 * t) + 0.009 * rng.normal(0.0, 1.0, t.size)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, fan, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    for denoise_flag in ("false", "true"):
        w = c.post("/api/v1/watch/chunk", files={"file": ("fan.wav", wav_bytes, "audio/wav")}, data={"denoise": denoise_flag})
        assert w.status_code == 200
        wj = w.json()
        assert wj["events"] == []
        assert wj["has_speech"] is False
        assert wj["urgent_alert"] is None

        a = c.post(
            "/api/v1/audio/analyze",
            files={"file": ("fan.wav", wav_bytes, "audio/wav")},
            data={"denoise": denoise_flag, "source_type": "upload", "save_log": "false"},
        )
        assert a.status_code == 200
        aj = a.json()
        assert aj["sound_events"] == []
        assert aj["speech_segments"] == []
        assert set(aj["metrics"]["scene_hmm"]["best_path"]) == {"Ambient"}

    # Room floor calibration suppresses even loud fan noise (-29 dB) when baseline_rms is supplied
    loud_fan = (fan * 4.0).astype(np.float32)
    lbuf = io.BytesIO()
    sf.write(lbuf, loud_fan, sr, format="WAV", subtype="PCM_16")
    cal_w = c.post(
        "/api/v1/watch/chunk",
        files={"file": ("loud_fan.wav", lbuf.getvalue(), "audio/wav")},
        data={"denoise": "false", "baseline_rms": "0.045"},
    )
    assert cal_w.status_code == 200
    assert cal_w.json()["events"] == []
    assert cal_w.json()["noise_rms"] > 0.01


def test_query_dtw_and_copilot_archive_qa(tmp_path, monkeypatch):
    import io
    import soundfile as sf
    from soniccaption.audio.io import load_audio

    monkeypatch.setenv("SONICCAPTION_DB_PATH", str(tmp_path / "copilot_test.db"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    c = TestClient(app)

    # Analyze clip so LAST_AUDIO_CONTEXT + SQLite archive hold a session
    with open("sample_test.wav", "rb") as f:
        r = c.post(
            "/api/v1/audio/analyze",
            files={"file": ("lecture_clip.wav", f, "audio/wav")},
            data={"denoise": "true", "source_type": "upload", "save_log": "true"},
        )
    assert r.status_code == 200
    assert "prosody_hmm" in r.json()["metrics"]

    # Extract 0.7s query excerpt from sample_test.wav and locate it via Unit 3 Subsequence DTW
    aud = load_audio("sample_test.wav", target_sr=16000, normalize_peak=False).samples
    q_buf = io.BytesIO()
    sf.write(q_buf, aud[: int(16000 * 0.7)], 16000, format="WAV", subtype="PCM_16")
    q_resp = c.post(
        "/api/v1/audio/query-dtw",
        files={"file": ("query.wav", q_buf.getvalue(), "audio/wav")},
        data={"denoise": "false"},
    )
    assert q_resp.status_code == 200
    matches = q_resp.json()["matches"]
    assert len(matches) >= 1 and matches[0]["similarity_pct"] > 0

    # Copilot spoken bookmarking + Archive Q&A (no reminders needed)
    bm = c.post(
        "/api/v1/agent/turn",
        data={"text": "Bookmark: Spectral subtraction suppresses stationary noise", "history_json": "[]"},
    )
    assert bm.status_code == 200
    assert len(bm.json()["new_notes"]) == 1

    qa = c.post(
        "/api/v1/agent/turn",
        data={"text": "Summarize my lecture_clip recordings", "history_json": "[]"},
    )
    assert qa.status_code == 200
    assert len(qa.json()["matched_sessions"]) >= 1
    assert "lecture_clip.wav" in qa.json()["reply"] or len(qa.json()["reply"]) > 10
