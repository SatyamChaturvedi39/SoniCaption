"""Unit tests for SQLite session store and acoustic-aware reminder agent."""

from soniccaption.services import agent_service, session_store


def test_session_store_crud_and_search(tmp_path):
    db = tmp_path / "test_sc.db"
    sid = session_store.save_session(
        filename="call_note.wav",
        source_type="mic_recording",
        duration=4.5,
        speech_source="groq-whisper",
        vad_speech_ratio=0.72,
        snr_before_db=8.5,
        snr_after_db=15.2,
        denoised=True,
        summary="Team check-in with applause.",
        timeline=[{"id": "t1", "start": 0.2, "end": 2.1, "category": "speech", "display_text": "Call Dr. Rao tomorrow"}],
        metrics={"test": True},
        db_path=db,
    )
    assert sid > 0
    sessions = session_store.list_sessions(search="applause", db_path=db)
    assert len(sessions) == 1
    assert sessions[0]["denoised"] is True
    assert sessions[0]["snr_after_db"] == 15.2

    rem = session_store.add_reminder(title="Call Dr. Rao", due_time="Tomorrow", priority="high", session_id=sid, db_path=db)
    assert rem["status"] == "pending"
    done = session_store.update_reminder_status(rem["id"], "completed", db_path=db)
    assert done["status"] == "completed"
    assert len(session_store.list_reminders(db_path=db)) == 1

    alert = session_store.log_alert(event_label="siren", confidence=0.91, severity="high", snippet_text="Urgent siren", db_path=db)
    assert alert["event_label"] == "siren"
    assert len(session_store.list_alerts(db_path=db)) == 1

    assert session_store.delete_reminder(rem["id"], db_path=db) is True
    assert session_store.delete_session(sid, db_path=db) is True


def test_agent_offline_insight_and_voice_turn(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    segs = [{"start": 0.1, "end": 3.0, "speaker": "Speaker 1", "text": "Remind me to submit the report tomorrow at 10 am."}]
    evs = [{"start": 3.2, "end": 4.8, "label": "Door knock", "confidence": 0.85}]

    insights = agent_service.extract_session_insights(segs, evs, pitch_stats={"mean_f0_hz": 142.0})
    assert insights["provider"] == "rule-based-local"
    assert any("submit the report" in r["title"].lower() for r in insights["reminders"])
    assert any(r["trigger_source"] == "acoustic_alert" for r in insights["reminders"])

    turn = agent_service.run_voice_agent_turn(
        "Remind me to email Huda today",
        sound_events=[{"start": 0.5, "end": 2.0, "label": "Door knock", "confidence": 0.78}],
        pitch_stats={"mean_f0_hz": 135.0},
    )
    assert "door knock" in turn["reply"].lower()
    assert any("email huda" in r["title"].lower() for r in turn["new_reminders"])
