import numpy as np

from soniccaption.algorithms.diarization import OnlineSpeakerTracker, assign_speakers, diarize, speaker_at

SR = 16000


def test_no_speech_means_no_speakers():
    r = diarize(np.zeros(SR, np.float32), [])
    assert r["n_speakers"] == 0 and r["turns"] == []


def test_forced_single_speaker():
    x = (np.random.default_rng(1).standard_normal(6 * SR) * 0.05).astype(np.float32)
    r = diarize(x, [{"start": 0.2, "end": 5.8}], speaker_mode="1")
    assert r["n_speakers"] == 1 and all(t["speaker_id"] == 1 for t in r["turns"])


def test_assign_speakers_splits_a_segment_at_the_speaker_change():
    turns = [{"start": 0.0, "end": 2.0, "speaker_id": 1}, {"start": 2.0, "end": 4.0, "speaker_id": 2}]
    words = [{"word": " hello", "start": 0.2, "end": 0.6}, {"word": " there", "start": 0.7, "end": 1.2},
             {"word": " hi", "start": 2.3, "end": 2.6}, {"word": " back", "start": 2.7, "end": 3.2}]
    out = assign_speakers([{"start": 0.2, "end": 3.2, "text": "hello there hi back", "words": words}], turns)
    assert [(o["speaker_id"], o["text"]) for o in out] == [(1, "hello there"), (2, "hi back")]


def test_speaker_at_and_tracker_ignore_tiny_audio():
    turns = [{"start": 1.0, "end": 2.0, "speaker_id": 3}]
    assert speaker_at(turns, 1.5) == 3 and speaker_at(turns, 9.0) is None
    assert OnlineSpeakerTracker().assign(np.zeros(100, np.float32))[0] == 0
