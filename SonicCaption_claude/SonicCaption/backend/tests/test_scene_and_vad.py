"""Scene classification on non-speech audio: noise must be reported as noise/silence, never as speech."""
import numpy as np

from soniccaption.audio.vad import detect_speech
from soniccaption.services.scene_classifier import analyze_scene

SR = 16000


def _noise(seconds, level, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * level).astype(np.float32)


def test_silence_is_silence():
    r = analyze_scene(np.zeros(4 * SR, np.float32), [])
    assert {s["state"] for s in r["segments"]} == {"silence"}


IMPULSIVE = {"dog", "door_wood_knock", "coughing", "sneezing", "glass_breaking"}


def test_steady_noise_is_never_speech_or_an_impulsive_sound():
    """White noise is out of the model's training distribution (it may be called rain, which is also steady noise);
    what must never happen is speech, or a dog/knock/cough that needs a burst the signal does not have."""
    x = _noise(6, 0.05)
    assert detect_speech(x, SR) == []
    r = analyze_scene(x, [])
    assert "speech" not in {s["state"] for s in r["segments"]}
    assert not IMPULSIVE & {s["key"] for s in r["segments"]}
    assert r["stats"]["silence"]["share"] == 0.0


def test_burst_guard_turns_a_burstless_impulsive_guess_into_noise():
    from soniccaption.services.scene_classifier import MIN_BURST_DB, classify_windows
    assert MIN_BURST_DB["door_wood_knock"] >= 10.0
    out = classify_windows([_noise(2, 0.05, seed=5)])
    assert out[0]["key"] not in IMPULSIVE


def test_hmm_smoothing_reports_path_and_model():
    r = analyze_scene(_noise(5, 0.04, seed=3), [])
    assert len(r["hmm"]["path"]) == len(r["hops"]) and r["hmm"]["model"]["states"] == ["Silence", "Speech", "SoundEffect", "BackgroundNoise"]


def test_scene_hmm_preset_rows_are_probabilities():
    from soniccaption.algorithms.hmm import load_preset
    m, _ = load_preset("scene_state")
    m.validate()
    assert np.allclose(m.A.sum(1), 1) and np.allclose(m.B.sum(1), 1)
