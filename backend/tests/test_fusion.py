import json

import pytest

from soniccaption.timeline.fusion import FusionConfig, fuse_timeline, merge_events

SPEECH = [
    {"start": 0.5, "end": 4.2, "text": " Good morning everyone.", "words": [{"word": "Good", "start": 0.5, "end": 0.8}]},
    {"start": 6.8, "end": 11.9, "text": "Today we demonstrate SonicCaption.", "words": []},
]
EVENTS = [
    {"start": 4.0, "end": 5.0, "label": "clapping", "confidence": 0.91, "model": "m"},
    {"start": 5.0, "end": 6.0, "label": "clapping", "confidence": 0.95, "model": "m"},
    {"start": 6.2, "end": 6.7, "label": "clapping", "confidence": 0.80, "model": "m"},
]


def ev(start, end, label, conf):
    return {"start": start, "end": end, "label": label, "confidence": conf, "model": "m"}


def test_basic_fusion_is_chronological_and_readable():
    out = fuse_timeline(SPEECH, EVENTS)
    tl = out["timeline"]
    assert [i["category"] for i in tl] == ["speech", "sound_event", "speech"]
    assert [i["id"] for i in tl] == ["t1", "t2", "t3"]
    assert tl[1]["display_text"] == "[Applause]" and tl[1]["label"] == "Clapping / Applause"
    assert (tl[1]["start"], tl[1]["end"], tl[1]["confidence"]) == (4.0, 6.7, 0.95)   # 3 windows merged
    assert tl[0]["display_text"] == "Good morning everyone." and tl[0]["confidence"] is None
    assert tl[0]["words"] == SPEECH[0]["words"]
    json.dumps(out)


def test_item_schema():
    for it in fuse_timeline(SPEECH, EVENTS)["timeline"]:
        assert set(it) == {"id", "start", "end", "category", "label", "display_text", "confidence",
                           "source", "overlaps_speech", "words"}
        assert it["start"] < it["end"]


def test_low_confidence_events_dropped_with_reason():
    out = fuse_timeline([], [ev(1, 2, "dog", 0.3), ev(3, 4, "siren", 0.8)])
    assert [i["display_text"] for i in out["timeline"]] == ["[Siren]"]
    assert "below threshold" in out["dropped"][0]["reason"]


def test_per_class_threshold():
    cfg = FusionConfig(class_thresholds={"coughing": 0.9})
    out = fuse_timeline([], [ev(1, 2, "coughing", 0.85), ev(3, 4, "dog", 0.85)], config=cfg)
    assert [i["label"] for i in out["timeline"]] == ["Dog barking"]


def test_gap_merging_respects_gap_and_label():
    m = merge_events([ev(0, 1, "dog", .9), ev(1.4, 2, "dog", .8), ev(3, 4, "dog", .7), ev(1, 2, "rain", .9)], gap_s=0.5)
    assert [(e["label"], e["start"], e["end"]) for e in m] == [("dog", 0, 2), ("rain", 1, 2), ("dog", 3, 4)]


def test_short_events_removed():
    out = fuse_timeline([], [ev(1, 1.3, "dog", 0.9)])
    assert out["timeline"] == [] and "shorter than" in out["dropped"][0]["reason"]


def test_events_under_speech_suppressed_even_if_confident():
    speech = [{"start": 0, "end": 10, "text": "talking"}]
    out = fuse_timeline(speech, [ev(2, 4, "laughing", 0.7), ev(5, 7, "dog", 0.99)])
    assert [i["label"] for i in out["timeline"]] == ["Speech"]
    assert all("covered by speech" in d["reason"] for d in out["dropped"])


def test_keep_if_confidence_option():
    speech = [{"start": 0, "end": 10, "text": "talking"}]
    out = fuse_timeline(speech, [ev(2, 4, "laughing", 0.7), ev(5, 7, "siren", 0.97)],
                        config=FusionConfig(keep_if_confidence=0.95))
    labels = [i["label"] for i in out["timeline"]]
    assert labels == ["Speech", "Siren"]
    assert out["timeline"][1]["overlaps_speech"] is True
    assert "covered by speech" in out["dropped"][0]["reason"]


def test_keep_policy_keeps_and_flags():
    speech = [{"start": 0, "end": 10, "text": "talking"}]
    out = fuse_timeline(speech, [ev(2, 4, "laughing", 0.7)], config=FusionConfig(overlap_policy="keep"))
    assert out["timeline"][1]["display_text"] == "[Laughter]" and out["timeline"][1]["overlaps_speech"]


def test_partial_overlap_is_kept():
    speech = [{"start": 0, "end": 4.5, "text": "hello"}]
    out = fuse_timeline(speech, [ev(4, 7, "clapping", 0.6)])            # only 1/6 under speech
    assert out["timeline"][1]["display_text"] == "[Applause]"


def test_vad_guard_drops_hallucinated_speech():
    speech = [{"start": 0, "end": 3, "text": "Real words."}, {"start": 10, "end": 12, "text": "Thank you."}]
    vad = [{"start": 0.2, "end": 2.9}]
    out = fuse_timeline(speech, [], vad_segments=vad)
    assert [i["display_text"] for i in out["timeline"]] == ["Real words."]
    assert "hallucination" in out["dropped"][0]["reason"]


def test_empty_text_and_empty_inputs():
    out = fuse_timeline([{"start": 0, "end": 1, "text": "   "}], [])
    assert out["timeline"] == [] and len(out["dropped"]) == 1
    assert fuse_timeline([], [])["timeline"] == []


def test_ties_put_speech_first_and_times_clipped():
    out = fuse_timeline([{"start": 2, "end": 3, "text": "hi"}], [ev(2, 3.5, "siren", 0.95)],
                        config=FusionConfig(duration=3.2, overlap_policy="keep"))
    tl = out["timeline"]
    assert [i["category"] for i in tl] == ["speech", "sound_event"]
    assert tl[1]["end"] == 3.2


def test_unknown_label_gets_generic_tag():
    out = fuse_timeline([], [ev(0, 2, "chainsaw_noise", 0.9)])
    assert out["timeline"][0]["display_text"] == "[Chainsaw Noise]"


def test_input_not_mutated():
    events = [dict(e) for e in EVENTS]
    fuse_timeline(SPEECH, events)
    assert events == EVENTS


def test_invalid_policy():
    with pytest.raises(ValueError):
        fuse_timeline([], [], config=FusionConfig(overlap_policy="drop"))


def test_competing_labels_keep_more_confident():
    out = fuse_timeline([], [ev(17, 19, "door_wood_knock", 0.92), ev(16.5, 18.5, "glass_breaking", 0.69),
                             ev(11, 15, "clapping", 0.99), ev(14.5, 15.5, "dog", 0.6)])
    assert [i["display_text"] for i in out["timeline"]] == ["[Applause]", "[Door knock]"]
    assert sum("more confident" in d["reason"] for d in out["dropped"]) == 2


def test_small_overlap_between_labels_is_allowed():
    out = fuse_timeline([], [ev(0, 4, "rain", 0.9), ev(3.5, 6, "siren", 0.8)])       # only 0.5/2.5 overlap
    assert len(out["timeline"]) == 2
