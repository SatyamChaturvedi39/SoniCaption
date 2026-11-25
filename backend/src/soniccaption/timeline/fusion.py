"""Dual-stream timeline fusion: Whisper speech segments + sound-model events -> one chronological caption timeline.

Pipeline (each step is deterministic, nothing is invented):
    1. Speech:  drop empty segments; if VAD segments are given, drop ASR segments that VAD says contain
                almost no speech (guards against Whisper "hallucinating" text such as "Thank you." in silence).
    2. Events:  drop events below the confidence threshold (global or per class), merge same-label events
                separated by small gaps, drop events shorter than the minimum duration.
    3. Overlap with speech: the sound model never heard speech during training (ESC-50 has no speech class), so it
                confidently mislabels voices (e.g. as dog or coughing). With policy "suppress", an event mostly covered
                by speech is dropped (optionally kept if above `keep_if_confidence`); with "keep", every event is kept
                and flagged `overlaps_speech`.
    3b. Overlap between events: when two different labels overlap, the less confident one is dropped if at least
                `event_overlap_ratio` of it lies inside the more confident one (one sound, two competing labels).
    4. Merge both streams, sort by start time (speech first on ties) and number the items.

Every dropped item is returned with the reason, so the UI/report can show what was filtered and why.

Timeline item (matches the mock API contract used by the frontend, plus provenance fields):
    {"id": "t1", "start": 0.5, "end": 4.2, "category": "speech" | "sound_event",
     "label": "Speech" | "Clapping / Applause", "display_text": "Good morning..." | "[Applause]",
     "confidence": None | 0.94, "source": "groq-whisper" | "soniccaption-esc50-resnet18",
     "overlaps_speech": bool, "words": [...]}
"""
from __future__ import annotations

from dataclasses import dataclass, field

# label -> (human-readable label, caption tag)
SOUND_TAGS: dict[str, tuple[str, str]] = {
    "clapping": ("Clapping / Applause", "[Applause]"),
    "laughing": ("Laughter", "[Laughter]"),
    "laughter": ("Laughter", "[Laughter]"),
    "coughing": ("Coughing", "[Coughing]"),
    "sneezing": ("Sneezing", "[Sneezing]"),
    "footsteps": ("Footsteps", "[Footsteps]"),
    "door_wood_knock": ("Door knock", "[Door knock]"),
    "glass_breaking": ("Glass breaking", "[Glass breaking]"),
    "siren": ("Siren", "[Siren]"),
    "dog": ("Dog barking", "[Dog barking]"),
    "rain": ("Rain", "[Rain]"),
}


@dataclass
class FusionConfig:
    event_threshold: float = 0.5                 # minimum confidence for a sound event
    class_thresholds: dict = field(default_factory=dict)   # optional per-class override, e.g. {"coughing": 0.7}
    merge_gap_s: float = 0.5                     # join same-label events separated by at most this gap
    min_event_s: float = 0.5                     # drop events shorter than this after merging
    overlap_policy: str = "suppress"             # "suppress" or "keep"
    speech_overlap_ratio: float = 0.5            # "mostly covered by speech" = at least this fraction
    keep_if_confidence: float | None = None      # if set, events at/above this confidence survive suppression
    event_overlap_ratio: float = 0.5             # resolve competing labels for the same sound
    min_vad_overlap: float = 0.2                 # ASR segment needs this fraction inside VAD speech
    duration: float | None = None                # clip length; times are clipped to [0, duration]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _covered(start: float, end: float, intervals: list[dict]) -> float:
    """Fraction of [start, end] covered by the union of `intervals` (which may overlap each other)."""
    if end <= start:
        return 0.0
    spans = sorted((max(start, i["start"]), min(end, i["end"])) for i in intervals if i["end"] > start and i["start"] < end)
    total, cur0, cur1 = 0.0, None, None
    for s, e in spans:
        if cur1 is None or s > cur1:
            if cur1 is not None:
                total += cur1 - cur0
            cur0, cur1 = s, e
        else:
            cur1 = max(cur1, e)
    if cur1 is not None:
        total += cur1 - cur0
    return total / (end - start)


def _clip(t: float, duration: float | None) -> float:
    t = max(0.0, float(t))
    return min(t, duration) if duration is not None else t


def merge_events(events: list[dict], gap_s: float) -> list[dict]:
    """Join events with the same label when they overlap or the gap between them is <= gap_s."""
    merged: list[dict] = []
    for e in sorted(events, key=lambda e: (e["label"], e["start"])):
        last = merged[-1] if merged else None
        if last and last["label"] == e["label"] and e["start"] - last["end"] <= gap_s:
            last["end"] = max(last["end"], e["end"])
            last["confidence"] = max(last["confidence"], e["confidence"])
        else:
            merged.append(dict(e))
    return sorted(merged, key=lambda e: e["start"])


def fuse_timeline(speech_segments: list[dict], sound_events: list[dict], vad_segments: list[dict] | None = None,
                  config: FusionConfig | None = None, speech_source: str = "whisper") -> dict:
    """Fuse the two streams. Returns {"timeline": [...], "dropped": [...], "stats": {...}}."""
    cfg = config or FusionConfig()
    if cfg.overlap_policy not in ("suppress", "keep"):
        raise ValueError("overlap_policy must be 'suppress' or 'keep'.")
    dropped: list[dict] = []

    # 1. speech stream
    speech = []
    for s in speech_segments:
        start, end = _clip(s["start"], cfg.duration), _clip(s["end"], cfg.duration)
        text = (s.get("text") or "").strip()
        if not text or end <= start:
            dropped.append({"item": s, "reason": "empty or zero-length speech segment"})
            continue
        if vad_segments is not None and _covered(start, end, vad_segments) < cfg.min_vad_overlap:
            dropped.append({"item": s, "reason": "VAD found almost no speech here (possible ASR hallucination)"})
            continue
        speech_item = {"start": start, "end": end, "text": text, "words": s.get("words", [])}
        for extra_k in ("speaker", "delivery_style", "mean_f0_hz", "pitch_slope_hz", "energy_ratio", "speaking_rate_wps"):
            if extra_k in s:
                speech_item[extra_k] = s[extra_k]
        speech.append(speech_item)

    # 2. sound-event stream
    kept = []
    for e in sound_events:
        thr = cfg.class_thresholds.get(e["label"], cfg.event_threshold)
        if e["confidence"] < thr:
            dropped.append({"item": e, "reason": f"confidence {e['confidence']:.2f} below threshold {thr:.2f}"})
        else:
            kept.append({**e, "start": _clip(e["start"], cfg.duration), "end": _clip(e["end"], cfg.duration)})
    events = []
    for e in merge_events(kept, cfg.merge_gap_s):
        if e["end"] - e["start"] < cfg.min_event_s:
            dropped.append({"item": e, "reason": f"shorter than {cfg.min_event_s:.2f} s"})
        else:
            events.append(e)

    # 3. overlap with speech
    final_events = []
    for e in events:
        ratio = _covered(e["start"], e["end"], speech)
        e["overlaps_speech"] = ratio > 0
        very_confident = cfg.keep_if_confidence is not None and e["confidence"] >= cfg.keep_if_confidence
        if cfg.overlap_policy == "suppress" and ratio >= cfg.speech_overlap_ratio and not very_confident:
            dropped.append({"item": e, "reason": f"{ratio:.0%} covered by speech (likely the voice, not a sound event)"})
        else:
            final_events.append(e)

    # 3b. competing labels for the same sound: keep the more confident one
    resolved = []
    for e in sorted(final_events, key=lambda e: -e["confidence"]):
        dur = e["end"] - e["start"]
        rival = next((r for r in resolved if r["label"] != e["label"]
                      and _overlap(e["start"], e["end"], r["start"], r["end"]) >= cfg.event_overlap_ratio * dur), None)
        if rival:
            dropped.append({"item": e, "reason": f"overlaps a more confident '{rival['label']}' event ({rival['confidence']:.2f})"})
        else:
            resolved.append(e)
    final_events = sorted(resolved, key=lambda e: e["start"])

    # 4. merge and sort
    items = []
    for s in speech:
        sp_out = {
            "start": round(s["start"], 3),
            "end": round(s["end"], 3),
            "category": "speech",
            "label": s.get("speaker", "Speech"),
            "display_text": s["text"],
            "confidence": None,
            "source": speech_source,
            "overlaps_speech": False,
            "words": s["words"],
        }
        for extra_k in ("speaker", "delivery_style", "mean_f0_hz", "pitch_slope_hz", "energy_ratio", "speaking_rate_wps"):
            if extra_k in s:
                sp_out[extra_k] = s[extra_k]
        items.append(sp_out)
    for e in final_events:
        label, tag = SOUND_TAGS.get(e["label"], (e["label"].replace("_", " ").title(),
                                                 f"[{e['label'].replace('_', ' ').title()}]"))
        items.append({"start": round(e["start"], 3), "end": round(e["end"], 3), "category": "sound_event",
                      "label": label, "display_text": tag, "confidence": round(float(e["confidence"]), 3),
                      "source": e.get("model", "sound-model"), "overlaps_speech": e["overlaps_speech"], "words": []})
    items.sort(key=lambda i: (i["start"], i["category"] != "speech", i["end"]))
    for n, it in enumerate(items, 1):
        it["id"] = f"t{n}"
    timeline = [{"id": it.pop("id"), **it} for it in items]

    stats = {"speech_items": len(speech), "sound_items": len(final_events), "dropped": len(dropped),
             "speech_seconds": round(sum(s["end"] - s["start"] for s in speech), 2)}
    return {"timeline": timeline, "dropped": dropped, "stats": stats}
