"""
Acoustic-Aware AI Agent, Smart Reminder Extractor, and Call Summary Service.

Combines spoken dialogue (Whisper), non-speech ESC-50 environmental tags
(ResNet-18), and vocal pitch cues (Member 1 F0 DSP). Uses Groq's OpenAI-compatible
Llama API when GROQ_API_KEY is available, and falls back deterministically to
structured rule-based extraction offline.
"""

from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional

from soniccaption.timeline.fusion import SOUND_TAGS

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

URGENT_SOUNDS = {
    "siren": ("high", "Emergency siren detected"),
    "glass_breaking": ("high", "Possible glass break / impact detected"),
    "door_wood_knock": ("medium", "Door knock heard — check entrance"),
    "dog": ("medium", "Dog barking detected"),
    "coughing": ("info", "Repeated coughing detected"),
}

_LABEL_TO_KEY = {display.lower(): key for key, (display, _) in SOUND_TAGS.items()}


def canonical_sound_key(label: str) -> str:
    """Map either raw ESC-50 labels or fused UI labels (e.g. 'Door knock') to canonical key."""
    low = str(label or "").strip().lower()
    if low in URGENT_SOUNDS or low in SOUND_TAGS:
        return low
    if low in _LABEL_TO_KEY:
        return _LABEL_TO_KEY[low]
    return low.replace(" ", "_")

_REMINDER_RE = re.compile(
    r"(?:remind me to|don't forget to|remember to|schedule|call back|follow up on|need to|have to)\s+([^.?!;\n]+)",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
    re.IGNORECASE,
)


def _call_groq_json(system_prompt: str, user_prompt: str, timeout: float = 20.0) -> Optional[dict]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        import requests

        resp = requests.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_CHAT_MODEL,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception:
        return None
    return None


def _heuristic_reminders(transcript: str, sound_events: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    seen = set()

    for m in _REMINDER_RE.finditer(transcript):
        phrase = m.group(1).strip(" ,.")
        if not phrase or phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        t_match = _TIME_RE.search(phrase)
        items.append(
            {
                "title": phrase[:1].upper() + phrase[1:],
                "due_time": t_match.group(1).title() if t_match else "Soon",
                "priority": "high" if any(k in phrase.lower() for k in ("urgent", "asap", "doctor", "exam", "call")) else "medium",
                "trigger_source": "speech_action",
            }
        )

    for ev in sound_events:
        raw_lbl = canonical_sound_key(ev.get("label", ""))
        if raw_lbl in URGENT_SOUNDS and float(ev.get("confidence", 0.0)) >= 0.60:
            sev, desc = URGENT_SOUNDS[raw_lbl]
            if sev in ("high", "medium") and desc.lower() not in seen:
                seen.add(desc.lower())
                items.append(
                    {
                        "title": desc,
                        "due_time": f"At {float(ev['start']):.1f}s",
                        "priority": sev,
                        "trigger_source": "acoustic_alert",
                    }
                )
    return items


def _fallback_summary(
    transcript: str,
    sound_events: List[Dict[str, Any]],
    speaker_count: int,
    pitch_stats: Optional[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    if transcript.strip():
        snippet = transcript.strip()
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        spk = f"{speaker_count} speaker{'s' if speaker_count > 1 else ''}"
        parts.append(f"Detected speech ({spk}): \"{snippet}\"")
    else:
        parts.append("No linguistic dialogue transcribed.")

    if sound_events:
        top_tags = ", ".join(f"{e['label']} ({int(float(e.get('confidence', 0)) * 100)}%)" for e in sound_events[:3])
        parts.append(f"Ambient context: {top_tags}.")
    else:
        parts.append("Quiet acoustic background.")

    if pitch_stats and pitch_stats.get("mean_f0_hz", 0) > 0:
        parts.append(f"Mean vocal F0: {pitch_stats['mean_f0_hz']} Hz.")
    return " ".join(parts)


def extract_session_insights(
    speech_segments: List[Dict[str, Any]],
    sound_events: List[Dict[str, Any]],
    pitch_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce executive session summary, acoustic context assessment, and reminders."""
    transcript = " ".join(
        str(s.get("text", "")).strip()
        for s in speech_segments
        if not str(s.get("text", "")).startswith("[Speech detected")
    ).strip()
    speakers = {s.get("speaker", "Speaker 1") for s in speech_segments if s.get("speaker")}
    speaker_count = max(1, len(speakers)) if speech_segments else 0

    prompt_payload = {
        "transcript": transcript,
        "speaker_turns": [{"speaker": s.get("speaker", "Speaker 1"), "text": s.get("text")} for s in speech_segments],
        "sound_events": sound_events,
        "pitch_stats": pitch_stats or {},
    }
    llm = None
    if transcript or sound_events:
        llm = _call_groq_json(
            "You are SonicCaption's multimodal audio intelligence assistant. "
            "Return JSON with keys: 'summary' (concise 1-2 sentence synthesis covering speech + sound events + mood), "
            "'reminders' (array of objects with 'title', 'due_time', 'priority' in {'low','medium','high'}, "
            "'trigger_source' in {'speech_action','acoustic_alert'}). Only extract genuine tasks/alerts.",
            json.dumps(prompt_payload),
        )

    if llm and isinstance(llm, dict) and isinstance(llm.get("summary"), str):
        reminders = []
        for r in llm.get("reminders", []) or []:
            if isinstance(r, dict) and r.get("title"):
                reminders.append(
                    {
                        "title": str(r["title"])[:140],
                        "due_time": str(r.get("due_time") or "Soon"),
                        "priority": str(r.get("priority") or "medium"),
                        "trigger_source": str(r.get("trigger_source") or "speech_action"),
                    }
                )
        return {
            "summary": llm["summary"].strip(),
            "reminders": reminders,
            "provider": "groq-llama",
        }

    return {
        "summary": _fallback_summary(transcript, sound_events, speaker_count, pitch_stats),
        "reminders": _heuristic_reminders(transcript, sound_events),
        "provider": "rule-based-local",
    }


def run_voice_agent_turn(
    user_text: str,
    sound_events: List[Dict[str, Any]],
    pitch_stats: Optional[Dict[str, Any]] = None,
    pending_reminders: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Generate an acoustic-aware spoken AI assistant reply + extracted follow-up reminders."""
    events_desc = [f"{e['label']} ({int(float(e.get('confidence', 0)) * 100)}%)" for e in sound_events]
    urgent = [e for e in sound_events if float(e.get("confidence", 0)) >= 0.55 and any(u in str(e.get("label", "")).lower() for u in ("siren", "glass", "knock", "dog"))]

    context_obj = {
        "user_said": user_text,
        "heard_ambient_sounds": events_desc,
        "vocal_pitch_hz": (pitch_stats or {}).get("mean_f0_hz", 0),
        "pending_reminders": [f"{r['title']} ({r['due_time']})" for r in (pending_reminders or [])[:6]],
        "recent_turns": (history or [])[-4:],
    }

    llm = _call_groq_json(
        "You are SonicCaption Copilot, a helpful, warm acoustic-aware AI voice agent. "
        "Unlike regular chatbots, you hear BOTH what the user says AND background ESC-50 sound events around them. "
        "Keep 'reply' natural and concise (1-3 short spoken sentences) so it sounds great via Text-to-Speech. "
        "If 'heard_ambient_sounds' includes a notable sound (e.g., Door knock, Siren, Laughing, Coughing, Dog barking, Rain), acknowledge it helpfully! "
        "Return JSON with keys: 'reply' (string) and 'new_reminders' (list of {'title','due_time','priority'}).",
        json.dumps(context_obj),
    )

    if llm and isinstance(llm, dict) and isinstance(llm.get("reply"), str):
        new_rem = [
            {
                "title": str(r["title"])[:140],
                "due_time": str(r.get("due_time") or "Soon"),
                "priority": str(r.get("priority") or "medium"),
                "trigger_source": "speech_action",
            }
            for r in (llm.get("new_reminders") or [])
            if isinstance(r, dict) and r.get("title")
        ]
        return {"reply": llm["reply"].strip(), "new_reminders": new_rem, "provider": "groq-llama"}

    # Clean offline acoustic-aware fallback
    extracted = _heuristic_reminders(user_text, sound_events)
    reply_parts: List[str] = []
    if urgent:
        lbl = urgent[0]["label"]
        reply_parts.append(f"Heads up—I also heard {lbl} in your background audio.")

    lower = user_text.lower()
    if extracted:
        titles = ", ".join(r["title"] for r in extracted if r["trigger_source"] == "speech_action")
        if titles:
            reply_parts.append(f"Got it, I saved that reminder for you: {titles}.")
    elif "reminder" in lower or "schedule" in lower or "task" in lower:
        if pending_reminders:
            top = "; ".join(f"{r['title']} ({r['due_time']})" for r in pending_reminders[:3])
            reply_parts.append(f"You currently have {len(pending_reminders)} active reminders: {top}.")
        else:
            reply_parts.append("Your reminder queue is currently clear.")
    elif user_text.strip() and not user_text.startswith("[Speech detected"):
        if events_desc and not urgent:
            reply_parts.append(f"I heard you say \"{user_text}\" alongside ambient {events_desc[0]}.")
        else:
            reply_parts.append(f"Understood: \"{user_text}\". I've logged this interaction to your SonicCaption call history.")
    else:
        if events_desc:
            reply_parts.append(f"I detected {events_desc[0]} in your audio. Set a free Groq API key in Settings anytime for full conversational replies.")
        else:
            reply_parts.append("Audio received and logged. Add a free Groq API key in Settings to unlock full Whisper transcription and Llama voice replies.")

    return {"reply": " ".join(reply_parts), "new_reminders": extracted, "provider": "rule-based-local"}
