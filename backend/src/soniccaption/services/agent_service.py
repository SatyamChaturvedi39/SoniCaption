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
DEFAULT_GROQ_CHAT_MODEL = "openai/gpt-oss-20b"
_FALLBACK_CHAT_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b")

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
_BOOKMARK_RE = re.compile(
    r"(?:bookmark(?:\s+this|\s+that)?|highlight(?:\s+this|\s+that)?|note down|key takeaway|key concept|flashcard|take a note(?:\s+that)?)\s*[:,-]?\s*([^.?!;\n]+)",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b((?:today|tonight|tomorrow|next\s+(?:week|month|sprint|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"this\s+(?:week|morning|afternoon|evening|sprint|friday|monday)|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"end\s+of\s+(?:the\s+)?(?:day|week|sprint)|eod|eow|by\s+noon|noon|midnight|asap|"
    r"within\s+(?:\d+\s+(?:minutes?|hours?|days?)|an?\s+hour))"
    r"(?:\s+(?:by|at|around|before)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?)?|"
    r"(?:by|at|before|around)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?|"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.))\b",
    re.IGNORECASE,
)

_SELF_COMMIT_RE = re.compile(
    r"(?:^|[.?!;,\s]+)(?:I\s+will|I'll|I\s+am\s+going\s+to|I'm\s+going\s+to|I\s+can|I\s+have\s+to|I\s+need\s+to|"
    r"I\s+have\s+got\s+to|I\s+gotta|let\s+me|let's|we\s+will|we'll|we\s+are\s+going\s+to|we're\s+going\s+to|"
    r"we\s+need\s+to|we\s+have\s+to|make\s+sure\s+to|remember\s+to)\s+([^.?!;\n]+)",
    re.IGNORECASE,
)
_DELEGATED_COMMIT_RE = re.compile(
    r"\b(Speaker\s*\d+|[A-Z][a-z]{2,15})\s+(?:will|is\s+going\s+to|can|needs\s+to|has\s+to|should|shall|must)\s+([^.?!;\n]+)",
)


def _format_deadline(raw_phrase: str) -> tuple[str, str]:
    """Extract clean deadline expression and strip trailing deadline words from deliverable phrase."""
    m = _TIME_RE.search(raw_phrase)
    if not m:
        return raw_phrase.strip(" ,."), "Unscheduled"
    dl = re.sub(r"^(?:by|at|before|around)\s+", "", m.group(0).strip(), flags=re.IGNORECASE).strip()
    dl_fmt = dl[:1].upper() + dl[1:]
    cleaned = (raw_phrase[: m.start()] + raw_phrase[m.end() :]).strip(" ,.")
    cleaned = re.sub(r"\s+(?:by|on|at|before)$", "", cleaned, flags=re.IGNORECASE).strip(" ,.")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return (cleaned or raw_phrase.strip(" ,.")), dl_fmt


def extract_corporate_action_items(speech_segments: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Parse spoken commitments into structured tasks mapped to responsible speaker, deliverable, and deadline."""
    items: List[Dict[str, Any]] = []
    seen = set()

    for seg in speech_segments or []:
        txt = str(seg.get("text", "")).strip()
        if not txt or txt.startswith("[Speech detected"):
            continue
        speaker = str(seg.get("speaker") or "Speaker 1")
        ts = round(float(seg.get("start", 0.0)), 2)

        for pat, explicit_owner in ((_SELF_COMMIT_RE, False), (_DELEGATED_COMMIT_RE, True)):
            for m in pat.finditer(txt):
                if explicit_owner:
                    candidate_owner = m.group(1).strip()
                    if candidate_owner.lower() in {"that", "this", "there", "here", "what", "which", "when", "they", "it", "everyone", "someone"}:
                        continue
                    owner = candidate_owner.title() if candidate_owner.lower().startswith("speaker") else f"{candidate_owner} ({speaker})"
                    raw_body = m.group(2).strip(" ,.")
                else:
                    owner = speaker
                    raw_body = m.group(1).strip(" ,.")

                if len(raw_body) < 4:
                    continue
                task_clean, deadline = _format_deadline(raw_body)
                if len(task_clean) < 3:
                    continue
                task_title = (task_clean[:1].upper() + task_clean[1:])[:120]
                dedup_key = f"{owner.lower()}::{task_title.lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                is_urgent = any(w in raw_body.lower() for w in ("urgent", "asap", "today", "eod", "blocker", "immediately"))
                items.append({
                    "owner": owner,
                    "task": task_title,
                    "deadline": deadline,
                    "priority": "high" if is_urgent or str(seg.get("emphasis_tag", "")) == "Urgent" else "medium",
                    "timestamp_s": ts,
                    "source_quote": txt[:160],
                    "status": "open",
                })
    return items


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    """Match whole words/phrases using regex word boundaries to avoid substring traps (e.g., incidentally/up and down)."""
    pattern = r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b"
    return bool(re.search(pattern, text))


def _derive_rule_based_scenario(
    speech_segments: List[Dict[str, Any]],
    sound_events: List[Dict[str, Any]],
    action_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Deterministically infer corporate meeting type, room atmosphere, and active cues."""
    transcript = " ".join(str(s.get("text", "")) for s in speech_segments).lower()
    speakers = sorted({str(s.get("speaker") or "Speaker 1") for s in speech_segments if s.get("text")})
    q_segs = [s for s in speech_segments if s.get("is_question") or str(s.get("text", "")).strip().endswith("?")]
    urgent_segs = [s for s in speech_segments if s.get("emphasis_tag") in ("Urgent", "Emphatic")]

    if _has_any_term(transcript, ("github", "commit", "pull request", "merge request", "deploy", "deployment", "api", "bug", "branch", "sprint", "codebase", "server", "database", "testing", "frontend", "backend", "repository", "ticket")):
        scenario_label = "Engineering Sprint Sync & Technical Review"
    elif _has_any_term(transcript, ("outage", "incident", "escalation", "server down", "service down", "production down", "site down", "blocker", "hotfix", "critical bug")):
        scenario_label = "Incident Escalation & Blocker Triage"
    elif _has_any_term(transcript, ("client", "contract", "budget", "pricing", "revenue", "invoice", "deal", "proposal", "sales", "quarter", "stakeholder", "roadmap")):
        scenario_label = "Client / Executive Strategy Sync"
    elif len(q_segs) >= 2 and len(speech_segments) >= 3:
        scenario_label = "Structured Q&A / Design Review"
    elif len(speakers) >= 2 and action_items:
        scenario_label = "Team Operational Planning & Task Handoff"
    elif len(speakers) >= 2:
        scenario_label = "Multi-Speaker Team Discussion"
    elif speech_segments:
        scenario_label = "Executive Briefing / Narrative Update"
    else:
        scenario_label = "Awaiting Meeting Audio"

    ev_labels = [str(e.get("label", "")).lower() for e in sound_events]
    active_cues: List[str] = []
    mood_parts: List[str] = []

    if q_segs:
        active_cues.append(f"{len(q_segs)} question inflection{'s' if len(q_segs) > 1 else ''} flagged")
    if urgent_segs:
        active_cues.append(f"{len(urgent_segs)} emphatic/urgent remark{'s' if len(urgent_segs) > 1 else ''}")
        mood_parts.append("High Energy / Urgent")
    if action_items:
        active_cues.append(f"{len(action_items)} deliverable{'s' if len(action_items) > 1 else ''} assigned")
    if any("clapping" in l or "applause" in l for l in ev_labels):
        active_cues.append("Applause / Group agreement")
        mood_parts.append("Consensus")
    if any("laugh" in l for l in ev_labels):
        active_cues.append("Room laughter heard")
        mood_parts.append("Lighthearted")
    if any("knock" in l for l in ev_labels):
        active_cues.append("Door knock / desk tap cue")
    if any("glass" in l or "bang" in l for l in ev_labels):
        active_cues.append("Sharp desk impact / bang")
    if any("siren" in l for l in ev_labels):
        active_cues.append("Building alarm / siren alert")
        mood_parts.append("Safety Alert")

    mood = " · ".join(dict.fromkeys(mood_parts)) if mood_parts else "Focused & Professional"
    return {
        "scenario_label": scenario_label,
        "meeting_mood": mood,
        "question_count": len(q_segs),
        "action_count": len(action_items),
        "speaker_count": len(speakers),
        "active_cues": active_cues,
    }


def search_archive_sessions(query: str, sessions: Optional[List[Dict[str, Any]]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve the most relevant saved audio sessions for Voice Copilot Archive Q&A."""
    if not sessions:
        return []
    tokens = {w for w in re.findall(r"[a-z0-9]{3,}", query.lower()) if w not in {"what", "when", "where", "which", "about", "there", "were", "have", "from", "with", "this", "that", "into", "recordings", "session", "sessions", "last"}}
    scored: List[tuple[float, Dict[str, Any]]] = []
    for idx, s in enumerate(sessions):
        timeline_text = " ".join(str(t.get("display_text", "")) + " " + str(t.get("label", "")) for t in (s.get("timeline") or []))
        blob = f"{s.get('filename', '')} {s.get('summary', '')} {timeline_text}".lower()
        hits = sum(1 for tok in tokens if tok in blob)
        recency_bonus = max(0.0, 0.4 - idx * 0.05)
        if hits > 0 or not tokens or any(k in query.lower() for k in ("last", "recent", "latest", "summarize", "summary", "all")):
            scored.append((hits + recency_bonus, s))
    scored.sort(key=lambda pair: -pair[0])
    out: List[Dict[str, Any]] = []
    for _, s in scored[:top_k]:
        out.append({
            "session_id": s.get("id"),
            "filename": s.get("filename", "recording"),
            "created_at": s.get("created_at", ""),
            "duration": s.get("duration", 0.0),
            "summary": s.get("summary", ""),
            "transcript": " ".join(str(t.get("display_text", "")) for t in (s.get("timeline") or []) if t.get("category") == "speech")[:320],
            "sounds": [t.get("label") for t in (s.get("timeline") or []) if t.get("category") == "sound_event"],
        })
    return out


def _call_groq_json(system_prompt: str, user_prompt: str, timeout: float = 20.0) -> Optional[dict]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    configured = os.environ.get("GROQ_CHAT_MODEL", DEFAULT_GROQ_CHAT_MODEL).strip() or DEFAULT_GROQ_CHAT_MODEL
    candidates = list(dict.fromkeys([configured, *_FALLBACK_CHAT_MODELS]))

    try:
        import requests
    except Exception:
        return None

    for model_id in candidates:
        try:
            resp = requests.post(
                GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model_id,
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
                content = (resp.json()["choices"][0]["message"]["content"] or "").strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL).strip()
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    parsed["_model_used"] = model_id
                    return parsed
            elif resp.status_code == 401:
                return None
            else:
                continue
        except Exception:
            continue
    return None


def _heuristic_notes_and_actions(
    transcript: str,
    sound_events: List[Dict[str, Any]],
    speech_segments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    seen = set()

    for m in _BOOKMARK_RE.finditer(transcript):
        phrase = m.group(1).strip(" ,.")
        if not phrase or phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        items.append(
            {
                "title": phrase[:1].upper() + phrase[1:],
                "due_time": "Voice Note",
                "priority": "medium",
                "trigger_source": "voice_bookmark",
            }
        )

    for m in _REMINDER_RE.finditer(transcript):
        phrase = m.group(1).strip(" ,.")
        if not phrase or phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        t_match = _TIME_RE.search(phrase)
        items.append(
            {
                "title": phrase[:1].upper() + phrase[1:],
                "due_time": t_match.group(1).title() if t_match else "Follow-up",
                "priority": "high" if any(k in phrase.lower() for k in ("urgent", "asap", "doctor", "exam", "call")) else "medium",
                "trigger_source": "speech_action",
            }
        )

    # Extract acoustic emphasis & question moments directly from vocal delivery
    for seg in speech_segments or []:
        style = str(seg.get("delivery_style", ""))
        txt = str(seg.get("text", "")).strip()
        if not txt or txt.startswith("[Speech detected"):
            continue
        if ("Emphasized" in style or "Question" in style) and len(txt) >= 12 and txt.lower() not in seen:
            seen.add(txt.lower())
            items.append(
                {
                    "title": txt[:135] + ("..." if len(txt) > 135 else ""),
                    "due_time": f"{float(seg.get('start', 0.0)):.1f}s ({style.split('/')[0].strip()})",
                    "priority": "high" if "Emphasized" in style else "medium",
                    "trigger_source": "prosody_highlight",
                }
            )
            if len(items) >= 5:
                break

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
    """Produce corporate meeting synthesis, structured action items, and dynamic meeting scenario."""
    transcript = " ".join(
        str(s.get("text", "")).strip()
        for s in speech_segments
        if not str(s.get("text", "")).startswith("[Speech detected")
    ).strip()
    speakers = {s.get("speaker", "Speaker 1") for s in speech_segments if s.get("speaker")}
    speaker_count = max(1, len(speakers)) if speech_segments else 0

    rule_actions = extract_corporate_action_items(speech_segments)
    rule_scenario = _derive_rule_based_scenario(speech_segments, sound_events, rule_actions)

    compact_pitch = {
        k: round(float(v), 2) if isinstance(v, (int, float)) else v
        for k, v in (pitch_stats or {}).items()
        if not isinstance(v, (list, tuple)) and k not in ("times", "f0_hz")
    }
    prompt_payload = {
        "transcript": transcript,
        "speaker_turns": [
            {
                "speaker": s.get("speaker", "Speaker 1"),
                "start": round(float(s.get("start", 0.0)), 2),
                "emphasis_tag": s.get("emphasis_tag", "Neutral"),
                "is_question": bool(s.get("is_question", False)),
                "text": s.get("text"),
            }
            for s in speech_segments[:40]
        ],
        "sound_events": sound_events[:25],
        "pitch_stats": compact_pitch,
    }
    llm = None
    if transcript or sound_events:
        llm = _call_groq_json(
            "You are SonicCaption Corporate Meeting Copilot assisting deaf workplace professionals. "
            "Given speaker turns (with emphasis tags & question flags) and room sound events (laughter, door knocks, clapping, desk bangs), return JSON with:\n"
            "1. 'summary': concise 1-2 sentence executive summary covering key spoken points, questions, and room cues.\n"
            "2. 'meeting_scenario': object with 'scenario_label' (short corporate meeting type e.g. 'Engineering Sprint Standup', 'Client Negotiation', 'Incident Triage', 'Executive Briefing', 'Team Brainstorming') and 'meeting_mood' (e.g. 'Collaborative & Urgent').\n"
            "3. 'action_items': array of spoken commitments mapping 'owner' (responsible speaker e.g. 'Speaker 3'), 'task' (concise deliverable e.g. 'Update GitHub'), 'deadline' (e.g. 'Tomorrow', '4:00 PM', 'Unscheduled'), 'priority' in {'low','medium','high'}, 'timestamp_s' (number).\n"
            "4. 'highlights': array of 1-3 key highlights with 'title', 'tag', 'priority', 'trigger_source' in {'prosody_highlight','speech_action','acoustic_alert'}.",
            json.dumps(prompt_payload, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)),
        )

    if llm and isinstance(llm, dict) and isinstance(llm.get("summary"), str):
        notes = []
        raw_items = llm.get("highlights") or llm.get("reminders") or []
        for r in raw_items:
            if isinstance(r, dict) and r.get("title"):
                notes.append(
                    {
                        "title": str(r["title"])[:140],
                        "due_time": str(r.get("tag") or r.get("due_time") or "Key Concept"),
                        "priority": str(r.get("priority") or "medium"),
                        "trigger_source": str(r.get("trigger_source") or "prosody_highlight"),
                    }
                )
        llm_actions = []
        for a in (llm.get("action_items") or []):
            if isinstance(a, dict) and a.get("task"):
                llm_actions.append({
                    "owner": str(a.get("owner") or "Speaker 1"),
                    "task": str(a["task"])[:120],
                    "deadline": str(a.get("deadline") or "Unscheduled"),
                    "priority": str(a.get("priority") or "medium"),
                    "timestamp_s": float(a.get("timestamp_s") or 0.0),
                    "source_quote": "",
                    "status": "open",
                })
        actions = llm_actions if llm_actions else rule_actions

        raw_scen = llm.get("meeting_scenario") if isinstance(llm.get("meeting_scenario"), dict) else {}
        scenario = {
            **rule_scenario,
            "scenario_label": str(raw_scen.get("scenario_label") or rule_scenario["scenario_label"]),
            "meeting_mood": str(raw_scen.get("meeting_mood") or rule_scenario["meeting_mood"]),
            "action_count": len(actions),
        }
        return {
            "summary": llm["summary"].strip(),
            "action_items": actions,
            "meeting_scenario": scenario,
            "reminders": notes,
            "highlights": notes,
            "provider": f"groq-{llm.get('_model_used', 'llm')}",
        }

    notes = _heuristic_notes_and_actions(transcript, sound_events, speech_segments=speech_segments)
    return {
        "summary": _fallback_summary(transcript, sound_events, speaker_count, pitch_stats),
        "action_items": rule_actions,
        "meeting_scenario": rule_scenario,
        "reminders": notes,
        "highlights": notes,
        "provider": "rule-based-local",
    }


def run_voice_agent_turn(
    user_text: str,
    sound_events: List[Dict[str, Any]],
    pitch_stats: Optional[Dict[str, Any]] = None,
    pending_reminders: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    recent_sessions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Voice Copilot turn: answers questions across recorded sessions, extracts voice bookmarks, and reads ambient sound."""
    events_desc = [f"{e['label']} ({int(float(e.get('confidence', 0)) * 100)}%)" for e in sound_events]
    urgent = [e for e in sound_events if float(e.get("confidence", 0)) >= 0.55 and any(u in str(e.get("label", "")).lower() for u in ("siren", "glass", "knock", "dog"))]
    matched_sessions = search_archive_sessions(user_text, recent_sessions, top_k=3) if user_text.strip() else []

    context_obj = {
        "user_said": user_text,
        "heard_ambient_sounds": events_desc,
        "vocal_pitch_hz": (pitch_stats or {}).get("mean_f0_hz", 0),
        "matching_recorded_sessions": matched_sessions,
        "saved_study_notes": [f"{r['title']} ({r['due_time']})" for r in (pending_reminders or [])[:6]],
        "recent_turns": (history or [])[-4:],
    }

    llm = _call_groq_json(
        "You are SonicCaption Voice Copilot — a speech-recognition & acoustic study assistant that lets users search their recorded audio archive, bookmark key spoken concepts, and inspect background sound events. "
        "Answer questions directly using 'matching_recorded_sessions' (citing filenames, spoken content, and sound events accurately when asked about past recordings). "
        "Keep 'reply' natural and concise (1-3 sentences suited for Text-to-Speech). "
        "If 'heard_ambient_sounds' includes a notable event (e.g., Door knock, Siren, Clapping, Coughing), mention it briefly. "
        "Only extract 'new_notes' if the user explicitly asks to bookmark, note, highlight, flashcard, or remember something. "
        "Return JSON with keys: 'reply' (string) and 'new_notes' (list of {'title','tag','priority','trigger_source'}).",
        json.dumps(context_obj),
    )

    if llm and isinstance(llm, dict) and isinstance(llm.get("reply"), str):
        raw_notes = llm.get("new_notes") or llm.get("new_reminders") or []
        new_rem = [
            {
                "title": str(r["title"])[:140],
                "due_time": str(r.get("tag") or r.get("due_time") or "Voice Note"),
                "priority": str(r.get("priority") or "medium"),
                "trigger_source": str(r.get("trigger_source") or "voice_bookmark"),
            }
            for r in raw_notes
            if isinstance(r, dict) and r.get("title")
        ]
        return {
            "reply": llm["reply"].strip(),
            "new_reminders": new_rem,
            "new_notes": new_rem,
            "matched_sessions": matched_sessions,
            "provider": f"groq-{llm.get('_model_used', 'llm')}",
        }

    # Clean deterministic offline Archive Q&A + acoustic-aware fallback
    extracted = _heuristic_notes_and_actions(user_text, sound_events)
    reply_parts: List[str] = []
    if urgent:
        lbl = urgent[0]["label"]
        reply_parts.append(f"Heads up—I also heard {lbl} in your audio.")

    lower = user_text.lower()
    if extracted:
        titles = ", ".join(r["title"] for r in extracted if r["trigger_source"] in ("voice_bookmark", "speech_action"))
        if titles:
            reply_parts.append(f"Saved voice note: {titles}.")
    elif any(k in lower for k in ("summarize", "summary", "recording", "recordings", "session", "find", "what did", "when", "speaker", "history", "search")):
        if matched_sessions:
            top = matched_sessions[0]
            snd = f" Sounds: {', '.join(top['sounds'][:3])}." if top["sounds"] else ""
            snip = top["transcript"] or top["summary"] or "Audio analyzed."
            reply_parts.append(f"From '{top['filename']}' ({top['duration']}s): {snip[:160]}.{snd}")
        else:
            reply_parts.append("No matching recordings found in your Audio Archive yet. Analyze a clip in Audio Studio first.")
    elif "note" in lower or "bookmark" in lower or "highlight" in lower or "reminder" in lower or "task" in lower:
        if pending_reminders:
            top = "; ".join(f"{r['title']} ({r['due_time']})" for r in pending_reminders[:3])
            reply_parts.append(f"You have {len(pending_reminders)} saved study notes: {top}.")
        else:
            reply_parts.append("Your saved study notes list is empty. Say 'Bookmark ...' or analyze a lecture clip to populate it.")
    elif user_text.strip() and not user_text.startswith("[Speech detected"):
        if events_desc and not urgent:
            reply_parts.append(f"I heard \"{user_text}\" alongside ambient {events_desc[0]}.")
        else:
            reply_parts.append(f"Logged: \"{user_text}\". Ask about past recordings or say 'Bookmark ...' to save key concepts.")
    else:
        if events_desc:
            reply_parts.append(f"Detected {events_desc[0]} in your audio.")
        else:
            reply_parts.append("Audio received. Ask about saved recordings or say 'Bookmark ...' to save a spoken concept.")

    return {
        "reply": " ".join(reply_parts),
        "new_reminders": extracted,
        "new_notes": extracted,
        "matched_sessions": matched_sessions,
        "provider": "rule-based-local",
    }
