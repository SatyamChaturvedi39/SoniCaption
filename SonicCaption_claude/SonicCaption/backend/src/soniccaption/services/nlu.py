"""Conversation understanding on top of the speaker-attributed transcript.

Answers the questions in the project workflow:
    * What situation is this?            -> scenario (meeting / phone call / lecture / interview / ...)
    * What did each person talk about?   -> per-speaker topics, key points, talk time
    * Who has to do what?                -> tasks, each *assigned to a speaker*
                                            "I'll make the changes to the repository by Friday" said by Speaker 2
                                            -> Task: "Make the changes to the repository" owner Speaker 2, due Friday

Everything runs offline with deterministic rules (so it always works and is testable). When GROQ_API_KEY is set the same
transcript is also sent to an LLM and its tasks/summaries are merged in, never replacing a rule-based result silently.

Input  : utterances = [{"start","end","text","speaker_id","speaker"}, ...] in time order
Output : analyse_conversation(utterances, sound_events=None, duration=None) -> dict (see bottom)
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

# --------------------------------------------------------------------------------------- vocabulary
STOP = set("""a about above after again all also am an and any are aren't as at be because been before being below between both but by can can't cannot
could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just let's like me more most mustn't
my myself no nor not now of off ok okay on once only or other ought our ours ourselves out over own really same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those
through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom
why why's will with won't would wouldn't yeah yes yep sure well um uh hmm right actually basically maybe thing things going gonna want need get got
make know think see say said one two three today tomorrow thanks thank please hello hey good morning afternoon evening bye everyone everybody someone
anyone anything everything something nothing decided great nice stuff lot bit kind sort next first last new old big small little long high low many much
few lots let's lets okay start started over also still back look looking take taking talk talking tell told ask asked give gave come came went""".split())

NOT_NAMES = set("""good fine sorry glad happy here back just not ready sure going trying calling looking afraid late early done okay ok great
thanks thank hello hey hi yes no well also still really very fine tired busy free sure wondering curious calling""".split())

DAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_DUE = re.compile(
    r"\b(?P<due>(?:by|before|until|on|this|next|at)?\s*(?:"
    r"end of (?:the )?(?:day|week|month|sprint)|eod|eow|cob|tonight|today|tomorrow|this (?:morning|afternoon|evening|week|weekend)|"
    r"next (?:week|month|"
    + DAYS + r")|(?:" + DAYS + r")|\d{1,2}(?::\d{2})?\s*(?:am|pm)|the \d{1,2}(?:st|nd|rd|th)|in \d+ (?:minutes?|hours?|days?|weeks?)|asap|right away|immediately))\b",
    re.IGNORECASE)
_URGENT = re.compile(r"\b(urgent(?:ly)?|asap|immediately|right away|critical|blocker|today|tonight|eod|end of (?:the )?day|emergency)\b", re.I)
_LOW = re.compile(r"\b(whenever|when you get a chance|eventually|sometime|if possible|no rush|at some point|low priority)\b", re.I)
_AFFIRM = re.compile(r"\b(sure|okay|ok|yes|yep|yeah|will do|on it|got it|no problem|sounds good|i'?ll|i will|i can|absolutely|of course|definitely|happy to)\b", re.I)
_DECLINE = re.compile(r"\b(can'?t|cannot|won'?t|not able|unable|no[, ]|sorry)\b", re.I)

# a commitment: first person + future / intent. (pattern, kind, confidence)
_FIRST = [
    (re.compile(r"\bI(?:'ll| will| shall)\s+(?!not\b|never\b|probably not\b|see\b|think\b|try to see\b)(?P<vp>[a-z].+)", re.I), "commitment", 0.9),
    (re.compile(r"\bI(?:'m| am)\s+(?:going to|gonna|about to)\s+(?!see\b|think\b)(?P<vp>[a-z].+)", re.I), "commitment", 0.85),
    (re.compile(r"\blet me\s+(?!know\b|think\b|see\b|just\b|say\b|explain\b)(?P<vp>[a-z].+)", re.I), "commitment", 0.8),
    (re.compile(r"\bI (?:can|could)\s+(?:also )?(?P<vp>(?:take|handle|do|work on|pick up|look into|check|fix|update|send|review|prepare|set up|schedule|follow up|write|draft|create|build|test|deploy|merge|push|change|make|finish|complete|fix|investigate|own)\b.+)", re.I), "offer", 0.7),
    (re.compile(r"\bI (?:need|have|got|must|want|should) to\s+(?P<vp>[a-z].+)", re.I), "obligation", 0.75),
    (re.compile(r"\bI(?:'ve| have) got to\s+(?P<vp>[a-z].+)", re.I), "obligation", 0.75),
    (re.compile(r"\b(?:remind me to|don't let me forget to|note to self[,:]? (?:to )?)\s*(?P<vp>[a-z].+)", re.I), "reminder", 0.9),
    (re.compile(r"\b(?:i'?ll|i will) take (?:care of|on|over|ownership of)\s+(?P<vp>.+)", re.I), "commitment", 0.9),
]
# a request to someone else. group "name" optional
_REQUEST = [
    re.compile(r"(?:^|[.?!,]\s*)(?:(?P<name>[A-Z][a-z]{2,})[,:]?\s+)?(?:can|could|would|will) you (?:please |also |just )?(?P<vp>[a-z].+?)\??$", re.I),
    re.compile(r"(?:^|[.?!,]\s*)(?:(?P<name>[A-Z][a-z]{2,})[,:]?\s+)?please\s+(?P<vp>[a-z].+)", re.I),
    re.compile(r"\byou (?:should|need to|have to|must|ought to)\s+(?P<vp>[a-z].+)", re.I),
]
_THIRD = re.compile(r"\b(?P<name>[A-Z][a-z]{2,})\s+(?:will|should|needs to|has to|is going to|is gonna|can you|to)\s+(?P<vp>[a-z].+)")
_TEAM = re.compile(r"\bwe(?:'ll| will| need to| should| must| have to| have got to)\s+(?!not\b|see\b|think\b)(?P<vp>[a-z].+)", re.I)
_ACTION_MARK = re.compile(r"\b(?:action item|to-?do|follow[- ]?up|next step)s?\s*[:,\-]\s*(?P<vp>.+)", re.I)

_TASK_VERBS = set("""update fix change make send review prepare set schedule write draft create build test deploy merge push finish complete
investigate check look call email message follow talk speak share submit upload download install configure refactor document implement
add remove delete clean organize book arrange confirm pay buy order pick drop bring ship release publish rewrite revise design plan
research analyze analyse compile gather collect compare run start open close assign handle take own approve sign print file report
contact reach inform notify ping loop circle sort resolve debug rebase commit raise log record prepare cancel reschedule move rename
renew apply register enrol enroll finalize finalise lead coordinate discuss present demo upgrade migrate monitor restart reset ask
get read study revise submit practice practise cook clean wash repair""".split())

_NON_TASK_START = re.compile(
    r"^(?:to )?(?:hear|see|tell|understand|believe|imagine|think|know|remember|recall|guess|say|explain|help me understand|confirm that you|be (?:there|here|available|on time)|"
    r"walk (?:you|us)|show (?:you|us)|take (?:you|us) through|go through|run (?:you|us)|give (?:you|us) (?:a|an|the) (?:overview|brief|quick|summary)|start (?:with|by)|begin (?:with|by)|"
    r"introduce|share (?:my|the) screen)\b", re.I)

_FILLER = re.compile(r"\b(um+|uh+|you know|like|basically|actually|kind of|sort of|i mean)\b[, ]*", re.I)

# --------------------------------------------------------------------------------------- scenarios
SCENARIOS: dict[str, dict] = {
    "Meeting": {"cues": r"meeting|agenda|stand-?up|sprint|status|deadline|action items?|follow[- ]?up|next steps?|minutes|ticket|release|deploy\w*|review|repo(?:sitory)?|pull request|\bpr\b|merge|roadmap|milestone|budget|client|stakeholders?|quarter\w*|sync|blockers?|circle back|assign\w*|backlog|deliverables?|commit|branch|update(?:s)? on|project|team|report|presentation|schedule",
                "w": 1.0},
    "Phone call": {"cues": r"hello|can you hear me|breaking up|hang up|call(?:ing)? (?:you )?back|call me|on the phone|voicemail|who'?s (?:this|speaking)|speaking|are you there|bad (?:line|connection|signal)|ring(?:ing)?|pick up|hold on|talk later|\bbye\b|take care",
                "w": 1.0},
    "Lecture / class": {"cues": r"today we(?:'ll| will)|lecture|chapter|exam|assignment|homework|students?|class|syllabus|professor|as you can see|formula|theorem|equation|slide|any questions|quiz|semester|textbook|learn(?:ing)?|concept|example",
                        "w": 1.0},
    "Interview": {"cues": r"tell me about yourself|experience|strengths?|weakness(?:es)?|candidate|position|\brole\b|resume|\bcv\b|hire|hiring|why do you want|salary|notice period|your background|interview|applying|opportunity|team fit|previous (?:job|company)",
                  "w": 1.0},
    "Customer support call": {"cues": r"customers?|order|refund|account|ticket number|how can i help|reference number|issue|calling about|warranty|subscription|billing|thank you for calling|replacement|complaint|invoice|delivery|tracking|support",
                              "w": 1.0},
    "Casual conversation": {"cues": r"lol|dude|movie|dinner|lunch|weekend|party|game|friends?|awesome|cool|funny|birthday|vacation|holiday|trip|weather|music|restaurant|coffee|netflix|sleep|girlfriend|boyfriend|family|mom|dad|brother|sister",
                            "w": 0.9},
    "Podcast / interview show": {"cues": r"welcome to the (?:show|podcast)|podcast|episode|listeners?|subscribe|our guest|today'?s guest|thanks for having me|tune in",
                                 "w": 1.2},
    "Presentation / talk": {"cues": r"welcome everyone|thank you (?:all )?for (?:coming|listening|joining)|in this (?:talk|presentation|session)|let me (?:walk|take) you through|next slide|agenda for today|to summari[sz]e|in conclusion|key takeaways?",
                            "w": 1.2},
    "Voice note / reminder": {"cues": r"remind me|note to self|reminder|don'?t forget|remember to|to-?do|i need to|i have to|my list|shopping list",
                              "w": 1.0},
}
_SCEN_RE = {k: re.compile(r"\b(?:" + v["cues"] + r")\b", re.I) for k, v in SCENARIOS.items()}


# --------------------------------------------------------------------------------------- helpers
def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\s*;\s+", text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) > 1]


def _clean_vp(vp: str) -> tuple[str, str | None]:
    """Strip fillers/trailing clauses and pull the due phrase out of a verb phrase. Returns (task_text, due)."""
    vp = vp.strip().rstrip(".?!,; ")
    vp = re.sub(r",\s*[A-Z][a-z]{2,}$", "", vp)                       # "... the report, Priya"
    vp = re.split(r"\s+(?:but|because|since|so that|and then|which means|otherwise)\s+|[,;]\s+(?:and |but |so )", vp, maxsplit=1)[0]
    due = None
    m = _DUE.search(vp)
    if m:
        due = re.sub(r"\s+", " ", m.group("due")).strip()
        due = re.sub(r"^(?:by|before|until|on|at)\s+", "", due, flags=re.I).strip() or None
        vp = (vp[:m.start()] + vp[m.end():]).strip(" ,")
        vp = re.sub(r"\s+(?:by|before|until|on|at)$", "", vp, flags=re.I)
    vp = _FILLER.sub(" ", vp)
    vp = re.sub(r"^(?:go ahead and|also|just|then|maybe)\s+", "", vp.strip(), flags=re.I)
    vp = re.sub(r"\s+(?:please|for you|for us|too|as well)$", "", vp, flags=re.I)
    vp = re.sub(r"\s+", " ", vp).strip(" ,.")
    if due:
        due = due[:1].upper() + due[1:]
    return vp, due


_PRONOUN_OBJ = {"it", "that", "this", "them", "those", "these", "so", "there", "now", "later", "soon", "then", "first"}


def _is_acknowledgement(vp: str) -> bool:
    """'do that', 'send it', 'take care of it': says yes to something already on the table, names no new work."""
    words = re.findall(r"[a-z']+", vp.lower())
    if not words or len(words) > 5:
        return False
    rest = [w for w in words[1:] if w not in ("care", "of", "on", "over", "with", "up", "out", "a", "the", "for", "you", "me", "us")]
    return all(w in _PRONOUN_OBJ for w in rest)


def _looks_like_task(vp: str) -> bool:
    words = re.findall(r"[A-Za-z']+", vp.lower())
    if len(words) < 2 or len(words) > 28:
        return False
    first = words[0]
    if first in ("be", "see", "think", "say", "know", "go") and len(words) < 3:
        return False
    if first in ("to",) and len(words) > 1:
        first = words[1]
    return first in _TASK_VERBS or (len(words) >= 3 and first not in STOP)


def _title(vp: str) -> str:
    vp = re.sub(r"^to\s+", "", vp.strip(), flags=re.I)
    vp = re.sub(r"\b(?:my|our)\b", lambda m: m.group(0), vp)
    return vp[:1].upper() + vp[1:]


def _priority(sentence: str, due: str | None) -> str:
    if _URGENT.search(sentence) or (due and re.search(r"today|tonight|eod|end of day|asap|right away|immediately", due, re.I)):
        return "high"
    if _LOW.search(sentence):
        return "low"
    return "medium"


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z\-']{2,}", text.lower()) if w not in STOP]


def _dedupe_key(text: str) -> frozenset:
    return frozenset(w[:5] for w in _content_words(text))


def label_of(spk_id: int, names: dict[int, str]) -> str:
    return f"Speaker {spk_id}" + (f" ({names[spk_id]})" if spk_id in names else "")


# --------------------------------------------------------------------------------------- names
def detect_names(utts: list[dict]) -> tuple[dict[int, str], dict[str, int]]:
    """Self-introductions ("Hi, this is Priya") -> {speaker_id: name}, plus reverse lookup lowercase-name -> speaker_id."""
    names: dict[int, str] = {}
    intro = re.compile(r"^(?i:(?:hi|hello|hey|good (?:morning|afternoon|evening))?[ ,.!]*(?:this is|i'?m|i am|my name is|it'?s))\s+(?P<n>[A-Z][a-z]{2,})\b")
    for u in utts:
        sid = u.get("speaker_id")
        if sid is None or sid in names:
            continue
        for s in _sentences(u["text"])[:2]:
            m = intro.match(s.strip())
            if m and m.group("n").lower() not in NOT_NAMES and m.group("n").lower() not in STOP:
                names[sid] = m.group("n")
                break
    return names, {n.lower(): s for s, n in names.items()}


def _addressed_name(sentence: str) -> str | None:
    m = re.match(r"^\s*(?:(?:hey|ok|okay|so|and|alright)[, ]+)?(?P<n>[A-Z][a-z]{2,})\s*[,:]", sentence)
    if m and m.group("n").lower() not in STOP and m.group("n").lower() not in NOT_NAMES:
        return m.group("n")
    m = re.search(r",\s*(?P<n>[A-Z][a-z]{2,})\s*[?.!]*$", sentence)
    if m and m.group("n").lower() not in STOP and m.group("n").lower() not in NOT_NAMES:
        return m.group("n")
    return None


# --------------------------------------------------------------------------------------- scenario
def classify_scenario(utts: list[dict], n_speakers: int, sound_events: list[dict] | None = None) -> dict:
    text = " ".join(u["text"] for u in utts)
    words = len(re.findall(r"\w+", text))
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for name, rx in _SCEN_RE.items():
        hits = [m.group(0).lower() for m in rx.finditer(text)]
        uniq = len(set(hits))
        scores[name] = SCENARIOS[name]["w"] * (min(len(hits), 8) * 0.5 + uniq * 1.0)
        evidence[name] = sorted(set(hits))[:6]
    # structural priors (who is talking) - small nudges, never enough to beat real vocabulary evidence
    if n_speakers == 1:
        scores["Presentation / talk"] += 0.8
        scores["Voice note / reminder"] += 0.8
        scores["Meeting"] -= 0.8
        scores["Phone call"] -= 1.0
        scores["Interview"] -= 1.0
        scores["Customer support call"] -= 1.0
        scores["Casual conversation"] -= 0.6
    elif n_speakers == 2:
        scores["Phone call"] += 0.7
        scores["Interview"] += 0.4
        scores["Customer support call"] += 0.3
        scores["Presentation / talk"] -= 1.0
        scores["Voice note / reminder"] -= 0.8
    elif n_speakers >= 3:
        scores["Meeting"] += 0.9
        scores["Phone call"] -= 1.2
        scores["Presentation / talk"] -= 0.5
        scores["Voice note / reminder"] -= 1.0
    for ev in sound_events or []:        # an acoustic cue: a door knock/footsteps => same room, not a phone line
        if str(ev.get("key", ev.get("label", ""))).lower().startswith(("door", "footstep")):
            scores["Phone call"] -= 0.4
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_s = ranked[0]
    total = sum(max(s, 0.0) for s in scores.values())
    if best_s < 1.5 or words < 6:
        label = "General conversation" if n_speakers >= 2 else "Monologue / voice note"
        return {"label": label, "confidence": 0.3 if words >= 6 else 0.1, "evidence": [],
                "alternatives": [{"label": k, "score": round(v, 2)} for k, v in ranked[:3] if v > 0], "speakers": n_speakers}
    conf = best_s / (total + 1e-9)
    conf = min(0.98, 0.35 + 0.65 * conf * (1 - math.exp(-best_s / 4)))
    return {"label": best, "confidence": round(conf, 2), "evidence": evidence[best],
            "alternatives": [{"label": k, "score": round(v, 2)} for k, v in ranked[1:4] if v > 0.5], "speakers": n_speakers}


# --------------------------------------------------------------------------------------- tasks
def extract_tasks(utts: list[dict], names: dict[int, str], name_to_spk: dict[str, int], n_speakers: int) -> list[dict]:
    """Rule-based action items, each with an owner. Order of preference: explicit commitment > assignment > team task."""
    tasks: list[dict] = []
    seen: set[frozenset] = set()

    def add(owner_id, owner_text, vp, kind, conf, sentence, utt, due_hint=None, requested_by=None):
        title_vp, due = _clean_vp(vp)
        due = due or due_hint
        if _NON_TASK_START.match(title_vp):
            return
        if _is_acknowledgement(title_vp):
            # "Sure, I'll do that (tomorrow)": confirms the latest task given to this speaker instead of creating a new one
            if owner_id is not None:
                for prev in reversed(tasks):
                    if prev["owner_id"] == owner_id and utt["start"] - prev["start"] <= 90:
                        prev["confirmed"] = True
                        prev["confirmed_by_text"] = sentence.strip()[:160]
                        if due and not prev["due"]:
                            prev["due"] = due
                        break
            return
        if not _looks_like_task(title_vp):
            return
        key = _dedupe_key(title_vp)
        if key in seen and key:
            return
        seen.add(key)
        if owner_id is not None:
            owner = label_of(owner_id, names)
        else:
            owner = owner_text or "Unassigned"
        tasks.append({
            "task": _title(title_vp), "owner": owner, "owner_id": owner_id, "owner_name": names.get(owner_id) if owner_id else (owner_text if owner_text and owner_text not in ("Unassigned", "Team") else None),
            "due": due, "priority": _priority(sentence, due), "kind": kind, "confidence": round(conf, 2),
            "source_text": sentence.strip()[:240], "start": round(float(utt["start"]), 2), "end": round(float(utt["end"]), 2),
            "said_by": label_of(utt.get("speaker_id") or 1, names), "said_by_id": utt.get("speaker_id"), "requested_by": requested_by,
        })

    for i, u in enumerate(utts):
        sid = u.get("speaker_id") or 1
        for sent in _sentences(u["text"]):
            if re.search(r"\b(?:if|when|unless|whether)\b.{0,25}\bI(?:'ll| will)\b", sent, re.I) and "?" in sent:
                continue                                              # hypotheticals / questions
            is_q = sent.rstrip().endswith("?")
            # 1. explicit first-person commitments
            if not is_q or re.search(r"\bI(?:'ll| will)\b", sent):
                for rx, kind, conf in _FIRST:
                    m = rx.search(sent)
                    if m:
                        # "I'll" inside a reply like "Sure, I'll do that" with no concrete verb phrase is handled below via the request
                        add(sid, None, m.group("vp"), kind, conf, sent, u)
                        break
            # 2. requests to someone else (assignment)
            for rx in _REQUEST:
                m = rx.search(sent)
                if not m:
                    continue
                vp = m.group("vp")
                nm = (m.groupdict().get("name") or _addressed_name(sent) or "")
                nm = nm if nm and nm.lower() not in STOP and nm.lower() not in NOT_NAMES else None
                assignee, assignee_text = None, None
                if nm and nm.lower() in name_to_spk and name_to_spk[nm.lower()] != sid:
                    assignee = name_to_spk[nm.lower()]
                elif nm:
                    assignee_text = nm
                else:
                    # the next speaker (someone else) who answers affirmatively takes it
                    for nxt in utts[i + 1:i + 4]:
                        if nxt.get("speaker_id") and nxt["speaker_id"] != sid:
                            if _AFFIRM.search(nxt["text"]) and not _DECLINE.search(nxt["text"][:30]):
                                assignee = nxt["speaker_id"]
                            break
                    if assignee is None and n_speakers == 2:
                        assignee = 1 if sid != 1 else 2             # in a two-person talk "you" can only be the other person
                if assignee is not None and nm and nm.lower() not in name_to_spk:
                    name_to_spk[nm.lower()] = assignee              # learn the name from the exchange
                    names.setdefault(assignee, nm)
                add(assignee, assignee_text or ("Unassigned" if assignee is None else None), vp, "assignment", 0.8 if assignee else 0.55,
                    sent, u, requested_by=label_of(sid, names))
                break
            else:
                # 3. "Priya will update the docs"
                m = _THIRD.search(sent)
                if m and m.group("name").lower() not in STOP and m.group("name").lower() not in NOT_NAMES:
                    nm = m.group("name")
                    owner_id = name_to_spk.get(nm.lower())
                    if owner_id is not None and owner_id == sid and re.search(r"\bI\b", sent):
                        pass
                    else:
                        add(owner_id, nm if owner_id is None else None, m.group("vp"), "assignment", 0.75, sent, u,
                            requested_by=label_of(sid, names))
                        continue
                # 4. shared / team tasks
                m = _TEAM.search(sent)
                if m and not is_q:
                    add(None, "Team", m.group("vp"), "team", 0.55, sent, u)
                    continue
                m = _ACTION_MARK.search(sent)
                if m:
                    add(sid, None, m.group("vp"), "action_item", 0.7, sent, u)
    tasks.sort(key=lambda t: t["start"])
    for k, t in enumerate(tasks, 1):
        t["id"] = f"task{k}"
    return tasks


# --------------------------------------------------------------------------------------- topics / per speaker
def _topics(per_speaker_text: dict[int, str], top_n: int = 5) -> dict[int, list[str]]:
    toks = {s: _content_words(t) for s, t in per_speaker_text.items()}
    df: Counter = Counter()
    for ws in toks.values():
        df.update(set(ws))
    n = max(1, len(toks))
    out: dict[int, list[str]] = {}
    for s, ws in toks.items():
        tf = Counter(ws)
        bigrams: Counter = Counter()
        for a, b in zip(ws, ws[1:]):
            bigrams[f"{a} {b}"] += 1
        scored: dict[str, float] = {}
        for w, c in tf.items():
            idf = math.log(1 + n / df[w]) if n > 1 else 1.0
            scored[w] = c * idf * (1 + 0.1 * min(len(w), 10))
        for bg, c in bigrams.items():
            if c >= 2:
                scored[bg] = c * 2.2
        ranked = [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])]
        picked: list[str] = []
        for w in ranked:
            if any(w in p.split() or p in w.split() for p in picked):
                continue
            picked.append(w)
            if len(picked) >= top_n:
                break
        out[s] = picked
    return out


_DECISION = re.compile(r"\b(?:we(?:'ve)? decided|let'?s go with|the plan is|we(?:'ll| will) go with|agreed|decision|approved|we(?:'re| are) going with|final(?:ly)? decided)\b", re.I)
_QUESTION_WORDS = re.compile(r"^(?:who|what|when|where|why|how|which|is|are|do|does|did|can|could|would|will|should|have|has)\b", re.I)


def speaker_profiles(utts: list[dict], names: dict[int, str], tasks: list[dict], duration: float | None) -> list[dict]:
    by: dict[int, list[dict]] = defaultdict(list)
    for u in utts:
        by[u.get("speaker_id") or 1].append(u)
    texts = {s: " ".join(x["text"] for x in us) for s, us in by.items()}
    topics = _topics(texts)
    total_talk = sum(max(0.0, x["end"] - x["start"]) for us in by.values() for x in us) or 1.0
    profiles = []
    for sid in sorted(by):
        us = by[sid]
        talk = sum(max(0.0, x["end"] - x["start"]) for x in us)
        sentences = [(s, x) for x in us for s in _sentences(x["text"])]
        questions = [s for s, _ in sentences if s.rstrip().endswith("?") or _QUESTION_WORDS.match(s)]
        decisions = [{"t": round(float(x["start"]), 2), "text": s} for s, x in sentences if _DECISION.search(s)]
        mine = [t for t in tasks if t["owner_id"] == sid]
        # key points: decisions first, then the longest information-bearing sentences
        info = sorted(((len(_content_words(s)), s, x) for s, x in sentences if len(_content_words(s)) >= 4), key=lambda r: -r[0])
        key_points = [{"t": d["t"], "text": d["text"], "kind": "decision"} for d in decisions[:2]]
        for _, s, x in info:
            if len(key_points) >= 3:
                break
            if all(s != k["text"] for k in key_points):
                key_points.append({"t": round(float(x["start"]), 2), "text": s[:200], "kind": "statement"})
        words = len(re.findall(r"\w+", texts[sid]))
        summary_bits = []
        if topics[sid]:
            summary_bits.append("Talked about " + ", ".join(topics[sid][:4]))
        if mine:
            summary_bits.append(f"took {len(mine)} task{'s' if len(mine) != 1 else ''}")
        if questions:
            summary_bits.append(f"asked {len(questions)} question{'s' if len(questions) != 1 else ''}")
        profiles.append({
            "speaker_id": sid, "speaker": f"Speaker {sid}", "label": label_of(sid, names), "name": names.get(sid),
            "talk_seconds": round(talk, 1), "share": round(talk / total_talk, 3), "turns": len(us), "words": words,
            "words_per_minute": round(words / (talk / 60), 0) if talk > 1 else 0,
            "topics": topics[sid], "key_points": key_points, "questions": questions[:5], "tasks": [t["id"] for t in mine],
            "summary": ("; ".join(summary_bits) + ".") if summary_bits else "No clear content detected.",
            "first_spoke": round(float(us[0]["start"]), 2),
        })
    return profiles


# --------------------------------------------------------------------------------------- LLM (optional)
def _llm_refine(utts: list[dict], base: dict) -> tuple[dict, list[str]]:
    """Merge an LLM's view in when GROQ_API_KEY is set. Never replaces rule-based output; only adds valid items."""
    import json
    import os
    warnings: list[str] = []
    if not os.environ.get("GROQ_API_KEY") or not utts:
        return base, warnings
    try:
        from soniccaption.services.agent_service import _call_groq_json
        transcript = "\n".join(f"[{u['start']:.1f}s] Speaker {u.get('speaker_id', 1)}: {u['text']}" for u in utts)[:12000]
        valid = {f"Speaker {s['speaker_id']}" for s in base["speakers"]}
        res = _call_groq_json(
            "You analyse a speaker-labelled transcript. Reply with JSON: "
            "{\"scenario\": string (meeting, phone call, lecture, interview, customer support, casual conversation, presentation, voice note), "
            "\"tasks\": [{\"owner\": \"Speaker N\" exactly as in the transcript, or \"Unassigned\", \"task\": short imperative, \"due\": string or null, \"priority\": low|medium|high}], "
            "\"speaker_summaries\": {\"Speaker N\": one sentence about what that person talked about}}. "
            "Only include tasks that someone actually committed to or was asked to do. Never invent speakers.",
            transcript)
        if not isinstance(res, dict):
            return base, ["LLM refinement returned no usable JSON; showing rule-based results only."]
        have = {_dedupe_key(t["task"]) for t in base["tasks"]}
        n_added = 0
        for t in res.get("tasks") or []:
            if not isinstance(t, dict) or not t.get("task"):
                continue
            owner = str(t.get("owner", "Unassigned"))
            if owner not in valid:
                owner = "Unassigned"
            key = _dedupe_key(str(t["task"]))
            if key in have:
                continue
            oid = int(owner.split()[-1]) if owner != "Unassigned" else None
            base["tasks"].append({"task": str(t["task"])[:160], "owner": owner, "owner_id": oid, "owner_name": None,
                                  "due": t.get("due"), "priority": t.get("priority") if t.get("priority") in ("low", "medium", "high") else "medium",
                                  "kind": "llm", "confidence": 0.7, "source_text": "", "start": 0.0, "end": 0.0,
                                  "said_by": owner, "said_by_id": oid, "requested_by": None, "id": f"task{len(base['tasks']) + 1}"})
            have.add(key)
            n_added += 1
        sums = res.get("speaker_summaries") or {}
        for sp in base["speakers"]:
            s = sums.get(f"Speaker {sp['speaker_id']}")
            if isinstance(s, str) and s.strip():
                sp["llm_summary"] = s.strip()[:300]
        if isinstance(res.get("scenario"), str):
            base["scenario"]["llm_label"] = res["scenario"][:60]
        base["provider"] = f"rules+groq(+{n_added} tasks)"
    except Exception as exc:  # noqa: BLE001 - optional enrichment must never break the analysis
        warnings.append(f"LLM refinement skipped: {exc}")
    return base, warnings


# --------------------------------------------------------------------------------------- public
def analyse_conversation(utterances: list[dict], sound_events: list[dict] | None = None, duration: float | None = None,
                         use_llm: bool = True) -> dict:
    """Scenario + per-speaker topics + speaker-assigned tasks from a speaker-labelled transcript.

    utterances are the ASR segments after diarization; placeholder text such as "[Speech detected ...]" is ignored.
    """
    utts = [dict(u) for u in utterances if str(u.get("text", "")).strip() and not str(u["text"]).startswith("[Speech detected")]
    utts.sort(key=lambda u: u["start"])
    for u in utts:
        u["text"] = re.sub(r"\s+", " ", u["text"]).strip()
        u.setdefault("speaker_id", 1)
    speaker_ids = sorted({u["speaker_id"] for u in utts})
    if not utts:
        return {"scenario": {"label": "Unknown", "confidence": 0.0, "evidence": [], "alternatives": [], "speakers": 0},
                "speakers": [], "tasks": [], "decisions": [], "questions": [], "names": {}, "provider": "rules",
                "warnings": ["No transcript text available, so scenario, topics and tasks could not be derived."]}
    names, name_to_spk = detect_names(utts)
    tasks = extract_tasks(utts, names, name_to_spk, len(speaker_ids))
    # extract_tasks may learn names from exchanges ("Priya, can you ..." answered by Speaker 2)
    scenario = classify_scenario(utts, len(speaker_ids), sound_events)
    profiles = speaker_profiles(utts, names, tasks, duration)
    for t in tasks:                                   # refresh labels with names learnt during extraction
        if t.get("owner_id"):
            t["owner"] = label_of(t["owner_id"], names)
            t["owner_name"] = names.get(t["owner_id"])
    decisions = [{"t": d["t"], "speaker": p["label"], "text": d["text"]} for p in profiles for d in p["key_points"] if d["kind"] == "decision"]
    questions = [{"speaker": p["label"], "text": q} for p in profiles for q in p["questions"]]
    result = {"scenario": scenario, "speakers": profiles, "tasks": tasks, "decisions": decisions, "questions": questions[:12],
              "names": {str(k): v for k, v in names.items()}, "provider": "rules", "warnings": []}
    if use_llm:
        result, w = _llm_refine(utts, result)
        result["warnings"].extend(w)
    return result
