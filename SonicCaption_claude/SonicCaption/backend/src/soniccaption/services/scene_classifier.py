"""Acoustic scene classifier: decides, for every moment of a recording, whether it is
SPEECH, a SOUND EFFECT (clap, rain, cough, door knock, ...), BACKGROUND NOISE (fan, hum, wind, traffic ...) or SILENCE.

This is the integration layer that makes the three members' work cooperate:

    Member 2  Silero VAD ............ decides *speech* (the sound model never heard speech, so it is not asked)
    Member 1  ResNet-18 (ESC-50) .... frozen feature extractor for non-speech audio
    new       scene head ............ open-set head on the ResNet embedding with an explicit `background_noise` class
    Member 2  discrete HMM / Viterbi  smooths the per-hop decisions (preset `scene_state`)

Why the head: the stock ResNet is closed-set and labelled steady noise as "rain" / speech as "coughing"
(see scripts/train_scene_head.py for the measurements). The head fixes that without retraining the backbone.

Public API
    analyze_scene(samples, vad_segments)  -> dict   offline / whole-file analysis (also used when a live session ends)
    LiveSceneClassifier                   -> class  incremental analysis for the live tab
    classify_window(window)               -> dict   one non-speech window
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 16000
WINDOW_S = 2.0
HOP_S = 0.5
SILENCE_RMS = 0.004              # ~ -48 dBFS: below this a non-speech window is "silence" (room tone)
EVENT_MIN_PROB = 0.50            # below this the top event class is not trusted -> "Unrecognised sound"
EVENT_LABELS = ["clapping", "coughing", "dog", "door_wood_knock", "footsteps",
                "glass_breaking", "laughing", "rain", "siren", "sneezing"]
BACKGROUND_TRAIN_CLASSES = ["wind", "vacuum_cleaner", "engine", "airplane", "washing_machine",
                            "sea_waves", "helicopter", "train", "insects", "chainsaw"]
BG_INDEX = len(EVENT_LABELS)

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Human readable names shown in the UI (kept in step with timeline.fusion.SOUND_TAGS)
EVENT_DISPLAY = {
    "clapping": "Clapping / Applause", "coughing": "Coughing", "dog": "Dog barking",
    "door_wood_knock": "Door knock", "footsteps": "Footsteps", "glass_breaking": "Glass breaking",
    "laughing": "Laughter", "rain": "Rain", "siren": "Siren", "sneezing": "Sneezing",
}

STATE_SPEECH, STATE_EFFECT, STATE_NOISE, STATE_SILENCE = "speech", "sound_effect", "background_noise", "silence"
_HMM_SYMBOL = {STATE_SILENCE: "silence", STATE_SPEECH: "speech", STATE_EFFECT: "effect", STATE_NOISE: "noise"}
_HMM_STATE_TO_APP = {"Silence": STATE_SILENCE, "Speech": STATE_SPEECH,
                     "SoundEffect": STATE_EFFECT, "BackgroundNoise": STATE_NOISE}


# ----------------------------------------------------------------------------- features
def acoustic_features(seg: np.ndarray) -> np.ndarray:
    """4 stationarity statistics: log10 rms, log10 p95 frame rms, dynamic range (dB), log10 spectral flatness."""
    from soniccaption.services.sound_cnn import _window_acoustics
    rms, p95, dyn, sfm = _window_acoustics(np.asarray(seg, dtype=np.float32))
    return np.array([np.log10(rms + 1e-5), np.log10(p95 + 1e-5), dyn, np.log10(sfm + 1e-6)], dtype=np.float32)


def _embed_tensor(model, feats):
    net = model.net
    h = net.maxpool(net.relu(net.bn1(net.conv1(feats))))
    h = net.layer4(net.layer3(net.layer2(net.layer1(h))))
    import torch
    return torch.flatten(net.avgpool(h), 1)


def window_embedding(model, seg: np.ndarray) -> np.ndarray:
    """512-d penultimate ResNet embedding of one window."""
    import torch
    seg = np.ascontiguousarray(seg, dtype=np.float32)
    if seg.size < model.min_samples:
        seg = np.pad(seg, (0, model.min_samples - seg.size))
    with torch.no_grad():
        return _embed_tensor(model, model.features(seg))[0].numpy()


def window_embeddings(model, segs: list[np.ndarray]) -> np.ndarray:
    """Batched embeddings. Matches `window_embedding` exactly (verified in tests) but ~5x faster."""
    import torch
    if not segs:
        return np.zeros((0, 512), np.float32)
    if model._mel is None:
        model.features(np.zeros(model.min_samples, np.float32))      # builds the cached mel transforms
    mel, to_db = model._mel
    L = max(max(len(s) for s in segs), model.min_samples)
    batch = torch.from_numpy(np.stack([np.pad(s, (0, L - len(s))) for s in segs]).astype(np.float32))
    with torch.no_grad():
        feats = to_db(mel(batch)[:, None])                            # (B, 1, n_mels, frames) -> per-sample top_db
        return _embed_tensor(model, feats).numpy()


# ----------------------------------------------------------------------------- the head
@dataclass
class SceneHead:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    labels: list[str]
    meta: dict = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(X) - self.mean) / np.maximum(self.scale, 1e-6)
        logits = z @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


@lru_cache(maxsize=1)
def load_head() -> SceneHead:
    npz, meta = WEIGHTS_DIR / "scene_head.npz", WEIGHTS_DIR / "scene_head.json"
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} is missing. Run backend/scripts/train_scene_head.py (needs the ESC-50 dataset) or restore the file.")
    d = np.load(npz)
    info = json.loads(meta.read_text()) if meta.exists() else {}
    return SceneHead(d["mean"], d["scale"], d["coef"], d["intercept"], info.get("labels", EVENT_LABELS + ["background_noise"]), info)


# ----------------------------------------------------------------------------- one window
def describe_background(seg: np.ndarray, sr: int = SR) -> str:
    """Plain-language descriptor of a steady noise from its spectral balance (a description, not a classifier)."""
    seg = np.asarray(seg, dtype=np.float32)
    if seg.size < 512:
        return "Background noise"
    spec = np.abs(np.fft.rfft(seg[: (seg.size // 512) * 512].reshape(-1, 512) * np.hanning(512), axis=1)) ** 2
    p = spec.mean(axis=0) + 1e-12
    f = np.fft.rfftfreq(512, 1 / sr)
    tot = p.sum()
    low, high = p[f < 300].sum() / tot, p[f > 2500].sum() / tot
    centroid = float((f * p).sum() / tot)
    if low > 0.60:
        return "Low-frequency hum / rumble"
    if high > 0.40 or centroid > 2800:
        return "Hiss / broadband noise (fan, air, static)"
    return "Steady ambient noise (room, traffic, crowd)"


# Sounds that are, by nature, short bursts. A window whose loud and quiet frames differ by less than this many dB has no burst in it,
# so an "impulsive" label there is the closed-set model guessing on steady noise. Measured on ESC-50 windows: the 5th percentile of
# true dog / knock / cough / sneeze / glass windows is 20-27 dB; the guard reverts <2% of correctly labelled windows.
MIN_BURST_DB = {"dog": 10.0, "door_wood_knock": 10.0, "coughing": 10.0, "sneezing": 10.0, "glass_breaking": 10.0, "footsteps": 5.0}


def classify_window(window: np.ndarray, model=None, head: SceneHead | None = None) -> dict:
    """Classify one NON-speech window -> {state, label, key, confidence, top}."""
    return classify_windows([window], model, head)[0]


def classify_windows(windows: list[np.ndarray], model=None, head: SceneHead | None = None) -> list[dict]:
    from soniccaption.services.sound_cnn import load_sound_model
    model = model or load_sound_model()
    head = head or load_head()
    if not windows:
        return []
    emb = window_embeddings(model, windows)
    ac = np.stack([acoustic_features(w) for w in windows])
    P = head.predict_proba(np.c_[emb, ac])
    out = []
    for w, p, a in zip(windows, P, ac):
        order = np.argsort(-p)
        top = [{"label": head.labels[i], "score": round(float(p[i]), 4)} for i in order[:3]]
        i0 = int(order[0])
        if head.labels[i0] in MIN_BURST_DB and float(a[2]) < MIN_BURST_DB[head.labels[i0]]:
            out.append({"state": STATE_NOISE, "label": describe_background(w), "key": "background_noise",
                        "confidence": round(float(max(p[BG_INDEX], 0.5)), 3), "top": top, "guard": "no_burst"})
            continue
        if i0 == BG_INDEX:
            out.append({"state": STATE_NOISE, "label": describe_background(w), "key": "background_noise",
                        "confidence": round(float(p[BG_INDEX]), 3), "top": top})
        elif p[i0] >= EVENT_MIN_PROB:
            key = head.labels[i0]
            out.append({"state": STATE_EFFECT, "label": EVENT_DISPLAY.get(key, key), "key": key,
                        "confidence": round(float(p[i0]), 3), "top": top})
        else:
            # the top class is weak: it is a non-speech, non-steady sound the model has no class for
            out.append({"state": STATE_EFFECT, "label": "Unrecognised sound", "key": "unknown",
                        "confidence": round(float(p[i0]), 3), "top": top})
    return out


# ----------------------------------------------------------------------------- whole-file analysis
def _frame_rms(x: np.ndarray, frame: int = 160) -> np.ndarray:
    n = (x.size // frame) * frame
    if n == 0:
        return np.zeros(0, np.float32)
    return np.sqrt(np.mean(x[:n].reshape(-1, frame) ** 2, axis=1) + 1e-12).astype(np.float32)


def _speech_mask(vad_segments: list[dict], n_samples: int) -> np.ndarray:
    m = np.zeros(n_samples, dtype=bool)
    for s in vad_segments:
        m[max(0, int(s["start"] * SR)):min(n_samples, int(s["end"] * SR))] = True
    return m


def _hop_window(x: np.ndarray, mask: np.ndarray, a: int, b: int, win_s: float):
    """Speech fraction of hop [a, b) and the 2 s window centred on it with speech spliced out (None if mostly speech)."""
    W = int(win_s * SR)
    sp_frac = float(mask[a:b].mean()) if b > a else 0.0
    if sp_frac >= 0.5:
        return sp_frac, None
    c = (a + b) // 2
    lo, hi = max(0, c - W // 2), min(x.size, c + W // 2)
    keep = ~mask[lo:hi]
    if keep.sum() < int(0.8 * SR):                           # almost the whole window is speech
        return max(sp_frac, 0.5), None
    return sp_frac, x[lo:hi][keep]


def _hop_windows(x: np.ndarray, mask: np.ndarray, n_hops: int, hop_s: float, win_s: float):
    H = int(hop_s * SR)
    for k in range(n_hops):
        sp, w = _hop_window(x, mask, k * H, min((k + 1) * H, x.size), win_s)
        yield k, sp, w


def _decide(x: np.ndarray, mask: np.ndarray, hop_s: float, win_s: float, model, head) -> list[dict]:
    n_hops = int(np.ceil(x.size / (hop_s * SR)))
    hops: list[dict] = [None] * n_hops                         # type: ignore[list-item]
    todo_idx, todo_win = [], []
    for k, sp_frac, win in _hop_windows(x, mask, n_hops, hop_s, win_s):
        t0, t1 = k * hop_s, min((k + 1) * hop_s, x.size / SR)
        if win is None:
            hops[k] = {"t0": t0, "t1": t1, "state": STATE_SPEECH, "label": "Speech", "key": "speech",
                       "confidence": round(max(sp_frac, 0.5), 3), "top": []}
        elif float(np.sqrt(np.mean(win ** 2))) < SILENCE_RMS:
            hops[k] = {"t0": t0, "t1": t1, "state": STATE_SILENCE, "label": "Silence", "key": "silence",
                       "confidence": 0.9, "top": []}
        else:
            todo_idx.append(k)
            todo_win.append(win.astype(np.float32))
    for lo in range(0, len(todo_win), 32):                    # batches of 32 windows
        res = classify_windows(todo_win[lo:lo + 32], model, head)
        for k, r in zip(todo_idx[lo:lo + 32], res):
            hops[k] = {"t0": k * hop_s, "t1": min((k + 1) * hop_s, x.size / SR), **r}
    return hops


def smooth_with_hmm(hops: list[dict]) -> dict:
    """Viterbi-smooth the per-hop states with Member 2's DiscreteHMM (preset `scene_state`)."""
    from soniccaption.algorithms.hmm import decode_for_ui, load_preset, viterbi
    model, _ = load_preset("scene_state")
    obs = [_HMM_SYMBOL[h["state"]] for h in hops]
    if not obs:
        return {"path": [], "observations": [], "model": model.to_dict()}
    v = viterbi(model, obs)
    path = [_HMM_STATE_TO_APP[s] for s in v["best_path"]]
    ui = decode_for_ui(model, obs[:40]) if len(obs) > 0 else {}     # trellis for the first 40 hops (UI size)
    return {"path": path, "observations": obs, "best_path_log_prob": v["best_path_log_prob"],
            "log_likelihood": float(model.forward(obs)["log_likelihood"]), "model": model.to_dict(),
            "trellis_preview": {k: ui.get(k) for k in ("observations", "best_path", "trellis_nodes", "trellis_edges")}}


def refine_span(x: np.ndarray, t0: float, t1: float, pad: float = 0.06) -> tuple[float, float]:
    """Trim a window-quantised event span to where the energy actually rises (a clap is 0.2 s, not 2 s)."""
    ctx_lo, ctx_hi = max(0, int((t0 - 1.0) * SR)), min(x.size, int((t1 + 1.0) * SR))
    fr = _frame_rms(x[ctx_lo:ctx_hi], 160)
    if fr.size < 8:
        return t0, t1
    base = float(np.percentile(fr, 20))
    thr = max(base * 3.0, 0.35 * float(np.percentile(fr, 98)), 0.006)
    f0, f1 = int((t0 * SR - ctx_lo) / 160), int((t1 * SR - ctx_lo) / 160)
    idx = np.where(fr[max(0, f0):max(0, f1)] >= thr)[0]
    if idx.size == 0:
        return t0, t1
    a = (ctx_lo + (max(0, f0) + idx[0]) * 160) / SR - pad
    b = (ctx_lo + (max(0, f0) + idx[-1] + 1) * 160) / SR + pad
    a, b = max(t0, a), min(t1, b)
    if b - a < 0.25:                                           # keep very short bursts visible
        mid = (a + b) / 2
        a, b = max(0.0, mid - 0.125), mid + 0.125
    return round(a, 2), round(b, 2)


def _runs(hops: list[dict], states: list[str]) -> list[dict]:
    """Merge consecutive hops into segments.

    speech hops are skipped (speech comes straight from the VAD). Silence and background-noise runs merge by state
    (the noise descriptor is the majority over the run, so one odd hop cannot relabel it); sound-effect runs merge by label.
    """
    segs: list[dict] = []
    for h, st in zip(hops, states):
        if st == STATE_SPEECH:
            continue
        if st == STATE_SILENCE:
            lab, key = "Silence", "silence"
        elif st == STATE_NOISE:
            lab, key = (h["label"], "background_noise") if h["state"] == STATE_NOISE else (None, "background_noise")
        else:
            lab, key = (h["label"], h.get("key", "")) if h["state"] == STATE_EFFECT else ("Unrecognised sound", "unknown")
        joinable = (segs and segs[-1]["state"] == st and h["t0"] - segs[-1]["end"] < 1e-6
                    and (st != STATE_EFFECT or segs[-1]["label"] == lab))
        if joinable:
            s = segs[-1]
            s["end"] = h["t1"]
            s["_conf"].append(h["confidence"])
            if lab:
                s["_labels"][lab] = s["_labels"].get(lab, 0) + 1
        else:
            segs.append({"state": st, "label": lab, "key": key, "start": h["t0"], "end": h["t1"],
                         "_conf": [h["confidence"]], "_labels": {lab: 1} if lab else {}})
    for s in segs:
        if s["state"] == STATE_NOISE:
            s["label"] = max(s["_labels"], key=s["_labels"].get) if s["_labels"] else "Background noise"
        s["confidence"] = round(float(np.max(s["_conf"])) if s["state"] == STATE_EFFECT else float(np.mean(s["_conf"])), 3)
        del s["_conf"], s["_labels"]
    return segs


def _relabel_effect_runs(hops: list[dict], states: list[str]) -> list[dict]:
    """Within an effect run take the confidence-weighted majority label (stops 'Clapping, Laughter, Clapping' flicker)."""
    out = [dict(h) for h in hops]
    i = 0
    while i < len(out):
        if states[i] == STATE_EFFECT:
            j = i
            while j < len(out) and states[j] == STATE_EFFECT:
                j += 1
            score: dict[str, float] = {}
            meta: dict[str, tuple[str, str]] = {}
            for h in out[i:j]:
                if h["state"] == STATE_EFFECT:
                    score[h["key"]] = score.get(h["key"], 0.0) + h["confidence"]
                    meta[h["key"]] = (h["label"], h["key"])
            if score:
                best = max(score, key=score.get)
                for h in out[i:j]:
                    h["label"], h["key"], h["state"] = meta[best][0], best, STATE_EFFECT
            i = j
        else:
            i += 1
    return out


def analyze_scene(samples: np.ndarray, vad_segments: list[dict], hop_s: float = HOP_S, win_s: float = WINDOW_S,
                  model=None, head: SceneHead | None = None) -> dict:
    """Whole-file analysis. `samples` must be mono 16 kHz float32.

    Returns
        segments    chronological speech / sound_effect / background_noise / silence segments with start, end, label, confidence
        hops        the raw per-hop decisions (what the classifier thought before smoothing)
        hmm         Viterbi smoothing details for the HMM tab
        stats       seconds and share of each state
    """
    from soniccaption.services.sound_cnn import load_sound_model
    model = model or load_sound_model()
    head = head or load_head()
    x = np.asarray(samples, dtype=np.float32)
    duration = x.size / SR
    if x.size == 0:
        return {"segments": [], "hops": [], "hmm": {}, "stats": {}, "duration": 0.0}
    mask = _speech_mask(vad_segments, x.size)
    hops = _decide(x, mask, hop_s, win_s, model, head)
    hmm = smooth_with_hmm(hops)
    states = hmm["path"]
    # never let smoothing erase real speech or fabricate it: speech hops stay as the VAD said
    states = [STATE_SPEECH if h["state"] == STATE_SPEECH else (s if s != STATE_SPEECH else h["state"])
              for h, s in zip(hops, states)]
    hops_rl = _relabel_effect_runs(hops, states)

    segments: list[dict] = []
    for s in vad_segments:
        segments.append({"state": STATE_SPEECH, "label": "Speech", "key": "speech", "start": round(float(s["start"]), 2),
                         "end": round(float(s["end"]), 2), "confidence": 0.95})
    for r in _runs(hops_rl, states):
        if r["state"] == STATE_EFFECT:
            r["start"], r["end"] = refine_span(x, r["start"], r["end"])
        else:
            r["start"], r["end"] = round(r["start"], 2), round(min(r["end"], duration), 2)
        # cut anything that overlaps speech out of non-speech segments
        pieces = [(r["start"], r["end"])]
        for s in vad_segments:
            nxt = []
            for a, b in pieces:
                if s["end"] <= a or s["start"] >= b:
                    nxt.append((a, b))
                else:
                    if s["start"] > a:
                        nxt.append((a, s["start"]))
                    if s["end"] < b:
                        nxt.append((s["end"], b))
            pieces = nxt
        for a, b in pieces:
            if b - a >= (0.2 if r["state"] == STATE_EFFECT else 0.4):
                segments.append({**r, "start": round(a, 2), "end": round(b, 2)})
    segments.sort(key=lambda s: (s["start"], s["end"]))
    for i, s in enumerate(segments, 1):
        s["id"] = f"s{i}"
    stats: dict[str, float] = {k: 0.0 for k in (STATE_SPEECH, STATE_EFFECT, STATE_NOISE, STATE_SILENCE)}
    for h, st in zip(hops, states):
        stats[st] += h["t1"] - h["t0"]
    stats = {k: {"seconds": round(v, 2), "share": round(v / max(duration, 1e-9), 3)} for k, v in stats.items()}
    return {"segments": segments, "hops": hops, "hop_states": states, "hmm": hmm, "stats": stats, "duration": round(duration, 3)}


# ----------------------------------------------------------------------------- live
class LiveSceneClassifier:
    """Incremental scene classification for the live tab.

    Feed 16 kHz float chunks with `feed()`. A 0.5 s hop is decided once 1 s of look-ahead exists (so its 2 s window can be
    centred on it) -> about 1.5 s latency from sound to label. Decisions are *provisional*: when recording stops the whole
    take is re-analysed with `analyze_scene` for the authoritative result, but speech / effect / noise labels are already
    reliable enough to read live. Work per hop is constant (only the neighbourhood of the hop is examined).
    """

    LOOKAHEAD_S = 1.0
    CONTEXT_S = 3.0

    def __init__(self, model=None, head: SceneHead | None = None):
        from soniccaption.audio.vad import StreamingVAD
        from soniccaption.services.sound_cnn import load_sound_model
        self.model = model or load_sound_model()
        self.head = head or load_head()
        self.vad = StreamingVAD()
        self.x = np.zeros(0, np.float32)
        self.next_hop = 0
        self.hops: list[dict] = []
        self._pending: dict | None = None
        self._cur_state: str | None = None
        self._flushing = False

    @property
    def seconds(self) -> float:
        return self.x.size / SR

    def feed(self, chunk: np.ndarray) -> list[dict]:
        chunk = np.asarray(chunk, dtype=np.float32)
        self.x = np.concatenate([self.x, chunk])
        self.vad.feed(chunk)
        out: list[dict] = []
        while self.seconds >= (self.next_hop + 1) * HOP_S + self.LOOKAHEAD_S:
            out.append(self._decide_hop(self.next_hop))
            self.next_hop += 1
        return out

    def finish(self) -> list[dict]:
        """Decide the remaining hops (end of the recording: no more look-ahead available)."""
        out: list[dict] = []
        while self.next_hop * HOP_S < self.seconds - 0.05:
            out.append(self._decide_hop(self.next_hop))
            self.next_hop += 1
        return out

    def _local_mask(self, lo: int, hi: int) -> np.ndarray:
        """Speech mask for samples [lo, hi) from the streaming VAD frame probabilities (gaps < 100 ms bridged)."""
        f0, f1 = lo // 512, min(len(self.vad.probs), hi // 512 + 1)
        flags = np.array(self.vad.probs[f0:f1]) >= 0.5
        if flags.size >= 3:                                   # close 1-2 frame holes (speech has micro-pauses)
            closed = flags.copy()
            for i in range(1, flags.size - 1):
                if not flags[i] and flags[i - 1] and flags[i + 1]:
                    closed[i] = True
            flags = closed
        m = np.repeat(flags, 512)
        out = np.zeros(hi - lo, dtype=bool)
        off = f0 * 512 - lo
        a, b = max(0, off), min(hi - lo, off + m.size)
        if b > a:
            out[a:b] = m[a - off:b - off]
        return out

    def _decide_hop(self, k: int) -> dict:
        a, b = int(k * HOP_S * SR), min(int((k + 1) * HOP_S * SR), self.x.size)
        lo, hi = max(0, a - int(self.CONTEXT_S * SR)), min(self.x.size, b + int(self.CONTEXT_S * SR))
        xs, mask = self.x[lo:hi], self._local_mask(lo, hi)
        sp_frac, win = _hop_window(xs, mask, a - lo, b - lo, WINDOW_S)
        t0, t1 = k * HOP_S, b / SR
        if win is None:
            h = {"t0": t0, "t1": t1, "state": STATE_SPEECH, "label": "Speech", "key": "speech",
                 "confidence": round(max(sp_frac, 0.5), 3), "top": []}
        elif float(np.sqrt(np.mean(win ** 2))) < SILENCE_RMS:
            h = {"t0": t0, "t1": t1, "state": STATE_SILENCE, "label": "Silence", "key": "silence",
                 "confidence": 0.9, "top": []}
        else:
            h = {"t0": t0, "t1": t1, **classify_window(win.astype(np.float32), self.model, self.head)}
        h = self._hysteresis(h)
        self.hops.append(h)
        return h

    def _hysteresis(self, h: dict) -> dict:
        """Speech and strong sound effects switch immediately; noise/silence need 2 hops in a row (anti-flicker)."""
        st = h["state"]
        immediate = st == STATE_SPEECH or (st == STATE_EFFECT and h["confidence"] >= 0.7)
        if self._cur_state is None or immediate or st == self._cur_state:
            self._cur_state, self._pending = st, None
            return h
        if self._pending and self._pending["state"] == st:
            self._cur_state, self._pending = st, None
            return h
        self._pending = h
        prev = self.hops[-1] if self.hops else h
        return {**prev, "t0": h["t0"], "t1": h["t1"], "provisional": True}
