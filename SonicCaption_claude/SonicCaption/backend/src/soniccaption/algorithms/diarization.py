"""Speaker diarization: *who spoke when*, for any number of speakers.

Replaces the 2-speaker K-Means split in `enhancement.diarize_speakers` (kept for reference/tests) with:

    1. Cut every VAD speech segment into short overlapping windows (1.5 s, hop 0.75 s).
    2. Embed each window. Default: pretrained 256-d speaker encoder (services/speaker_encoder.py).
       Fallback when torch weights are unavailable: Member 1's hand-crafted 19-d MFCC + F0 + LPC-PARCOR vector
       (`enhancement.extract_speaker_embedding`).
    3. Cluster with average-linkage agglomerative clustering on cosine distance. The number of speakers is *not*
       fixed: it is whatever the distance threshold produces, bounded by min/max speakers. Tiny clusters (a stray
       cough-like window) are absorbed into the nearest real speaker.
    4. Median-smooth window labels, then merge neighbouring windows into speaker turns that stay inside VAD segments.

`speaker_mode`:  "auto" (estimate), "single"/"1" (one speaker), or an integer string "2".."8" (force that many).

Contract
    diarize(samples, vad_segments, ...) -> {"turns": [{"start","end","speaker","speaker_id"}], "n_speakers": int,
        "backend": str, "centroids": ..., "projection": [{"x","y","speaker","t"}], ...}
    OnlineSpeakerTracker.assign(audio16k) -> ("Speaker 2", distance)         (live tab)
"""
from __future__ import annotations

import numpy as np

SR = 16000
WIN_S, HOP_S, MIN_WIN_S = 1.5, 0.75, 0.6

# cosine-distance thresholds tuned on real multi-speaker conversations (see scripts/diarization_bench.py)
THRESHOLD = {"neural": 0.46, "dsp": 0.40}


# ------------------------------------------------------------------------------ windows & embeddings
def make_windows(vad_segments: list[dict], win_s: float = WIN_S, hop_s: float = HOP_S, min_s: float = MIN_WIN_S):
    """[(start, end, seg_index)] windows that never cross a VAD segment boundary."""
    out = []
    for si, s in enumerate(vad_segments):
        a, b = float(s["start"]), float(s["end"])
        if b - a < min_s * 0.6:
            continue
        if b - a <= win_s + 0.25:
            out.append((a, b, si))
            continue
        t = a
        while t + win_s <= b + 1e-6:
            out.append((t, t + win_s, si))
            t += hop_s
        if out and out[-1][2] == si and b - out[-1][1] > 0.4:      # cover the tail
            out.append((b - win_s, b, si))
    return out


def embed_windows(x: np.ndarray, windows, backend: str = "auto") -> tuple[np.ndarray, str]:
    """Return (embeddings, backend_used). Rows are L2-normalised for the neural backend, z-scored for dsp."""
    chunks = [x[int(a * SR):int(b * SR)] for a, b, _ in windows]
    if backend in ("auto", "neural"):
        try:
            from soniccaption.services import speaker_encoder
            if speaker_encoder.available():
                return speaker_encoder.embed_utterances(chunks), "neural"
            if backend == "neural":
                raise FileNotFoundError("speaker encoder weights not found")
        except Exception:
            if backend == "neural":
                raise
    from soniccaption.algorithms.enhancement import extract_speaker_embedding
    E = np.stack([extract_speaker_embedding(c, sr=SR) for c in chunks]) if chunks else np.zeros((0, 19), np.float32)
    if len(E) > 1:
        E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    return E.astype(np.float32), "dsp"


# ------------------------------------------------------------------------------ clustering
def _cluster(E: np.ndarray, threshold: float, min_k: int, max_k: int, forced_k: int | None) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    n = len(E)
    if n == 1 or max_k <= 1:
        return np.zeros(n, dtype=int)
    Z = linkage(E, method="average", metric="cosine")
    if forced_k:
        labels = fcluster(Z, t=min(forced_k, n), criterion="maxclust") - 1
    else:
        labels = fcluster(Z, t=threshold, criterion="distance") - 1
        k = labels.max() + 1
        if k > max_k:
            labels = fcluster(Z, t=max_k, criterion="maxclust") - 1
        elif k < min_k:
            labels = fcluster(Z, t=min(min_k, n), criterion="maxclust") - 1
    return labels.astype(int)


def _absorb_tiny(E: np.ndarray, labels: np.ndarray, durations: np.ndarray, min_total_s: float = 1.6, min_windows: int = 2) -> np.ndarray:
    """Merge clusters with too little evidence into the closest substantial cluster."""
    labels = labels.copy()
    while True:
        ids = np.unique(labels)
        if len(ids) <= 1:
            return labels
        sup = {i: (np.sum(durations[labels == i]), int(np.sum(labels == i))) for i in ids}
        tiny = [i for i in ids if sup[i][0] < min_total_s and sup[i][1] < min_windows + 1]
        big = [i for i in ids if i not in tiny]
        if not tiny or not big:
            return labels
        t = min(tiny, key=lambda i: sup[i][0])
        ct = E[labels == t].mean(0)
        dists = {b: 1 - float(np.dot(ct, E[labels == b].mean(0)) / (np.linalg.norm(ct) * np.linalg.norm(E[labels == b].mean(0)) + 1e-9)) for b in big}
        labels[labels == t] = min(dists, key=dists.get)


def _median_smooth(labels: np.ndarray, seg_idx: np.ndarray, k: int = 3) -> np.ndarray:
    out = labels.copy()
    for i in range(len(labels)):
        lo, hi = max(0, i - k // 2), min(len(labels), i + k // 2 + 1)
        nb = [labels[j] for j in range(lo, hi) if seg_idx[j] == seg_idx[i] or abs(j - i) == 1]
        if nb:
            vals, cnt = np.unique(nb, return_counts=True)
            if cnt.max() > len(nb) / 2:
                out[i] = vals[np.argmax(cnt)]
    return out


def _projection(E: np.ndarray, labels: np.ndarray, windows) -> list[dict]:
    if len(E) < 2:
        return [{"x": 0.0, "y": 0.0, "speaker": int(labels[0]) + 1 if len(labels) else 1, "t": round(windows[0][0], 2)}] if len(E) else []
    X = E - E.mean(0)
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        P = X @ vt[:2].T
    except np.linalg.LinAlgError:
        P = np.zeros((len(E), 2))
    if P.shape[1] < 2:
        P = np.c_[P, np.zeros(len(P))]
    scale = float(np.abs(P).max()) or 1.0
    return [{"x": round(float(p[0] / scale), 3), "y": round(float(p[1] / scale), 3), "speaker": int(l) + 1,
             "t": round(float(w[0]), 2)} for p, l, w in zip(P, labels, windows)]


def diarize(samples: np.ndarray, vad_segments: list[dict], speaker_mode: str = "auto", max_speakers: int = 8,
            backend: str = "auto", threshold: float | None = None) -> dict:
    x = np.asarray(samples, dtype=np.float32)
    empty = {"turns": [], "n_speakers": 0, "backend": "none", "centroids": [], "projection": [], "window_count": 0}
    if not vad_segments:
        return empty
    mode = str(speaker_mode or "auto").lower()
    single = mode in ("1", "single")
    forced = int(mode) if mode.isdigit() and int(mode) >= 2 else (2 if mode == "dialogue" else None)

    windows = make_windows(vad_segments)
    if not windows:                                           # every segment is shorter than one window: one speaker per VAD segment
        return {**empty, "turns": [{"start": float(s["start"]), "end": float(s["end"]), "speaker": "Speaker 1", "speaker_id": 1}
                                   for s in vad_segments], "n_speakers": 1}
    E, used = embed_windows(x, windows, backend)
    durs = np.array([b - a for a, b, _ in windows])
    seg_idx = np.array([s for _, _, s in windows])

    if single or len(windows) == 1:
        labels = np.zeros(len(windows), dtype=int)
    else:
        thr = THRESHOLD[used] if threshold is None else threshold
        labels = _cluster(E, thr, 1, max(1, int(max_speakers)), forced)
        if not forced:
            labels = _absorb_tiny(E, labels, durs)
        labels = _median_smooth(labels, seg_idx)

    # relabel in order of first appearance -> Speaker 1 is whoever speaks first
    order: dict[int, int] = {}
    for l in labels[np.argsort([w[0] for w in windows], kind="stable")]:
        order.setdefault(int(l), len(order) + 1)
    sid = np.array([order[int(l)] for l in labels])

    # build turns: per VAD segment, assign time spans to the speaker of the window covering them (nearest window centre)
    turns: list[dict] = []
    centres = np.array([(a + b) / 2 for a, b, _ in windows])
    for si, s in enumerate(vad_segments):
        a, b = float(s["start"]), float(s["end"])
        idx = np.where(seg_idx == si)[0]
        if idx.size == 0:                                      # too short to embed: inherit from nearest window in time
            j = int(np.argmin(np.abs(centres - (a + b) / 2)))
            turns.append({"start": round(a, 3), "end": round(b, 3), "speaker_id": int(sid[j])})
            continue
        cur_spk, cur_start = int(sid[idx[0]]), a
        for p, q in zip(idx[:-1], idx[1:]):
            if sid[p] != sid[q]:
                cut = (windows[p][1] + windows[q][0]) / 2      # midway between the two windows' inner edges
                cut = min(max(cut, cur_start + 0.05), b - 0.05)
                turns.append({"start": round(cur_start, 3), "end": round(cut, 3), "speaker_id": cur_spk})
                cur_spk, cur_start = int(sid[q]), cut
        turns.append({"start": round(cur_start, 3), "end": round(b, 3), "speaker_id": cur_spk})
    turns.sort(key=lambda t: t["start"])
    # merge adjacent same-speaker turns separated by < 0.35 s
    merged: list[dict] = []
    for t in turns:
        if merged and merged[-1]["speaker_id"] == t["speaker_id"] and t["start"] - merged[-1]["end"] < 0.35:
            merged[-1]["end"] = t["end"]
        else:
            merged.append(dict(t))
    for t in merged:
        t["speaker"] = f"Speaker {t['speaker_id']}"

    n = int(sid.max())
    cents = []
    for k in range(1, n + 1):
        m = sid == k
        c = E[m].mean(0)
        cents.append({"speaker": f"Speaker {k}", "speaker_id": k, "windows": int(m.sum()),
                      "seconds": round(float(sum(t["end"] - t["start"] for t in merged if t["speaker_id"] == k)), 2),
                      "centroid": [round(float(v), 4) for v in (c / (np.linalg.norm(c) + 1e-9))][:64]})
    return {"turns": merged, "n_speakers": n, "backend": used, "centroids": cents,
            "projection": _projection(E, sid - 1, windows), "window_count": len(windows),
            "threshold": THRESHOLD[used] if threshold is None else threshold,
            "_embeddings": E, "_labels": sid}


# ------------------------------------------------------------------------------ assigning words / segments to speakers
def speaker_at(turns: list[dict], t: float, tol: float = 0.25) -> int | None:
    best, bd = None, 1e9
    for tr in turns:
        if tr["start"] <= t <= tr["end"]:
            return tr["speaker_id"]
        d = min(abs(t - tr["start"]), abs(t - tr["end"]))
        if d < bd:
            best, bd = tr["speaker_id"], d
    return best if bd <= tol else None


def assign_speakers(asr_segments: list[dict], turns: list[dict], min_split_gap: float = 0.0) -> list[dict]:
    """Attach a speaker to every ASR segment; split a segment when its words fall in different speakers' turns.

    Whisper segments can span a change of speaker; word timestamps (when present) let us split at the change.
    """
    out: list[dict] = []
    if not turns:
        return [{**s, "speaker": "Speaker 1", "speaker_id": 1} for s in asr_segments]
    for seg in asr_segments:
        words = seg.get("words") or []
        if not words:
            ov: dict[int, float] = {}
            for tr in turns:
                o = min(seg["end"], tr["end"]) - max(seg["start"], tr["start"])
                if o > 0:
                    ov[tr["speaker_id"]] = ov.get(tr["speaker_id"], 0.0) + o
            spk = max(ov, key=ov.get) if ov else (speaker_at(turns, (seg["start"] + seg["end"]) / 2, 1.0) or 1)
            out.append({**seg, "speaker_id": spk, "speaker": f"Speaker {spk}"})
            continue
        groups: list[list[dict]] = []
        cur_spk = None
        for w in words:
            spk = speaker_at(turns, (w["start"] + w["end"]) / 2, 0.6) or cur_spk or 1
            if spk != cur_spk:
                groups.append([])
                cur_spk = spk
            groups[-1].append({**w, "_spk": spk})
        # absorb 1-word flips (usually an alignment error at a turn edge)
        for gi in range(1, len(groups) - 1):
            if len(groups[gi]) == 1 and groups[gi - 1][0]["_spk"] == groups[gi + 1][0]["_spk"]:
                for w in groups[gi]:
                    w["_spk"] = groups[gi - 1][0]["_spk"]
        flat = [w for g in groups for w in g]
        regroup: list[list[dict]] = []
        for w in flat:
            if not regroup or regroup[-1][0]["_spk"] != w["_spk"]:
                regroup.append([])
            regroup[-1].append(w)
        for g in regroup:
            spk = g[0]["_spk"]
            text = "".join(w.get("word", "") if str(w.get("word", "")).startswith(" ") else " " + str(w.get("word", "")) for w in g).strip()
            out.append({"start": float(g[0]["start"]), "end": float(g[-1]["end"]), "text": text,
                        "words": [{k: v for k, v in w.items() if k != "_spk"} for w in g],
                        "speaker_id": spk, "speaker": f"Speaker {spk}"})
    return out


# ------------------------------------------------------------------------------ live tracking
class OnlineSpeakerTracker:
    """Assigns finished speech segments to speakers as they arrive (live tab), creating new speakers on demand."""

    def __init__(self, backend: str = "auto", max_speakers: int = 8, threshold: float | None = None):
        self.backend_pref = backend
        self.max_speakers = max_speakers
        self.threshold = threshold
        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []
        self.backend = None

    def assign(self, audio: np.ndarray) -> tuple[int, float]:
        """Return (speaker_id starting at 1, distance to its centroid). Very short audio returns (0, 1.0)."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < int(0.5 * SR):
            return 0, 1.0
        E, used = embed_windows(audio, [(0.0, audio.size / SR, 0)], self.backend_pref)
        if used != "neural":                                    # dsp features are only meaningful relative to a whole file
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        self.backend = used
        e = E[0] / (np.linalg.norm(E[0]) + 1e-9)
        thr = self.threshold if self.threshold is not None else THRESHOLD[used]
        if self.centroids:
            d = [1 - float(np.dot(e, c / (np.linalg.norm(c) + 1e-9))) for c in self.centroids]
            j = int(np.argmin(d))
            if d[j] <= thr or len(self.centroids) >= self.max_speakers:
                self.centroids[j] = (self.centroids[j] * self.counts[j] + e) / (self.counts[j] + 1)
                self.counts[j] += 1
                return j + 1, d[j]
        self.centroids.append(e.copy())
        self.counts.append(1)
        return len(self.centroids), 0.0
