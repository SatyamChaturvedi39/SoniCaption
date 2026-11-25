"""Ground-truth scene benchmark: composes long recordings from known pieces and scores the scene classifier per 0.5 s hop.

    python backend/scripts/scene_bench.py --esc50 /path/to/ESC-50-master --speech-dir /path/to/wavs [--scenes 20]
        [--holdout-head head_f14.npz]   # head fitted on ESC-50 folds 1-4 (see train_scene_head.py --save-holdout)

Sound effects and noise are taken from ESC-50 fold 5 only (never used to fit the head when --holdout-head is given).
Speech comes from any folder of real recordings. Two conditions are scored:
    clean    pieces separated by near-silence
    bed      a steady background noise runs underneath everything (a fan in the room), speech/effects sit on top of it
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SR = 16000
EVENTS = ["clapping", "coughing", "dog", "door_wood_knock", "footsteps", "glass_breaking", "laughing", "rain", "siren", "sneezing"]
BG = ["wind", "vacuum_cleaner", "engine", "airplane", "washing_machine", "sea_waves", "helicopter", "train", "insects", "chainsaw"]
CLASSES = ["speech", "sound_effect", "background_noise", "silence"]


def _load(path: Path) -> np.ndarray:
    import librosa
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    return librosa.resample(y, orig_sr=sr, target_sr=SR).astype(np.float32) if sr != SR else y


def _active(y: np.ndarray, max_s: float) -> np.ndarray:
    """Cut the audible part of an ESC-50 clip (they are padded with digital silence)."""
    n = (y.size // 160) * 160
    fr = np.sqrt(np.mean(y[:n].reshape(-1, 160) ** 2, axis=1))
    idx = np.where(fr > 0.08 * fr.max())[0]
    if idx.size == 0:
        return y[: int(max_s * SR)]
    a, b = idx[0] * 160, (idx[-1] + 1) * 160
    return y[a:min(b, a + int(max_s * SR))]


def _scale(y: np.ndarray, rms: float) -> np.ndarray:
    return y * (rms / (float(np.sqrt(np.mean(y ** 2))) + 1e-9))


class Pieces:
    def __init__(self, esc: Path, speech_dir: Path, fold: int = 5):
        rows = [r for r in csv.DictReader(open(esc / "meta" / "esc50.csv")) if int(r["fold"]) == fold]
        self.esc = esc
        self.ev = {c: [r["filename"] for r in rows if r["category"] == c] for c in EVENTS}
        self.bg = {c: [r["filename"] for r in rows if r["category"] == c] for c in BG}
        self.speech = [_load(p) for p in sorted(speech_dir.glob("*.wav")) + sorted(speech_dir.glob("*.flac"))]
        self.speech = [s for s in self.speech if s.size > SR]

    def audio(self, fn: str) -> np.ndarray:
        return _load(self.esc / "audio" / fn)


def compose(rng: np.random.Generator, P: Pieces, bed: bool, n_items: int = 12):
    parts, truth, t = [], [], 0.0
    bed_cat = rng.choice(BG)
    for _ in range(n_items):
        kind = rng.choice(CLASSES, p=[0.38, 0.27, 0.2, 0.15])
        if kind == "speech":
            u = P.speech[rng.integers(len(P.speech))]
            y = _scale(u[: int(rng.uniform(2.0, 5.0) * SR)], 0.05)
            lab = "speech"
        elif kind == "sound_effect":
            c = rng.choice(EVENTS)
            y = _scale(_active(P.audio(rng.choice(P.ev[c])), 3.0), rng.uniform(0.04, 0.09))
            lab = c
        elif kind == "background_noise":
            c = rng.choice(BG)
            y = _scale(_active(P.audio(rng.choice(P.bg[c])), 5.0), rng.uniform(0.02, 0.05))
            lab = c
        else:
            y = rng.standard_normal(int(rng.uniform(1.2, 2.5) * SR)).astype(np.float32) * 0.0008
            lab = "silence"
        if y.size < int(0.3 * SR):
            continue
        parts.append(y)
        truth.append({"start": t, "end": t + y.size / SR, "cls": kind, "label": lab})
        t += y.size / SR
    x = np.concatenate(parts)
    if bed:
        b = P.audio(rng.choice(P.bg[bed_cat]))
        b = _scale(b[: int(5 * SR)] if b.size >= 5 * SR else b, 0.012)
        reps = int(np.ceil(x.size / b.size))
        x = x + np.tile(b, reps)[: x.size]
        for s in truth:                                   # silence under a fan is background noise, not silence
            if s["cls"] == "silence":
                s["cls"], s["label"] = "background_noise", bed_cat
    return np.clip(x, -1, 1).astype(np.float32), truth


def score(pred_segments: list[dict], truth: list[dict], duration: float, hop: float = 0.5):
    def pred_at(t: float) -> tuple[str, str]:
        best = None
        for s in pred_segments:
            if s["start"] <= t < s["end"]:
                if s["state"] == "speech":
                    return "speech", "speech"
                best = best or s
        return (best["state"], best.get("key", "")) if best else ("silence", "silence")

    conf = np.zeros((4, 4), int)
    ev_total = ev_ok = 0
    for k in range(int(duration / hop)):
        a, b = k * hop, (k + 1) * hop
        c = (a + b) / 2
        gt = [s for s in truth if s["start"] <= c < s["end"]]
        if not gt:
            continue
        g = gt[0]
        # skip hops that straddle a boundary (ambiguous by half a hop either way)
        if any(abs(c - e) < 0.35 for e in (g["start"], g["end"])):
            continue
        ps, pk = pred_at(c)
        conf[CLASSES.index(g["cls"]), CLASSES.index(ps)] += 1
        if g["cls"] == "sound_effect":
            ev_total += 1
            ev_ok += int(ps == "sound_effect" and pk == g["label"])
    return conf, ev_ok, ev_total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", type=Path, required=True)
    ap.add_argument("--speech-dir", type=Path, required=True)
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--holdout-head", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from soniccaption.audio.vad import detect_speech
    from soniccaption.services import scene_classifier as sc
    from soniccaption.services.sound_cnn import load_sound_model

    head = None
    if args.holdout_head:
        d = np.load(args.holdout_head)
        head = sc.SceneHead(d["mean"], d["scale"], d["coef"], d["intercept"], sc.EVENT_LABELS + ["background_noise"])
    model = load_sound_model()
    P = Pieces(args.esc50, args.speech_dir)
    print(f"{len(P.speech)} speech recordings, fold-5 event/noise clips")
    for cond in ("clean", "bed"):
        rng = np.random.default_rng(args.seed)
        tot = np.zeros((4, 4), int)
        eo = et = 0
        for _ in range(args.scenes):
            x, truth = compose(rng, P, bed=cond == "bed")
            vad = detect_speech(x, SR)
            res = sc.analyze_scene(x, vad, model=model, head=head)
            c, o, n = score(res["segments"], truth, x.size / SR)
            tot += c
            eo += o
            et += n
        print(f"\n=== condition: {cond}  ({args.scenes} scenes) ===")
        print("rows = truth, cols = predicted        " + "  ".join(f"{c[:9]:>9}" for c in CLASSES))
        for i, c in enumerate(CLASSES):
            n = tot[i].sum()
            print(f"{c:>18} (n={n:4d}) recall {tot[i, i] / max(n, 1):5.1%} | " + "  ".join(f"{v:9d}" for v in tot[i]))
        print(f"overall hop accuracy {np.trace(tot) / tot.sum():.1%}")
        print(f"sound-effect hops with the right *label*: {eo / max(et, 1):.1%} ({eo}/{et})")


if __name__ == "__main__":
    main()
