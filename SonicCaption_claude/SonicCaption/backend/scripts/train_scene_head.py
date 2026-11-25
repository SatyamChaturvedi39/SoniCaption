"""Train the open-set "scene head" that sits on top of Member 1's ESC-50 ResNet-18.

WHY THIS EXISTS
---------------
The shipped ResNet-18 is a *closed-set* classifier: ESC-50 gives it 10 sound-effect classes and nothing else, so it
has to put every input into one of them. Measured on real ESC-50 audio the original network labelled
  * 100 % of steady background noise (wind, vacuum, engine, airplane, washing machine ...) as a sound effect,
    ~65 % of it as "rain";
  * human speech as coughing / dog / laughing (it never heard speech).
That is why the background-noise model "was not working" inside the app.

WHAT THIS DOES
--------------
Keeps Member 1's ResNet-18 as a frozen feature extractor and trains a small multinomial logistic-regression head
on its 512-d embeddings (+ 4 stationarity statistics) with an extra class, `background_noise`, built from ESC-50
stationary-noise classes. Speech is handled upstream by Silero VAD, not by this head.

Training windows are 2 s (the live window size) so training matches deployment.
Held-out protocol (same as notebook 02): fit on folds 1-4, report on fold 5. The shipped head is then refit on
all folds. Leave-one-noise-type-out numbers show how it copes with noise types it never saw.

USAGE
-----
    # 1. download ESC-50 (CC BY-NC 3.0): https://github.com/karolpiczak/ESC-50  -> ESC-50-master/
    # 2. python backend/scripts/train_scene_head.py --esc50 /path/to/ESC-50-master
Writes backend/src/soniccaption/services/weights/scene_head.npz and scene_head.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soniccaption.services.scene_classifier import (  # noqa: E402
    BACKGROUND_TRAIN_CLASSES, EVENT_LABELS, WINDOW_S, acoustic_features, window_embedding,
)
from soniccaption.services.sound_cnn import load_sound_model  # noqa: E402

BG_LABEL = len(EVENT_LABELS)               # index 10 = background_noise


def extract(esc_dir: Path, limit: int | None = None):
    import librosa
    import soundfile as sf
    model = load_sound_model()
    rows = list(csv.DictReader(open(esc_dir / "meta" / "esc50.csv")))
    wanted = set(EVENT_LABELS) | set(BACKGROUND_TRAIN_CLASSES)
    rows = [r for r in rows if r["category"] in wanted]
    if limit:
        rows = rows[:limit]
    W, H = int(WINDOW_S * 16000), 16000
    X, y, fold, cat = [], [], [], []
    for i, r in enumerate(rows):
        a, sr = sf.read(esc_dir / "audio" / r["filename"], dtype="float32")
        a = a.mean(1) if a.ndim > 1 else a
        a = librosa.resample(a, orig_sr=sr, target_sr=16000).astype(np.float32)
        for s in range(0, len(a) - W + 1, H):
            seg = a[s:s + W]
            if float(np.sqrt(np.mean(seg ** 2))) < 0.004:      # ESC-50 pads clips with digital silence
                continue
            X.append(np.r_[window_embedding(model, seg), acoustic_features(seg)])
            c = r["category"]
            y.append(EVENT_LABELS.index(c) if c in EVENT_LABELS else BG_LABEL)
            fold.append(int(r["fold"]))
            cat.append(c)
        if i % 200 == 0:
            print(f"  embedded {i}/{len(rows)} clips", flush=True)
    return np.array(X, np.float32), np.array(y), np.array(fold), np.array(cat)


def fit(X, y, mask, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X[mask])
    clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced").fit(sc.transform(X[mask]), y[mask])
    return sc, clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", required=True, type=Path)
    ap.add_argument("--C", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None, help="debug: only embed the first N clips")
    ap.add_argument("--save-holdout", type=Path, default=None,
                    help="also save the folds-1-4 head here (used to evaluate scenes built from fold-5 clips)")
    args = ap.parse_args()

    from sklearn.metrics import f1_score
    X, y, fold, cat = extract(args.esc50, args.limit)
    print(f"{len(X)} windows")

    tr, te = fold <= 4, fold == 5
    sc, clf = fit(X, y, tr, args.C)
    pred = clf.predict(sc.transform(X))
    ev = te & (y < BG_LABEL)
    bg = te & (y == BG_LABEL)
    metrics = {
        "protocol": "fit on ESC-50 folds 1-4, evaluate on held-out fold 5 (2 s windows)",
        "event_accuracy_fold5": round(float((pred[ev] == y[ev]).mean()), 4),
        "macro_f1_11class_fold5": round(float(f1_score(y[te], pred[te], average="macro")), 4),
        "background_recall_fold5": round(float((pred[bg] == BG_LABEL).mean()), 4),
        "events_mislabelled_as_background_fold5": round(float((pred[ev] == BG_LABEL).mean()), 4),
        "leave_one_noise_type_out_background_recall": {},
    }
    for b in BACKGROUND_TRAIN_CLASSES:
        m = (fold <= 4) & (cat != b)
        s2, c2 = fit(X, y, m, args.C)
        p2 = c2.predict(s2.transform(X[cat == b]))
        metrics["leave_one_noise_type_out_background_recall"][b] = round(float((p2 == BG_LABEL).mean()), 3)
    print(json.dumps(metrics, indent=2))

    if args.save_holdout:
        np.savez_compressed(args.save_holdout, mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
                            coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32))
    sc, clf = fit(X, y, np.ones(len(X), bool), args.C)       # final head uses every fold
    out = ROOT / "src" / "soniccaption" / "services" / "weights"
    np.savez_compressed(out / "scene_head.npz", mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
                        coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32))
    (out / "scene_head.json").write_text(json.dumps({
        "labels": EVENT_LABELS + ["background_noise"],
        "window_seconds": WINDOW_S,
        "backbone": "soniccaption_esc50_resnet.pt (Member 1, frozen)",
        "head": "multinomial logistic regression, C=%g, class_weight=balanced" % args.C,
        "features": "512-d ResNet embedding + [log10 rms, log10 p95 frame rms, dynamic range dB, log10 spectral flatness]",
        "background_training_classes": BACKGROUND_TRAIN_CLASSES,
        "n_training_windows": int(len(X)),
        "held_out_metrics": metrics,
        "note": "held_out_metrics come from the folds-1-4 fit; the shipped weights are refit on all folds.",
    }, indent=2))
    print("saved scene_head.npz / scene_head.json")


if __name__ == "__main__":
    main()
