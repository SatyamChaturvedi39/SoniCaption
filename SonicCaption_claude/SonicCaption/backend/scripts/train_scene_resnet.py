"""Fine-tune Member 1's ESC-50 ResNet-18 into an 11-class *scene* model (10 sound effects + background_noise).

Same protocol as notebook 02 so the numbers are comparable:  train folds 1-3 | validate fold 4 (epoch selection) | test fold 5.
What is new compared with the notebook:
    * an 11th class, `background_noise`, built from stationary ESC-50 classes (wind, vacuum, engine, airplane, ...);
    * 2 s training windows (the live window size) instead of whole 5 s clips;
    * augmentation that mimics real recordings: random gain, mixing with a different noise bed at 5-25 dB SNR,
      SpecAugment time/frequency masks.
Starts from the shipped weights (soniccaption_esc50_resnet.pt): only the last Linear layer gets one new output row.

    python backend/scripts/train_scene_resnet.py --esc50 /path/to/ESC-50-master [--epochs 18]
Writes services/weights/soniccaption_scene_resnet.pt, scene_resnet_config.json, scene_resnet_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SR = 16000
W = 2 * SR


def main() -> None:
    import librosa
    import soundfile as sf
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchaudio.transforms as T
    from sklearn.metrics import f1_score

    from soniccaption.services.scene_classifier import BACKGROUND_TRAIN_CLASSES, EVENT_LABELS
    from soniccaption.services.sound_cnn import SonicAcousticResNet, load_sound_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--crops", type=int, default=4, help="random crops per clip per epoch")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    classes = EVENT_LABELS + ["background_noise"]
    rows = list(csv.DictReader(open(args.esc50 / "meta" / "esc50.csv")))
    rows = [r for r in rows if r["category"] in EVENT_LABELS or r["category"] in BACKGROUND_TRAIN_CLASSES]
    clips = []
    for i, r in enumerate(rows):
        a, sr = sf.read(args.esc50 / "audio" / r["filename"], dtype="float32")
        a = a.mean(1) if a.ndim > 1 else a
        a = librosa.resample(a, orig_sr=sr, target_sr=SR).astype(np.float32)
        y = EVENT_LABELS.index(r["category"]) if r["category"] in EVENT_LABELS else 10
        clips.append({"x": a, "y": y, "fold": int(r["fold"]), "cat": r["category"]})
    print(f"{len(clips)} clips loaded", flush=True)
    train = [c for c in clips if c["fold"] <= 3]
    bed_pool = [c["x"] for c in train if c["y"] == 10]

    base = load_sound_model()
    net = SonicAcousticResNet(11)
    sd = {k: v for k, v in base.net.state_dict().items() if not k.startswith("classifier.4")}
    net.load_state_dict(sd, strict=False)
    old = base.net.state_dict()
    with torch.no_grad():
        net.classifier[4].weight[:10] = old["classifier.4.weight"]
        net.classifier[4].bias[:10] = old["classifier.4.bias"]
    cfg = base.config
    mel = T.MelSpectrogram(sample_rate=SR, n_fft=cfg["n_fft"], win_length=cfg.get("win_length", cfg["n_fft"]),
                           hop_length=cfg["hop_length"], n_mels=cfg["n_mels"])
    to_db = T.AmplitudeToDB()

    def feats(batch: np.ndarray, augment: bool) -> torch.Tensor:
        with torch.no_grad():
            f = to_db(mel(torch.from_numpy(batch))[:, None])
            if augment:
                for i in range(f.shape[0]):
                    for _ in range(rng.integers(1, 3)):
                        w = int(rng.integers(4, 17)); s = int(rng.integers(0, f.shape[2] - w))
                        f[i, :, s:s + w, :] = f[i].mean()
                    for _ in range(rng.integers(1, 3)):
                        w = int(rng.integers(3, 11)); s = int(rng.integers(0, f.shape[3] - w))
                        f[i, :, :, s:s + w] = f[i].mean()
        return f

    def crop(c, augment: bool) -> np.ndarray:
        x = c["x"]
        for _ in range(8):                                   # find an audible crop (ESC-50 pads with digital silence)
            s = int(rng.integers(0, max(1, x.size - W)))
            seg = x[s:s + W]
            if np.sqrt(np.mean(seg ** 2)) >= 0.004:
                break
        seg = np.pad(seg, (0, W - seg.size)).astype(np.float32)
        if augment:
            seg = seg * float(10 ** (rng.uniform(-10, 8) / 20))
            if rng.random() < 0.5 and bed_pool:                 # mix with a different noise bed
                bed = bed_pool[int(rng.integers(len(bed_pool)))]
                bs = int(rng.integers(0, max(1, bed.size - W)))
                bed = np.pad(bed[bs:bs + W], (0, max(0, W - bed[bs:bs + W].size)))
                if c["y"] != 10:
                    snr = rng.uniform(5, 25)
                    ps, pb = np.mean(seg ** 2) + 1e-9, np.mean(bed ** 2) + 1e-9
                    seg = seg + bed * np.sqrt(ps / (pb * 10 ** (snr / 10)))
            if rng.random() < 0.3:
                seg = seg + rng.standard_normal(W).astype(np.float32) * float(10 ** rng.uniform(-3.8, -2.4))
        return np.clip(seg, -1, 1).astype(np.float32)

    def eval_windows(fold: int):
        X, y = [], []
        for c in clips:
            if c["fold"] != fold:
                continue
            for s in range(0, c["x"].size - W + 1, SR):
                seg = c["x"][s:s + W]
                if np.sqrt(np.mean(seg ** 2)) >= 0.004:
                    X.append(seg); y.append(c["y"])
        return np.stack(X).astype(np.float32), np.array(y)

    def evaluate(Xw, yw):
        net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xw), 64):
                preds.append(net(feats(Xw[i:i + 64], False)).argmax(1).numpy())
        p = np.concatenate(preds)
        return p, float((p == yw).mean()), float(f1_score(yw, p, average="macro"))

    Xv, yv = eval_windows(4)
    Xt, yt = eval_windows(5)
    print(f"train clips {len(train)} | val windows {len(Xv)} | test windows {len(Xt)}", flush=True)

    # earlier layers move slowly, later layers + head faster
    early = [p for n, p in net.named_parameters() if n.startswith(("conv1", "bn1", "layer1", "layer2"))]
    late = [p for n, p in net.named_parameters() if not n.startswith(("conv1", "bn1", "layer1", "layer2"))]
    opt = torch.optim.AdamW([{"params": early, "lr": 5e-5}, {"params": late, "lr": 3e-4}], weight_decay=1e-3)
    steps_per_epoch = int(np.ceil(len(train) * args.crops / 32))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 6e-4], total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
    counts = np.bincount([c["y"] for c in train], minlength=11).astype(np.float32)
    # bg class is much larger than each event class -> weight inversely to frequency
    weights = torch.tensor(counts.sum() / (11 * np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    best, best_state, log = -1.0, None, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        order = np.repeat(np.arange(len(train)), args.crops)
        rng.shuffle(order)
        tot = 0.0
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            xb = np.stack([crop(train[j], True) for j in idx])
            yb = torch.tensor([train[j]["y"] for j in idx])
            loss = crit(net(feats(xb, True)), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(idx)
        _, va, vf = evaluate(Xv, yv)
        log.append({"epoch": ep, "train_loss": round(tot / len(order), 4), "val_acc": round(va, 4), "val_macro_f1": round(vf, 4)})
        print(f"epoch {ep:2d}/{args.epochs}  loss {tot / len(order):.3f}  val acc {va:.3f}  macro-F1 {vf:.3f}  [{time.time() - t0:.0f}s]", flush=True)
        if vf > best:
            best, best_state = vf, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    pt, ta, tf = evaluate(Xt, yt)
    ev = yt < 10
    metrics = {
        "protocol": "train ESC-50 folds 1-3, select epoch on fold 4, test on fold 5 (2 s windows)",
        "best_val_macro_f1": round(best, 4), "test_accuracy_11class": round(ta, 4), "test_macro_f1_11class": round(tf, 4),
        "test_event_accuracy": round(float((pt[ev] == yt[ev]).mean()), 4),
        "test_background_recall": round(float((pt[~ev] == 10).mean()), 4),
        "test_events_mislabelled_as_background": round(float((pt[ev] == 10).mean()), 4),
        "per_class_recall": {classes[k]: round(float((pt[yt == k] == k).mean()), 3) for k in range(11) if (yt == k).any()},
        "history": log,
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    out = ROOT / "src" / "soniccaption" / "services" / "weights"
    torch.save(best_state, out / "soniccaption_scene_resnet.pt")
    (out / "scene_resnet_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "scene_resnet_config.json").write_text(json.dumps({**cfg, "architecture": "SonicAcousticResNet18",
                                                              "num_classes": 11, "classes": classes, "clip_duration_seconds": 2.0}, indent=2))
    print("saved soniccaption_scene_resnet.pt")


if __name__ == "__main__":
    main()
