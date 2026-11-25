"""Diarization benchmark on composed conversations with known speakers.

    python backend/scripts/diarization_bench.py --speech-dir DIR [--backend neural|dsp] [--noise-wav FILE]

DIR holds one sub-folder per speaker (e.g. spk1/, spk2/, jfk/) with that person's wavs. Conversations of 2-5 speakers are
composed with random turn order; we report how often the speaker COUNT is right and the frame-level label accuracy after
the best one-to-one mapping of predicted to true speakers (Hungarian), on speech frames only (no overlap in these mixes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SR = 16000


def load_pool(d: Path) -> dict[str, list[np.ndarray]]:
    import librosa
    import soundfile as sf
    pool = {}
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        utts = []
        for f in sorted(list(sub.glob("*.wav")) + list(sub.glob("*.flac"))):
            y, sr = sf.read(f, dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            y = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
            utts.append((y * 0.05 / (np.sqrt(np.mean(y ** 2)) + 1e-9)).astype(np.float32))
        if utts:
            pool[sub.name] = utts
    return pool


def conversation(rng, pool, n_spk: int, turns: int = 14, noise: np.ndarray | None = None, snr_db: float = 15.0):
    names = list(rng.choice(list(pool), size=n_spk, replace=False))
    parts, truth, t, prev = [], [], 0.0, None
    order = list(names) + [str(rng.choice(names)) for _ in range(turns - n_spk)]
    rng.shuffle(order)
    for i, name in enumerate(order):
        if name == prev and n_spk > 1:
            name = [n for n in names if n != prev][int(rng.integers(n_spk - 1))]
        utt = pool[name][int(rng.integers(len(pool[name])))]
        parts.append(utt)
        truth.append((t, t + utt.size / SR, name))
        t += utt.size / SR
        gap = rng.standard_normal(int(rng.uniform(0.35, 0.9) * SR)).astype(np.float32) * 0.001
        parts.append(gap)
        t += gap.size / SR
        prev = name
    x = np.concatenate(parts)
    if noise is not None:
        nz = np.tile(noise, int(np.ceil(x.size / noise.size)))[: x.size]
        nz = nz * (0.05 / (np.sqrt(np.mean(nz ** 2)) + 1e-9)) * 10 ** (-snr_db / 20)
        x = x + nz
    return x.astype(np.float32), truth, len(set(n for _, _, n in truth))


def frame_accuracy(pred_turns, truth) -> float:
    from scipy.optimize import linear_sum_assignment
    T = int(max(e for _, e, _ in truth) * 100) + 1
    tr = np.full(T, -1)
    pr = np.full(T, -1)
    names = sorted({n for _, _, n in truth})
    for a, b, n in truth:
        tr[int(a * 100):int(b * 100)] = names.index(n)
    for t in pred_turns:
        pr[int(t["start"] * 100):min(T, int(t["end"] * 100))] = t["speaker_id"] - 1
    m = (tr >= 0)
    kp = max(pr.max() + 1, len(names))
    C = np.zeros((len(names), kp), int)
    for i in range(len(names)):
        for j in range(kp):
            C[i, j] = np.sum((tr == i) & (pr == j) & m)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / max(m.sum(), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-dir", type=Path, required=True)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--noise-wav", type=Path, default=None)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--thresholds", type=float, nargs="*", default=None)
    ap.add_argument("--denoise", action="store_true", help="apply Member 1's spectral subtraction first, as the app pipeline does")
    args = ap.parse_args()
    from soniccaption.algorithms.diarization import diarize
    from soniccaption.audio.vad import detect_speech
    pool = load_pool(args.speech_dir)
    print("speakers:", {k: len(v) for k, v in pool.items()})
    noise = None
    if args.noise_wav:
        import soundfile as sf
        import librosa
        nz, sr = sf.read(args.noise_wav, dtype="float32")
        noise = librosa.resample(nz.mean(1) if nz.ndim > 1 else nz, orig_sr=sr, target_sr=SR)
    for thr in args.thresholds or [None]:
        for cond in (["clean"] + (["noisy"] if noise is not None else [])):
            ok_cnt = tot = 0
            accs = []
            for n_spk in (1, 2, 3, 4, 5):
                if n_spk > len(pool):
                    continue
                rng = np.random.default_rng(100 + n_spk)
                hits, a_list = 0, []
                for _ in range(args.trials):
                    x, truth, k = conversation(rng, pool, n_spk, noise=noise if cond == "noisy" else None)
                    if args.denoise:
                        from soniccaption.algorithms.enhancement import spectral_subtraction_denoise
                        x = spectral_subtraction_denoise(x, sr=SR)["denoised"]
                    vad = detect_speech(x, SR)
                    r = diarize(x, vad, backend=args.backend, threshold=thr)
                    hits += int(r["n_speakers"] == k)
                    a_list.append(frame_accuracy(r["turns"], truth))
                print(f"  thr={thr} {cond:5s} {n_spk} speakers: count right {hits}/{args.trials}  frame acc {np.mean(a_list):.1%}  [{r['backend']}]")
                ok_cnt += hits; tot += args.trials; accs += a_list
            print(f"thr={thr} {cond:5s} OVERALL count right {ok_cnt}/{tot}  frame acc {np.mean(accs):.1%}\n", flush=True)


if __name__ == "__main__":
    main()
