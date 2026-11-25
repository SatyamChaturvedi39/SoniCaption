"""Discrete Hidden Markov Model (Unit 4).

A discrete HMM is the 5-tuple lambda = (S, V, A, B, pi):
    S   hidden states            s_1 .. s_N
    V   observation symbols      v_1 .. v_M
    A   transition matrix        a_ij = P(q_{t+1} = s_j | q_t = s_i)      shape (N, N)
    B   emission matrix          b_j(k) = P(o_t = v_k | q_t = s_j)        shape (N, M)
    pi  initial distribution     pi_i = P(q_1 = s_i)                      shape (N,)

The three basic problems (Rabiner, 1989):
    1. Evaluation  P(O | lambda)          -> forward algorithm      (this file)
    2. Decoding    best state sequence Q* -> Viterbi algorithm      (this file)
    3. Learning    adjust lambda to fit O -> Baum-Welch (not needed for the demo)

All recursions run in log space so long sequences do not underflow to zero.
This is an educational implementation for the HMM Lab; it is not the decoder used by Whisper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

_TOL = 1e-6


def _log(x: np.ndarray) -> np.ndarray:
    """log with log(0) = -inf and no warning."""
    with np.errstate(divide="ignore"):
        return np.log(x)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log(sum(exp(a))) that returns -inf when every term is -inf."""
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)) + m_safe
    return np.squeeze(out, axis=axis)


@dataclass
class DiscreteHMM:
    states: list[str]
    symbols: list[str]
    pi: np.ndarray
    A: np.ndarray
    B: np.ndarray
    name: str = "custom"
    description: str = ""
    _sym_index: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.states = [str(s) for s in self.states]
        self.symbols = [str(v) for v in self.symbols]
        self.pi = np.asarray(self.pi, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.validate()
        self._sym_index = {v: k for k, v in enumerate(self.symbols)}

    # ------------------------------------------------------------------ checks
    @property
    def N(self) -> int:
        return len(self.states)

    @property
    def M(self) -> int:
        return len(self.symbols)

    def validate(self) -> None:
        N, M = len(self.states), len(self.symbols)
        if N == 0 or M == 0:
            raise ValueError("An HMM needs at least one state and one observation symbol.")
        if len(set(self.states)) != N or len(set(self.symbols)) != M:
            raise ValueError("State names and symbol names must be unique.")
        if self.pi.shape != (N,):
            raise ValueError(f"pi must have shape ({N},), got {self.pi.shape}.")
        if self.A.shape != (N, N):
            raise ValueError(f"A must have shape ({N}, {N}), got {self.A.shape}.")
        if self.B.shape != (N, M):
            raise ValueError(f"B must have shape ({N}, {M}), got {self.B.shape}.")
        for name, mat in (("pi", self.pi), ("A", self.A), ("B", self.B)):
            if not np.all(np.isfinite(mat)) or np.any(mat < 0):
                raise ValueError(f"{name} must contain finite, non-negative probabilities.")
        if abs(self.pi.sum() - 1) > _TOL:
            raise ValueError(f"pi must sum to 1 (sums to {self.pi.sum():.6f}).")
        for name, mat in (("A", self.A), ("B", self.B)):
            bad = np.where(np.abs(mat.sum(axis=1) - 1) > _TOL)[0]
            if bad.size:
                raise ValueError(f"Each row of {name} must sum to 1; row(s) {bad.tolist()} do not.")

    def encode(self, observations: Sequence[str | int]) -> np.ndarray:
        """Map symbol names (or integer indices) to integer indices."""
        if len(observations) == 0:
            raise ValueError("The observation sequence is empty.")
        out = []
        for o in observations:
            if isinstance(o, (int, np.integer)) and not isinstance(o, bool):
                if not 0 <= o < self.M:
                    raise ValueError(f"Observation index {o} is outside 0..{self.M - 1}.")
                out.append(int(o))
            elif str(o) in self._sym_index:
                out.append(self._sym_index[str(o)])
            else:
                raise ValueError(f"Unknown observation symbol {o!r}; valid symbols: {self.symbols}.")
        return np.asarray(out, dtype=int)

    # -------------------------------------------------------- problem 1: forward
    def forward(self, observations: Sequence[str | int]) -> dict:
        """Forward algorithm.

        alpha_1(i)     = pi_i * b_i(o_1)
        alpha_{t+1}(j) = [ sum_i alpha_t(i) * a_ij ] * b_j(o_{t+1})
        P(O | lambda)  = sum_i alpha_T(i)
        """
        obs = self.encode(observations)
        T = len(obs)
        logA, logB = _log(self.A), _log(self.B)
        log_alpha = np.empty((T, self.N))
        log_alpha[0] = _log(self.pi) + logB[:, obs[0]]
        for t in range(1, T):
            log_alpha[t] = _logsumexp(log_alpha[t - 1][:, None] + logA, axis=0) + logB[:, obs[t]]
        log_likelihood = float(_logsumexp(log_alpha[-1], axis=0))
        return {
            "log_alpha": log_alpha,
            "alpha": np.exp(log_alpha),
            "log_likelihood": log_likelihood,
            "likelihood": float(np.exp(log_likelihood)),
        }

    def backward(self, observations: Sequence[str | int]) -> np.ndarray:
        """Backward variables in log space: beta_T(i) = 1, beta_t(i) = sum_j a_ij b_j(o_{t+1}) beta_{t+1}(j)."""
        obs = self.encode(observations)
        T = len(obs)
        logA, logB = _log(self.A), _log(self.B)
        log_beta = np.zeros((T, self.N))
        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(logA + logB[:, obs[t + 1]][None, :] + log_beta[t + 1][None, :], axis=1)
        return log_beta

    def state_posteriors(self, observations: Sequence[str | int]) -> np.ndarray:
        """gamma_t(i) = P(q_t = s_i | O, lambda) = alpha_t(i) beta_t(i) / P(O | lambda)."""
        f = self.forward(observations)
        if not np.isfinite(f["log_likelihood"]):
            raise ValueError("This observation sequence is impossible under the model (P(O|lambda) = 0).")
        return np.exp(f["log_alpha"] + self.backward(observations) - f["log_likelihood"])

    # ------------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "states": self.states, "symbols": self.symbols,
                "pi": self.pi.tolist(), "A": self.A.tolist(), "B": self.B.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "DiscreteHMM":
        return cls(states=d["states"], symbols=d["symbols"], pi=d["pi"], A=d["A"], B=d["B"],
                   name=d.get("name", "custom"), description=d.get("description", ""))


# ---------------------------------------------------------------------- presets
PRESETS: dict[str, dict] = {
    "weather": {
        "name": "weather",
        "description": "Hidden weather (Rainy/Sunny) inferred from a friend's daily activity "
                       "(the classic example used to teach HMMs).",
        "states": ["Rainy", "Sunny"],
        "symbols": ["walk", "shop", "clean"],
        "pi": [0.6, 0.4],
        "A": [[0.7, 0.3], [0.4, 0.6]],
        "B": [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
        "example_observations": ["walk", "shop", "clean"],
    },
    "coin_toss": {
        "name": "coin_toss",
        "description": "Two hidden coins, one fair and one biased towards heads; we only see Heads/Tails "
                       "(Rabiner's coin-toss model).",
        "states": ["Fair", "Biased"],
        "symbols": ["H", "T"],
        "pi": [0.5, 0.5],
        "A": [[0.9, 0.1], [0.2, 0.8]],
        "B": [[0.5, 0.5], [0.85, 0.15]],
        "example_observations": ["H", "T", "H", "H", "H", "H", "T", "H", "T", "T"],
    },
    "urn_ball": {
        "name": "urn_ball",
        "description": "Three hidden urns with different mixes of coloured balls; a ball is drawn from the "
                       "current urn each step (Rabiner's urn-and-ball model).",
        "states": ["Urn1", "Urn2", "Urn3"],
        "symbols": ["Red", "Blue", "Green"],
        "pi": [1 / 3, 1 / 3, 1 / 3],
        "A": [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]],
        "B": [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]],
        "example_observations": ["Red", "Red", "Blue", "Green", "Green", "Blue"],
    },
    "acoustic_scene": {
        "name": "acoustic_scene",
        "description": "Production audio scene HMM smoothing transitions between Ambient silence, Speech, and Environmental Event windows.",
        "states": ["Ambient", "Speech", "SoundEvent"],
        "symbols": ["quiet", "voice", "mixed", "burst"],
        "pi": [0.5, 0.35, 0.15],
        "A": [
            [0.70, 0.20, 0.10],
            [0.15, 0.75, 0.10],
            [0.20, 0.20, 0.60],
        ],
        "B": [
            [0.78, 0.08, 0.06, 0.08],
            [0.08, 0.72, 0.15, 0.05],
            [0.08, 0.10, 0.22, 0.60],
        ],
        "example_observations": ["quiet", "voice", "burst", "voice"],
    },
    "scene_state": {
        "name": "scene_state",
        "description": "Four-state acoustic scene HMM used to smooth per-hop decisions (silence / speech / sound effect / background noise). Sticky transitions remove one-hop flicker; emissions say how far each hop's raw decision is trusted.",
        "states": ["Silence", "Speech", "SoundEffect", "BackgroundNoise"],
        "symbols": ["silence", "speech", "effect", "noise"],
        "pi": [0.40, 0.30, 0.10, 0.20],
        "A": [
            [0.82, 0.07, 0.04, 0.07],
            [0.07, 0.84, 0.04, 0.05],
            [0.14, 0.12, 0.62, 0.12],
            [0.08, 0.07, 0.05, 0.80],
        ],
        "B": [
            [0.84, 0.03, 0.03, 0.10],
            [0.03, 0.88, 0.04, 0.05],
            [0.04, 0.06, 0.82, 0.08],
            [0.10, 0.03, 0.07, 0.80],
        ],
        "example_observations": ["silence", "speech", "speech", "effect", "noise", "noise", "silence"],
    },
    "prosody_discourse": {
        "name": "prosody_discourse",
        "description": "Unit 4 spoken prosody & lecture engagement HMM decoding conversational states from pitch/energy observations.",
        "states": ["TurnPause", "Explanation", "KeyEmphasis", "QuestionPrompt"],
        "symbols": ["silent_pause", "steady_voice", "stressed_loud", "rising_pitch"],
        "pi": [0.25, 0.50, 0.15, 0.10],
        "A": [
            [0.55, 0.35, 0.05, 0.05],
            [0.12, 0.63, 0.15, 0.10],
            [0.10, 0.35, 0.45, 0.10],
            [0.25, 0.30, 0.10, 0.35],
        ],
        "B": [
            [0.82, 0.10, 0.04, 0.04],
            [0.08, 0.68, 0.14, 0.10],
            [0.04, 0.20, 0.66, 0.10],
            [0.05, 0.18, 0.12, 0.65],
        ],
        "example_observations": ["silent_pause", "steady_voice", "stressed_loud", "rising_pitch", "steady_voice"],
    },
}


def load_preset(name: str) -> tuple[DiscreteHMM, list[str]]:
    """Return (model, example observation sequence) for a named preset."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}.")
    p = PRESETS[name]
    return DiscreteHMM.from_dict(p), list(p["example_observations"])


# ============================================================== problem 2: Viterbi
"""Viterbi decoding for a discrete HMM (Unit 4, basic problem 2).

    delta_1(i)     = pi_i * b_i(o_1)                          psi_1(i) = 0
    delta_t(j)     = max_i [ delta_{t-1}(i) * a_ij ] * b_j(o_t)
    psi_t(j)       = argmax_i [ delta_{t-1}(i) * a_ij ]        (backpointer)
    P*             = max_i delta_T(i)
    q*_T           = argmax_i delta_T(i)
    q*_t           = psi_{t+1}(q*_{t+1})                       (backtracking)

Forward sums over all paths (evaluation); Viterbi keeps only the best path (decoding).
Computed in log space; ties are broken towards the lower state index.
"""



def viterbi(model: DiscreteHMM, observations: Sequence[str | int]) -> dict:
    """Decode the most likely hidden state sequence.

    Returns indices and names for the best path, its (log) probability, and the full
    delta / backpointer trellis for visualisation.
    """
    obs = model.encode(observations)
    T, N = len(obs), model.N
    logA, logB = _log(model.A), _log(model.B)

    log_delta = np.full((T, N), -np.inf)
    psi = np.zeros((T, N), dtype=int)
    log_delta[0] = _log(model.pi) + logB[:, obs[0]]
    for t in range(1, T):
        scores = log_delta[t - 1][:, None] + logA          # scores[i, j] = delta_{t-1}(i) + log a_ij
        psi[t] = np.argmax(scores, axis=0)
        log_delta[t] = scores[psi[t], np.arange(N)] + logB[:, obs[t]]

    best_log_prob = float(np.max(log_delta[-1]))
    if not np.isfinite(best_log_prob):
        raise ValueError("This observation sequence is impossible under the model: every path has probability 0.")

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(log_delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]

    return {
        "best_path_idx": path.tolist(),
        "best_path": [model.states[i] for i in path],
        "best_path_log_prob": best_log_prob,
        "best_path_prob": float(np.exp(best_log_prob)),
        "log_delta": log_delta,
        "backpointers": psi,
    }


def decode_for_ui(model: DiscreteHMM, observations: Sequence[str | int]) -> dict:
    """JSON-ready result for the HMM Lab / `POST /labs/hmm/viterbi`.

    Contains the forward likelihood (problem 1), the Viterbi path (problem 2), the delta value of
    every trellis node and every edge with whether it lies on the optimal path.
    """
    obs = model.encode(observations)
    v = viterbi(model, obs)
    f = model.forward(obs)
    T, N = len(obs), model.N
    path = v["best_path_idx"]

    def num(x: float):
        return float(x) if np.isfinite(x) else None          # -inf -> null in JSON

    nodes = [{"step": t, "state": model.states[j], "observation": model.symbols[obs[t]],
              "log_delta": num(v["log_delta"][t, j]), "delta": float(np.exp(v["log_delta"][t, j])),
              "alpha": float(f["alpha"][t, j]),
              "backpointer": None if t == 0 else model.states[v["backpointers"][t, j]],
              "on_best_path": path[t] == j}
             for t in range(T) for j in range(N)]
    edges = [{"step": t, "from": model.states[i], "to": model.states[j], "transition_prob": float(model.A[i, j]),
              "is_backpointer": int(v["backpointers"][t, j]) == i,
              "chosen": path[t - 1] == i and path[t] == j}
             for t in range(1, T) for i in range(N) for j in range(N)]

    return {
        "model": model.to_dict(),
        "observations": [model.symbols[k] for k in obs],
        "log_likelihood": num(f["log_likelihood"]),
        "likelihood": f["likelihood"],
        "best_path": v["best_path"],
        "best_path_log_prob": v["best_path_log_prob"],
        "best_path_prob": v["best_path_prob"],
        "trellis_nodes": nodes,
        "trellis_edges": edges,
    }


def decode_acoustic_scene_hmm(
    duration: float,
    vad_segments: list[dict],
    sound_events: list[dict],
    step_s: float = 1.0,
    max_steps: int = 30,
) -> dict:
    """Apply the discrete acoustic_scene HMM to discretised intervals of a clip (pass max_steps/step_s to cover long files)."""
    total_steps = max(1, min(max_steps, int(np.ceil(max(duration, step_s) / step_s))))
    obs: list[str] = []
    for k in range(total_steps):
        t0, t1 = k * step_s, (k + 1) * step_s
        has_vad = any(min(t1, float(s["end"])) > max(t0, float(s["start"])) for s in vad_segments)
        has_ev = any(
            min(t1, float(e["end"])) > max(t0, float(e["start"])) and float(e.get("confidence", 0.0)) >= 0.40
            for e in sound_events
        )
        if has_vad and has_ev:
            obs.append("mixed")
        elif has_ev:
            obs.append("burst")
        elif has_vad:
            obs.append("voice")
        else:
            obs.append("quiet")

    model, _ = load_preset("acoustic_scene")
    decoded = decode_for_ui(model, obs)
    return {
        "step_s": step_s,
        "observations": decoded["observations"],
        "best_path": decoded["best_path"],
        "log_likelihood": decoded["log_likelihood"],
        "best_path_log_prob": decoded["best_path_log_prob"],
    }


def decode_prosody_hmm(
    speech_segments: list[dict],
    duration: float,
    step_s: float = 1.0,
    max_steps: int = 30,
) -> dict:
    """Decode Unit 4 conversational prosody states across windows using Unit 1-2 acoustic cues."""
    total_steps = max(1, min(max_steps, int(np.ceil(max(duration, step_s) / step_s))))
    obs: list[str] = []
    for k in range(total_steps):
        t0, t1 = k * step_s, (k + 1) * step_s
        active = [
            s for s in speech_segments
            if min(t1, float(s.get("end", 0.0))) > max(t0, float(s.get("start", 0.0)))
        ]
        if not active:
            obs.append("silent_pause")
            continue
        styles = " ".join(str(s.get("delivery_style", "")) for s in active)
        if "Question" in styles or "Rising" in styles:
            obs.append("rising_pitch")
        elif "Emphasized" in styles or max(float(s.get("energy_ratio", 1.0)) for s in active) >= 1.25:
            obs.append("stressed_loud")
        else:
            obs.append("steady_voice")

    model, _ = load_preset("prosody_discourse")
    decoded = decode_for_ui(model, obs)
    return {
        "step_s": step_s,
        "observations": decoded["observations"],
        "best_path": decoded["best_path"],
        "log_likelihood": decoded["log_likelihood"],
        "best_path_log_prob": decoded["best_path_log_prob"],
    }
