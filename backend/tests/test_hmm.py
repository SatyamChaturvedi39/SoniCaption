import itertools
import json

import numpy as np
import pytest

from soniccaption.algorithms.hmm import PRESETS, DiscreteHMM, load_preset
from soniccaption.algorithms.hmm import decode_for_ui, viterbi


def brute_force(model: DiscreteHMM, obs):
    """Enumerate every state path: returns P(O|lambda) and the best path with its probability."""
    o = model.encode(obs)
    total, best_p, best_q = 0.0, -1.0, None
    for q in itertools.product(range(model.N), repeat=len(o)):
        p = model.pi[q[0]] * model.B[q[0], o[0]]
        for t in range(1, len(o)):
            p *= model.A[q[t - 1], q[t]] * model.B[q[t], o[t]]
        total += p
        if p > best_p:
            best_p, best_q = p, list(q)
    return total, best_q, best_p


# ------------------------------------------------------------ hand-checked example
def test_weather_example_by_hand():
    """Classic example: O = walk, shop, clean.

    t=1  delta(R)=0.6*0.1=0.06            delta(S)=0.4*0.6=0.24
    t=2  delta(R)=max(0.06*.7, 0.24*.4)*0.4=0.0384  delta(S)=max(0.06*.3, 0.24*.6)*0.3=0.0432
    t=3  delta(R)=max(0.0384*.7, 0.0432*.4)*0.5=0.01344  delta(S)=max(...)*0.1=0.002592
    Best path: Sunny -> Rainy -> Rainy, P* = 0.01344
    """
    m, obs = load_preset("weather")
    v = viterbi(m, obs)
    assert v["best_path"] == ["Sunny", "Rainy", "Rainy"]
    assert v["best_path_prob"] == pytest.approx(0.01344)
    assert np.exp(v["log_delta"][1]) == pytest.approx([0.0384, 0.0432])
    assert m.forward(obs)["likelihood"] == pytest.approx(0.033612)


# --------------------------------------------- agreement with brute-force enumeration
@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_match_brute_force(name):
    m, obs = load_preset(name)
    obs = obs[:8]
    total, best_q, best_p = brute_force(m, obs)
    assert m.forward(obs)["likelihood"] == pytest.approx(total, rel=1e-9)
    v = viterbi(m, obs)
    assert v["best_path_prob"] == pytest.approx(best_p, rel=1e-9)


def test_random_models_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(25):
        N, M, T = rng.integers(2, 4), rng.integers(2, 4), rng.integers(1, 6)
        norm = lambda x: x / x.sum(axis=-1, keepdims=True)
        m = DiscreteHMM(states=[f"s{i}" for i in range(N)], symbols=[f"v{k}" for k in range(M)],
                        pi=norm(rng.random(N)), A=norm(rng.random((N, N))), B=norm(rng.random((N, M))))
        obs = rng.integers(0, M, T).tolist()
        total, _, best_p = brute_force(m, obs)
        assert m.forward(obs)["likelihood"] == pytest.approx(total, rel=1e-9)
        v = viterbi(m, obs)
        assert v["best_path_prob"] == pytest.approx(best_p, rel=1e-9)
        # the returned path really has probability best_p
        q, o = v["best_path_idx"], m.encode(obs)
        p = m.pi[q[0]] * m.B[q[0], o[0]] * np.prod([m.A[q[t - 1], q[t]] * m.B[q[t], o[t]] for t in range(1, T)])
        assert p == pytest.approx(best_p, rel=1e-9)


# ------------------------------------------------------------------- properties
def test_path_length_equals_observation_length():
    m, obs = load_preset("coin_toss")
    assert len(viterbi(m, obs)["best_path"]) == len(obs)


def test_single_observation():
    m, _ = load_preset("weather")
    v = viterbi(m, ["walk"])
    assert v["best_path"] == ["Sunny"]                       # 0.4*0.6 > 0.6*0.1
    assert m.forward(["walk"])["likelihood"] == pytest.approx(0.6 * 0.1 + 0.4 * 0.6)


def test_viterbi_never_exceeds_forward():
    m, obs = load_preset("urn_ball")
    assert viterbi(m, obs)["best_path_prob"] <= m.forward(obs)["likelihood"] + 1e-15


def test_posteriors_sum_to_one():
    m, obs = load_preset("coin_toss")
    g = m.state_posteriors(obs)
    assert g.shape == (len(obs), m.N)
    assert g.sum(axis=1) == pytest.approx(np.ones(len(obs)))


def test_long_sequence_does_not_underflow():
    m, _ = load_preset("coin_toss")
    obs = ["H", "T"] * 1000                                   # 2000 steps: raw product would be 0.0
    f = m.forward(obs)
    assert np.isfinite(f["log_likelihood"]) and f["log_likelihood"] < -1000
    assert len(viterbi(m, obs)["best_path"]) == 2000


def test_zero_probabilities_are_respected():
    # state "B" can never emit "x", and A forbids B -> A
    m = DiscreteHMM(states=["A", "B"], symbols=["x", "y"], pi=[0.5, 0.5],
                    A=[[0.5, 0.5], [0.0, 1.0]], B=[[0.9, 0.1], [0.0, 1.0]])
    v = viterbi(m, ["x", "y", "y"])
    assert v["best_path"][0] == "A"
    total, best_q, _ = brute_force(m, ["x", "y", "y"])
    assert v["best_path_idx"] == best_q
    assert m.forward(["x", "y", "y"])["likelihood"] == pytest.approx(total)


def test_impossible_sequence_raises_clear_error():
    m = DiscreteHMM(states=["A"], symbols=["x", "y"], pi=[1.0], A=[[1.0]], B=[[1.0, 0.0]])
    assert m.forward(["y"])["likelihood"] == 0.0
    with pytest.raises(ValueError, match="impossible"):
        viterbi(m, ["y"])


def test_integer_and_symbol_observations_agree():
    m, obs = load_preset("weather")
    assert viterbi(m, obs)["best_path"] == viterbi(m, [0, 1, 2])["best_path"]


# ------------------------------------------------------------------- validation
@pytest.mark.parametrize("kwargs, msg", [
    (dict(pi=[0.5, 0.6]), "pi must sum to 1"),
    (dict(A=[[0.5, 0.4], [0.5, 0.5]]), "row"),
    (dict(B=[[1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]), "B must have shape"),
    (dict(A=[[1.2, -0.2], [0.5, 0.5]]), "non-negative"),
])
def test_invalid_parameters_rejected(kwargs, msg):
    base = dict(states=["a", "b"], symbols=["x", "y"], pi=[0.5, 0.5], A=[[0.5, 0.5], [0.5, 0.5]], B=[[0.5, 0.5], [0.5, 0.5]])
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        DiscreteHMM(**base)


def test_unknown_symbol_and_empty_sequence_rejected():
    m, _ = load_preset("weather")
    with pytest.raises(ValueError, match="Unknown observation"):
        viterbi(m, ["walk", "swim"])
    with pytest.raises(ValueError, match="empty"):
        m.forward([])


# ----------------------------------------------------------------- UI payload
def test_ui_payload_is_json_and_consistent():
    m, obs = load_preset("weather")
    out = decode_for_ui(m, obs)
    json.dumps(out)                                           # serialisable
    assert out["best_path"] == ["Sunny", "Rainy", "Rainy"]
    assert len(out["trellis_nodes"]) == len(obs) * m.N
    assert len(out["trellis_edges"]) == (len(obs) - 1) * m.N * m.N
    assert sum(e["chosen"] for e in out["trellis_edges"]) == len(obs) - 1
    on_path = [n["state"] for n in out["trellis_nodes"] if n["on_best_path"]]
    assert on_path == out["best_path"]
