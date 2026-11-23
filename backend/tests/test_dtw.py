import numpy as np
import pytest
from soniccaption.algorithms.dtw import dtw, compute_frame_distance_matrix


def test_dtw_identity():
    """Identity test: DTW distance between identical sequences must be exactly 0.0."""
    rng = np.random.default_rng(42)
    seq = rng.normal(0.0, 1.0, (30, 13)).astype(np.float32)

    res = dtw(seq, seq, metric="euclidean")
    assert res.distance == pytest.approx(0.0, abs=1e-5)
    assert res.normalized_distance == pytest.approx(0.0, abs=1e-5)

    # Path should be pure diagonal: (0, 0), (1, 1), ..., (29, 29)
    expected_path = [(i, i) for i in range(30)]
    assert res.path == expected_path


def test_dtw_boundary_and_monotonicity():
    """Verify alignment path boundary and monotonicity constraints."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, (25, 13)).astype(np.float32)
    y = rng.normal(0.0, 1.0, (40, 13)).astype(np.float32)

    res = dtw(x, y, metric="euclidean")

    # Boundary conditions
    assert res.path[0] == (0, 0)
    assert res.path[-1] == (24, 39)

    # Monotonicity & continuity
    for k in range(1, len(res.path)):
        prev_i, prev_j = res.path[k - 1]
        curr_i, curr_j = res.path[k]

        step_i = curr_i - prev_i
        step_j = curr_j - prev_j

        assert step_i >= 0
        assert step_j >= 0
        assert (step_i, step_j) in [(1, 0), (0, 1), (1, 1)]


def test_dtw_time_warped_alignment():
    """A time-stretched sequence should align with low distance."""
    base = np.sin(np.linspace(0, 4 * np.pi, 50))[:, np.newaxis].astype(np.float32)
    # Stretched by a factor of 2 (100 samples)
    stretched = np.sin(np.linspace(0, 4 * np.pi, 100))[:, np.newaxis].astype(np.float32)

    res = dtw(base, stretched, metric="euclidean")
    assert res.distance < 5.0
    assert res.normalized_distance < 0.1
