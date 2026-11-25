import numpy as np
import pytest
from soniccaption.algorithms.vq import kmeans_vq, lbg_vq


def test_kmeans_vq_clustering():
    """Verify K-means vector quantisation convergence and shapes."""
    rng = np.random.default_rng(42)
    # 200 vectors in 13-dimensional space (simulating 13-MFCC frames)
    features = rng.normal(0.0, 1.0, (200, 13)).astype(np.float32)

    k = 8
    res = kmeans_vq(features, k=k, max_iter=30)

    assert res.codebook.shape == (k, 13)
    assert len(res.quantized_indices) == 200
    assert np.all(res.quantized_indices >= 0)
    assert np.all(res.quantized_indices < k)
    assert res.distortion >= 0.0
    # Distortion history should decrease monotonically or plateau
    assert res.distortion_history[-1] <= res.distortion_history[0]


def test_lbg_vq_binary_splitting():
    """Verify LBG binary splitting generates exact target K codewords."""
    rng = np.random.default_rng(101)
    features = rng.normal(5.0, 2.0, (150, 13)).astype(np.float32)

    target_k = 16
    res = lbg_vq(features, target_k=target_k, epsilon=0.01)

    assert res.codebook.shape == (target_k, 13)
    assert len(res.quantized_indices) == 150
    assert res.distortion > 0.0
