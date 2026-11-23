"""
Vector Quantisation (VQ) Codebook Generation.
Fulfills Unit 2 (Vector Quantisation & Pattern Recognition) requirements:
- K-Means Lloyd Vector Quantisation
- Linde-Buzo-Gray (LBG) binary splitting algorithm
- Codeword index quantization and distortion tracking
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np

from soniccaption.core.schemas import VQResult


def compute_euclidean_distance_matrix(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Computes pairwise squared Euclidean distance between N data vectors and K centroids.
    Formula: ||x_n - c_k||^2 = ||x_n||^2 - 2 * x_n . c_k^T + ||c_k||^2

    Returns:
        (N, K) distance matrix.
    """
    # data: (N, D), centroids: (K, D)
    n_sq = np.sum(data ** 2, axis=1, keepdims=True)  # (N, 1)
    c_sq = np.sum(centroids ** 2, axis=1, keepdims=True)  # (K, 1)
    dot = np.dot(data, centroids.T)  # (N, K)
    dist_sq = np.maximum(n_sq - 2.0 * dot + c_sq.T, 0.0)
    return dist_sq.astype(np.float32)


def kmeans_vq(
    features: np.ndarray,
    k: int = 16,
    max_iter: int = 50,
    tol: float = 1e-4,
    seed: int = 42
) -> VQResult:
    """
    Generates a VQ codebook using standard K-Means (Lloyd's Algorithm).

    Parameters:
        features: (N, D) matrix of speech feature vectors (e.g. MFCCs or LPC vectors).
        k: Codebook size (number of acoustic codewords).
        max_iter: Maximum number of iterations.
        tol: Relative distortion convergence threshold.
        seed: Random seed for initialization.

    Returns:
        VQResult dataclass.
    """
    N, D = features.shape
    if N == 0:
        return VQResult(codebook=np.zeros((k, D), dtype=np.float32), quantized_indices=np.array([]), distortion=0.0)

    # If N < k, clamp k to N
    actual_k = min(k, N)

    # K-Means++ style initialization or uniform random sample
    rng = np.random.default_rng(seed)
    init_indices = rng.choice(N, size=actual_k, replace=False)
    centroids = features[init_indices].copy()

    prev_distortion = float("inf")
    distortion_history: List[float] = []

    for iteration in range(max_iter):
        # 1. Assignment step: find nearest centroid for each vector
        dist_sq = compute_euclidean_distance_matrix(features, centroids)  # (N, actual_k)
        quantized_indices = np.argmin(dist_sq, axis=1)  # (N,)

        # 2. Compute current average distortion
        min_dists = np.min(dist_sq, axis=1)
        curr_distortion = float(np.mean(min_dists))
        distortion_history.append(curr_distortion)

        # Check convergence
        if abs(prev_distortion - curr_distortion) / max(prev_distortion, 1e-9) < tol:
            break
        prev_distortion = curr_distortion

        # 3. Update step: recompute centroids as mean of assigned vectors
        new_centroids = np.zeros_like(centroids)
        for cluster_idx in range(actual_k):
            members = features[quantized_indices == cluster_idx]
            if len(members) > 0:
                new_centroids[cluster_idx] = np.mean(members, axis=0)
            else:
                # Handle empty cluster by re-seeding from random data point
                new_centroids[cluster_idx] = features[rng.choice(N)]

        centroids = new_centroids

    # Compute final cluster counts
    counts = [int(np.sum(quantized_indices == c)) for c in range(actual_k)]

    return VQResult(
        codebook=centroids.astype(np.float32),
        quantized_indices=quantized_indices.astype(np.int32),
        distortion=distortion_history[-1] if distortion_history else 0.0,
        distortion_history=distortion_history,
        cluster_counts=counts
    )


def lbg_vq(
    features: np.ndarray,
    target_k: int = 16,
    epsilon: float = 0.01,
    max_iter: int = 30
) -> VQResult:
    """
    Linde-Buzo-Gray (LBG) binary splitting algorithm for VQ codebook design.

    Algorithm:
    1. Initialize 1 centroid: c_0 = mean of all feature vectors.
    2. Split each centroid: c_plus = c * (1 + epsilon), c_minus = c * (1 - epsilon).
    3. Run Lloyd iterations to optimize centroids at current size.
    4. Repeat splitting until target_k codewords are reached.

    Returns:
        VQResult dataclass.
    """
    N, D = features.shape
    if N == 0:
        return VQResult(codebook=np.zeros((target_k, D), dtype=np.float32), quantized_indices=np.array([]), distortion=0.0)

    # 1. Initial 1-vector codebook (global centroid)
    centroids = np.mean(features, axis=0, keepdims=True)  # (1, D)
    distortion_history: List[float] = []

    while len(centroids) < target_k:
        # 2. Binary splitting: double number of centroids
        split_plus = centroids * (1.0 + epsilon)
        split_minus = centroids * (1.0 - epsilon)
        centroids = np.vstack([split_plus, split_minus])  # (2*M, D)

        if len(centroids) > target_k:
            centroids = centroids[:target_k]

        # 3. Lloyd iterations on current split
        for _ in range(max_iter):
            dist_sq = compute_euclidean_distance_matrix(features, centroids)
            quantized_indices = np.argmin(dist_sq, axis=1)

            # Recompute centroids
            new_centroids = np.zeros_like(centroids)
            for c_idx in range(len(centroids)):
                members = features[quantized_indices == c_idx]
                if len(members) > 0:
                    new_centroids[c_idx] = np.mean(members, axis=0)
                else:
                    new_centroids[c_idx] = centroids[c_idx]
            centroids = new_centroids

    # Final assignment
    dist_sq = compute_euclidean_distance_matrix(features, centroids)
    quantized_indices = np.argmin(dist_sq, axis=1)
    distortion = float(np.mean(np.min(dist_sq, axis=1)))
    distortion_history.append(distortion)
    counts = [int(np.sum(quantized_indices == c)) for c in range(len(centroids))]

    return VQResult(
        codebook=centroids.astype(np.float32),
        quantized_indices=quantized_indices.astype(np.int32),
        distortion=distortion,
        distortion_history=distortion_history,
        cluster_counts=counts
    )
