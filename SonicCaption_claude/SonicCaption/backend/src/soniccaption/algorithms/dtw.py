"""
Dynamic Time Warping (DTW) Sequence Alignment.
Fulfills Unit 3 (Pattern Comparison & Time Alignment) requirements:
- Frame-level local distortion metrics (Euclidean, Cosine, Squared Euclidean)
- Accumulated cost matrix dynamic programming recurrence
- Boundary, monotonicity, and continuity constraints
- Optimal warp path backtracking
- Normalized alignment distance
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np

from soniccaption.core.schemas import DTWResult


def compute_frame_distance_matrix(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Computes pairwise local distance matrix between two feature sequences.

    Parameters:
        x: (N, D) array of feature vectors (e.g. MFCCs from Clip A).
        y: (M, D) array of feature vectors (e.g. MFCCs from Clip B).
        metric: 'euclidean', 'sqeuclidean', or 'cosine'.

    Returns:
        (N, M) matrix of pairwise distances.
    """
    N, D1 = x.shape
    M, D2 = y.shape
    if D1 != D2:
        raise ValueError(f"Feature dimension mismatch: x has dimension {D1}, y has {D2}")

    metric_lower = metric.lower()

    try:
        from scipy.spatial.distance import cdist
        scipy_metric = "sqeuclidean" if metric_lower in ("sqeuclidean", "sq_euclidean") else metric_lower
        dist = cdist(x, y, metric=scipy_metric)
        return dist.astype(np.float32)
    except Exception:
        # Fallback exact difference broadcasting for moderate sequences
        if metric_lower == "cosine":
            norm_x = np.linalg.norm(x, axis=1, keepdims=True)
            norm_y = np.linalg.norm(y, axis=1, keepdims=True)
            norm_x = np.maximum(norm_x, 1e-12)
            norm_y = np.maximum(norm_y, 1e-12)
            x_normed = x / norm_x
            y_normed = y / norm_y
            dot = np.dot(x_normed, y_normed.T)
            dist = 1.0 - np.clip(dot, -1.0, 1.0)
            return dist.astype(np.float32)
        else:
            diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]  # (N, M, D)
            if metric_lower in ("sqeuclidean", "sq_euclidean"):
                return np.sum(diff ** 2, axis=-1, dtype=np.float32)
            return np.linalg.norm(diff, axis=-1).astype(np.float32)


def dtw(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "euclidean",
    window: Optional[int] = None
) -> DTWResult:
    """
    Computes Dynamic Time Warping (DTW) between feature sequences x and y.

    Mathematical Formulation:
    Accumulated cost recurrence:
    D(i, j) = d(i, j) + min( D(i-1, j), D(i, j-1), D(i-1, j-1) )

    Constraints:
    1. Boundary condition: Path starts at (0, 0) and ends at (N-1, M-1).
    2. Monotonicity: i_{k-1} <= i_k and j_{k-1} <= j_k.
    3. Continuity / Step condition: (i_k - i_{k-1}, j_k - j_{k-1}) in {(1, 0), (0, 1), (1, 1)}.

    Parameters:
        x: (N, D) sequence A feature vectors.
        y: (M, D) sequence B feature vectors.
        metric: 'euclidean', 'sqeuclidean', or 'cosine'.
        window: Optional Sakoe-Chiba band constraint width (|i - j| <= window).

    Returns:
        DTWResult dataclass containing cost matrix, optimal warp path, and normalized distance.
    """
    # Ensure 2D arrays
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]

    N, M = len(x), len(y)
    if N == 0 or M == 0:
        return DTWResult(
            distance=0.0,
            normalized_distance=0.0,
            path_length=0,
            path=[],
            cost_matrix=np.zeros((N, M), dtype=np.float32),
            local_distance_matrix=np.zeros((N, M), dtype=np.float32),
            metric=metric
        )

    # 1. Compute pairwise local distance matrix d(i, j)
    local_dist = compute_frame_distance_matrix(x, y, metric=metric)

    # 2. Dynamic programming accumulated cost matrix
    cost_matrix = np.full((N, M), np.inf, dtype=np.float32)

    # Base case
    cost_matrix[0, 0] = local_dist[0, 0]

    # Initialize first column
    for i in range(1, N):
        if window is None or abs(i) <= window:
            cost_matrix[i, 0] = cost_matrix[i - 1, 0] + local_dist[i, 0]

    # Initialize first row
    for j in range(1, M):
        if window is None or abs(j) <= window:
            cost_matrix[0, j] = cost_matrix[0, j - 1] + local_dist[0, j]

    # Fill accumulated cost matrix
    for i in range(1, N):
        j_min = 1
        j_max = M
        if window is not None:
            j_min = max(1, i - window)
            j_max = min(M, i + window + 1)

        for j in range(j_min, j_max):
            prev_min = min(
                cost_matrix[i - 1, j],      # Insertion (vertical)
                cost_matrix[i, j - 1],      # Deletion (horizontal)
                cost_matrix[i - 1, j - 1]   # Match / Diagonal step
            )
            cost_matrix[i, j] = local_dist[i, j] + prev_min

    # 3. Backtracking to recover optimal warp path
    path: List[Tuple[int, int]] = []
    curr_i, curr_j = N - 1, M - 1
    path.append((curr_i, curr_j))

    while curr_i > 0 or curr_j > 0:
        if curr_i == 0:
            curr_j -= 1
        elif curr_j == 0:
            curr_i -= 1
        else:
            # Step to minimum cost neighbor
            neighbors = [
                (cost_matrix[curr_i - 1, curr_j - 1], curr_i - 1, curr_j - 1),  # Diagonal
                (cost_matrix[curr_i - 1, curr_j], curr_i - 1, curr_j),          # Vertical
                (cost_matrix[curr_i, curr_j - 1], curr_i, curr_j - 1)           # Horizontal
            ]
            neighbors.sort(key=lambda item: item[0])
            _, curr_i, curr_j = neighbors[0]

        path.append((curr_i, curr_j))

    # Reverse path so it begins at (0, 0) and ends at (N-1, M-1)
    path.reverse()

    total_dist = float(cost_matrix[N - 1, M - 1])
    path_len = len(path)
    norm_dist = total_dist / float(path_len) if path_len > 0 else 0.0

    return DTWResult(
        distance=total_dist,
        normalized_distance=norm_dist,
        path_length=path_len,
        path=path,
        cost_matrix=cost_matrix,
        local_distance_matrix=local_dist,
        metric=metric
    )
