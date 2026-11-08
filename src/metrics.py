from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .core import Index, Matrix, solve_h


def accuracy(K: Index, K_true: Index) -> float:
    return len(set(K.tolist()) & set(K_true.tolist())) / len(K_true)


def q2(W: Matrix, X: Matrix, K: Index) -> float:
    """max_k min_{j in K} ||W[:, k] - X[:, j]||_2 (Gillis, eq. 7.1)."""
    D = np.linalg.norm(W[:, :, None] - X[:, K][:, None, :], axis=0)
    return float(D.min(axis=1).max())


def spectral_angle(W: Matrix, X: Matrix, K: Index) -> float:
    """Mean angle (degrees) between W and X[:, K] after optimal matching of columns."""
    A = W / np.linalg.norm(W, axis=0)
    B = X[:, K] / np.linalg.norm(X[:, K], axis=0)
    angles = np.degrees(np.arccos(np.clip(A.T @ B, -1.0, 1.0)))
    rows, cols = linear_sum_assignment(angles)
    return float(angles[rows, cols].mean())


def fit_error(X: Matrix, K: Index, iters: int = 200) -> float:
    """||X - X[:, K] H||_F / ||X||_F with H in the capped simplex."""
    W = X[:, K]
    return float(np.linalg.norm(X - W @ solve_h(X, W, iters)) / np.linalg.norm(X))
