from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
Index = NDArray[np.int64]


@dataclass
class Selection:
    """Indices K of the selected columns, plus method-specific extras (e.g. swaps)."""

    K: Index
    info: dict = field(default_factory=dict)


class Selector(ABC):
    """A separable NMF column-selection method: X (m x n) -> K with |K| = r."""

    name: str = "selector"

    @abstractmethod
    def select(self, X: Matrix, r: int) -> Selection: ...

    def fit(self, X: Matrix, r: int, iters: int = 500) -> tuple[Index, Matrix, Matrix]:
        """K, W = X[:, K] and H >= 0 with columns summing to at most 1."""
        K = self.select(X, r).K
        W = X[:, K]
        return K, W, solve_h(X, W, iters)

    def __repr__(self) -> str:
        return self.name


def l1_normalize(X: Matrix) -> Matrix:
    return X / (X.sum(axis=0, keepdims=True) + 1e-16)


def project_simplex(Y: Matrix) -> Matrix:
    """Project each column onto {y >= 0, sum(y) = 1}."""
    r, n = Y.shape
    U = -np.sort(-Y, axis=0)
    css = np.cumsum(U, axis=0) - 1.0
    steps = np.arange(1, r + 1)[:, None]
    rho = r - 1 - np.argmax((U - css / steps > 0)[::-1], axis=0)
    theta = css[rho, np.arange(n)] / (rho + 1)
    return np.maximum(Y - theta, 0.0)


def project_capped_simplex(Y: Matrix) -> Matrix:
    """Project each column onto {y >= 0, sum(y) <= 1}."""
    Z = np.maximum(Y, 0.0)
    over = Z.sum(axis=0) > 1.0
    if over.any():
        Z[:, over] = project_simplex(Y[:, over])
    return Z


def solve_h(
    X: Matrix,
    W: Matrix,
    iters: int = 500,
    H0: Matrix | None = None,
    sum_to_one: bool = False,
) -> Matrix:
    """min ||X - W H||_F over columns of H in the (capped) simplex, by accelerated projected gradient."""
    WtW, WtX = W.T @ W, W.T @ X
    L = float(np.linalg.norm(WtW, 2)) or 1.0
    proj = project_simplex if sum_to_one else project_capped_simplex
    H = np.zeros((W.shape[1], X.shape[1])) if H0 is None else proj(H0)
    Z, t = H.copy(), 1.0
    for _ in range(iters):
        H_next = proj(Z - (WtW @ Z - WtX) / L)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = H_next + ((t - 1.0) / t_next) * (H_next - H)
        H, t = H_next, t_next
    return H


def argmax_tiebreak(scores: NDArray, prior: NDArray, rtol: float = 1e-6) -> int:
    """Index of the largest score; near-ties go to the largest prior (as in Gillis' SPA.m)."""
    top = scores.max()
    ties = np.flatnonzero((top - scores) <= rtol * abs(top))
    return int(ties[np.argmax(prior[ties])])
