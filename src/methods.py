from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Matrix, Selection, Selector, argmax_tiebreak, l1_normalize, solve_h
from .qr import strong_rrqr


def _spa(X: Matrix, r: int) -> list[int]:
    R = np.array(X, dtype=float, copy=True)
    norms0 = np.sum(R**2, axis=0)
    norms = norms0.copy()
    K: list[int] = []
    for _ in range(r):
        if norms.max() <= 1e-12 * norms0.max():
            break
        p = argmax_tiebreak(norms, norms0)
        K.append(p)
        u = R[:, p] / np.linalg.norm(R[:, p])
        R -= np.outer(u, u @ R)
        norms = np.sum(R**2, axis=0)
    return K


# --------------------------------------------------------------------------- SPA
@dataclass
class SPAConfig:
    normalize: bool = False


class SPA(Selector):
    """Successive projection algorithm (Gillis, Alg. 7.1)."""

    def __init__(self, cfg: SPAConfig | None = None):
        self.cfg = cfg or SPAConfig()
        self.name = "SPA"

    def select(self, X: Matrix, r: int) -> Selection:
        Y = l1_normalize(X) if self.cfg.normalize else X
        return Selection(np.array(_spa(Y, r), dtype=np.int64))


# --------------------------------------------------------------------------- FAW
@dataclass
class FAWConfig:
    normalize: bool = False
    passes: int = 1  # 1 = FAW; more passes = N-FINDR-style local search


class FAW(Selector):
    """SPA, then replace each index by the column farthest from the span of the others (Alg. 7.2)."""

    def __init__(self, cfg: FAWConfig | None = None):
        self.cfg = cfg or FAWConfig()
        self.name = "FAW" if self.cfg.passes == 1 else f"FAW x{self.cfg.passes}"

    def select(self, X: Matrix, r: int) -> Selection:
        Y = l1_normalize(X) if self.cfg.normalize else X
        K = _spa(Y, r)
        swaps = 0
        for _ in range(self.cfg.passes):
            for i in range(len(K)):
                Q, _ = np.linalg.qr(Y[:, K[:i] + K[i + 1 :]])
                R = Y - Q @ (Q.T @ Y)
                best = int(np.argmax(np.sum(R**2, axis=0)))
                swaps += best != K[i]
                K[i] = best
        return Selection(np.array(K, dtype=np.int64), {"swaps": swaps})


# -------------------------------------------------------------------------- SNPA
@dataclass
class SNPAConfig:
    iters: int = 500
    normalize: bool = False
    sum_to_one: bool = False  # False: sum(h) <= 1, True: sum(h) = 1


class SNPA(Selector):
    """Successive nonnegative projection algorithm (Gillis, Alg. 7.3)."""

    def __init__(self, cfg: SNPAConfig | None = None):
        self.cfg = cfg or SNPAConfig()
        self.name = "SNPA"

    def select(self, X: Matrix, r: int) -> Selection:
        Y = l1_normalize(X) if self.cfg.normalize else X
        n = Y.shape[1]
        norms0 = np.sum(Y**2, axis=0)
        residual = norms0.copy()
        K: list[int] = []
        H = None
        for _ in range(r):
            if residual.max() <= 1e-12 * norms0.max():
                break
            p = argmax_tiebreak(residual, norms0)
            K.append(p)
            W = Y[:, K]
            H0 = None if H is None else np.vstack([H, np.eye(1, n, p)])
            H = solve_h(Y, W, self.cfg.iters, H0, self.cfg.sum_to_one)
            residual = np.sum((Y - W @ H) ** 2, axis=0)
        return Selection(np.array(K, dtype=np.int64), {"H": H})


# ------------------------------------------------------------------------- SRRQR
@dataclass
class SRRQRConfig:
    f: float = 1.01  # np.inf = QR with column pivoting (no swaps)
    normalize: bool = False
    max_swaps: int = 10_000


class SRRQR(Selector):
    """Strong rank-revealing QR (Gu & Eisenstat): QRCP + swaps that grow the volume by more than f."""

    def __init__(self, cfg: SRRQRConfig | None = None):
        self.cfg = cfg or SRRQRConfig()
        self.name = "QRCP" if np.isinf(self.cfg.f) else f"SRRQR f={self.cfg.f:g}"
        if self.cfg.normalize:
            self.name += " (L1)"

    def select(self, X: Matrix, r: int) -> Selection:
        Y = l1_normalize(X) if self.cfg.normalize else X
        _, _, P, _, swaps = strong_rrqr(
            Y, r=r, f=self.cfg.f, max_swaps=self.cfg.max_swaps
        )
        return Selection(P[:r].copy(), {"swaps": swaps})


# ----------------------------------------------------------------------- Precond
def mve_transform(X: Matrix, r: int, tol: float = 1e-3, max_iter: int = 5000) -> Matrix:
    """Project onto the top-r subspace, then map the minimum-volume centered ellipsoid
    containing the columns to the unit ball (Khachiyan / Todd iterations)."""
    Y = np.linalg.svd(X, full_matrices=False)[0][:, :r].T @ X
    n = Y.shape[1]
    u = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        M = (Y * u) @ Y.T
        g = np.sum(Y * np.linalg.solve(M, Y), axis=0)
        j = int(np.argmax(g))
        if g[j] <= r * (1 + tol):
            break
        step = (g[j] / r - 1) / (g[j] - 1)
        u *= 1 - step
        u[j] += step
    A = np.linalg.inv((Y * u) @ Y.T) / r
    return np.linalg.cholesky(A).T @ Y


@dataclass
class PrecondConfig:
    kind: str = (
        "pinv"  # "pinv": pinv(X[:, K0]) with K0 from `first`; "svd": top-r subspace; "mve": min-volume ellipsoid
    )
    mve_tol: float = 1e-3


class Precond(Selector):
    """Run `method` on P @ X; indices refer to the original columns."""

    def __init__(
        self,
        method: Selector,
        first: Selector | None = None,
        cfg: PrecondConfig | None = None,
    ):
        self.method, self.first = method, first or SPA()
        self.cfg = cfg or PrecondConfig()
        prefix = {"svd": "SVD", "mve": "MVE"}.get(self.cfg.kind, self.first.name)
        self.name = f"{prefix}-{method.name}"

    def transform(self, X: Matrix, r: int) -> Matrix:
        if self.cfg.kind == "mve":
            return mve_transform(X, r, self.cfg.mve_tol)
        if self.cfg.kind == "svd":
            U = np.linalg.svd(X, full_matrices=False)[0]
            return U[:, :r].T @ X
        K0 = self.first.select(X, r).K
        return np.linalg.pinv(X[:, K0]) @ X

    def select(self, X: Matrix, r: int) -> Selection:
        return self.method.select(self.transform(X, r), r)


# --------------------------------------------------------------------------- VCA
def _normalize(X: Matrix, norm: str) -> Matrix:
    if norm == "l2":
        return X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-16)
    if norm == "l1":
        return X / (np.sum(np.abs(X), axis=0, keepdims=True) + 1e-16)
    return X


@dataclass
class VCAConfig:
    normalize: str = "none"  # "none", "l1" or "l2"
    seed: int = 0


class VCA(Selector):
    """Vertex component analysis: project onto the top-r subspace, then pick argmax |f^T x|
    for random directions f orthogonal to the columns already selected (Nascimento & Bioucas-Dias).
    """

    def __init__(self, cfg: VCAConfig | None = None):
        self.cfg = cfg or VCAConfig()
        self.name = "VCA"

    def select(self, X: Matrix, r: int) -> Selection:
        Y = _normalize(X, self.cfg.normalize)
        Y = np.linalg.svd(Y, full_matrices=False)[0][:, :r].T @ Y
        rng = np.random.default_rng(self.cfg.seed)
        A = np.zeros((r, 0))
        K: list[int] = []
        for _ in range(r):
            w = rng.standard_normal(r)
            f = w - A @ np.linalg.lstsq(A, w, rcond=None)[0] if A.shape[1] else w
            k = int(np.argmax(np.abs((f / np.linalg.norm(f)) @ Y)))
            K.append(k)
            A = np.column_stack([A, Y[:, k]])
        return Selection(np.array(K, dtype=np.int64))
