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
@dataclass
class PrecondConfig:
    kind: str = (
        "pinv"  # "pinv": pinv(X[:, K0]) with K0 from `first`; "svd": project onto top-r singular vectors
    )


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
        prefix = "SVD" if self.cfg.kind == "svd" else self.first.name
        self.name = f"{prefix}-{method.name}"

    def transform(self, X: Matrix, r: int) -> Matrix:
        if self.cfg.kind == "svd":
            U = np.linalg.svd(X, full_matrices=False)[0]
            return U[:, :r].T @ X
        K0 = self.first.select(X, r).K
        return np.linalg.pinv(X[:, K0]) @ X

    def select(self, X: Matrix, r: int) -> Selection:
        return self.method.select(self.transform(X, r), r)
