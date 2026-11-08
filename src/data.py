from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from .core import Index, Matrix


@dataclass
class Dataset:
    X: Matrix
    r: int
    K_true: Index | None = None  # indices of the pure columns, when known
    W: Matrix | None = None  # true or reference W (m x r), when known
    name: str = "data"

    @classmethod
    def load(
        cls,
        path: str | Path,
        r: int,
        key: str = "X",
        ref_key: str | None = None,
        transpose: bool = False,
    ) -> "Dataset":
        """Load X (features x samples) and an optional reference W from .npy, .npz or .mat."""
        path = Path(path)
        if path.suffix == ".npy":
            store = {key: np.load(path)}
        elif path.suffix == ".npz":
            store = dict(np.load(path))
        elif path.suffix == ".mat":
            from scipy.io import loadmat

            store = loadmat(path)
        else:
            raise ValueError(f"Unsupported file type {path.suffix}")
        X = np.asarray(store[key], dtype=float)
        X = X.T if transpose else X
        W = np.asarray(store[ref_key], dtype=float) if ref_key else None
        return cls(X=X, r=r, W=W, name=path.stem)


@dataclass
class SyntheticConfig:
    m: int = 40
    r: int = 20
    n: int = 200  # mixed columns (Dirichlet only); middle points use all r(r-1)/2 pairs
    mixing: str = "dirichlet"  # "dirichlet" or "middle"
    ill_conditioned: bool = False
    cond: float = 6.0  # kappa(W) = 10**cond when ill-conditioned
    theta: float = 0.5  # Dirichlet parameter


class Synthetic:
    """Near-separable data X = W [I, H'] Pi + N as in Gillis, Section 7.4.6."""

    def __init__(self, cfg: SyntheticConfig | None = None):
        self.cfg = cfg or SyntheticConfig()

    def sample_w(self, rng: np.random.Generator) -> Matrix:
        c = self.cfg
        W = rng.random((c.m, c.r))
        if c.ill_conditioned:
            U, _, Vt = np.linalg.svd(W, full_matrices=False)
            W = (U * np.logspace(-c.cond, 0, c.r)) @ Vt
        return W

    def sample(self, delta: float, rng: np.random.Generator) -> Dataset:
        c = self.cfg
        W = self.sample_w(rng)
        if c.mixing == "dirichlet":
            H = np.hstack([np.eye(c.r), rng.dirichlet(np.full(c.r, c.theta), c.n).T])
            X = W @ H
            N = rng.standard_normal(X.shape)
            N *= delta * np.linalg.norm(X) / np.linalg.norm(N)
        elif c.mixing == "middle":
            pairs = list(combinations(range(c.r), 2))
            Hm = np.zeros((c.r, len(pairs)))
            for col, (i, j) in enumerate(pairs):
                Hm[[i, j], col] = 0.5
            H = np.hstack([np.eye(c.r), Hm])
            X = W @ H
            N = np.zeros_like(X)
            N[:, c.r :] = delta * (X[:, c.r :] - W.mean(axis=1, keepdims=True))
        else:
            raise ValueError(f"Unknown mixing {c.mixing!r}")

        perm = rng.permutation(X.shape[1])
        K_true = np.flatnonzero(perm < c.r)
        return Dataset(X=(X + N)[:, perm], r=c.r, K_true=K_true, W=W, name="synthetic")
