"""Strong rank-revealing QR, Gu & Eisenstat (1996) Algorithm 4.

Port of Dirk Toewe's srrqr_l2r (2019), stopped at a target rank r with a tunable threshold f.
"""
import numpy as np
import numpy.linalg as la

from .core import Index, Matrix


def strong_rrqr(
    M: Matrix, r: int | None = None, f: float = 1.01, max_swaps: int = 10_000
) -> tuple[Matrix, Matrix, Index, int, int]:
    """Returns Q, R, P, rank, n_swaps with M[:, P] = Q @ R; the selected columns are P[:r]."""
    if not f >= 1:
        raise ValueError("f must be >= 1")
    M = np.array(M, dtype=np.float64, copy=True)
    m, n = M.shape
    Q, R, P = np.eye(m), M.copy(), np.arange(n, dtype=np.int64)
    R_norm = la.norm(R)
    if R_norm == 0:
        return Q, R, P, 0, 0

    AB = np.zeros_like(R)  # [inv(A), A \ B]
    v = la.norm(R, axis=0, keepdims=True)  # column norms of C
    rank = min(m, n)
    K = rank if r is None else min(int(r), rank)
    k = 0

    def pivot(j: int) -> None:
        R[:, [k, j]] = R[:, [j, k]]
        AB[:, [k, j]] = AB[:, [j, k]]
        P[[k, j]] = P[[j, k]]

    def eliminate(k: int) -> bool:
        nonlocal rank
        u = R[k:, k].copy()
        u[0] += ((u[0] > 0) * 2 - 1) * la.norm(u)
        if u[0] == 0:
            rank = k
            return False
        u /= np.abs(u).max()
        u *= np.sqrt(2) / la.norm(u)
        u = u.reshape(-1, 1)
        R[k:, :] -= u @ (u.T @ R[k:, :])
        Q[:, k:] -= (Q[:, k:] @ u) @ u.T
        R[k + 1 :, k] = 0
        return True

    def update() -> None:
        nonlocal k, v
        AB[k, k] = -1
        AB[: k + 1, k] /= -R[k, k]
        k += 1
        AB[:k, k:] += AB[:k, k - 1 : k] @ R[k - 1 : k, k:]
        v = la.norm(R[k:, k:], axis=0, keepdims=True)

    def downdate() -> None:
        nonlocal k
        AB[:k, k:] -= AB[:k, k - 1 : k] @ R[k - 1 : k, k:]
        k -= 1
        AB[: k + 1, k] *= -R[k, k]
        AB[k, k:] = 0

    ztol = np.finfo(float).eps * max(m, n) * R_norm
    n_swaps, done = 0, False

    while k < K and not done:
        pivot(k + int(v.argmax()))
        while True:
            if not eliminate(k):
                done = True
                break
            update()
            if k >= n:
                break

            # ratio[i, j] = |det A after swapping column i with column k + j| / |det A|
            ratio = np.hypot(AB[:k, k:], la.norm(AB[:k, :k], axis=1, keepdims=True) * v)
            i, j = np.unravel_index(ratio.argmax(), ratio.shape)
            if not ratio[i, j] > f or n_swaps >= max_swaps:
                if la.norm(v) <= ztol:
                    rank, k = k, K
                break

            j += k
            R[:, i : k - 1], R[:, k - 1] = R[:, i + 1 : k].copy(), R[:, i].copy()
            P[i : k - 1], P[k - 1] = P[i + 1 : k].copy(), P[i]
            AB[i : k - 1], AB[k - 1] = AB[i + 1 : k].copy(), AB[i].copy()

            for h in range(i, k - 1):  # re-triangulate A with Givens rotations
                c, s = R[h : h + 2, h]
                if s != 0:
                    hyp = np.hypot(c, s)
                    G = np.array([[c, s], [-s, c]]) / hyp
                    R[h : h + 2, h:] = G @ R[h : h + 2, h:]
                    AB[:, h : h + 2] = AB[:, h : h + 2] @ G.T
                    Q[:, h : h + 2] = Q[:, h : h + 2] @ G.T
            R[i:, i:k] = np.triu(R[i:, i:k])
            AB[i:, i:k] = np.triu(AB[i:, i:k])

            downdate()
            pivot(j)
            n_swaps += 1

    return Q, R, P, int(rank), n_swaps
