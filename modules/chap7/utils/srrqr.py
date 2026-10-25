"""
Strong rank-revealing QR (Gu & Eisenstat, Algorithm 4), left-to-right variant.

Original implementation: Dirk Toewe, Dec 22, 2019.
Patched for benchmarking separable NMF:
  - `r`: stop at target rank r, so P[:r] satisfies the strong RRQR condition at rank r
    (without this, swaps at later steps k > r can move columns out of the first r slots).
  - `f`: swap threshold exposed as a parameter (f = inf gives plain QR with column pivoting).
  - `debug`: all expensive consistency checks only run when debug=True.
  - `max_swaps`: safety cap against cycling caused by round-off when f is close to 1.
  - A zero remaining column now stops the factorization cleanly instead of producing NaNs.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla


def srrqr_l2r(M, r=None, f=1.01, debug=False, max_swaps=10_000):
    """
    Returns Q, R, P, rank, n_swaps with M[:, P] = Q @ R.

    For column subset selection, the selected columns are P[:r].
    """
    if not f >= 1:
        raise ValueError("f must be >= 1")

    M = np.array(M, dtype=np.float64, copy=True)
    M.flags.writeable = False
    m, n = M.shape

    Q = np.eye(m, dtype=np.float64)
    R = M.copy()
    P = np.arange(n, dtype=np.int64)

    R_norm = la.norm(R)
    if R_norm == 0:
        return Q, R, P, 0, 0

    # R = [[A, B], [0, C]], A upper triangular k x k.
    # AB keeps inv(A) in its first k columns and A \ B in the rest.
    AB = np.zeros_like(R)
    v = la.norm(R, axis=0, keepdims=True)  # column norms of C

    rank = min(m, n)
    K = rank if r is None else min(int(r), rank)
    k = 0

    def pivot(j):
        R[:, [k, j]] = R[:, [j, k]]
        AB[:, [k, j]] = AB[:, [j, k]]
        P[[k, j]] = P[[j, k]]

    def eliminate(k):
        """Householder elimination of column k. Returns False if the column is zero."""
        nonlocal rank
        u = R[k:, k].copy()
        u[0] += ((u[0] > 0) * 2 - 1) * la.norm(u)
        if u[0] == 0:
            rank = k
            return False
        u /= np.abs(u).max()  # underflow-safe scaling
        u *= np.sqrt(2) / la.norm(u)
        u = u.reshape(-1, 1)
        R[k:, :] -= u @ (u.T @ R[k:, :])
        Q[:, k:] -= (Q[:, k:] @ u) @ u.T
        R[k + 1 :, k] = 0
        return True

    def update():
        nonlocal k, v
        AB[k, k] = -1
        AB[: k + 1, k] /= -R[k, k]
        k += 1
        AB[:k, k:] += AB[:k, k - 1 : k] @ R[k - 1 : k, k:]
        v = la.norm(R[k:, k:], axis=0, keepdims=True)

    def downdate():
        nonlocal k
        AB[:k, k:] -= AB[:k, k - 1 : k] @ R[k - 1 : k, k:]
        k -= 1
        AB[: k + 1, k] *= -R[k, k]
        if debug:
            assert np.allclose(-1, AB[k, k])
            assert np.allclose(0, AB[k, k + 1 :])
        AB[k, k:] = 0  # remove cancellation errors from row k

    ztol = np.finfo(R.dtype).eps * max(m, n) * R_norm
    n_swaps = 0
    done = False

    while k < K and not done:
        # Greedy pivot: largest remaining column norm (same as QRCP / SPA).
        pivot(k + int(v.argmax()))

        while True:
            if not eliminate(k):
                done = True  # remaining block is zero
                break
            update()
            if k >= n:
                break

            if debug:
                assert np.allclose(v, la.norm(R[k:, k:], axis=0, keepdims=True))
                assert np.allclose(Q @ Q.T, np.eye(m))
                assert np.allclose(M[:, P], Q @ R)
                assert np.allclose(0, np.tril(R[:, :k], -1))
                assert np.allclose(
                    sla.solve_triangular(R[:k, :k], R[:k, k:]), AB[:k, k:]
                )
                assert np.allclose(np.eye(k), R[:k, :k] @ AB[:k, :k])
                logdet_a = np.sum(np.log(np.abs(np.diag(R)[:k])))

            # Wm[i, j] = |det(A after swapping column i with column k+j)| / |det(A)|
            row_norms = la.norm(AB[:k, :k], axis=1, keepdims=True)
            Wm = np.hypot(AB[:k, k:], row_norms * v)
            i, j = np.unravel_index(Wm.argmax(), Wm.shape)
            F = Wm[i, j]

            if not F > f or n_swaps >= max_swaps:
                if la.norm(v) <= ztol:  # trailing block numerically zero
                    rank, k = k, K
                break

            j += k

            # Move column i to the right end of A (cyclic permutation).
            R[:, i : k - 1], R[:, k - 1] = R[:, i + 1 : k].copy(), R[:, i].copy()
            P[i : k - 1], P[k - 1] = P[i + 1 : k].copy(), P[i]
            AB[i : k - 1], AB[k - 1] = AB[i + 1 : k].copy(), AB[i].copy()

            # Re-triangulate A with Givens rotations.
            for h in range(i, k - 1):
                c, s = R[h : h + 2, h]
                if s != 0:
                    hyp = np.hypot(c, s)
                    c /= hyp
                    s /= hyp
                    if s != 0:
                        G = np.array([[c, s], [-s, c]])
                        R[h : h + 2, h:] = G @ R[h : h + 2, h:]
                        AB[:, h : h + 2] = AB[:, h : h + 2] @ G.T
                        Q[:, h : h + 2] = Q[:, h : h + 2] @ G.T

            R[i:, i:k] = np.triu(R[i:, i:k])
            AB[i:, i:k] = np.triu(AB[i:, i:k])

            downdate()
            pivot(j)
            n_swaps += 1

            if debug:
                logdet_new = np.log(np.abs(np.diag(R)[:k])).sum() + np.log(
                    la.norm(R[k:, k])
                )
                assert np.isclose(logdet_new, logdet_a + np.log(F))

    return Q, R, P, int(rank), n_swaps
