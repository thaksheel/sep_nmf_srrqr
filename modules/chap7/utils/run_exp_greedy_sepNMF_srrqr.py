"""
Drop-in replacement for run_exp_greedy_sepNMF with extra SRRQR variants.

Same call signature and return value as the original:
    results, timings = run_exp_greedy_sepNMF(m, n, r, xp, condW, delta, nummat, nalgo, diri)

Algorithm codes 1-6 behave exactly as in the original file; 7-12 are new:
    1  SPA                      7  QRCP (SRRQR with f = inf, no swaps)
    2  VCA                      8  SRRQR f = 1.01
    3  FAW                      9  SRRQR f = 1.1
    4  SNPA                    10  SRRQR f = 2
    5  MVE-SPA                 11  SRRQR f = 1.01, no L1 normalization
    6  SPA-SPA (precond.)      12  SRRQR f = 1.01 on SPA-preconditioned data

Optional keyword arguments (the driver script does not need them):
    seed=None            None keeps the original behaviour (fixed permutation, global RNG
                         untouched). An integer reseeds each trial with seed + t, so trial t
                         sees the same matrix at every noise level.
    return_details=False True adds a third return value with exact-recovery rates,
                         mean SRRQR swaps and failure counts.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.algorithms.separable_nmf.spa_matlab import spa_matlab, SPAOptions
from modules.algorithms.separable_nmf.snpa_matlab import snpa_matlab, SNPAOptions
from modules.algorithms.separable_nmf.separable_nmf import (
    vca as vca_light,
    faw as faw_light,
)
from synthdatasetSepNMF import synthdatasetSepNMF
from srrqr import srrqr_l2r  # patched version with r, f, debug

ALGO_NAMES: Dict[int, str] = {
    1: "SPA",
    2: "VCA",
    3: "FAW",
    4: "SNPA",
    5: "MVE-SPA",
    6: "SPA-SPA",
    7: "QRCP",
    8: "SRRQR f=1.01",
    9: "SRRQR f=1.1",
    10: "SRRQR f=2",
    11: "SRRQR f=1.01 raw",
    12: "SPA-SRRQR f=1.01",
}

# code -> (f, normalize, precondition)
SRRQR_VARIANTS: Dict[int, Tuple[float, bool, bool]] = {
    7: (np.inf, True, False),
    8: (1.01, True, False),
    9: (1.1, True, False),
    10: (2.0, True, False),
    11: (1.01, False, False),
    12: (1.01, True, True),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _l1_normalize(X: np.ndarray) -> np.ndarray:
    return X / (X.sum(axis=0, keepdims=True) + 1e-16)


def _spa(X: np.ndarray, r: int, normalize: int = 1) -> List[int]:
    return [
        int(k) for k in spa_matlab(X, r, SPAOptions(normalize=normalize, display=0))
    ]


def _spa_precondition(X: np.ndarray, r: int) -> np.ndarray:
    """pinv(X[:, K_SPA]) @ X, the preconditioning used by SPA-SPA."""
    return np.linalg.pinv(X[:, _spa(X, r)]) @ X


def _mve_preprocess(X: np.ndarray, r: int, mode: int) -> np.ndarray:
    """mode 0: none, mode 1: truncated SVD to r dims, mode 3: random projection to r dims."""
    if mode == 1:
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        return U[:, :r].T @ X
    if mode == 3:
        P = np.random.default_rng(0).standard_normal((r, X.shape[0]))
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-16
        return P @ X
    return X


def srrqr_indices(
    X: np.ndarray,
    r: int,
    f: float = 1.01,
    normalize: bool = True,
    precondition: bool = False,
) -> Tuple[List[int], int]:
    """Column selection K = P[:r] from a strong RRQR stopped at rank r. Returns (K, n_swaps)."""
    Y = _spa_precondition(X, r) if precondition else X
    if normalize:
        Y = _l1_normalize(Y)
    _, _, P, _, n_swaps = srrqr_l2r(Y, r=r, f=f)
    return [int(p) for p in P[:r]], int(n_swaps)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run_exp_greedy_sepNMF(
    m: int,
    n: int,
    r: int,
    xp: int,
    condW: int,
    delta: np.ndarray,
    nummat: int,
    nalgo: List[int],
    diri: float,
    seed: Optional[int] = None,
    return_details: bool = False,
):
    unknown = [a for a in nalgo if a not in ALGO_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown algorithm codes {unknown}; valid codes: {ALGO_NAMES}"
        )

    print("*************************************************")
    print(
        {
            1: "    Experiment: well-conditioned Dirichlet",
            2: "    Experiment: well-conditioned Middle points",
            3: "    Experiment: ill-conditioned Dirichlet",
            4: "    Experiment: ill-conditioned Middle points",
        }.get(xp, "    Experiment")
    )
    print("***********************************************")

    delta = np.asarray(delta, dtype=float)
    na, nd = len(nalgo), len(delta)
    results = np.zeros((na, nd), dtype=float)
    timings = np.zeros((na,), dtype=float)
    exact = np.zeros((na, nd), dtype=float)
    swap_sum = np.zeros((na, nd), dtype=float)
    failures = np.zeros((na,), dtype=int)

    print(f"Total number of noise levels: {nd:2.0f} ")
    for i, d in enumerate(delta):
        for t in range(nummat):
            if seed is not None:
                np.random.seed(seed + t)
                perm_rng = np.random.default_rng(seed + t)
            else:
                perm_rng = np.random.default_rng(0)  # original behaviour

            W, H, Noise = synthdatasetSepNMF(m, n, r, float(d), xp, condW, diri)
            Xt = W @ H + Noise
            permu = perm_rng.permutation(Xt.shape[1])
            Xt = Xt[:, permu]
            Kstar = set(np.where(permu < r)[0].tolist())

            ldr = 1  # MVE-SPA variant counter, as in the original
            for aidx, algo in enumerate(nalgo):
                n_swaps = None
                t0 = time.process_time()
                try:
                    if algo == 1:
                        K = _spa(Xt, r)
                    elif algo == 2:
                        K = list(vca_light(Xt, r, normalize="l2", random_state=0))
                    elif algo == 3:
                        K = list(faw_light(Xt, r, normalize="l1"))
                    elif algo == 4:
                        K, _ = snpa_matlab(
                            Xt, r, SNPAOptions(maxitn=500, proj=1, display=0)
                        )
                    elif algo == 5:
                        mode = {1: 0, 2: 1, 3: 3}.get(ldr, 0)
                        K = _spa(_mve_preprocess(Xt, r, mode), r)
                        ldr += 1
                    elif algo == 6:
                        K = _spa(_spa_precondition(Xt, r), r)
                    else:
                        f, normalize, precondition = SRRQR_VARIANTS[algo]
                        K, n_swaps = srrqr_indices(Xt, r, f, normalize, precondition)
                except Exception as exc:  # keep the run going; count as zero recovery
                    failures[aidx] += 1
                    print(
                        f"\n[{ALGO_NAMES[algo]}] failed at delta={d:.3g}, trial {t}: {exc}"
                    )
                    K = []
                timings[aidx] += time.process_time() - t0

                hits = len(set(int(k) for k in K) & Kstar)
                results[aidx, i] += hits / nummat / r
                exact[aidx, i] += (hits == r) / nummat
                if n_swaps is not None:
                    swap_sum[aidx, i] += n_swaps / nummat

        print(f"{i+1}...", end="", flush=True)
        if (i + 1) % 10 == 0:
            print()
    print()

    if failures.any():
        print(
            "Failures:", {ALGO_NAMES[a]: int(c) for a, c in zip(nalgo, failures) if c}
        )

    if not return_details:
        return results, timings

    is_srrqr = np.array([a in SRRQR_VARIANTS for a in nalgo])
    swaps = np.where(is_srrqr[:, None], swap_sum, np.nan)
    details = {
        "names": [ALGO_NAMES[a] for a in nalgo],
        "exact": exact,  # fraction of trials where all r vertices were found
        "swaps": swaps,  # mean SRRQR swaps per call (NaN for non-SRRQR methods)
        "failures": failures,
    }
    return results, timings, details


# --------------------------------------------------------------------------- #
# Sanity checks (optional, run before the full benchmark)
# --------------------------------------------------------------------------- #
def sanity_checks(m: int = 20, n: int = 60, r: int = 8, trials: int = 5) -> None:
    rng = np.random.default_rng(123)
    same = 0
    for _ in range(trials):
        X = rng.random((m, n))
        K_qr, _ = srrqr_indices(X, r, f=np.inf, normalize=False)
        same += set(K_qr) == set(_spa(X, r, normalize=0))
    print(f"QRCP vs SPA(normalize=0): {same}/{trials} identical index sets")

    for xp in (1, 3):
        W, H, Noise = synthdatasetSepNMF(m, n, r, 0.0, xp, 1, 1.0)
        K, s = srrqr_indices(W @ H + Noise, r, f=1.01)
        ok = set(K) == set(range(r))
        print(f"Noiseless recovery, xp={xp}: {'OK' if ok else 'FAILED'} (swaps={s})")
