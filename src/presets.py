from __future__ import annotations

import numpy as np

from .data import SyntheticConfig
from .experiment import ExperimentConfig


def book_experiments(
    levels: int = 30, trials: int = 20, seed: int = 2020, m: int = 40, r: int = 20
) -> list[ExperimentConfig]:
    """The four experiments of Gillis, Section 7.4.6 (n = 200 mixtures, theta = 0.5, kappa = 1e6)."""

    def cfg(mixing: str, ill: bool) -> SyntheticConfig:
        return SyntheticConfig(
            m=m, r=r, n=200, mixing=mixing, ill_conditioned=ill, cond=6, theta=0.5
        )

    return [
        ExperimentConfig(
            "dd_well",
            "Dirichlet, well-conditioned W",
            cfg("dirichlet", False),
            np.linspace(0.0, 1.3, levels),
            trials,
            seed,
        ),
        ExperimentConfig(
            "mp_well",
            "Middle points, well-conditioned W",
            cfg("middle", False),
            np.linspace(0.0, 0.7, levels),
            trials,
            seed,
        ),
        ExperimentConfig(
            "dd_ill",
            "Dirichlet, ill-conditioned W",
            cfg("dirichlet", True),
            np.logspace(-5, 1, levels),
            trials,
            seed,
        ),
        ExperimentConfig(
            "mp_ill",
            "Middle points, ill-conditioned W",
            cfg("middle", True),
            np.logspace(-3, np.log10(2), levels),
            trials,
            seed,
        ),
    ]
