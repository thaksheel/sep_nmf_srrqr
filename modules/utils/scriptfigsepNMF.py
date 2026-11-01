import os
from typing import List, Dict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Legend labels and line styles per algorithm code (codes match run_exp_greedy_sepNMF_srrqr).
# Original methods use solid lines; QR/SRRQR variants use dashed lines.
LABELS = {
    1: "SPA",
    2: "VCA",
    3: "FastAnchorWords",
    4: "SNPA",
    5: "MVE-SPA",
    6: "SPA-SPA",
    7: "QRCP",
    8: "SRRQR f=1.01",
    9: "SRRQR f=1.1",
    10: "SRRQR f=2",
    11: "SRRQR f=1.01 (raw)",
    12: "SPA-SRRQR f=1.01",
}
MARKERS = {
    1: "o-",
    2: "s-",
    3: "x-",
    4: "d-",
    5: "^-.",
    6: "v-",
    7: "<--",
    8: "*--",
    9: "P--",
    10: "h--",
    11: "o--",
    12: ">--",
}
EXPERIMENTS: Dict[int, str] = {
    1: "dd_well_cond",
    2: "mp_well_cond",
    3: "dd_ill_cond",
    4: "mp_ill_cond",
}


def scriptfigsepNMF(
    results: np.ndarray,
    timings: np.ndarray,
    delta: np.ndarray,
    nalgo: List[int],
    xp: int,
    figs_dir: str,
    title: str, 
) -> None:
    """
    Plot success rate vs noise level for selected algorithms.
    Saves a PDF per experiment.
    """
    delta = np.asarray(delta, dtype=float)
    plt.figure(figsize=(7.5, 4.5))
    # Plot in the order given by nalgo, so every requested method appears,
    # including codes without a predefined label.
    for idx, a in enumerate(nalgo):
        lab = LABELS.get(a, f"algo {a}")
        plt.plot(delta, results[idx, :], MARKERS.get(a, "-"), label=lab, markersize=5, alpha=0.5)

    # Log-spaced noise levels (experiments 3 and 4) read better on a log x-axis.
    if delta.min() > 0 and delta.max() / delta.min() > 100:
        plt.xscale("log")

    plt.xlabel("noise level (delta)")
    plt.ylabel("success rate")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", alpha=0.3)
    # With up to 12 methods the legend goes outside the axes so it doesn't cover the curves.
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    os.makedirs(figs_dir, exist_ok=True)
    out = os.path.join(figs_dir, f"gns_nmf_{EXPERIMENTS[xp]}.pdf")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    # Timing summary in the console (total CPU seconds over all noise levels and trials).
    print(f"Total CPU time, experiment {xp}:")
    for idx, a in enumerate(nalgo):
        print(f"  {LABELS.get(a, f'algo {a}'):20s} {timings[idx]:8.2f} s")
