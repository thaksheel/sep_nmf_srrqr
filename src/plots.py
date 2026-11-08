from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curves(results, out_dir: str | Path, metric: str = "accuracy") -> None:
    """One PDF per experiment: metric versus noise level, one line per method."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = results.df
    for name, part in df.groupby("experiment", sort=False):
        curves = part.groupby(["delta", "method"], sort=False)[metric].mean().unstack()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for method in part["method"].unique():
            style = "--" if ("RRQR" in method or "QRCP" in method) else "-"
            ax.plot(curves.index, curves[method], style, marker="o", ms=3, label=method)
        deltas = curves.index.to_numpy()
        if deltas.min() > 0 and deltas.max() / deltas.min() > 100:
            ax.set_xscale("log")
        ax.set(xlabel="noise level (delta)", ylabel=metric, title=part["title"].iloc[0])
        if metric in ("accuracy", "exact"):
            ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)
