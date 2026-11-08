from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import metrics
from .core import Selection, Selector
from .data import Dataset, Synthetic, SyntheticConfig


def evaluate(method: Selector, data: Dataset, fit_iters: int = 0) -> dict:
    """Run one method on one data set and score it with every metric the data allows."""
    t0 = time.perf_counter()
    try:
        sel = method.select(data.X, data.r)
    except Exception as exc:
        sel = Selection(np.array([], dtype=np.int64), {"error": repr(exc)})
    row = {
        "method": method.name,
        "time_s": time.perf_counter() - t0,
        "swaps": sel.info.get("swaps", np.nan),
        "failed": "error" in sel.info,
    }
    K = sel.K
    if data.K_true is not None:
        row["accuracy"] = metrics.accuracy(K, data.K_true)
        row["exact"] = float(row["accuracy"] == 1.0)
    if data.W is not None and len(K):
        row["q2"] = metrics.q2(data.W, data.X, K)
        row["angle_deg"] = metrics.spectral_angle(data.W, data.X, K)
    if fit_iters and len(K):
        row["fit_error"] = metrics.fit_error(data.X, K, fit_iters)
    return row


# -------------------------------------------------------------------- synthetic
@dataclass
class ExperimentConfig:
    name: str
    title: str
    data: SyntheticConfig = field(default_factory=SyntheticConfig)
    deltas: NDArray[np.float64] = field(default_factory=lambda: np.linspace(0, 1, 10))
    trials: int = 20
    seed: int = 2020
    fit_iters: int = 0  # > 0 also records the fit error (slower)


class Experiment:
    """Accuracy of each method versus noise level on synthetic near-separable data."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.data = Synthetic(cfg.data)

    def run(self, methods: Sequence[Selector], progress: bool = True) -> "Results":
        c = self.cfg
        rows = []
        for i, delta in enumerate(c.deltas):
            for t in range(c.trials):
                # trial t sees the same W, H and noise direction at every delta (paired curves)
                data = self.data.sample(
                    float(delta), np.random.default_rng([c.seed, t])
                )
                for method in methods:
                    rows.append(
                        {
                            "experiment": c.name,
                            "title": c.title,
                            "delta": float(delta),
                            "trial": t,
                        }
                        | evaluate(method, data, c.fit_iters)
                    )
            if progress:
                print(
                    f"\r{c.name}: noise level {i + 1}/{len(c.deltas)}",
                    end="",
                    flush=True,
                )
        if progress:
            print()
        return Results(pd.DataFrame(rows))


# ------------------------------------------------------------------------- real
@dataclass
class RealConfig:
    fit_iters: int = 200


class RealExperiment:
    """Score each method on a real data set (fit error, and angle to a reference W if given)."""

    def __init__(self, data: Dataset, cfg: RealConfig | None = None):
        self.data, self.cfg = data, cfg or RealConfig()

    def run(self, methods: Sequence[Selector]) -> "Results":
        rows = [
            {"experiment": self.data.name, "title": self.data.name}
            | evaluate(m, self.data, self.cfg.fit_iters)
            for m in methods
        ]
        return Results(pd.DataFrame(rows))


# ---------------------------------------------------------------------- results
class Results:
    """One row per (experiment, noise level, trial, method)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def concat(cls, parts: Sequence["Results"]) -> "Results":
        return cls(pd.concat([p.df for p in parts], ignore_index=True))

    def curves(self, metric: str = "accuracy") -> pd.DataFrame:
        """Mean metric per (experiment, delta) x method."""
        return self.df.pivot_table(
            index=["experiment", "delta"],
            columns="method",
            values=metric,
            aggfunc="mean",
            sort=False,
        )

    def summary(self) -> pd.DataFrame:
        """One row per (experiment, method): mean of every metric, plus Table 7.2-style thresholds."""
        keys = ["experiment", "method"]
        numeric = [
            c
            for c in self.df.select_dtypes("number").columns
            if c not in ("delta", "trial")
        ]
        out = self.df.groupby(keys, sort=False)[numeric].mean().reset_index()
        if "delta" in self.df and "accuracy" in self.df:
            curve = self.df.groupby(keys + ["delta"])["accuracy"].mean().sort_index()
            for level, col in [(1.0, "max_delta_100"), (0.95, "max_delta_95")]:
                out[col] = [
                    _last_delta_above(curve.loc[e, m], level)
                    for e, m in zip(out.experiment, out.method)
                ]
        return out

    def save(self, path: str | Path, summary: bool = False) -> None:
        """Save the rows (or the summary) as .csv, .xlsx, .parquet or .pkl."""
        df = self.summary() if summary else self.df
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writers = {".csv": df.to_csv, ".xlsx": df.to_excel, ".parquet": df.to_parquet}
        if path.suffix == ".pkl":
            df.to_pickle(path)
        elif path.suffix in writers:
            writers[path.suffix](path, index=False)
        else:
            raise ValueError(f"Unsupported file type {path.suffix}")

    @classmethod
    def load(cls, path: str | Path) -> "Results":
        path = Path(path)
        readers = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".parquet": pd.read_parquet,
            ".pkl": pd.read_pickle,
        }
        return cls(readers[path.suffix](path))

    def plot(self, out_dir: str | Path, metric: str = "accuracy") -> None:
        from .plots import plot_curves

        plot_curves(self, out_dir, metric)


def _last_delta_above(curve: pd.Series, level: float) -> float:
    curve = curve.sort_index()
    ok = (curve >= level - 1e-12).to_numpy()
    if not ok[0]:
        return np.nan
    return float(curve.index[len(ok) - 1 if ok.all() else np.argmin(ok) - 1])
