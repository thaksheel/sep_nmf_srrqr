import os, pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Union
import numpy as np

from ..utils.run_exp_greedy_sepNMF_srrqr import run_exp_greedy_sepNMF, ALGO_NAMES
from ..utils.scriptfigsepNMF import scriptfigsepNMF
from ..utils.silence_warnings import silence_numpy_warnings

ALGO_GROUPS: Dict[str, List[int]] = {
    "all": sorted(ALGO_NAMES),
    "baseline": [1, 2, 3, 4, 5, 6],
    "fast": [1, 6, 7, 8, 9, 10, 11, 12],  # everything except VCA, FAW, SNPA, MVE-SPA
    "srrqr": [7, 8, 9, 10, 11, 12],
}


@dataclass(frozen=True)
class ExperimentConfig:
    xp: int
    title: str
    delta: np.ndarray


def default_experiments(numnoiselevels: int = 20) -> Dict[int, ExperimentConfig]:
    """The four standard experiments, keyed by xp."""
    k = numnoiselevels
    return {
        1: ExperimentConfig(
            1, "Dirichlet Dist. Well Conditioned W", np.linspace(0.0, 1.3, k)
        ),
        2: ExperimentConfig(
            2, "Middle Point Well Conditioned W", np.linspace(0.0, 0.7, k)
        ),
        3: ExperimentConfig(
            3, "Dirichlet Dist. Ill-Conditioned W", np.logspace(-7, 1, k)
        ),
        4: ExperimentConfig(
            4, "Middle Point Ill-Conditioned W", np.logspace(-3, np.log10(2), k)
        ),
    }


def resolve_algos(algos: Iterable[Union[int, str]]) -> List[int]:
    """
    Turn a mix of codes (8), names ("SRRQR f=1.01", case-insensitive)
    and group names ("srrqr") into a list of unique codes, keeping order.
    """
    by_name = {name.lower(): code for code, name in ALGO_NAMES.items()}
    codes: List[int] = []
    for a in algos:
        if isinstance(a, (int, np.integer)) or (isinstance(a, str) and a.isdigit()):
            new = [int(a)]
        elif a.lower() in ALGO_GROUPS:
            new = ALGO_GROUPS[a.lower()]
        elif a.lower() in by_name:
            new = [by_name[a.lower()]]
        else:
            raise ValueError(
                f"Unknown algorithm {a!r}. Codes/names: {ALGO_NAMES}; groups: {list(ALGO_GROUPS)}"
            )
        for c in new:
            if c not in ALGO_NAMES:
                raise ValueError(f"Unknown algorithm code {c}. Valid: {ALGO_NAMES}")
            if c not in codes:
                codes.append(c)
    return codes


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    nalgo: List[int]
    results: np.ndarray  # (n_algos, n_delta) success rate
    timings: np.ndarray  # (n_algos,) total CPU seconds
    details: Optional[dict] = None  # exact recovery, swaps, failures

    @property
    def names(self) -> List[str]:
        return [ALGO_NAMES[a] for a in self.nalgo]

    def curve(self, algo: Union[int, str]) -> np.ndarray:
        """Success-rate curve of one algorithm, by code or name."""
        return self.results[self.nalgo.index(resolve_algos([algo])[0])]

    def summary(self) -> str:
        w = max(len(s) for s in self.names) + 2
        lines = [
            f"[xp={self.config.xp}] {self.config.title}",
            f"{'Algorithm':{w}}{'mean success':>14}{'CPU s':>10}",
        ]
        for name, row, t in zip(self.names, self.results, self.timings):
            lines.append(f"{name:{w}}{row.mean():14.3f}{t:10.2f}")
        return "\n".join(lines)


@dataclass
class StudyResults:
    """Everything produced by one call to main()."""

    settings: dict
    experiments: Dict[int, ExperimentResult] = field(default_factory=dict)

    def __getitem__(self, xp: int) -> ExperimentResult:
        return self.experiments[xp]

    def summary(self) -> str:
        return "\n\n".join(e.summary() for e in self.experiments.values())

    def to_dict(self) -> dict:
        """Plain dicts and arrays only, so the file loads without this module."""
        return {
            "settings": self.settings,
            "experiments": {
                xp: {
                    "xp": e.config.xp,
                    "title": e.config.title,
                    "delta": e.config.delta,
                    "nalgo": e.nalgo,
                    "names": e.names,
                    "results": e.results,
                    "timings": e.timings,
                    "details": e.details,
                }
                for xp, e in self.experiments.items()
            },
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.to_dict(), fh)

    @staticmethod
    def load(path: str) -> "StudyResults":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        study = StudyResults(settings=d["settings"])
        for xp, e in d["experiments"].items():
            cfg = ExperimentConfig(e["xp"], e["title"], e["delta"])
            study.experiments[xp] = ExperimentResult(
                cfg, e["nalgo"], e["results"], e["timings"], e["details"]
            )
        return study


def main(
    m: int = 40,
    r: int = 20,
    n: int = 200,
    condW: int = 6,
    nummat: int = 10,
    algos: Sequence[Union[int, str]] = ("fast",),
    diri: float = 1.0,  # larger than 0.5 due to smaller m
    numnoiselevels: int = 20,
    xps: Sequence[int] = (1, 2, 3, 4),
    deltas: Optional[Dict[int, np.ndarray]] = None,  # override noise grid per xp
    seed: int = 2020,
    paired_trials: bool = False,  # True: trial t uses seed + t at every noise level
    figs: Optional[str] = "./exports/base/",  # None: no plots
    save_path: Optional[str] = None,  # e.g. "./exports/base/results.pkl"
    verbose: bool = True,
) -> StudyResults:
    silence_numpy_warnings()
    np.random.seed(seed)

    nalgo = resolve_algos(algos)
    configs = default_experiments(numnoiselevels)
    for xp, d in (deltas or {}).items():
        configs[xp] = ExperimentConfig(
            xp, configs[xp].title, np.asarray(d, dtype=float)
        )

    unknown_xp = [xp for xp in xps if xp not in configs]
    if unknown_xp:
        raise ValueError(f"Unknown experiments {unknown_xp}; valid: {list(configs)}")

    study = StudyResults(
        settings=dict(
            m=m,
            r=r,
            n=n,
            condW=condW,
            nummat=nummat,
            nalgo=nalgo,
            diri=diri,
            numnoiselevels=numnoiselevels,
            xps=list(xps),
            seed=seed,
            paired_trials=paired_trials,
        )
    )

    for xp in xps:
        cfg = configs[xp]
        results, timings, details = run_exp_greedy_sepNMF(
            m,
            n,
            r,
            cfg.xp,
            condW,
            cfg.delta,
            nummat,
            nalgo,
            diri,
            seed=seed if paired_trials else None,
            return_details=True,
        )
        res = ExperimentResult(cfg, nalgo, results, timings, details)
        study.experiments[xp] = res

        if figs is not None:
            scriptfigsepNMF(results, timings, cfg.delta, nalgo, cfg.xp, figs, cfg.title)
        if verbose:
            print(res.summary(), "\n")

    if save_path is not None:
        study.save(save_path)
        if verbose:
            print(f"Saved results to {save_path}")
    return study


if __name__ == "__main__":
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
    study = main(
        m=40,
        r=20,
        n=200,
        condW=6,
        nummat=10,
        # Codes, names or groups ("all", "baseline", "fast", "srrqr"), mixed freely.
        algos=[1, 2, 3, 4, 5, 6],
        diri=1.0,
        numnoiselevels=20,
        xps=[1, 2, 3, 4],
        deltas=None,  # e.g. {3: np.logspace(-6, -2, 10)} to override a noise grid
        seed=2020,
        paired_trials=False,
        figs="./exports/base/",
        save_path=None,
        verbose=True,
    )
    print("Done.")
