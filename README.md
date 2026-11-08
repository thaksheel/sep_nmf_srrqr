# Separable NMF with strong rank-revealing QR

Code for studying **strong rank-revealing QR (SRRQR)** as a column-selection method for
separable NMF, compared with SPA, FAW and SNPA.

## Research

**Separable NMF.** The data matrix is assumed to have the form X ≈ X(:, K) H with H ≥ 0:
a few columns of X (pure pixels, anchor words) generate all the others as nonnegative
mixtures. The task is to find those r columns K from noisy data. Because the basis
vectors are real data points, the factorization is directly interpretable, and the
vertices of the data's convex hull are unique, so the answer is identifiable.

**SRRQR.** SPA is QR with column pivoting: it picks columns greedily to maximize the volume
of the selected set. SRRQR (Gu and Eisenstat, 1996) starts from that greedy choice and then
swaps a selected column for an unselected one whenever the swap increases the volume by more
than a factor `f`. It is a local search for maximum volume, in the same family as FAW and
N-FINDR, with a cheap test for every swap and a guaranteed stop.

**Findings so far.** On the four synthetic experiments of Gillis (*Nonnegative Matrix
Factorization*, Section 7.4.6), SNPA is best on Dirichlet data and SRRQR is best on the
adversarial middle-point data.

**Goals**

1. Prove identifiability and robustness guarantees for SRRQR: when it returns the true
   columns, and how its error depends on the noise level, the conditioning of W and `f`.
2. Design a preconditioned SRRQR whose guarantees do not depend on the conditioning of W.
3. Combine SNPA and SRRQR into a hybrid that is robust on both data types.
4. Test each guarantee on synthetic data, then on real data (hyperspectral images,
   documents).

## Code structure

```
run.py                 entry point: pick methods and experiments, run, save, plot
sepnmf/
  core.py              Selector base class, Selection result, simplex projections, H solver
  methods.py           SPA, FAW, SNPA, SRRQR, Precond and their config dataclasses
  qr.py                strong_rrqr: the strong RRQR kernel (Gu-Eisenstat, Algorithm 4)
  data.py              Dataset (load real data), SyntheticConfig, Synthetic (generator)
  experiment.py        ExperimentConfig, Experiment, RealExperiment, Results
  metrics.py           accuracy, q2 error, spectral angle, fit error
  plots.py             accuracy versus noise level, one PDF per experiment
  presets.py           book_experiments(): the four experiments with the book's settings
```

Every method has the same interface, `method.select(X, r)`, which returns the selected
indices `K` and extras such as the number of swaps. Variants are built by combining classes:

| Method | Code |
| --- | --- |
| SPA | `SPA()` |
| FAW | `FAW()`; `FAW(FAWConfig(passes=3))` for repeated passes |
| SNPA | `SNPA(SNPAConfig(iters=500))` |
| QRCP | `SRRQR(SRRQRConfig(f=np.inf))` |
| SRRQR | `SRRQR(SRRQRConfig(f=1.01))` |
| SPA-SPA | `Precond(SPA())` |
| SPA-SRRQR | `Precond(SRRQR())` |
| SVD-SRRQR | `Precond(SRRQR(), cfg=PrecondConfig(kind="svd"))` |

## Usage

Requirements: Python 3.10+, numpy, scipy, pandas, matplotlib (openpyxl for .xlsx).

```bash
python run.py
```

Edit the method list and the experiment settings at the top of `run.py`. A minimal run:

```python
from sepnmf import SPA, SRRQR, Experiment, Results, book_experiments

results = Results.concat([Experiment(cfg).run([SPA(), SRRQR()]) for cfg in book_experiments()])
results.save("exports/results.csv")                 # one row per trial
results.save("exports/summary.csv", summary=True)   # one row per experiment and method
results.plot("exports/figs")
```

Real data (features x samples, with an optional reference W):

```python
from sepnmf import Dataset, RealExperiment

data = Dataset.load("data/samson.mat", r=3, key="V", ref_key="M")
RealExperiment(data).run([SPA(), SRRQR()]).save("exports/samson.csv")
```

## Metrics

| Column | Meaning |
| --- | --- |
| `accuracy` | Fraction of the true columns recovered |
| `exact` | 1 if all r true columns were recovered |
| `q2` | max over true columns of the distance to the nearest selected column (Gillis, eq. 7.1) |
| `angle_deg` | Mean angle between W and the selected columns after matching |
| `fit_error` | ‖X − X(:, K) H‖ / ‖X‖ with H in the capped simplex (real data) |
| `max_delta_100`, `max_delta_95` | Largest noise level with 100% and 95% accuracy (as in Table 7.2) |

## Notes

- No method normalizes columns by default. The ill-conditioned W of the book has negative
  entries, and L1 normalization breaks the geometry on that data.
- Trial `t` uses the same seed at every noise level, so all methods see the same matrices
  and the curves are paired.
- The synthetic generator follows Section 7.4.6, except that it does not redraw W until
  γ₂(W) ≥ 10⁻ᶜᵒⁿᵈ.
