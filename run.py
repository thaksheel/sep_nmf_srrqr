import numpy as np

from src import (
    FAW,
    SNPA,
    SPA,
    SRRQR,
    Dataset,
    Experiment,
    Precond,
    RealExperiment,
    Results,
    SNPAConfig,
    SRRQRConfig,
    book_experiments,
)

OUT = "exports/w45"

methods = [
    SPA(),
    FAW(),
    SNPA(SNPAConfig(iters=500)),
    Precond(SPA()),  # SPA-SPA
    SRRQR(SRRQRConfig(f=np.inf)),  # QRCP
    SRRQR(SRRQRConfig(f=1.01)),
    SRRQR(SRRQRConfig(f=2.0)),
    Precond(SRRQR(SRRQRConfig(f=1.01))),  # SPA-SRRQR
]

experiments = book_experiments(levels=30, trials=20)

results = Results.concat([Experiment(cfg).run(methods) for cfg in experiments])
results.save(f"{OUT}/results.csv")
results.save(f"{OUT}/summary.csv", summary=True)
results.plot(f"{OUT}/figs")
print(results.summary().to_string(index=False))

# Real data: X is features x samples, W is an optional reference (features x r).
# real = Dataset.load("data/samson.mat", r=3, key="V", ref_key="M")
# RealExperiment(real).run(methods).save(f"{OUT}/samson.csv")

print("END")