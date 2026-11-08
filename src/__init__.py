from .core import Selection, Selector, solve_h
from .data import Dataset, Synthetic, SyntheticConfig
from .experiment import (
    Experiment,
    ExperimentConfig,
    RealConfig,
    RealExperiment,
    Results,
)
from .methods import (
    FAW,
    SNPA,
    SPA,
    SRRQR,
    FAWConfig,
    Precond,
    PrecondConfig,
    SNPAConfig,
    SPAConfig,
    SRRQRConfig,
    VCA, 
    VCAConfig, 
)
from .presets import book_experiments
from .qr import strong_rrqr
