# ensemble_stats

`ensemble_stats` is a lightweight Python package for statistical analysis of hierarchical ensemble timeseries, with particular focus on climate-model large ensembles and predictability studies.

The package currently includes:

- sliding-window moving-block-bootstrap (SWMBB) analysis
- hierarchical ANOVA variance decomposition
- synthetic ensemble generation
- plotting utilities for ensemble diagnostics
- support for temporally correlated ensemble variability

The code was developed for analysis of Arctic sea-ice predictability within the CANARI project.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/NOC-MSM/CANARI_analysis.git
```

Move to the package directory:

```bash
cd CANARI_analysis/synth-rapid-ice-loss/python
```

Create the conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate anova_xarray_env
```

Install the package in editable mode:

```bash
pip install -e .
```

---

# Quick test

Launch Python or Jupyter and test:

```python
import ensemble_stats

from ensemble_stats.analysis import *
from ensemble_stats.synthetic import *
from ensemble_stats.plotting import *
```

---

# Package structure

```text
python/
│
├── environment.yml
├── pyproject.toml
├── README.md
│
├── notebooks/
│   └── v0.2_SWMBB_ANOVA_CANARI_LE.ipynb [demonstrating functionality]
│
├── scripts/
│
├── src/
│   └── ensemble_stats/
│       ├── __init__.py
│       ├── analysis.py
│       ├── plotting.py
│       └── synthetic.py
│
└── tests/
```

---

# Main modules

## `ensemble_stats.analysis`

Core statistical routines, including:

- sliding-window moving-block-bootstrap analysis
- hierarchical ANOVA decomposition
- variance partitioning
- signal-to-noise diagnostics
- preprocessing utilities

---

## `ensemble_stats.synthetic`

Synthetic hierarchical ensemble generators for testing and demonstration.

Supports:
- nested ensemble structure `(j, k, t)`
- seasonal cycles
- AR(2) persistence
- macro-state variability
- member variability
- stochastic residual noise
- 360-day calendars

---

## `ensemble_stats.plotting`

Plotting utilities for:
- ensemble spaghetti plots
- ANOVA variance diagnostics
- STL decomposition
- confidence intervals
- predictability metrics

---

# Demonstration notebook

The primary demonstration notebook is:

```text
notebooks/SWMBB_ANOVA_CANARI_LE.ipynb
```

This notebook demonstrates:

- generation of synthetic ensembles
- preprocessing workflows
- STL decomposition
- SWMBB analysis
- ANOVA variance decomposition
- bootstrap confidence intervals
- predictability diagnostics
- publication-style figures

It serves as the main worked example for the package.

---

# Typical workflow

```python
from ensemble_stats.synthetic import *
from ensemble_stats.analysis import *
from ensemble_stats.plotting import *

# Generate synthetic ensemble
g = generate_synthetic_nested_ensemble()

# Run SWMBB ANOVA analysis
results = sliding_window_MBB_ensemble_analysis(
    g,
    window_length=240,
    bootstrap=True,
)

# Plot diagnostics
plot_member_spaghetti(g)
```

---

# Dependencies

Main runtime dependencies include:

- numpy
- pandas
- xarray
- scipy
- matplotlib
- statsmodels
- scikit-learn
- cftime

---

# Development status

This package is currently research-oriented and under active development.

Interfaces and function signatures may evolve.

---

# License

See `LICENSE`.

---

# Acknowledgements

Parts of the development workflow, documentation, and code refinement were assisted using OpenAI ChatGPT.
