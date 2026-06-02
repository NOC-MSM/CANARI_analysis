# ensemble_stats

`ensemble_stats` is a lightweight Python package for statistical analysis of hierarchical ensemble timeseries, with particular focus on climate-model large ensembles and predictability studies.

The package currently includes:

- sliding-window moving-block-bootstrap (SWMBB) analysis
- hierarchical ANOVA variance decomposition
- synthetic ensemble generation
- plotting utilities for ensemble diagnostics
- support for temporally correlated ensemble variability

The code was developed for analysis of Arctic sea-ice predictability within the CANARI project.

Changelog for this notebook and accompanying ensemble_stats package:

11 May 2026: v0.1.0 - Chris Wilson - first release version.

14 May 2026: v0.1.1 - Chris Wilson - bugfixes and extra plots in this notebook.

26 May 2026: v0.2 - Chris Wilson - bugfixes and tidying.

2 June 2026: v0.2.1 - CW - fixed bug in pyproject.toml, tidied environment.yml, tidied notebook.

OpenAI ChatGPT was used to assist with aspects of code development and refinement.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/NOC-MSM/CANARI_analysis.git
```

Move to the package directory:

```bash
cd CANARI_analysis/synth-rapid-ice-loss/python
```

Create the Conda environment:

```bash
conda env create -f environment.yml
```

This installs all required dependencies and automatically installs the package in editable mode.

Activate the environment:

```bash
conda activate ensemble-stats
```

Verify the installation:

```bash
python -c "import ensemble_stats; print(ensemble_stats.__version__)"
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
│   ├── ANOVA_MBB_draft.ipynb
│   └── v0.2.1_SWMBB_ANOVA_CANARI_LE.ipynb
│
├── scripts/
│   └── fig_sep_SIE_trends_RILEs_overview.py
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

- sliding-window moving-block-bootstrap (SWMBB) analysis
- hierarchical ANOVA decomposition
- variance partitioning
- signal-to-noise diagnostics
- trend and seasonal-cycle removal
- STL decomposition
- preprocessing utilities

---

## `ensemble_stats.synthetic`

Synthetic hierarchical ensemble generators for testing and demonstration.

Supports configurable synthetic ensemble generation, including:

- nested ensemble structure `(j, k, t)`
- seasonal cycles
- temporal persistence
- macro-state variability
- member variability
- stochastic residual noise

---

## `ensemble_stats.plotting`

Plotting utilities for:

- ensemble spaghetti plots
- ANOVA variance diagnostics
- STL decomposition visualisation
- bootstrap confidence intervals
- predictability and signal-to-noise diagnostics

---

# Demonstration notebook

The primary demonstration notebook is:

```text
notebooks/v0.2_SWMBB_ANOVA_CANARI_LE.ipynb
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
import ensemble_stats as es

# Generate a synthetic ensemble
g = es.generate_synthetic_nested_ensemble(...)

# Run SWMBB ANOVA analysis
results = es.sliding_window_MBB_ensemble_analysis(
    g,
    ...
)

# Plot diagnostics
es.plot_member_spaghetti(g)
es.plot_variance_decomposition_timeseries(results)
```

See the demonstration notebook for a complete, reproducible example.

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

Additional development, documentation, notebook, and I/O dependencies are provided through `environment.yml`.

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
