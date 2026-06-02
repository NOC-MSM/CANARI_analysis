"""
ensemble_stats

Sliding-window moving-block-bootstrap ANOVA tools for hierarchical
ensemble analysis.
"""

from importlib.metadata import version as _version
from importlib.metadata import (
    version as _version,
    PackageNotFoundError as _PackageNotFoundError,
)

try:
    __version__ = _version("ensemble-stats")
except _PackageNotFoundError:
    __version__ = "unknown"


from .analysis import (
    validate_input_dims,
    compute_STL_decomposition,
    remove_linear_trend,
    remove_seasonal_cycle,
    estimate_autocorrelation_timescale,
    estimate_mutual_information_timescale,
    define_block_length,
    moving_block_bootstrap_indices,
    compute_block_statistics_anova_numpy,
    sliding_window_MBB_ensemble_analysis,
    compute_relative_nonstationarity,
    compute_preprocessing_variance_statistics,
)

from .plotting import (
    plot_STL_decomposition,
    plot_groups_members,
    plot_preprocessing_components,
    plot_normalised_variance,
    plot_distribution_comparison,
    plot_variance_decomposition_timeseries,
    plot_fractional_variance_decomposition,
    plot_bootstrap_confidence_intervals,
    plot_gamma_time_var_bootstrap_median,
    plot_gamma_time_var_bootstrap_envelopes,
    plot_gamma_to_epsilon_snr_bootstrap_subset,
    plot_gamma_to_epsilon_snr_bootstrap,
    plot_member_spaghetti,
    plot_example_window_decomposition,
)

from .synthetic import (
    generate_synthetic_nested_ensemble,
)

try:
    __version__ = _version("ensemble-stats")
except _PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    # analysis
    "validate_input_dims",
    "compute_STL_decomposition",
    "remove_linear_trend",
    "remove_seasonal_cycle",
    "estimate_autocorrelation_timescale",
    "estimate_mutual_information_timescale",
    "define_block_length",
    "moving_block_bootstrap_indices",
    "compute_block_statistics_anova_numpy",
    "sliding_window_MBB_ensemble_analysis",
    "compute_relative_nonstationarity",
    "compute_preprocessing_variance_statistics",

    # plotting
    "plot_STL_decomposition",
    "plot_groups_members",
    "plot_preprocessing_components",
    "plot_normalised_variance",
    "plot_distribution_comparison",
    "plot_variance_decomposition_timeseries",
    "plot_fractional_variance_decomposition",
    "plot_bootstrap_confidence_intervals",
    "plot_gamma_time_var_bootstrap_median",
    "plot_gamma_time_var_bootstrap_envelopes",
    "plot_gamma_to_epsilon_snr_bootstrap_subset",
    "plot_gamma_to_epsilon_snr_bootstrap",
    "plot_member_spaghetti",
    "plot_example_window_decomposition",

    # synthetic
    "generate_synthetic_nested_ensemble",

    # metadata
    "__version__",
]
