# Changelog:
 
# 11 May 2026: v0.1.0 - Chris Wilson - first release version.
 
# 14 May 2026: v0.1.1 - Chris Wilson - bugfixes and extra plots in the notebook.

# 26 May 2026: v0.2 - Chris Wilson - bugfixes and tidying.

# OpenAI ChatGPT was used to assist with aspects of code development and refinement.



import matplotlib.pyplot as plt
import numpy as np


def plot_STL_decomposition(
    ds_stl,
    t_dim="time",
    aspect=0.4,
):
    """
    Plot STL decomposition components of the grand ensemble-mean.

    Parameters
    ----------
    ds_stl : xarray.Dataset
        Dataset containing:
        - STL_trend
        - STL_seasonal
        - STL_resid
    t_dim : str, optional
        Name of temporal dimension.
    aspect : float, optional
        Axes box aspect ratio.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects.
    """

    # Extract STL components
    trend = ds_stl.STL_trend
    seasonal = ds_stl.STL_seasonal
    resid = ds_stl.STL_resid

    trend_plus_seasonal = trend + seasonal
    alert = resid + 1

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ------------------------------------------------------
    # Trend
    # ------------------------------------------------------
    trend.plot(ax=axes[0, 0])
    axes[0, 0].set_title("trend")

    # ------------------------------------------------------
    # Seasonal component
    # ------------------------------------------------------
    seasonal.plot(ax=axes[0, 1])
    axes[0, 1].set_title("seasonal")

    # ------------------------------------------------------
    # Trend + seasonal
    # ------------------------------------------------------
    trend_plus_seasonal.plot(
        ax=axes[1, 0],
        label="trend + seasonal",
    )

    alert.plot(
        ax=axes[1, 0],
        color="grey",
        label="residual + $10^6$ km$^2$",
    )

    axes[1, 0].axhline(
        y=1,
        color="red",
        linestyle="dotted",
        linewidth=1,
        alpha=0.7,
    )

    axes[1, 0].legend()

    axes[1, 0].set_title("trend + seasonal")

    # ------------------------------------------------------
    # Residual
    # ------------------------------------------------------
    resid.plot(
        ax=axes[1, 1],
        color="grey",
    )

    axes[1, 1].set_title(
        "residual of the temporal decomposition\n"
        "of the grand ensemble-mean timeseries"
    )

    # ------------------------------------------------------
    # Shared formatting
    # ------------------------------------------------------
    for ax in axes.flat:
        ax.set_xlabel("year")
        ax.set_ylabel("Sea-ice extent ($10^6$ km$^2$)")
        ax.set_box_aspect(aspect)

    # Overall title
    fig.suptitle(
        "Temporal components of the grand ensemble-mean,\n"
        "monthly-mean Arctic sea-ice extent ($10^6$ km$^2$)",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def plot_groups_members(
    g,
    ntimes=1800,
    jsel=0,
    ksel=0,
    j_dim="j",
    k_dim="k",
    t_dim="time",
):
    """
    Plot hierarchical ensemble-member timeseries and highlight
    one selected member.

    Parameters
    ----------
    g : xarray.DataArray
        Ensemble dataset with dimensions (j, k, time).
    ntimes : int, optional
        Number of time samples to display.
    jsel : int, optional
        Index of highlighted parent group.
    ksel : int, optional
        Index of highlighted child member.
    j_dim, k_dim, t_dim : str, optional
        Names of group, member, and temporal dimensions.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    """

    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    # Subset in time
    g_subset = g.isel({t_dim: slice(0, ntimes)})

    # Highlighted indices
    j_highlight = jsel
    k_highlight = ksel

    # Colour map
    cmap = plt.get_cmap("tab10")
    colors = [
        cmap(j)
        for j in range(g_subset.sizes[j_dim])
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot all members except highlighted member
    for j in range(g_subset.sizes[j_dim]):

        color = colors[j]

        if j == j_highlight:
            alpha = 0.8
            lw = 0.4
        else:
            alpha = 0.3
            lw = 0.2

        for k in range(g_subset.sizes[k_dim]):

            if k != k_highlight:

                ax.plot(
                    g_subset[t_dim].values,
                    g_subset.isel({j_dim: j, k_dim: k}).values,
                    color=color,
                    alpha=alpha,
                    linewidth=lw,
                )

    # Highlight selected member
    ax.plot(
        g_subset[t_dim].values,
        g_subset.isel({
            j_dim: j_highlight,
            k_dim: k_highlight,
        }).values,
        color='black',
        linewidth=1.0,
        linestyle=":",
        alpha=1.0,
        zorder=10,
    )

    ax.set_title(
        f"First {ntimes} samples of hierarchical ensemble dataset"
    )

    ax.set_xlabel(t_dim)
    ax.set_ylabel("Data value")

    ax.grid(True)

    # --------------------------------------------------
    # Legend 1: groups
    # --------------------------------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            color=colors[j],
            lw=3 if j == j_highlight else 2,
            alpha=1.0 if j == j_highlight else 0.6,
            label=f"{j_dim}={j}",
        )
        for j in range(g_subset.sizes[j_dim])
    ]

    legend1 = ax.legend(
        handles=legend_handles,
        title="Groups",
        loc="lower right",
    )

    # --------------------------------------------------
    # Legend 2: highlighted member
    # --------------------------------------------------
    selected_handle = Line2D(
        [0], [0],
        color='black',
        linewidth=1,
        linestyle=":",
        label=(
            f"Selected member "
            f"({j_dim}={j_highlight}, "
            f"{k_dim}={k_highlight})"
        ),
    )

    ax.legend(
        handles=[selected_handle],
        loc="upper right",
        handlelength=4,
    )

    # Keep both legends
    ax.add_artist(legend1)

    plt.tight_layout()

    return fig, ax



    
def plot_preprocessing_components(
    g,
    g_detrended,
    g_preprocessed,
    j_example=0,
    k_example=0,
    j_dim="j",
    k_dim="k",
    t_dim="t",
):
    """
    Plot preprocessing effects on:
    (1) the grand-ensemble mean; and
    (2) one example ensemble member.
    """

    grand_ensemble_mean = g.mean(dim=[j_dim, k_dim])
    grand_mean_detrended = g_detrended.mean(dim=[j_dim, k_dim])
    grand_mean_final = g_preprocessed.mean(dim=[j_dim, k_dim])

    example_member_original = g.sel(
        {j_dim: j_example, k_dim: k_example}
    )

    example_member_detrended = g_detrended.sel(
        {j_dim: j_example, k_dim: k_example}
    )

    example_member_final = g_preprocessed.sel(
        {j_dim: j_example, k_dim: k_example}
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        sharex=True,
    )

    # ----------------------------------------------------------
    # Top panel: grand-ensemble mean evolution
    # ----------------------------------------------------------
    axes[0].plot(
        grand_ensemble_mean[t_dim],
        grand_ensemble_mean,
        label="Original grand-ensemble mean",
    )

    axes[0].plot(
        grand_mean_detrended[t_dim],
        grand_mean_detrended,
        label="After detrending",
    )

    axes[0].plot(
        grand_mean_final[t_dim],
        grand_mean_final,
        label="After seasonality removal",
    )

    axes[0].set_title(
        "Effect of preprocessing on grand-ensemble mean"
    )
    axes[0].set_ylabel("Grand-ensemble mean before and after preprocessing steps")
    axes[0].legend()

    # ----------------------------------------------------------
    # Bottom panel: one example member through preprocessing
    # ----------------------------------------------------------
    axes[1].plot(
        example_member_original[t_dim],
        example_member_original,
        label=f"Original member ({j_dim}={j_example}, {k_dim}={k_example})",
    )

    axes[1].plot(
        example_member_detrended[t_dim],
        example_member_detrended,
        label="After detrending",
    )

    axes[1].plot(
        example_member_final[t_dim],
        example_member_final,
        label="After seasonality removal",
    )

    axes[1].axhline(
        0.0,
        color="k",
        linestyle="--",
        linewidth=1,
    )

    axes[1].set_title(
        "Effect of preprocessing on an example ensemble member"
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Member before and after preprocessing steps")
    axes[1].legend()

    plt.tight_layout()

    return fig, axes

# --------------------------------------------------------------
# Compare across-ensemble variance through time before and after
# preprocessing
# --------------------------------------------------------------



def plot_normalised_variance(
    normalised_variance_original,
    normalised_variance_residual,
    t_dim="time"):
    """Plot normalised grand-ensemble variance diagnostics through time."""
    fig, ax = plt.subplots(figsize=(10, 5))

    normalised_variance_original.plot(ax=ax, label="Normalised grand-ensemble variance of g", alpha=0.5, color='red')
    normalised_variance_residual.plot(ax=ax, label="Normalised grand-ensemble variance of residual component of g", alpha=1, color='grey',linewidth=1)

    ax.legend()
    ax.set_title("Pre- and post-processed grand-ensemble variance timeseries, normalised by their time-mean variance")

    plt.tight_layout()

    return fig, ax

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_distribution_comparison(g_original, g_preprocessed, bins=40):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Flatten data
    x1 = g_original.values.ravel()
    x2 = g_preprocessed.values.ravel()

    # Remove NaNs if needed
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    # Histograms as densities
    ax.hist(
        x1,
        bins=bins,
        density=True,
        alpha=0.4,
        label="Original"
    )

    ax.hist(
        x2,
        bins=bins,
        density=True,
        alpha=0.4,
        label="Preprocessed",
        color="green"
    )

    # KDE / PDF estimates
    xs = np.linspace(
        min(x1.min(), x2.min()),
        max(x1.max(), x2.max()),
        500
    )

    kde1 = gaussian_kde(x1)
    kde2 = gaussian_kde(x2)

    ax.plot(xs, kde1(xs), linewidth=2, label="Original PDF (Gaussian KDE)")
    ax.plot(xs, kde2(xs), linewidth=2, color="green", label="Preprocessed PDF (Gaussian KDE)")

    ax.legend()
    ax.set_title("Distribution comparison")
    ax.set_ylabel("Density")

    plt.tight_layout()

    return fig, ax



def plot_variance_decomposition_timeseries(results, t_coord=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    for col in [
        "variance_between_j",
        "variance_between_k_within_j",
        "residual_variance",
        "total_variance",
    ]:
        if col in results:
            ax.plot(results[col], label=col)

    ax.legend()
    ax.set_title("Variance decomposition through time")

    plt.tight_layout()
    
    return fig, ax


def plot_fractional_variance_decomposition(results, t_coord=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    for col in [
        "fraction_variance_between_j",
        "fraction_variance_between_k_within_j",
        "fraction_residual_variance",
    ]:
        if col in results:
            ax.plot(results[col], label=col)

    ax.legend()
    ax.set_title("Fractional variance decomposition")

    plt.tight_layout()
    
    return fig, ax


def plot_bootstrap_confidence_intervals(results, component="fraction_variance_between_j", t_coord=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(results[component], label=component)

    if f"{component}_lower" in results and f"{component}_upper" in results:
        ax.fill_between(
            np.arange(len(results)),
            results[f"{component}_lower"],
            results[f"{component}_upper"],
            alpha=0.3,
        )

    ax.legend()
    ax.set_title(f"Bootstrap confidence intervals: {component}")

    plt.tight_layout()
    
    return fig, ax

def plot_gamma_time_var_bootstrap_median(
    results,
    var_name="gamma_time_var_boot",
    cmap="viridis",
    figsize=(12, 5),
    xtick_step=60):
    """
    Plot bootstrap-median temporal variance field.

    Parameters
    ----------
    results : xarray.Dataset
        Output dataset from
        sliding_window_MBB_ensemble_analysis().

    var_name : str, default="gamma_time_var_boot"
        Name of bootstrap ensemble variable.

    cmap : str, default="viridis"
        Matplotlib colormap.

    figsize : tuple, default=(12, 5)
        Figure size.

    xtick_step : int, default=60
        Spacing between x-axis ticks.
        For monthly data, 60 corresponds to 5 years.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # ------------------------------------------------------
    # Bootstrap ensemble
    # Dimensions:
    # (boot, j, time)
    # ------------------------------------------------------
    boot = (
        results[var_name]
        .transpose("boot", "j", "time")
    )

    # ------------------------------------------------------
    # Median across bootstrap dimension
    # Result dims:
    # (j, time)
    # ------------------------------------------------------
    p50 = boot.quantile(
        0.5,
        dim="boot",
    )

    # ----------------------------------------------------------
    # Font scaling
    # ----------------------------------------------------------
    base_tick_size = 10
    base_colorbar_size = 10
    base_title_size = 12

    tick_size = base_tick_size * 1.5
    colorbar_size = base_colorbar_size * 1.5
    title_size = base_title_size * 1.2

    # ------------------------------------------------------
    # Figure
    # ------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=figsize
    )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------
    im = p50.plot(
        ax=ax,
        cmap=cmap,
        cbar_kwargs={
            "label": r"$\sigma_\gamma^2$",
        },
    )

    # ------------------------------------------------------
    # Colorbar formatting
    # ------------------------------------------------------
    cbar = im.colorbar

    cbar.ax.tick_params(
        labelsize=colorbar_size
    )

    cbar.set_label(
        r"$\sigma_\gamma^2$",
        fontsize=colorbar_size,
    )

    # ------------------------------------------------------
    # Labels/titles
    # ------------------------------------------------------
    ax.set_ylabel(
        "Group j",
        fontsize=tick_size,
    )

    ax.set_xlabel(
        "Year",
        fontsize=tick_size,
    )

    ax.set_title(
        "Time-evolving macro-state temporal variance "
        "(bootstrap median)",
        fontsize=title_size,
    )

    # ------------------------------------------------------
    # Tick every N samples
    # ------------------------------------------------------
    tick_idx = np.arange(
        0,
        len(results["time"]),
        xtick_step,
    )

    tick_times = (
        results["time"]
        .values[tick_idx]
    )

    # Major ticks
    ax.set_xticks(
        tick_times
    )

    # Labels
    ax.set_xticklabels(
        [
            f"{t.year:04d}"
            for t in tick_times
        ],
        rotation=-90,
        ha="center",
    )

    # ----------------------------------------------------------
    # Tick formatting
    # ----------------------------------------------------------
    ax.tick_params(
        axis="both",
        labelsize=tick_size,
    )

    # ------------------------------------------------------
    # Y-grid at cell edges
    # ------------------------------------------------------
    j_vals = results["j"].values

    y_edges = np.arange(
        j_vals.min() - 0.5,
        j_vals.max() + 1.5,
        1,
    )

    ax.set_yticks(j_vals)

    ax.set_yticks(
        y_edges,
        minor=True,
    )

    # ------------------------------------------------------
    # Grid configuration
    # ------------------------------------------------------
    # X-grid at cell centres
    ax.grid(
        which="major",
        axis="x",
        color="white",
        linewidth=1.0,
        alpha=0.5,
    )

    # Y-grid at cell edges
    ax.grid(
        which="minor",
        axis="y",
        color="white",
        linewidth=1.5,
    )

    # ------------------------------------------------------
    # Tight axis/layout
    # ------------------------------------------------------
    ax.axis("tight")

    plt.tight_layout()

    return fig, ax


def plot_gamma_time_var_bootstrap_envelopes(
    results,
    var_name="gamma_time_var_boot",
    cmap_name="tab10",
    figsize=(12, 5),
    year_tick_spacing=5,
):
    """
    Plot bootstrap median and percentile envelopes
    for gamma_time_var_boot.

    Parameters
    ----------
    results : xarray.Dataset
        Output dataset from
        sliding_window_MBB_ensemble_analysis().

    var_name : str, default="gamma_time_var_boot"
        Name of bootstrap ensemble variable.

    cmap_name : str, default="tab10"
        Matplotlib colormap name.

    figsize : tuple, default=(12, 5)
        Figure size.

    year_tick_spacing : int, default=5
        Spacing between x-axis year ticks.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=figsize
    )

    cmap = plt.get_cmap(
        cmap_name
    )

    time = results["time"].values

    # ----------------------------------------------------------
    # Font scaling
    # ----------------------------------------------------------
    base_tick_size = 10
    base_legend_size = 9
    base_title_size = 12

    tick_size = base_tick_size * 1.5
    legend_size = base_legend_size * 1.5
    title_size = base_title_size * 1.2

    # ----------------------------------------------------------
    # Loop over groups
    # ----------------------------------------------------------
    for j in results["j"].values:

        color = cmap(j)

        # ------------------------------------------------------
        # Bootstrap ensemble
        # Shape: (boot, time)
        # ------------------------------------------------------
        boot = (
            results[var_name]
            .sel(j=j)
            .transpose("boot", "time")
            .values
        )

        # ------------------------------------------------------
        # Percentile intervals
        # ------------------------------------------------------
        p05 = np.percentile(
            boot,
            5,
            axis=0,
        )

        p25 = np.percentile(
            boot,
            25,
            axis=0,
        )

        p75 = np.percentile(
            boot,
            75,
            axis=0,
        )

        p95 = np.percentile(
            boot,
            95,
            axis=0,
        )

        # ------------------------------------------------------
        # 90% interval
        # ------------------------------------------------------
        ax.fill_between(
            time,
            p05,
            p95,
            color=color,
            alpha=0.15,
        )

        # ------------------------------------------------------
        # 50% interval
        # ------------------------------------------------------
        ax.fill_between(
            time,
            p25,
            p75,
            color=color,
            alpha=0.35,
        )

        # ------------------------------------------------------
        # Median estimate
        # ------------------------------------------------------
        boot_median = np.percentile(
            boot,
            50,
            axis=0,
        )

        ax.plot(
            time,
            boot_median,
            color=color,
            linewidth=2.5,
            label=f"j={j}",
        )

    # ----------------------------------------------------------
    # Axis labels and title
    # ----------------------------------------------------------
    ax.set_ylabel(
        r"$\sigma_\gamma^2$",
        fontsize=tick_size,
    )

    ax.set_title(
        "Macro-state temporal variance "
        "with bootstrap median and "
        "50%, 90% percentile envelopes",
        fontsize=title_size,
    )

    # ----------------------------------------------------------
    # X-axis ticks
    # ----------------------------------------------------------
    years = np.array(
        [t.year for t in time]
    )

    tick_years = np.arange(
        (years.min() // year_tick_spacing)
        * year_tick_spacing,
        (
            (years.max() // year_tick_spacing)
            + 1
        )
        * year_tick_spacing,
        year_tick_spacing,
    )

    # ----------------------------------------------------------
    # Representative positions
    # ----------------------------------------------------------
    tick_positions = [
        time[
            np.argmin(
                np.abs(years - y)
            )
        ]
        for y in tick_years
    ]

    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        [
            f"{y:04d}"
            for y in tick_years
        ],
        rotation=-90,
        ha="center",
    )

    # ----------------------------------------------------------
    # Tick formatting
    # ----------------------------------------------------------
    ax.tick_params(
        axis="both",
        labelsize=tick_size,
    )

    # ----------------------------------------------------------
    # Grid and legend
    # ----------------------------------------------------------
    ax.grid(True)

    ax.legend(
        ncol=2,
        fontsize=legend_size,
    )

    # ----------------------------------------------------------
    # Layout
    # ----------------------------------------------------------
    plt.tight_layout()

    plt.show()

    return fig, ax

def plot_gamma_to_epsilon_snr_bootstrap_subset(
    results,
    j_subset=None,
    gamma_var="gamma_time_var",
    gamma_boot_var="gamma_time_var_boot",
    epsilon_var="epsilon_time_var",
    epsilon_boot_var="epsilon_time_var_boot",
    cmap_name="tab10",
    figsize=(12, 5),
    xtick_step=60,
    save_path=None,
):
    """
    Plot macro-state signal-to-noise ratio:

        sigma_gamma^2 / sigma_epsilon^2

    for a selected subset of j-groups using bootstrap
    median and percentile envelopes.

    Parameters
    ----------
    results : xarray.Dataset
        Output dataset from
        sliding_window_MBB_ensemble_analysis().

    j_subset : sequence or None, default=None
        Subset of j indices to plot.
        Example:
            [2, 5]

        If None, all groups are plotted.

    gamma_var : str
        Central gamma variance variable.

    gamma_boot_var : str
        Bootstrap gamma variance variable.

    epsilon_var : str
        Central epsilon variance variable.

    epsilon_boot_var : str
        Bootstrap epsilon variance variable.

    cmap_name : str, default="tab10"
        Matplotlib colormap.

    figsize : tuple, default=(12, 5)
        Figure size.

    xtick_step : int, default=60
        Tick spacing in samples.
        For monthly data, 60 = 5 years.

    save_path : str or None, default=None
        Optional output file path.
        If provided, the figure is saved.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # ----------------------------------------------------------
    # Default: plot all groups
    # ----------------------------------------------------------
    if j_subset is None:

        j_subset = results["j"].values

    # ----------------------------------------------------------
    # Font scaling
    # ----------------------------------------------------------
    base_tick_size = 10
    base_legend_size = 9
    base_title_size = 12

    tick_size = base_tick_size * 1.5
    legend_size = base_legend_size * 2.0
    title_size = base_title_size * 1.2

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=figsize
    )

    cmap = plt.get_cmap(
        cmap_name
    )

    time = results["time"].values

    # ----------------------------------------------------------
    # Ensemble-mean epsilon bootstrap distribution
    #
    # Dimensions:
    #   (boot, time)
    # ----------------------------------------------------------
    epsilon_boot = (
        results[epsilon_boot_var]
        .mean(dim=("j", "k"))
        .transpose("boot", "time")
    )

    # ----------------------------------------------------------
    # Ensemble-mean epsilon central estimate
    # ----------------------------------------------------------
    epsilon_central = (
        results[epsilon_var]
        .mean(dim=("j", "k"))
    )

    # ----------------------------------------------------------
    # Loop over selected groups
    # ----------------------------------------------------------
    for j in j_subset:

        color = cmap(j)

        # ------------------------------------------------------
        # Central gamma estimate
        # ------------------------------------------------------
        gamma_central = (
            results[gamma_var]
            .sel(j=j)
        )

        # ------------------------------------------------------
        # Central SNR estimate
        # ------------------------------------------------------
        snr_central = (
            gamma_central
            / epsilon_central
        )

        # ------------------------------------------------------
        # Bootstrap gamma distribution
        #
        # Dimensions:
        #   (boot, time)
        # ------------------------------------------------------
        gamma_boot = (
            results[gamma_boot_var]
            .sel(j=j)
            .transpose("boot", "time")
        )

        # ------------------------------------------------------
        # Bootstrap SNR distribution
        # ------------------------------------------------------
        snr_boot = (
            gamma_boot
            / epsilon_boot
        )

        # ------------------------------------------------------
        # Bootstrap percentile envelopes
        # ------------------------------------------------------
        p05 = snr_boot.quantile(
            0.05,
            dim="boot",
        )

        p25 = snr_boot.quantile(
            0.25,
            dim="boot",
        )

        p50 = snr_boot.quantile(
            0.50,
            dim="boot",
        )

        p75 = snr_boot.quantile(
            0.75,
            dim="boot",
        )

        p95 = snr_boot.quantile(
            0.95,
            dim="boot",
        )

        # ------------------------------------------------------
        # 90% interval
        # ------------------------------------------------------
        ax.fill_between(
            time,
            p05.values,
            p95.values,
            color=color,
            alpha=0.15,
        )

        # ------------------------------------------------------
        # 50% interval
        # ------------------------------------------------------
        ax.fill_between(
            time,
            p25.values,
            p75.values,
            color=color,
            alpha=0.35,
        )

        # ------------------------------------------------------
        # Bootstrap median
        # ------------------------------------------------------
        ax.plot(
            time,
            p50.values,
            color=color,
            linewidth=2.5,
            label=f"j={j}",
        )

    # ----------------------------------------------------------
    # Reference line
    # ----------------------------------------------------------
    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    # ----------------------------------------------------------
    # Labels/titles
    # ----------------------------------------------------------
    ax.set_ylabel(
        r"$\sigma_\gamma^2 / \sigma_\epsilon^2$",
        fontsize=tick_size,
    )

    ax.set_xlabel(
        "Year",
        fontsize=tick_size,
    )

    ax.set_title(
        r"Macro-state signal-to-noise ratio "
        r"($\sigma_\gamma^2 / \sigma_\epsilon^2$)"
        "\n"
        r"(bootstrap median with 50% and 90% percentile envelopes)",
        fontsize=title_size,
    )

    # ----------------------------------------------------------
    # Tick spacing
    # ----------------------------------------------------------
    tick_idx = np.arange(
        0,
        len(time),
        xtick_step,
    )

    tick_times = time[tick_idx]

    ax.set_xticks(
        tick_times
    )

    ax.set_xticklabels(
        [
            f"{t.year:04d}"
            for t in tick_times
        ],
        rotation=-90,
        ha="center",
    )

    # ----------------------------------------------------------
    # Tick formatting
    # ----------------------------------------------------------
    ax.tick_params(
        axis="both",
        labelsize=tick_size,
    )

    # ----------------------------------------------------------
    # Grid
    # ----------------------------------------------------------
    ax.grid(
        which="major",
        axis="x",
        color="lightgray",
        linewidth=1.0,
        alpha=0.5,
    )

    ax.grid(
        which="major",
        axis="y",
        color="lightgray",
        linewidth=1.0,
        alpha=0.5,
    )

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------
    ax.legend(
        ncol=2,
        fontsize=legend_size,
    )

    # ----------------------------------------------------------
    # Tight axis/layout
    # ----------------------------------------------------------
    ax.axis("tight")

    plt.tight_layout()

    # ----------------------------------------------------------
    # Optional save
    # ----------------------------------------------------------
    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return fig, ax

def plot_gamma_to_epsilon_snr_bootstrap(
    results,
    gamma_var="gamma_time_var",
    gamma_boot_var="gamma_time_var_boot",
    epsilon_var="epsilon_time_var",
    epsilon_boot_var="epsilon_time_var_boot",
    cmap_name="tab10",
    figsize=(12, 5),
    xtick_step=60,
):
    """
    Plot macro-state signal-to-noise ratio:

        sigma_gamma^2 / sigma_epsilon^2

    using bootstrap median and percentile envelopes.

    Parameters
    ----------
    results : xarray.Dataset
        Output dataset from
        sliding_window_MBB_ensemble_analysis().

    gamma_var : str
        Central gamma variance variable.

    gamma_boot_var : str
        Bootstrap gamma variance variable.

    epsilon_var : str
        Central epsilon variance variable.

    epsilon_boot_var : str
        Bootstrap epsilon variance variable.

    cmap_name : str, default="tab10"
        Matplotlib colormap.

    figsize : tuple, default=(12, 5)
        Figure size.

    xtick_step : int, default=60
        Tick spacing in samples.
        For monthly data, 60 = 5 years.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # ----------------------------------------------------------
    # Font scaling
    # ----------------------------------------------------------
    base_tick_size = 10
    base_legend_size = 9
    base_title_size = 12

    tick_size = base_tick_size * 1.5
    legend_size = base_legend_size * 2.0
    title_size = base_title_size * 1.2

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=figsize
    )

    cmap = plt.get_cmap(
        cmap_name
    )

    time = results["time"].values

    # ----------------------------------------------------------
    # Ensemble-mean epsilon bootstrap distribution
    #
    # Dimensions:
    #   (boot, time)
    # ----------------------------------------------------------
    epsilon_boot = (
        results[epsilon_boot_var]
        .mean(dim=("j", "k"))
        .transpose("boot", "time")
    )

    # ----------------------------------------------------------
    # Ensemble-mean epsilon central estimate
    # ----------------------------------------------------------
    epsilon_central = (
        results[epsilon_var]
        .mean(dim=("j", "k"))
    )

    # ----------------------------------------------------------
    # Loop over groups
    # ----------------------------------------------------------
    for j in results["j"].values:

        color = cmap(j)

        # ------------------------------------------------------
        # Central gamma estimate
        # ------------------------------------------------------
        gamma_central = (
            results[gamma_var]
            .sel(j=j)
        )

        # ------------------------------------------------------
        # Central SNR estimate
        # ------------------------------------------------------
        snr_central = (
            gamma_central
            / epsilon_central
        )

        # ------------------------------------------------------
        # Bootstrap gamma distribution
        #
        # Dimensions:
        #   (boot, time)
        # ------------------------------------------------------
        gamma_boot = (
            results[gamma_boot_var]
            .sel(j=j)
            .transpose("boot", "time")
        )

        # ------------------------------------------------------
        # Bootstrap SNR distribution
        # ------------------------------------------------------
        snr_boot = (
            gamma_boot
            / epsilon_boot
        )

        # ------------------------------------------------------
        # Bootstrap percentile envelopes
        # ------------------------------------------------------
        p05 = snr_boot.quantile(
            0.05,
            dim="boot",
        )

        p25 = snr_boot.quantile(
            0.25,
            dim="boot",
        )

        p50 = snr_boot.quantile(
            0.50,
            dim="boot",
        )

        p75 = snr_boot.quantile(
            0.75,
            dim="boot",
        )

        p95 = snr_boot.quantile(
            0.95,
            dim="boot",
        )

        # ------------------------------------------------------
        # 90% interval
        # ------------------------------------------------------
        ax.fill_between(
            time,
            p05.values,
            p95.values,
            color=color,
            alpha=0.15,
        )

        # ------------------------------------------------------
        # 50% interval
        # ------------------------------------------------------
        ax.fill_between(
            time,
            p25.values,
            p75.values,
            color=color,
            alpha=0.35,
        )

        # ------------------------------------------------------
        # Bootstrap median
        # ------------------------------------------------------
        ax.plot(
            time,
            p50.values,
            color=color,
            linewidth=2.5,
            label=f"j={j}",
        )

    # ----------------------------------------------------------
    # Reference line
    # ----------------------------------------------------------
    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    # ----------------------------------------------------------
    # Labels/titles
    # ----------------------------------------------------------
    ax.set_ylabel(
        r"$\sigma_\gamma^2 / \sigma_\epsilon^2$",
        fontsize=tick_size,
    )

    ax.set_xlabel(
        "Year",
        fontsize=tick_size,
    )

    ax.set_title(
        r"Macro-state signal-to-noise ratio "
        r"($\sigma_\gamma^2 / \sigma_\epsilon^2$)"
        "\n"
        r"(bootstrap median with 50% and 90% percentile envelopes)",
        fontsize=title_size,
    )

    # ----------------------------------------------------------
    # Tick spacing
    # ----------------------------------------------------------
    tick_idx = np.arange(
        0,
        len(time),
        xtick_step,
    )

    tick_times = time[tick_idx]

    ax.set_xticks(
        tick_times
    )

    ax.set_xticklabels(
        [
            f"{t.year:04d}"
            for t in tick_times
        ],
        rotation=-90,
        ha="center",
    )

    # ----------------------------------------------------------
    # Tick formatting
    # ----------------------------------------------------------
    ax.tick_params(
        axis="both",
        labelsize=tick_size,
    )

    # ----------------------------------------------------------
    # Grid
    # ----------------------------------------------------------
    ax.grid(
        which="major",
        axis="x",
        color="lightgray",
        linewidth=1.0,
        alpha=0.5,
    )

    ax.grid(
        which="major",
        axis="y",
        color="lightgray",
        linewidth=1.0,
        alpha=0.5,
    )

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------
    ax.legend(
        ncol=2,
        fontsize=legend_size,
    )

    # ----------------------------------------------------------
    # Tight axis/layout
    # ----------------------------------------------------------
    ax.axis("tight")

    plt.tight_layout()

    plt.show()

    return fig, ax


def plot_member_spaghetti(g, j_dim="j", k_dim="k", t_dim="t", alpha=0.4, title="Ensemble member spaghetti plot"):

    fig, ax = plt.subplots(figsize=(12, 5))

    # Total number of members
    n_members = len(g[j_dim]) * len(g[k_dim])

    # Generate distinct colours
    cmap = plt.get_cmap("tab20", n_members)

    member_idx = 0

    for j in g[j_dim].values:
        for k in g[k_dim].values:

            color = cmap(member_idx)

            g.sel({j_dim: j, k_dim: k}).plot(
                ax=ax,
                alpha=alpha,
                color=color,
                linewidth=1
            )

            member_idx += 1

    # Ensemble mean
    g.mean(dim=[j_dim, k_dim]).plot(
        ax=ax,
        linewidth=3,
        label="Grand ensemble-mean",
        color="gray"
    )

    ax.legend()
    ax.set_title(title)

    plt.tight_layout()

    return fig, ax


def plot_example_window_decomposition(g_window, j_dim="j", k_dim="k", t_dim="t"):
    fig, ax = plt.subplots(figsize=(12, 5))

    g_window.mean(dim=t_dim).plot(ax=ax)
    ax.set_title("Example window mean structure")

    plt.tight_layout()
    
    return fig, ax