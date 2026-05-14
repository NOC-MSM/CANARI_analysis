# Changelog:
# 
# 11 May 2026: Chris Wilson - first release version.
#
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

def plot_distribution_comparison(g_original, g_preprocessed, bins=40):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(g_original.values.ravel(), bins=bins, alpha=0.5, label="Original")
    ax.hist(g_preprocessed.values.ravel(), bins=bins, alpha=0.5, label="Preprocessed", color='green')
    ax.legend()
    ax.set_title("Distribution comparison")

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


def plot_member_spaghetti(g, j_dim="j", k_dim="k", t_dim="t", alpha=0.2):
    fig, ax = plt.subplots(figsize=(12, 5))

    for j in g[j_dim].values:
        for k in g[k_dim].values:
            g.sel({j_dim: j, k_dim: k}).plot(ax=ax, alpha=alpha, color="blue")

    g.mean(dim=[j_dim, k_dim]).plot(ax=ax, linewidth=3, label="Ensemble mean", color="gray")
    ax.legend()
    ax.set_title("Ensemble member spaghetti plot")

    plt.tight_layout()
    
    return fig, ax


def plot_example_window_decomposition(g_window, j_dim="j", k_dim="k", t_dim="t"):
    fig, ax = plt.subplots(figsize=(12, 5))

    g_window.mean(dim=t_dim).plot(ax=ax)
    ax.set_title("Example window mean structure")

    plt.tight_layout()
    
    return fig, ax