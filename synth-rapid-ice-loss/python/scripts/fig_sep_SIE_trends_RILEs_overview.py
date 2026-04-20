"""Script to make a series of plots showcasing sea ice extent, trend time series, and
to explore the RILE definition in the CANARI-LE, CMIP6 data, and observational products.
"""

__author__ = "Jake R. Aylmer"

from argparse import ArgumentParser
import calendar
from pathlib import Path
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.stats import linregress


# Data parameters
# ---------------
members_le = np.arange(1, 41, 1)
j_le_show  = 12    # member to pick out on plots


# Plot appearance
# ---------------
color_ob = "tab:blue"
color_c6 = "tab:grey"
cmap_c6  = "Greys"


# Global customisation of matplotlib
# ----------------------------------
mpl.rcParams["lines.linewidth"] *= .5
mpl.rcParams["axes.titlesize"]   = mpl.rcParams["axes.labelsize"]
mpl.rcParams["figure.dpi"] *= 3

# Data paths
# ----------
# Note: directory 'data_path' and its contents below do not actually exist in
# repository. The directory is just a link set up in the author's local copy for
# convenience and to save putting explicit paths to local storage volumes in
# committed scripts.
#
# TODO: create some sort of framework in repository to handle data inputs/paths?
#
data_path        = Path("..", "data")
data_file_le     = Path(data_path, "cle_sie_pan-Arctic_r1-40_1950-2099.npy")
data_file_c6     = Path(data_path, "cmip6_sie_pan-Arctic_20m_1850-2100.pickle")
data_file_ob_esa = Path(data_path, "seaice/esa_cci_l4/ESA_CCI_L4_Sept_SIE.nc")
data_file_ob_had = Path(data_path, "seaice/hadisst/HadISST_SIE.nc")
data_file_ob_sbt = Path(data_path, "seaice/ssmi/gn/siextent_bt_nh_gn_sep_1979-2024.nc")
data_file_ob_snt = Path(data_path, "seaice/ssmi/gn/siextent_nt_nh_gn_sep_1979-2024.nc")


def trend_time_series(t, x, n_step=5):
    """Compute the moving trend of x as a function of time t over periods of n_step steps
    in t. Trends and their standard errors are computed using the SciPy linregress module.


    Parameters
    ----------
    t : 1D array (nt,) of float
        The nt time coordinates.

    x : 2D array (nt, nx) of float
        The nx time series datasets [e.g., nx ensemble members each of time-length nt].


    Optional parameters
    -------------------
    n_step : int, default = 5
        The length of the moving window to compute trends over expressed as an integer
        number of steps/indices of t. Must have nt >= n_step.


    Returns
    -------
    t_trend : 1D array (nt - n_step + 1,) of float
        Time coordinates at the centre of the moving trend windows.

    x_trend, s_trend : 2D array (nt - n_step + 1, nx) of float
        Moving trends and the standard errors as a function of t_trend for each dataset.

    """

    nt, nx = np.shape(x)

    # The number of moving/overlapping trends of size n_step time steps:
    nt_trend = nt - n_step + 1

    t_trend = np.zeros( nt_trend)         # time at centre of trend interval
    x_trend = np.zeros((nt_trend, nx))    # trends
    s_trend = np.zeros((nt_trend, nx))    # standard error of trends

    for k in range(nt_trend):
        t_trend[k] = np.mean(t[k:k+n_step])
        for j in range(nx):
            x_trend[k,j], _, _, _, s_trend[k,j] = \
                linregress(t[k:k+n_step], x[k:k+n_step,j])

    return t_trend, x_trend, s_trend


def get_riles(t_trend, x_trend, threshold=-.3, n_step=4):
    """Identify locations (in time) of rapid ice loss events (RILEs) following the method
    outlined most recently by Sticker et al. (2025; Cryosphere): the local trend (defined
    and calculated externally, input to this function) must fall below a specified
    threshold for at least n_step time steps.


    Parameters
    ----------
    t_trend : 1D array (nt,) of float
        Time coordinates of the moving trend time series.

    x_trend : 2D array (nt, nx) of float
        Moving trends as a function of t_trend for nx datasets.


    Optional parameters
    -------------------
    threshold : float, default = -0.3
        Threshold trend below which is considered 'rapid loss'.

    n_step : int, default = 4
        Number of time steps (indices of t_trend) that the threshold must be
        exceeded to count as a rapid loss event.


    Returns
    -------
    t_rile : 1D array (nt - n_step + 1,) of float
        Time coordinates at the centre of each n_step moving window of t_trend.

    is_rile : 2D array (nt - n_step + 1, nx) of boolean
        Indicates whether the n_step moving window of t_trend centred on the
        corresponding coordinate in t_rile is a rapid loss event (True) or not
        (False), for each input dataset (second dimension of size nx).

    """

    nt, nx = np.shape(x_trend)

    # Number of moving/overlapping n_step trends of size n_step input time steps:
    nt_rile = nt - n_step + 1

    t_rile  = np.zeros(nt_rile)
    is_rile = np.zeros((nt_rile, nx), dtype=bool)

    for k in range(nt_rile):
        t_rile[k]  = np.mean(t_trend[k:k+n_step])
        is_rile[k] = np.all(x_trend[k:k+n_step,:] < threshold, axis=0)

    return t_rile, is_rile


def get_year_ice_free(year, sie, threshold=1., n_step=5):
    """Return the first year of the first n_step year period for which sea ice
    extent (sie) falls below a threshold. Assumes year is in steps of 1 year.
    See Sentfleben et al. (2020, J. Clim.).


    Parameters
    ----------
    year, sie : 1D array (nt,) of float
        Time coordinates as years and SIE time series data.


    Optional parameters
    -------------------
    threshold : float, default = 1.0
        Threshold for 'ice free' conditions.

    n_step : int, default = 5
        Number of consecutive years for which the threshold condition in sie is
        met, from which the first year of that period is returned.


    Returns
    -------
    year_if : float
        Year of onset of ice-free conditions as described above. If all years
        satisfy the criteria, the first year is returned. If no years do, then
        the last year is returned.

    """

    if all(sie < threshold):
        year_if = year[0]
    else:
        j = 0
        while j < len(year):
            if all(sie[j:j+n_step] < threshold):
                year_if = year[j]
                break
            else:
                j += 1

        if j == len(year):
            year_if = year[-1]

    return year_if


def plot_ensemble(ax, xdata, ydata, color="k", alpha=.25, label=None, p=[10,90],
                  percentiles=True, mean=False, minmax=False, members=None):
    """Function to plot a (set of) dataset(s) on some existing axes with various choices
    of what statistic(s) to display.


    Parameters
    ----------
    ax : matplotlib Axes instance
        Axes on which to plot.

    xdata : 1D array (nx,) of float
        X-axis data in common to all datasets in 'ydata'.

    ydata : 2D array (nx, n_datasets) of float
        Y-axis datasets to plot.


    Optional parameters
    -------------------
    color : str or length-3 list of int or float; default = 'k'
        Colour recognised by matplotlib.

    alpha : float in the range 0-1, default = 0.25
        Alpha value used for shading when quantile range is plotted (see below).

    label : str or None (default)
        Label for plot elements (for legends if added later).

    p : length-2 list of int in the range 0-100, default = [10, 90]
        Lower and upper percentiles of range to plot when percentiles = True.

    The following parameters are all booleans and dictate what to plot:
        percentiles : shaded range between two percentiles across datasets    (True )
        mean        : line for the mean across datasets                       (False)
        minmax      : thin line for the running min. and max. across datasets (False)

    members : list of int or None (default)
        If a list of int, plot lines for each of the specific datasets at these indices.

    """

    if percentiles:
        ax.fill_between(xdata, *np.percentile(ydata, p, axis=1),
                        facecolor=color, alpha=alpha)

    if mean:
        ax.plot(xdata, np.mean(ydata, axis=1), color=color, label=label)

    if minmax:
        ax.plot(xdata, np.min(ydata, axis=1), color=color,
                linewidth=.5*mpl.rcParams["lines.linewidth"])
        ax.plot(xdata, np.max(ydata, axis=1), color=color,
                linewidth=.5*mpl.rcParams["lines.linewidth"])

    if members is not None:
        ax.plot(xdata, ydata[:,members], color=color)


def main():

    # Provide command line arguments to change script behaviour:
    prsr = ArgumentParser()
    prsr.add_argument("--month"                        , type=int  , default=9)
    prsr.add_argument("-m", "--m-years-below-threshold", type=int  , default=4)
    prsr.add_argument("-n", "--n-years-trend"          , type=int  , default=5)
    prsr.add_argument("-t", "--rile-threshold"         , type=float, default=-.3)
    prsr.add_argument("-i", "--ifree-threshold"        , type=float, default=1.)
    prsr.add_argument("-c", "--color-le"               , type=str  , default="tab:orange")
    prsr.add_argument("-s", "--save-fig", action="store_true")
    cmd = prsr.parse_args()

    # Prepare common keywords for data processing functions defined at
    # the top of the script, used in the same way for different datasets:
    trend_kw = {"n_step"   : cmd.n_years_trend}
    riles_kw = {"threshold": cmd.rile_threshold, "n_step": cmd.m_years_below_threshold}
    ifree_kw = {"threshold": cmd.ifree_threshold}

    color_le = cmd.color_le

    # Attempt to match colormap (panel d) with color, else fall back to viridis:
    if color_le.startswith("tab:"):
        cmap_le = f"{color_le[4].upper()}{color_le[5:]}s"
    else:
        cmap_le = f"{color_le[0].upper()}{color_le[1:]}s"

    if cmap_le not in plt.colormaps():
        cmap_le = "viridis"

    jmon = cmd.month - 1  # time array index of month (cmd.month == 1 --> Jan, etc.)


    # ============================= #
    # ---- Load CANARI-LE data ---- #
    # ============================= #
    data_le  = np.load(data_file_le)
    sie_le   = data_le[:,jmon,members_le-1]    # extract month and selected members
    yr_le    = np.arange(1950, 2100, 1)        # time axis as year

    ny_le, ne_le = np.shape(sie_le)

    # Calculate trend time series:
    yr_trend_le, sie_trend_le, sie_trend_err_le = \
        trend_time_series(yr_le, sie_le, **trend_kw)

    # Calculate RILE occurrences:
    yr_riles_le, riles_le = get_riles(yr_trend_le, sie_trend_le, **riles_kw)

    # Year of onset of seasonally-ice-free conditions for each ensemble member:
    yr_sif_le = np.zeros(ne_le)
    for e in range(ne_le):
        yr_sif_le[e] = get_year_ice_free(yr_le, sie_le[:,e], **ifree_kw)


    # ========================= #
    # ---- Load CMIP6 data ---- #
    # ========================= #

    # This data has been pre-processed from Aylmer et al. (2024, U. Reading)
    # data and saved into a convenient format, unpacked below:
    #
    with open(data_file_c6, "rb") as file:
        data_c6 = pickle.load(file)

    yr_c6      = yr_le         # it is actually 1850-2100, but match to LE
    models_c6  = data_c6[0]    # list of str, model names
    members_c6 = data_c6[1]    # list of list of str, each model's member labels

    # Extract 1950-2099, month, all members, for all models:
    sie_c6_m   = [x[100:-1,jmon,:] for x in data_c6[2]]

    # Merge all models into one multi-model ensemble:
    sie_c6     = np.concatenate(sie_c6_m, axis=1)

    nm_c6 = len(models_c6)
    ne_c6 = np.array([len(members_c6[m]) for m in range(nm_c6)])

    # Calculate trend time series, RILE occurrence, and ice-free onset:
    yr_trend_c6, sie_trend_c6, sie_trend_err_c6 = \
        trend_time_series(yr_c6, sie_c6, **trend_kw)

    yr_riles_c6, riles_c6 = get_riles(yr_trend_c6, sie_trend_c6, **riles_kw)

    yr_sif_c6_m = [np.zeros(ne_c6[m]) for m in range(nm_c6)]
    for m in range(nm_c6):
        for e in range(ne_c6[m]):
            yr_sif_c6_m[m][e] = get_year_ice_free(yr_c6, sie_c6_m[m][:,e], **ifree_kw)

    yr_sif_c6 = np.concatenate(yr_sif_c6_m)


    # ======================== #
    # ---- Load obs. data ---- #
    # ======================== #

    ne_ob = 4    # there are four observational datasets

    # Observations are in different sources, but combine all four into one array:
    yr_ob  = np.arange(1980, 2025, 1)      # common time period for all
    ny_ob = len(yr_ob)
    sie_ob = np.zeros((ny_ob, ne_ob))

    # ---- ESA CCI:
    with nc.Dataset(data_file_ob_esa, "r") as ncdat:
        # Raw year range is 1980 to 2025 inclusive (so need to remove last year):
        sie_ob[:,0] = np.array(ncdat.variables["sept_SIE"][:-1]) * 1.e-6

    # ---- HadISST:
    with nc.Dataset(data_file_ob_had, "r") as ncdat:
        # Raw year range is 1980 to 2025 inclusive (so need to remove last year):
        sie_ob[:,1] = np.array(ncdat.variables["HadISST_sept_SIE"][:-1]) * 1.e-6

    # ---- SSM/I NASA Team and Bootstrap:
    for j, data_file in zip([2,3], [data_file_ob_sbt, data_file_ob_snt]):
        with nc.Dataset(data_file, "r") as ncdat:
            # Raw year range is 1978 to 2024 inclusive (so remove first two years):
            sie_ob[:,j] = np.array(ncdat.variables["siextent"][2:]) * 1.e-12

    # Calculate moving trends (we do not calculate RILEs because there is not enough
    # data, and we do not calculate ice free conditions because it hasn't happened yet:
    yr_trend_ob, sie_trend_ob, sie_trend_err_ob = \
        trend_time_series(yr_ob, sie_ob, **trend_kw)


    # ======================= #
    # ---- Create figure ---- #
    # ======================= #

    fig, axs = plt.subplots(ncols=2, nrows=2,
                            height_ratios=[3,1], figsize=(6.4, 3.2))

    # (a) Time series
    # ---------------
    plot_ensemble(axs[0,0], yr_c6, sie_c6, color=color_c6)
    plot_ensemble(axs[0,0], yr_le, sie_le, color=color_le, members=[j_le_show-1])
    plot_ensemble(axs[0,0], yr_ob, sie_ob, color=color_ob, alpha=.5, p=[0,100])

    axs[0,0].axhline(cmd.ifree_threshold, linestyle="--", color="tab:grey")
    axs[0,0].set_ylim(0,10)
    axs[0,0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))

    # In-line labels for LE highlighted member and ice-free threshold
    # (hard-code positions, for now):
    axs[0,0].annotate(r"$r=%i$" % j_le_show, (2025, .5), xycoords="data",
                      fontsize="x-small", color=color_le, va="center", ha="right")

    axs[0,0].annotate(r"Ice free", (1953, 1.4), xycoords="data", fontsize="x-small",
                      color="tab:grey", fontstyle="italic", va="center", ha="left")


    # (b) Trend time series
    # ---------------------
    plot_ensemble(axs[0,1], yr_trend_c6, sie_trend_c6, color=color_c6)
    plot_ensemble(axs[0,1], yr_trend_le, sie_trend_le, color=color_le, members=[j_le_show-1])
    plot_ensemble(axs[0,1], yr_trend_ob, sie_trend_ob, color=color_ob, alpha=.5, p=[0,100])

    # For the highlighted LE member, show also the uncertainty range in the trend:
    for x in [1., -1.]:
        axs[0,1].plot(yr_trend_le, (sie_trend_le + x*sie_trend_err_le)[:,j_le_show-1],
                      color=color_le, linestyle="--",
                      linewidth=.5*mpl.rcParams["lines.linewidth"]/2)

    axs[0,1].axhline(cmd.rile_threshold, linestyle="--", color="tab:grey")
    axs[0,1].set_ylim(-0.9, 0.3)
    axs[0,1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, steps=[1,2,3,4,5]))


    # (c) Time of ice free box plots (LE vs CMIP6)
    # --------------------------------------------

    # Keyword arguments to format box plots:
    kw = {"orientation" : "horizontal", "widths": .7,
          "boxprops"    : {"color": color_le, "facecolor":"w"},
          "patch_artist": True,
          "capprops"    : {"color": color_le},
          "whiskerprops": {"color": color_le},
          "medianprops" : {"color": color_le}
    }

    # Draw the LE box at arbitrary height of 1.5, then below the CMIP6 at 0.5
    # (these used later for alignment):
    axs[1,0].boxplot(yr_sif_le, positions=[1.5], **kw)

    # Replace the colour in the boxplot keyword arguments for CMIP6:
    for x in ["box", "cap", "whisker", "median"]:
        kw[f"{x}props"]["color"] = color_c6

    axs[1,0].boxplot(yr_sif_c6, positions=[.5], **kw)

    axs[1,0].set_ylim(0,2)
    axs[1,0].set_yticks([.5, 1.5])
    # Hack for spacing, for now (puts labels roughly in the centre):
    axs[1,0].set_yticklabels(["    CMIP6", "CANARI-LE"])


    # (d) Number of ensemble members experiencing a RILE
    # --------------------------------------------------

    # We plot this as pcolormesh, where each 'cell' represents one year.
    # On the axes, the x-coordinates of the cell edges are then centred on each
    # year, and the y-coordinates of the cell edges just specify the heights:
    #
    xbnds = np.arange(yr_riles_le[0] - .5, yr_riles_le[-1] + 1., 1.)
    ybnds = np.array([-.175, .175])

    # Plot along the same arbitrary y-coordinates ('yoffset') as in panel (c):
    for riles, yoffset, cmap, color in zip([riles_le, riles_c6], [1.5, 0.5],
                                           [cmap_le , cmap_c6 ],
                                           [color_le, color_c6]):

        # Total RILEs across members at each year:
        n_riles_y = np.sum(riles, axis=1)

        # Set to missing (NaN) so it doesn't get plotted with pcolormesh:
        n_riles_y = np.where(n_riles_y == 0, np.nan, n_riles_y)

        axs[1,1].pcolormesh(xbnds, yoffset + ybnds, n_riles_y[np.newaxis,:],
                            cmap=cmap, vmin=-1, vmax=np.nanmax(n_riles_y))

        # Label the maximum number of RILEs in any given year corresponding to the
        # darkest colour shade (again we hard-code the position, for now):
        axs[1,1].annotate(f"(max. = {np.nanmax(n_riles_y):.0f})", (2100, yoffset),
                          ha="right", va="center", color=color, xycoords="data",
                          fontsize="x-small", backgroundcolor="white")

    axs[1,1].set_ylim(0,2)
    axs[1,1].set_yticks([])  # remove ticks/labels


    # General formatting and layout
    # -----------------------------
    axs[0,0].set_ylabel(r"$10^6$ km$^2$"            , loc="top")
    axs[0,1].set_ylabel(r"$10^6$ km$^2$ year$^{-1}$", loc="top")

    axs[0,0].set_title(f"(a) {calendar.month_name[cmd.month]} sea ice extent", loc="left")
    axs[0,1].set_title(f"(b) Moving {cmd.n_years_trend}-year trend"          , loc="left")
    axs[1,0].set_xlabel("(c) Onset of ice-free conditions"                   , loc="left")
    axs[1,1].set_xlabel("(d) RILE occurrence"                                , loc="left")

    for ax in axs.flatten():  # all axes/subplots are on the same time axis limits
        ax.set_xlim(1950, 2100)

    for ax in axs[0,:]:  # top row (panels a and b)
        ax.tick_params(axis="x", pad=4.)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axs[1,:]:  # bottom row (panels c and d)

        # Faint grid line through box plots/RILE occurrence:
        for yoffset in [.5, 1.5]:
            ax.axhline(yoffset, *ax.get_ylim(), color="lightgrey", zorder=0,
                       linewidth=.5*mpl.rcParams["lines.linewidth"])

        # Remove all ticks:
        ax.tick_params(which="both", axis="both", bottom=False, labelbottom=False,
                       left=False, labelleft=False, right=False, labelright=True,
                       top=True, labeltop=False, pad=1.)

        # Remove all spines except top (visually, use time axis of a/b):
        for spine in ["left", "right", "bottom"]:
            ax.spines[spine].set_visible(False)

    # Needs to be done after tick_params (I guess? Doesn't work unless here)
    # Update the color of the 'CMIP'/'CANARI-LE' labels on panel c:
    for t, col in zip(axs[1,0].yaxis.get_ticklabels(), [color_c6, color_le]):
        t.set_color(col)
        t.set_fontweight("bold")

    fig.tight_layout()  # set layout roughly, then adjust below
    fig.subplots_adjust(hspace=0.3, wspace=0.425)  # from trial and error

    if cmd.save_fig:
        fig.savefig(f"cle_cmip6_{calendar.month_abbr[cmd.month].lower()}_riles_"
                    + f"N{cmd.n_years_trend}_M{cmd.m_years_below_threshold}_"
                    + f"T{str(cmd.rile_threshold).replace('.','p')}.png",
                    dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
