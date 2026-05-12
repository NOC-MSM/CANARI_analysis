# Changelog:
# 
# 11 May 2026: Chris Wilson - first release version.
#
# OpenAI ChatGPT was used to assist with aspects of code development and refinement.

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL

def validate_input_dims(g, j_dim="j", k_dim="k", t_dim="t"):
    """
    Validate that an input array contains required dimensions.

    Parameters
    ----------
    g : xarray.DataArray or xarray.Dataset
        Input data structure.
    j_dim, k_dim, t_dim : str, optional
        Names of required spatial and temporal dimensions.

    Raises
    ------L
    ValueError
        If one or more required dimensions are missing.
    """

    # Define the set of dimensions required by downstream analysis
    required_dims = {j_dim, k_dim, t_dim}

    # Identify any missing dimensions in the input object
    missing = required_dims.difference(g.dims)

    # Raise an error, if so
    if missing:
        raise ValueError(f"Missing required dimensions: {missing}")


def compute_STL_decomposition(
    g,
    j_dim="j",
    k_dim="k",
    t_dim="time",
    period=12,
    robust=False
):
    """
    Compute STL decomposition of the grand ensemble-mean timeseries.
    (Seasonal Decomposition of Time Series by Loess (STL))
    
    Parameters
    ----------
    g : xarray.DataArray
        Input ensemble dataset.
    j_dim, k_dim : str, optional
        Names of ensemble dimensions.
    t_dim : str, optional
        Name of temporal dimension.
    period : int, optional
        Seasonal period for STL decomposition.
    robust : bool, optional
        Use robust STL fitting.

    Returns
    -------
    xarray.Dataset
        Dataset containing STL trend, seasonal, and residual
        components of the grand ensemble-mean.
    """

    # Grand ensemble-mean timeseries
    g_jk_mean = g.mean(dim=(j_dim, k_dim))

    # STL decomposition performed on NumPy array
    stl = STL(
        g_jk_mean.values,
        period=period,
        robust=robust,
    ).fit()

    # Convert STL output back to xarray Dataset
    ds_stl = xr.Dataset(
        {
            "STL_trend": (t_dim, stl.trend),
            "STL_seasonal": (t_dim, stl.seasonal),
            "STL_resid": (t_dim, stl.resid),
        },
        coords={t_dim: g_jk_mean[t_dim]},
    )

    return ds_stl
    

def remove_linear_trend(g, t_dim="t", a=0.0, b=0.0):
    """
    Remove a prescribed linear trend from a time-dependent array.

    The trend is defined as:
        trend(t) = a + b * tau

    where tau is elapsed time in units of 30-day months relative
    to the first timestamp.

    Parameters
    ----------
    g : xarray.DataArray
        Input data array.
    t_dim : str, optional
        Name of the temporal dimension.
    a : float, optional
        Trend intercept.
    b : float, optional
        Trend slope per 30-day month.

    Returns
    -------
    xarray.DataArray
        Detrended data array.
    """

    # Use the first timestamp as the temporal reference point
    t0 = pd.Timestamp(g[t_dim].values[0])

    # Compute elapsed time in units of 30-day months relative to t0
    # (appropriate for monthly means on a 360-day calendar)
    tau = (
        pd.to_datetime(g[t_dim].values) - t0
    ) / np.timedelta64(30, "D")  #if input monthly-means are 30 days apart (i.e. 360d calendar)

    # Convert elapsed time array into an xarray DataArray
    # with matching coordinates and dimensions
    tau = xr.DataArray(
        tau.astype(float),
        dims=[t_dim],
        coords={t_dim: g[t_dim]},
    )

    # Construct linear trend model:
    # trend(t) = a + b * tau
    trend = a + b * tau

    # Return detrended data
    return g - trend


def remove_seasonal_cycle(g, t_dim="t"):
    """
    Remove the mean monthly seasonal cycle from a time series.

    Parameters
    ----------
    g : xarray.DataArray
        Input data array with a datetime-like time coordinate.
    t_dim : str, optional
        Name of the temporal dimension.

    Returns
    -------
    xarray.DataArray
        Data array containing anomaly after seasonal cycle removed.
        Seasonal cycle can be recovered by subtracting this from original.
    """

    # Compute mean climatology for each calendar month
    monthly_climatology = g.groupby(f"{t_dim}.month").mean(dim=t_dim)

    # Subtract monthly climatology from each corresponding month
    return g.groupby(f"{t_dim}.month") - monthly_climatology


def estimate_autocorrelation_timescale(x, max_lag=None):

    """
    Estimate the autocorrelation decay timescale of a 1D signal.

    The function computes the autocorrelation function (ACF) and
    returns the first lag where the autocorrelation falls below
    exp(-1).

    Parameters
    ----------
    x : array-like
        Input 1D time series.
    max_lag : int, optional
        Maximum lag to evaluate. Defaults to len(x) // 4.

    Returns
    -------
    int
        Estimated autocorrelation timescale in samples.
    """
    
    # Convert input to floating-point NumPy array
    x = np.asarray(x, dtype=float)

    # Remove NaN and infinite values
    x = x[np.isfinite(x)]

    # Require a minimum number of samples for stable estimation
    if len(x) < 5:
        return 1

    # Default maximum lag: one quarter of the signal length
    if max_lag is None:
        max_lag = max(1, len(x) // 4)

    # Compute autocorrelation function up to max_lag
    # using FFT-based implementation for efficiency
    acf_values = acf(x, nlags=max_lag, fft=True, missing="drop")

    # Identify lags where autocorrelation drops below exp(-1)
    below_threshold = np.where(acf_values < np.exp(-1))[0]

    # If no threshold crossing occurs, return maximum tested lag
    if len(below_threshold) == 0:
        return max_lag

    # Return first lag below threshold
    return int(max(1, below_threshold[0]))


def estimate_mutual_information_timescale(x, max_lag=None):
    """
    Estimate the characteristic information-decay timescale of a 1D signal.

    The function computes time-delayed mutual information between
    x[t] and x[t + lag] for increasing lag values, then returns the
    first lag where the mutual information falls below exp(-1) of its
    value at lag 1.

    Parameters
    ----------
    x : array-like
        Input 1D time series.
    max_lag : int, optional
        Maximum lag to evaluate. Defaults to len(x) // 4.

    Returns
    -------
    int
        Estimated decorrelation timescale in samples.

    Notes
    -----
    Mutual information is estimated using sklearn's
    ``mutual_info_regression`` nearest-neighbor estimator.

    This measure captures nonlinear temporal dependence and may be
    interpreted as an information-theoretic autocorrelation time.
    """
    # Convert input to floating-point NumPy array
    x = np.asarray(x, dtype=float)

    # Remove NaN and infinite values
    x = x[np.isfinite(x)]

    # Require a minimum number of samples for stable estimation
    if len(x) < 10:
        return 1

    # Default maximum lag: one quarter of the signal length
    if max_lag is None:
        max_lag = max(2, len(x) // 4)

    # Store mutual information values for each lag
    mi = []

    # Compute delayed mutual information over increasing lags
    for lag in range(1, max_lag + 1):

        # Construct paired sequences:
        # x1 = x[t]
        # x2 = x[t + lag]
        x1 = x[:-lag]
        x2 = x[lag:]

        # Skip lags with insufficient remaining samples
        if len(x1) < 5:
            mi.append(np.nan)
            continue

        # Estimate mutual information between present and delayed signal
        # using sklearn's nearest-neighbour MI estimator
        score = mutual_info_regression(
            x1.reshape(-1, 1),
            x2,
            discrete_features=False,
            random_state=0,
        )[0]

        # Store MI estimate for this lag
        mi.append(score)

    # Convert list to NumPy array for vectorised operations
    mi = np.asarray(mi)

    # Identify finite (valid) MI estimates
    valid = np.isfinite(mi)

    # Fallback if all estimates failed
    if not np.any(valid):
        return 1

    # Define characteristic decay threshold as exp(-1) of the
    # initial mutual information value
    threshold = np.exp(-1) * mi[valid][0]
    below_threshold = np.where(mi < threshold)[0]

    # If no decay crossing occurs, return maximum tested lag
    if len(below_threshold) == 0:
        return max_lag

    # Return first lag below threshold (convert from zero-based index)
    return int(max(1, below_threshold[0] + 1))


def define_block_length(
    g,
    j_dim="j",
    k_dim="k",
    t_dim="t",
    max_lag=None,
    persistence_factor=4, #usually choose a value between 3 and 5
    quantile=0.75):
    """
    Estimate an appropriate moving-block bootstrap length from
    ensemble persistence timescales.

    The function computes both autocorrelation and mutual-information
    persistence timescales for each ensemble member, then derives a
    representative block length from upper-quantile persistence
    behaviour.

    Parameters
    ----------
    g : xarray.DataArray
        Input ensemble dataset.
    j_dim, k_dim, t_dim : str, optional
        Names of spatial and temporal dimensions.
    max_lag : int, optional
        Maximum lag used in timescale estimation.
    persistence_factor : float, optional
        Scaling factor applied to the dominant persistence scale.
    quantile : float, optional
        Quantile used to define representative persistence behaviour.

    Returns
    -------
    dict
        Dictionary containing estimated timescales and recommended
        block length.
    """

    # Store persistence estimates for all ensemble members
    acf_scales = []
    mi_scales = []

    # Loop over all ensemble coordinates
    for j in g[j_dim].values:
        for k in g[k_dim].values:
            x = g.sel({j_dim: j, k_dim: k}).values

            acf_scales.append(
                estimate_autocorrelation_timescale(x, max_lag=max_lag)
            )

            mi_scales.append(
                estimate_mutual_information_timescale(x, max_lag=max_lag)
            )

    # Convert results to NumPy arrays
    acf_scales = np.asarray(acf_scales)
    mi_scales = np.asarray(mi_scales)

    # Define representative persistence scales using upper quantiles
    representative_acf = int(np.ceil(np.quantile(acf_scales, quantile)))
    representative_mi = int(np.ceil(np.quantile(mi_scales, quantile)))

    # Use the more conservative (larger) persistence estimate
    dominant_scale = max(representative_acf, representative_mi)

    return {
        "acf_timescales": acf_scales,
        "mi_timescales": mi_scales,
        "representative_acf": representative_acf,
        "representative_mi": representative_mi,
        "dominant_persistence_scale": dominant_scale,
        
        # Scale dominant persistence to obtain bootstrap block length
        # with a hard lower bound of 3 samples
        "block_length": max(
            3,
            int(np.ceil(persistence_factor * dominant_scale)),
        ),
    }


def moving_block_bootstrap_indices(n, block_length, rng=None):
    """
    Generate moving-block bootstrap resampling indices.

    Parameters
    ----------
    n : int
        Length of the original time series.
    block_length : int
        Length of each resampled block.
    rng : numpy.random.Generator, optional
        Random number generator instance.

    Returns
    -------
    numpy.ndarray
        Bootstrap index array of length n.
    """

    # Create default random number generator if needed
    if rng is None:
        rng = np.random.default_rng()

    possible_starts = np.arange(0, max(1, n - block_length + 1))

    # Accumulate bootstrap indices until target length is reached
    idx = []

    while len(idx) < n:

        # Randomly select block starting point
        start = rng.choice(possible_starts)

        # Append contiguous block indices
        idx.extend(np.arange(start, min(start + block_length, n)))

    # Truncate to exact target length
    return np.asarray(idx[:n])





def compute_block_statistics_anova_numpy(g_block):
    r"""
    Fast NumPy implementation of ANOVA decomposition.

    Parameters
    ----------
    g_block : ndarray
        Array with shape (j, k, t)

    Returns
    -------
    dict
        Dictionary of statistics.

    --------------------------------------------------------------------------
    
    Compute ANOVA-style variance decomposition statistics for a block.
    
    Motivated by: 
    Hodson, D. L. R. and R. T. Sutton (2008). doi: 10.1007/s00382-008-0372-z.
    Hawkins, E. and R. T. Sutton (2009). doi: 10.1175/2009BAMS2607.1.
    Deser, C. et al. (2025). doi: 10.1007/s00382-024-07553-z
    
    The decomposition separates the signal into:
    - mu = time-mean of grand ensemble-mean
    - alpha = time variation of grand ensemble-mean w.r.t. mu
    - beta = group bias w.r.t. mu
    - gamma = time variation of unbiased group-mean
    - epsilon = internal variability

    Parameters
    ----------
    g_block : xarray.DataArray
        Input data block.
    j_dim, k_dim, t_dim : str, optional
        Names of spatial and temporal dimensions.

    Returns
    -------
    dict
        Dictionary of decomposition statistics and variance estimates.




    HIERARCHICAL ENSEMBLE STRUCTURE
    ──────────────────────────────────────────────────────────────
    
    Dimensions:
    
        t = time
        j = parent group (e.g. macro)
        k = child member (e.g. micro)

        See below for more examples of parent/child
    

    Data structure:

    g_block(j,k,t)

    Hierarchy:
    
        parent group j=0
        │
        ├── child member k=0
        │      x000  x001  x002  x003 ...
        │
        ├── child member k=1
        │      x010  x011  x012  x013 ...
        │
        └── child member k=2
               x020  x021  x022  x023 ...
    
    
        parent group j=1
        │
        ├── child member k=0
        │      x100  x101  x102  x103 ...
        │
        ├── child member k=1
        │      x110  x111  x112  x113 ...
        │
        └── child member k=2
               x120  x121  x122  x123 ...
    
    
                         time ─────────────────────────────►
        
    
 
    
    parent j may represent one of: 
        emissions forcing scenario
        model configuration (e.g. HadGEM3 GC3.1 MM or CESM abc) for multi-model ensemble
        macro initial condition perturbation state ONLY IF single model, single scenario

    child k may represent one of:
        model configuration (e.g. HadGEM3 GC3.1 MM or CESM abc) ONLY IF single forcing scenario is parent
        micro initial condition perturbation state 
        
    ================================================================
    ANOVA-TYPE DECOMPOSITION
    ================================================================
    
    Goal:
    decompose variability into hierarchical components
    
    
    Observed signal:
    
        g(j,k,t)
    
    
    Decomposition:
    
        g(j,k,t)
          =
          μ
          + α(t)
          + β(j)
          + γ(j,t)
          + ε(j,k,t)
    
    
    Or visually:
    
    
                        OBSERVED SIGNAL
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    
          GLOBAL            GROUP             INTERNAL
           MEAN              EFFECT           VARIABILITY
    
    
    ================================================================
    1. GLOBAL MEAN : μ
    ================================================================
    
                       μ = mean over ALL dimensions
    
    
                    time
                      │
                      ▼
    
              ensemble mean over (j,k)
                      │
                      ▼
    
                time mean over t
                      │
                      ▼
    
                      μ
    
    
    Interpretation:
        overall baseline state
        grand ensemble climatology

    
                    all members
                 ╱ ╱ ╱ ╱ ╱ ╱ ╱
    t1  ─────── ● ● ● ● ● ● ●
    t2  ─────── ● ● ● ● ● ● ●
    t3  ─────── ● ● ● ● ● ● ●
    t4  ─────── ● ● ● ● ● ● ●
    
                    ↓
    
             single grand mean
    
    
    ================================================================
    2. TEMPORAL FORCING : α(t)
    ================================================================
    
              α(t) = grand ensemble mean anomaly
    
    
    For each time:
    
        ensemble mean(t)
            minus
        global mean μ
    
    grand ensemble mean:
    
            time ─────────────────────────►
    
                   μ
    
    
    α(t) captures:
        externally forced signal
        common temporal evolution
    
    
    Mathematically:
    
        α(t)
          =
          mean_jk[g(j,k,t)]
          − μ
    
    
    ================================================================
    3. GROUP BIAS : β(j)
    ================================================================
    
              β(j) = persistent group offset
    
    
    For each parent group:
    
        group mean(j)
            minus
        global mean μ
    
    
                     β(j)
    
    group 1     ────────────────
    group 2     ───────
    group 3     ──────────────────────
    group 4     ───────────
    
                     μ
    
    
    Interpretation:
        model bias
        macro-member offset
    
    Mathematically:
    
        β(j)
          =
          mean_tk[g(j,k,t)]
          − μ
    
    
    ================================================================
    4. TIME-VARYING GROUP RESPONSE : γ(t,j)
    ================================================================
    
             γ(t,j) = group-specific temporal evolution
    
    
    First compute:
    
        g_tj = mean over k
    
    giving:
    
               time-dependent group means
    
    
    Then remove:
        μ
        α(t)
        β(j)
    
    remaining structure:
    
        γ(t,j)
      
    Common forced signal:
    
               /\        /\
              /  \      /  \
    
    Group-specific temporal deviations:
    
    group 1      /\_
    group 2    _/  \__
    group 3        /\
    group 4   __/\
    
    
    γ(t,j) captures:
        differing temporal responses
        time-dependent group disagreement (e.g. between macro I.C.s or multiple models) 
        forced-response diversity
    
    
    Centering constraints:
    
        mean_t[γ] = 0
        mean_j[γ] = 0
    
    
    This prevents overlap with:
        α(t)
        β(j)
    
    
    ================================================================
    5. INTERNAL VARIABILITY : ε(j,k,t)
    ================================================================
    
    Residual:
    
        ε
          =
          g − μ − α − β − γ
    
    
    ASCII:
    
    signal
      │
      ├── forced structure
      ├── group structure
      ├── interaction structure
      ▼
     residual fluctuations
    
    
    Represents:
        internal variability
        realization noise
        unresolved stochastic structure
    
    
    Within each group:
    
            group mean
    ────────────●────────────
    
    members:
          x   x  x   x x  x
    
    scatter around mean:
            ↑
            ε
    
    
    ================================================================
    VARIANCE DECOMPOSITION
    ================================================================
    
    The code then computes temporal variances:
    
    
    α_var
        variance of common forced signal
    
    γ_var
        variance of group-specific temporal response
    
    ε_var
        variance of internal variability
    
    
    
    TOTAL VARIANCE
    ────────────────────────────────────────────
    
            forced
              │
              ▼
          α_var
    
        + group-time interaction
              │
              ▼
          γ_var
    
        + internal variability
              │
              ▼
          ε_var
    
    ────────────────────────────────────────────
    
    
    Interpretation:
    
        α_var
            externally forced variability
    
        γ_var
            group disagreement in temporal evolution
    
        ε_var
            internal variability / noise
    
    
    ================================================================
    HIERARCHICAL VIEW
    ================================================================
    
                         g(j,k,t)
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
    
          α(t)             β(j)            ε(j,k,t)
    
     common forcing      group bias        internal noise
    
             │
             ▼
    
          γ(j,t)
    
     group-specific temporal evolution
    
    
    Final residual structure:
    
        g = μ + α + β + γ + ε
    """

    # ------------------------------------------------------
    # Mean structure
    # ------------------------------------------------------

    mu = np.mean(g_block)

    # alpha(t)
    alpha = np.mean(g_block, axis=(0, 1)) - mu

    # beta(j)
    beta = np.mean(g_block, axis=(1, 2)) - mu

    # gamma(j,t)
    g_tj = np.mean(g_block, axis=1)

    gamma = (
        g_tj
        - mu
        - alpha[None, :]
        - beta[:, None]
    )

    # Center gamma
    gamma = (
        gamma
        - gamma.mean(axis=1, keepdims=True)
    )

    gamma = (
        gamma
        - gamma.mean(axis=0, keepdims=True)
    )

    # epsilon(j,k,t)
    epsilon = (
        g_block
        - mu
        - alpha[None, None, :]
        - beta[:, None, None]
        - gamma[:, None, :]
    )

    # ------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------

    alpha_mean = np.mean(alpha)

    gamma_mean = np.mean(gamma, axis=1)

    epsilon_mean = np.mean(epsilon, axis=2)

    alpha_var = np.var(alpha, ddof=1)

    gamma_var = np.var(gamma, axis=1, ddof=1)

    epsilon_var = np.var(epsilon, axis=2, ddof=1)

    total_var = np.var(g_block, axis=2, ddof=1)

    # ------------------------------------------------------
    # Predictable-fraction diagnostics
    # ------------------------------------------------------

    # Expand dimensions for broadcasting
    alpha_2d = alpha_var * np.ones_like(total_var)

    gamma_2d = gamma_var[:, None]

    # Fraction of total variance associated with:
    #
    # alpha     : grand forced signal
    # gamma     : group/macroscopic response
    # epsilon   : internal variability
    # pred      : total potentially predictable variance
    #
    # Exact closure:
    #
    # F_alpha + F_gamma + F_epsilon = 1
    #
    # modulo floating-point precision.

    F_alpha = alpha_2d / total_var

    F_gamma = gamma_2d / total_var

    F_epsilon = epsilon_var / total_var

    F_pred = (
        alpha_2d + gamma_2d
    ) / total_var

    return {
        "mu": mu,

        "alpha_mean": alpha_mean,
        "alpha_var": alpha_var,

        "gamma_mean": gamma_mean,
        "gamma_var": gamma_var,

        "epsilon_mean": epsilon_mean,
        "epsilon_var": epsilon_var,

        "beta": beta,

        "total_var": total_var,

        # Predictable fractions
        "F_alpha": F_alpha,
        "F_gamma": F_gamma,
        "F_epsilon": F_epsilon,
        "F_pred": F_pred,
    }



def sliding_window_MBB_ensemble_analysis(
    g,
    j_dim="j",
    k_dim="k",
    t_dim="t",
    window_length=None,
    bootstrap=True,
    n_boot=200,
    block_length=None,
    random_seed=123,
    store_bootstrap_samples=False):

    """
    Sliding-window moving-block-bootstrap ensemble analysis.

    Fast NumPy-based implementation of the ANOVA framework.
    
    Perform sliding-window ensemble statistical analysis with optional
    moving-block bootstrap uncertainty estimation.

    We further embed this statistical analysis over the structure of
    an ensemble of timeseries with hierarchical nested structure, such as
    a single-model large ensemble (SMILE) with parent (macro) and 
    child (micro) initial condition ensemble 
    or 
    a multi-model ensemble with parent (model) and child (micro I.C. members).


    If bootstrap=False, only the central estimator of the statistic is computed
    over a sliding temporal window, for each of the timeseries in j_dim, k_dim.

    If bootstrap=True, moving-block bootstrap (MBB) resampling
    is applied along the temporal dimension within each sliding
    window in order to estimate the empirical sampling distribution
    of the statistic under temporally correlated variability.
    The hierarchical ensemble structure (j_dim, k_dim) is preserved
    exactly during resampling.
    
    The statistic could be, e.g. variance, mean or linear trend parameters.

    Sliding-window MBB is used so that a moderately non-stationary timeseries 
    can be characterised.  Standard MBB assumes stationarity within the window.

    
    Parameters
    ----------
    g : xarray.DataArray
        Input ensemble dataset.
    j_dim, k_dim, t_dim : str, optional
        Names of ensemble (parent group (j), child member (k)) 
        and temporal dimensions.  
        e.g. j may refer to macro and k to micro initial condition
        ensemble member dimensions.
    window_length : int
        sliding-window length in samples.
    bootstrap : bool, optional
        If True, estimate confidence intervals via bootstrap.
    n_boot : int, optional
        Number of bootstrap resamples.
    block_length : int, optional
        Moving-block bootstrap block length.
    random_seed : int, optional
        Random seed for reproducibility.
    store_bootstrap_samples : bool, optional
        If True, retain full bootstrap realizations for each statistic.
        This allows statistically consistent uncertainty propagation
        for nonlinear derived quantities (e.g. ratios or variance fractions).

    Returns
    -------
    xarray.Dataset
        Dataset containing sliding statistics and optional
        bootstrap confidence intervals.

        FULL TIME SERIES
    ───────────────────────────────────────────────────────────────► t
    
    x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x
    
    
    ================================================================
    STEP 1: SLIDING WINDOW
    ================================================================
    
                    ┌─────────────────────┐
                    │   sliding window    │
                    └─────────────────────┘
    x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x
              ^                                         ^
            start                                      end
    
    
    Within this window:
    
        g_win = g[t_start : t_end]
    
    
    Compute CENTRAL ESTIMATOR from ORIGINAL data:
    
        θ̂ = statistic(g_win)
    
    Examples:
        mean
        variance
        trend slope
        ANOVA variance components
        ensemble statistics
    
    
    ================================================================
    STEP 2: MOVING BLOCK BOOTSTRAP INSIDE WINDOW
    ================================================================
    
    Original window:
    
    t:   0 1 2 3 4 5 6 7 8 9
         ───────────────────
    x = [a b c d e f g h i j]
    
    
    Choose contiguous blocks of length L:
    
    Block 1:
            [c d e]
    
    Block 2:
                  [f g h]
    
    Block 3:
        [a b c]
    
    Block 4:
                      [h i j]
    
    
    Concatenate sampled blocks:
    
    [c d e] + [f g h] + [a b c] + [h i j]
          ↓
    [c d e f g h a b c h]
    
    
    Truncate to original length if needed:
    
    [c d e f g h a b c h]
    
    
    This forms ONE bootstrap realization:
    
        g_boot^(1)
    
    
    Repeat many times:
    
        g_boot^(1)
        g_boot^(2)
        g_boot^(3)
        ...
        g_boot^(N)
    
    
    ================================================================
    STEP 3: BOOTSTRAP DISTRIBUTION
    ================================================================
    
    For each bootstrap realization:
    
        θ̂*(1) = statistic(g_boot^(1))
        θ̂*(2) = statistic(g_boot^(2))
        ...
        θ̂*(N)
    
    
    Build empirical sampling distribution:
    
                    ^
                    |
    frequency       |                         *
                    |                      *  *
                    |                   *  *  *  *
                    |                *  *  *  *  *
                    |             *  *  *  *  *  *
                    +------------------------------------► θ
    
                        lower      θ̂      upper
                         2.5%               97.5%
    
    
    Where:
    
        θ̂      = central estimator from original window
        lower   = bootstrap CI lower bound
        upper   = bootstrap CI upper bound
    
    
    ================================================================
    STEP 4: SLIDE WINDOW FORWARD
    ================================================================
    
    Window position t:
    
            ┌─────────────────────┐
    x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x
    
    
    Window position t+1:
    
              ┌─────────────────────┐
    x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x
    
    
    Repeat:
        central estimate
        bootstrap resampling
        confidence intervals
    
    
    ================================================================
    FINAL OUTPUT
    ================================================================
    
    time ─────────────────────────────────────────────────────►
    
    central estimate:
            ●────●────●────●────●────●────●
    
    95% confidence interval:
           ╱│╲  ╱│╲  ╱│╲  ╱│╲  ╱│╲  ╱│╲
          ╱ │ ╲╱ │ ╲╱ │ ╲╱ │ ╲╱ │ ╲╱ │ ╲
    
    Each point:
        ● = statistic from original sliding window
    
    Each envelope:
        CI estimated from moving-block bootstrap

    """

    # ----------------------------------------------------------
    # Validate dimensions
    # ----------------------------------------------------------
    validate_input_dims(
        g,
        j_dim=j_dim,
        k_dim=k_dim,
        t_dim=t_dim,
    )

    # ----------------------------------------------------------
    # Convert once to NumPy
    # Internal shape: (j, k, t)
    # ----------------------------------------------------------
    g_values = g.transpose(
        j_dim,
        k_dim,
        t_dim,
    ).values

    # ----------------------------------------------------------
    # Window setup
    # ----------------------------------------------------------
    half_window = window_length // 2

    nt = g.sizes[t_dim]

    rng = np.random.default_rng(random_seed)

    results = {}

    output_times = []

    # ----------------------------------------------------------
    # Main sliding-window loop
    # ----------------------------------------------------------
    for it in range(nt):

        start = max(0, it - half_window)

        end = min(nt, it + half_window + 1)

        # ------------------------------------------------------
        # Extract NumPy window
        # Shape: (j, k, t)
        # ------------------------------------------------------
        g_win = g_values[:, :, start:end]

        # ------------------------------------------------------
        # Central estimator
        # ------------------------------------------------------
        stats = compute_block_statistics_anova_numpy(
            g_win
        )

        for key, value in stats.items():

            results.setdefault(key, []).append(value)

        # ------------------------------------------------------
        # Bootstrap uncertainty
        # ------------------------------------------------------
        if bootstrap:

            boot_results = {
                key: []
                for key in stats.keys()
            }

            n_t = g_win.shape[2]

            effective_block_length = (
                n_t
                if block_length is None
                else min(block_length, n_t)
            )

            # --------------------------------------------------
            # Bootstrap loop
            # --------------------------------------------------
            for _ in range(n_boot):

                t_idx = moving_block_bootstrap_indices(
                    n=n_t,
                    block_length=effective_block_length,
                    rng=rng,
                )

                # Resample only along time dimension
                g_boot = g_win[:, :, t_idx]

                boot_stats = (
                    compute_block_statistics_anova_numpy(
                        g_boot
                    )
                )
                
                for key, value in boot_stats.items():

                    boot_results[key].append(value)

            # --------------------------------------------------
            # Confidence intervals + optional storage
            # --------------------------------------------------
            for key, values in boot_results.items():

                # Shape:
                # scalar statistic:
                #   (n_boot,)
                #
                # j statistic:
                #   (n_boot, j)
                #
                # jk statistic:
                #   (n_boot, j, k)
                arr = np.stack(values)

                # ----------------------------------------------
                # Optionally retain full bootstrap ensemble
                # ----------------------------------------------
                if store_bootstrap_samples:

                    results.setdefault(
                        f"{key}_boot",
                        [],
                    ).append(arr)

                # ----------------------------------------------
                # Percentile confidence intervals
                # ----------------------------------------------
                lower = np.percentile(
                    arr,
                    2.5,
                    axis=0,
                )

                upper = np.percentile(
                    arr,
                    97.5,
                    axis=0,
                )

                results.setdefault(
                    f"{key}_ci_lower",
                    [],
                ).append(lower)

                results.setdefault(
                    f"{key}_ci_upper",
                    [],
                ).append(upper)

        output_times.append(
            g[t_dim].values[it]
        )

    # ----------------------------------------------------------
    # Convert accumulated results back to xarray
    # ----------------------------------------------------------
    data_vars = {}

    boot_coord = np.arange(n_boot)

    for key, values in results.items():

        arr = np.stack(values)

        coords = {
            t_dim: output_times
        }

        # ------------------------------------------------------
        # Bootstrap samples
        # ------------------------------------------------------
        if key.endswith("_boot"):

            # ----------------------------------------------
            # scalar bootstrap statistics
            # arr shape: (time, boot)
            # ----------------------------------------------
            if arr.ndim == 2:

                dims = [t_dim, "boot"]

                coords["boot"] = boot_coord

            # ----------------------------------------------
            # j statistics
            # arr shape: (time, boot, j)
            # ----------------------------------------------
            elif arr.ndim == 3:

                dims = [t_dim, "boot", j_dim]

                coords["boot"] = boot_coord

                coords[j_dim] = g[j_dim]

            # ----------------------------------------------
            # jk statistics
            # arr shape: (time, boot, j, k)
            # ----------------------------------------------
            elif arr.ndim == 4:

                dims = [
                    t_dim,
                    "boot",
                    j_dim,
                    k_dim,
                ]

                coords["boot"] = boot_coord

                coords[j_dim] = g[j_dim]

                coords[k_dim] = g[k_dim]

            else:

                raise ValueError(
                    f"{key}: unsupported ndim {arr.ndim}"
                )

        # ------------------------------------------------------
        # Standard statistics
        # ------------------------------------------------------
        else:

            # scalar statistic
            # arr shape: (time,)
            if arr.ndim == 1:

                dims = [t_dim]

            # j statistic
            # arr shape: (time, j)
            elif arr.ndim == 2:

                dims = [t_dim, j_dim]

                coords[j_dim] = g[j_dim]

            # jk statistic
            # arr shape: (time, j, k)
            elif arr.ndim == 3:

                dims = [
                    t_dim,
                    j_dim,
                    k_dim,
                ]

                coords[j_dim] = g[j_dim]

                coords[k_dim] = g[k_dim]

            else:

                raise ValueError(
                    f"{key}: unsupported ndim {arr.ndim}"
                )

        data_vars[key] = xr.DataArray(
            arr,
            dims=dims,
            coords=coords,
        )

    # ----------------------------------------------------------
    # Construct dataset
    # ----------------------------------------------------------
    ds = xr.Dataset(data_vars)

    # ----------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------
    ds.attrs["window_length"] = int(window_length)

    ds.attrs["block_length"] = int(block_length)

    ds.attrs["bootstrap"] = bool(bootstrap)

    ds.attrs["n_boot"] = int(n_boot)

    ds.attrs["store_bootstrap_samples"] = bool(
        store_bootstrap_samples
    )

    return ds


def compute_relative_nonstationarity(
    variance_series):
    
    """
    Compute normalized temporal variability of a variance series.

    Parameters
    ----------
    variance_series : array-like
        Time series of variance estimates.

    Returns
    -------
    dict
        Mean variance, temporal standard deviation, and relative
        nonstationarity metric.
    """

    # Mean variance level
    mean_variance = float(variance_series.mean())

    # Temporal variability of variance
    std_variance = float(variance_series.std())

    return {
        "mean": mean_variance,
        "std": std_variance,

        # Normalized measure of variance nonstationarity
        "relative_nonstationarity": std_variance / mean_variance,
    }

def compute_preprocessing_variance_statistics(
    g,
    g_preprocessed,
    j_dim="j",
    k_dim="k"):
    """
    Compute ensemble variance statistics before and after preprocessing.

    Parameters
    ----------
    g : xarray.DataArray
        Original dataset.
    g_preprocessed : xarray.DataArray
        Preprocessed dataset.
    j_dim, k_dim : str, optional
        Names of ensemble dimensions.

    Returns
    -------
    dict
        Variance diagnostics for original and preprocessed data.
    """

    # Ensemble variance before preprocessing
    ensemble_variance_original = g.var(
        dim=[j_dim, k_dim],
        ddof=1,
    )

    # Temporal mean ensemble variance
    ensemble_variance_residual = g_preprocessed.var(
        dim=[j_dim, k_dim],
        ddof=1,
    )

    # Temporal mean ensemble variance
    tmean_var_original = float(
        ensemble_variance_original.mean()
    )

    tmean_var_residual = float(
        ensemble_variance_residual.mean()
    )

    # Normalise variance series by temporal mean
    ensemble_variance_original_norm = (
        ensemble_variance_original / tmean_var_original
    )

    ensemble_variance_residual_norm = (
        ensemble_variance_residual / tmean_var_residual
    )

    # Temporal standard deviation of variance
    tstd_var_original = float(
        ensemble_variance_original.std()
    )

    tstd_var_residual = float(
        ensemble_variance_residual.std()
    )

    return {
        "ensemble_variance_original": ensemble_variance_original,
        "ensemble_variance_residual": ensemble_variance_residual,
        "ensemble_variance_original_norm": ensemble_variance_original_norm,
        "ensemble_variance_residual_norm": ensemble_variance_residual_norm,
        "tmean_var_original": tmean_var_original,
        "tmean_var_residual": tmean_var_residual,
        "tstd_var_original": tstd_var_original,
        "tstd_var_residual": tstd_var_residual,
    }








    '''
# These two functions are not currently used in the analysis framework:
# - compute_block_statistics_sliding_forwards_trend_params
# - forward_sliding_trend_nan_robust
# They were part of the testing of whether the trend-based rapid ice loss event (RILE)
# could be analysed consistently in the framework.

def compute_block_statistics_sliding_forwards_trend_params(y, trwindow=5, min_valid=3):
    b0, b1 = forward_sliding_trend_nan_robust(y, trwindow, min_valid)
    return np.stack([b0, b1], axis=0)

def forward_sliding_trend_nan_robust(y, window, min_valid=3):
    """
    Calculates the parameters \beta_0 (intercept) and \beta_1 (slope)
    over a sliding trend window, using ordinary least squares.
    Note that this uses the analytical version of the optimal 
    ordinary least-squares solution for the trend parameters, 
    which is a unique solution.
    """
    
    y = np.asarray(y)

    # Force strict 1D behavior
    if y.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape {y.shape}")

    n = len(y)
    t_idx = np.arange(n)

    beta0 = np.full(n, np.nan)
    beta1 = np.full(n, np.nan)

    for t in range(n):
        end = min(t + window, n)

        ys = y[t:end]
        s = t_idx[t:end]

        mask = np.isfinite(ys)   # safer than ~np.isnan

        if mask.sum() < min_valid:
            continue

        ys = ys[mask]
        s = s[mask]

        s_mean = s.mean()
        y_mean = ys.mean()

        cov = np.mean((s - s_mean) * (ys - y_mean))
        var = np.mean((s - s_mean) ** 2)

        beta1[t] = cov / var if var > 0 else np.nan
        beta0[t] = y_mean - beta1[t] * s_mean

    return beta0, beta1

'''