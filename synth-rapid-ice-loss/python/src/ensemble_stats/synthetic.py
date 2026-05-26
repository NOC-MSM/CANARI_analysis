# Changelog:
 
# 11 May 2026: v0.1.0 - Chris Wilson - first release version.
 
# 14 May 2026: v0.1.1 - Chris Wilson - bugfixes and extra plots in the notebook.

# 26 May 2026: v0.2 - Chris Wilson - bugfixes and tidying.

# OpenAI ChatGPT was used to assist with aspects of code development and refinement.

import numpy as np
import pandas as pd
import xarray as xr
import cftime

def generate_synthetic_nested_ensemble(
    nt=180,
    gintercept=0.0,
    gslope=0.0,
    gseasonal_amp=1.0,
    gAR2_scale=1.0,
    nj=8,
    nk=5,
    R_jt_scale = 0.5,
    R_jkt_scale = 1.0,
    residual_scale=0.000,
    start_date="1950-01-21",
    seed=123):

    """
        SYNTHETIC HIERARCHICAL ENSEMBLE GENERATOR
    ══════════════════════════════════════════════════════════════════
    
    
    Goal:
    Construct a synthetic ensemble field with nested structure:
    
        g(j,k,t)
    
    containing:
    
        1. shared grand signal
        2. group-dependent variability
        3. member-dependent variability
        4. residual noise
    
    
    Hierarchy:
    
        grand signal
             │
             ▼
        parent groups (j)
             │
             ▼
        child members (k)
             │
             ▼
          time (t)

          
    INPUT PARAMETERS
    ═══════════════════════════════════════════════════════════════
    
    TIME DIMENSION
    ───────────────────────────────────────────────────────────────
    
    nt
        Number of time samples.
    
        Example:
            nt = 180  →  15 years of monthly data
    
    
    GRAND SIGNAL PARAMETERS
    ───────────────────────────────────────────────────────────────
    
    gintercept
        Constant offset of the shared grand signal.
    
    gslope
        Linear trend slope applied through time.
    
    gseasonal_amp
        Amplitude of sinusoidal seasonal cycle.
    
    gAR2_scale
        Amplitude of shared AR(2) red-noise variability.
    
    
    ENSEMBLE STRUCTURE
    ───────────────────────────────────────────────────────────────
    
    nj
        Number of parent groups.
    
        Examples:
            climate models
            macro-members
            forcing groups
    
    nk
        Number of child members within each parent group.
    
        Examples:
            initial-condition realizations
            ensemble members
            
    
    
    GROUP / MEMBER VARIABILITY
    ───────────────────────────────────────────────────────────────
    
    R_jt_scale
        Magnitude of group-dependent time-varying variability.
    
        Controls spread BETWEEN parent groups.
    
    R_jkt_scale
        Magnitude of member-dependent variability.
    
        Controls spread WITHIN parent groups.
    
    
    RESIDUAL NOISE
    ───────────────────────────────────────────────────────────────
    
    residual_scale
        Standard deviation of independent residual Gaussian noise.
    
    
    RANDOM SEED
    ───────────────────────────────────────────────────────────────
    
    seed
        Random seed for reproducible ensemble generation.
    
    
    
    OUTPUT
    ═══════════════════════════════════════════════════════════════
    
    Returns:
    
        xarray.DataArray
    
    
    Dimensions:
    ───────────────────────────────────────────────────────────────
    
    (j, k, t)
    
    where:
    
        j = parent group
        k = child member
        t = time
    
    
    Shape:
    ───────────────────────────────────────────────────────────────
    
    (nj, nk, nt)
    
    
    Coordinates:
    ───────────────────────────────────────────────────────────────
    
    j
        Integer parent-group index.
    
    k
        Integer child-member index.
    
    t
        Monthly datetime coordinate.
    
    
    Variable name:
    ───────────────────────────────────────────────────────────────
    
    "synthetic_field"
    
    
    DATA CONTENT
    ═══════════════════════════════════════════════════════════════
    
    Each time series contains:
    
        shared grand signal
        + group variability
        + member variability
        + residual noise
    
    
    Mathematically:
    
        g(j,k,t)
          =
          R_t
          + R_jt
          + R_jkt
          + ε
    
    
    Interpretation:
    ───────────────────────────────────────────────────────────────
    
    The output mimics a hierarchical nested ensemble with:
    
        externally forced variability
        inter-group variability
        intra-group variability
        temporally correlated persistence
        stochastic internal variability
    """
    
    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------
    # Construct monthly time coordinate using a CF-compliant
    # 360-day calendar
    # ----------------------------------------------------------
    start = pd.Timestamp(start_date)
    
    time = xr.date_range(
        start=cftime.Datetime360Day(
            start.year,
            start.month,
            start.day,
        ),
        periods=nt,
        freq="30D",
        calendar="360_day",
        use_cftime=True,
    )
    
    month_index = np.arange(nt)
    # ----------------------------------------------------------
    # Shared grand signal, R_t
    # ----------------------------------------------------------
    seasonal_cycle = gseasonal_amp * np.sin(
        2.0 * np.pi * month_index / 12.0
    )

    ar_noise = rng.normal(0.0, 1.0, nt)
    red_component = np.zeros(nt)

    phi1 = 1.6
    phi2 = -0.65

    for t in range(2, nt):
        red_component[t] = (
            phi1 * red_component[t - 1]
            + phi2 * red_component[t - 2]
            + ar_noise[t]
        )

    red_component = (
        red_component - np.mean(red_component)
    ) / np.std(red_component)

    red_component = gAR2_scale * red_component

    linear_trend = gintercept + gslope * month_index

    grand_signal = (
        linear_trend
        + seasonal_cycle
        + red_component
    )

    # ----------------------------------------------------------
    # Time-varying R_jt component
    # ----------------------------------------------------------
    R_jt_component = np.zeros((nj, nt))

    for j in range(nj):
        innovations = rng.normal(0.0, 1.0, nt)
        series = np.zeros(nt)

        phi1_j = 1.5
        phi2_j = -0.55
        
        for t in range(2, nt):
            series[t] = (
                phi1_j * series[t - 1]
                + phi2_j * series[t - 2]
                + innovations[t]
            )

        series = (series - np.mean(series)) / np.std(series)
        R_jt_component[j, :] = R_jt_scale * series

    # ----------------------------------------------------------
    # Time-varying R_jkt component
    # ----------------------------------------------------------
    R_jkt_component = np.zeros((nj, nk, nt))

    # Persistent mean offset for each (j, k)
    k_member_offsets = rng.normal(0.0, R_jkt_scale, (nj, nk))
    
    for j in range(nj):
        for k in range(nk):
            innovations = rng.normal(0.0, 1.0, nt)
            series = np.zeros(nt)
    
            phi1_k = 1.5
            phi2_k = -0.6
    
            for t in range(2, nt):
                series[t] = (
                    phi1_k * series[t - 1]
                    + phi2_k * series[t - 2]
                    + innovations[t]
                )
    
            series = (series - np.mean(series)) / np.std(series)
    
            # Persistent member-specific offset plus
            # time-varying fluctuation
            R_jkt_component[j, k, :] = (
                k_member_offsets[j, k]
                + 2.5 * R_jkt_scale * series
            )

    # ----------------------------------------------------------
    # Residual noise
    # ----------------------------------------------------------
    noise = rng.normal(0.0, residual_scale, (nj, nk, nt))

    # ----------------------------------------------------------
    # Final synthetic field
    # ----------------------------------------------------------
    data = (
        grand_signal[None, None, :]
        + R_jt_component[:, None, :]
        + R_jkt_component
        + noise
    )

    return xr.DataArray(
        data,
        dims=["j", "k", "t"],
        coords={
            "j": np.arange(nj),
            "k": np.arange(nk),
            "t": time,
        },
        name="synthetic_field",
    )

