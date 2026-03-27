"""
models/simulation.py
Monte Carlo simulation: 100,000 paths over 15 years using GBM.

Simulates future portfolio value using Geometric Brownian Motion,
which assumes log-normally distributed returns with constant drift and
volatility estimated from historical data.
"""

import numpy as np
import pandas as pd
from arch import arch_model

def run_gbm_simulation(
    returns: pd.Series,
    initial_value: float,
    years: int = 15,
    n_paths: int = 100_000,
    trading_days: int = 252,
) -> np.ndarray:
    """
    Simulate future portfolio value using Geometric Brownian Motion.

    GBM formula:
        S(t+1) = S(t) * exp((mu - 0.5 * sigma**2) + sigma * Z)
        where Z ~ N(0, 1) and parameters are in daily units.

    Parameters
    returns         : historic log returns of the portfolio
    initial_value   : starting portfolio value of the simulation
    years           : simulation horizon in years
    n_paths         : number of simulated paths
    trading_days    : trading days per year

    Returns
    paths : np.ndarray of shape (n_steps + 1, n_paths)
            paths[0] = starting portfolio for all paths
    """
    n_steps = years * trading_days

    # Estimate mean and volatility from historical returns
    mu    = returns.mean()
    sigma = returns.std()

    # Simulate paths step by step, drawing fresh shocks each day
    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = initial_value

    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * Z)

    return paths

def run_garch_simulation(
    returns: pd.Series,
    initial_value: float,
    years: int = 15,
    n_paths: int = 100_000,
    trading_days: int = 252,
) -> np.ndarray:
    """
    Simulate future portfolio value using GARCH(1,1).

    The variance update follows:
        sigma(t+1)^2 = omega + alpha * X(t)^2 + beta * sigma(t)^2
        where X(t) is the simulated return at time t.

    Parameters
    returns         : daily log returns of the portfolio (historical)
    initial_value   : starting portfolio value
    years           : simulation horizon in years
    n_paths         : number of simulated paths
    trading_days    : trading days per year

    Returns
    paths : np.ndarray of shape (n_steps + 1, n_paths)
    """

    n_steps = years * trading_days

    # Scale returns by 100 for numerical stability during fitting
    scaled_returns = returns * 100

    model = arch_model(
        scaled_returns,
        vol="Garch",   # GARCH volatility model
        p=1,           # lag order of squared returns (alpha)
        q=1,           # lag order of variance (beta)
        dist="normal",
    )

    result = model.fit(disp="off")

    # Extract fitted parameters
    omega = result.params["omega"]
    alpha = result.params["alpha[1]"]
    beta  = result.params["beta[1]"]
    mu = result.params["mu"]

    # Starting variance from last fitted conditional variance
    initial_variance = result.conditional_volatility.iloc[-1] ** 2

    # Simulate paths
    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = initial_value

    # All paths start at the last observed conditional variance
    variances = np.full(n_paths, initial_variance)
    
    # Initialise with the last observed scaled return
    r = returns.iloc[-1] * 100

    for t in range(1, n_steps + 1):
        
        # Update variance using previous sigma and return
        variances = omega + alpha * (r - mu)**2 + beta * variances
        
        # Draw random shocks
        Z = np.random.standard_normal(n_paths)

        # Simulate return using updated variance
        sigma = np.sqrt(variances)
        r = mu + sigma * Z
        daily_return = np.exp(r / 100)
        paths[t] = paths[t - 1] * daily_return
        
    return paths


def simulation_stats(paths: np.ndarray) -> dict:
    """
    Compute summary statistics of the simulation at the final time step.

    Returns a dictionary with the following keys:
        mean, median, std, p5, p25, p75, p95, min, max
    """
    final = paths[-1]
    return {
        "mean"  : float(np.mean(final)),
        "median": float(np.median(final)),
        "std"   : float(np.std(final)),
        "p5"    : float(np.percentile(final, 5)),
        "p25"   : float(np.percentile(final, 25)),
        "p75"   : float(np.percentile(final, 75)),
        "p95"   : float(np.percentile(final, 95)),
        "min"   : float(np.min(final)),
        "max"   : float(np.max(final)),
    }

def expected_shortfall(paths: np.ndarray, confidence: float = 0.95) -> float:
    """
    Compute Expected Shortfall (ES) at the given confidence level.

    Parameters
    paths       : simulation output from run_monte_carlo
    confidence  : confidence level, default 0.95

    Returns
    float : average value in the worst tail
    """
    final  = paths[-1]
    cutoff = np.percentile(final, (1 - confidence) * 100)
    tail   = final[final <= cutoff]
    return float(np.mean(tail))