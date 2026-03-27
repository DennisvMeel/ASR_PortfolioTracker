"""
models/simulation.py
Monte Carlo simulation: 100,000 paths over 15 years using GBM.

Simulates future portfolio value using Geometric Brownian Motion,
which assumes log-normally distributed returns with constant drift and
volatility estimated from historical data.
"""

import numpy as np
import pandas as pd


def run_monte_carlo(
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