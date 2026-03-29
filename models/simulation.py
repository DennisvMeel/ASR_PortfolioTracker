"""
models/simulation.py
Monte Carlo simulation: 100,000 paths over 15 years using GBM.

Simulates future portfolio value using Geometric Brownian Motion,
which assumes log-normally distributed returns with constant drift and
volatility estimated from historical data.
"""

import numpy as np
import pandas as pd
import os
import warnings

from arch import arch_model
from scipy import stats
from scipy.stats import t as student_t
from hmmlearn.hmm import GaussianHMM

os.environ.setdefault("OMP_NUM_THREADS", "5")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Model is not converging*")

def draw_shocks(n: int, method: str = "normal", fitted_returns: pd.Series = None, df: float = None):
    if method == "normal":
        return np.random.standard_normal(n)
    elif method == "student-t":
        raw = np.random.standard_t(df, size=n)
        return raw / np.sqrt(df / (df - 2))
    elif method == "edf":
        std_returns = (fitted_returns - fitted_returns.mean()) / fitted_returns.std()
        return np.random.choice(std_returns.values, size=n, replace=True)
    else:
        raise ValueError(f"Unknown distribution: {method}")

def run_gbm_simulation(
    returns: pd.Series,
    initial_value: float,
    years: int = 15,
    n_paths: int = 100_000,
    trading_days: int = 252,
    dist: str = "normal",
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
    
    # Fit df once before the loop (if applicable)
    df = None
    if dist == "student-t":
        standardised = (returns - returns.mean()) / returns.std()
        df, _, _ = student_t.fit(standardised, floc=0, fscale=1)

    for t in range(1, n_steps + 1):
        Z = draw_shocks(n_paths, method=dist, fitted_returns=returns, df=df)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * Z)

    return paths

def run_garch_simulation(
    returns: pd.Series,
    initial_value: float,
    years: int = 15,
    n_paths: int = 100_000,
    trading_days: int = 252,
    dist: str = "normal",
) -> np.ndarray:
    """
    Simulate future portfolio value using GARCH(1,1).

    The variance update follows:
        sigma(t+1)^2 = omega + alpha * r(t)^2 + beta * sigma(t)^2
        where r(t) is the simulated return at time t.

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
    
    # Fit df once before the loop
    df = None
    if dist == "student-t":
        standardised = (returns - returns.mean()) / returns.std()
        df, _, _ = student_t.fit(standardised, floc=0, fscale=1)

    for t in range(1, n_steps + 1):
        
        # Update variance using previous sigma and return
        variances = omega + alpha * (r - mu)**2 + beta * variances
        
        # Draw random shocks from specified distribution
        Z = draw_shocks(n_paths, method=dist, fitted_returns=returns, df=df)

        # Simulate return using updated variance
        sigma = np.sqrt(variances)
        r = mu + sigma * Z
        daily_return = np.exp(r / 100)
        paths[t] = paths[t - 1] * daily_return
        
    return paths

def run_regime_simulation(
    returns: pd.Series,
    initial_value: float,
    years: int = 15,
    n_paths: int = 100_000,
    trading_days: int = 252,
    n_regimes: int = 2,
) -> np.ndarray:
    """
    Simulate future portfolio value using a Regime-Switching model.

    Fits a Hidden Markov Model (HMM) with n_regimes states on the
    historical returns. Each regime has its own mean and variance.
    At each step the simulation probabilistically switches between
    regimes based on the fitted transition matrix.

    Parameters
    returns      : daily log returns of the portfolio (historical)
    initial_value: starting portfolio value
    years        : simulation horizon in years
    n_paths      : number of simulated paths
    trading_days : trading days per year
    n_regimes    : number of regimes, default 2 (bull and bear)

    Returns
    paths : np.ndarray of shape (n_steps + 1, n_paths)
    """

    n_steps = years * trading_days
    
    # Standardize
    mu = returns.mean()
    std = returns.std()
    X = ((returns - mu) / std).values.reshape(-1, 1)

    # Fit HMM on historical returns
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=3000,
        tol=1e-6, 
        random_state=42,
    )
    model.fit(X)

    # Extract fitted parameters
    means   	= model.means_.flatten() * std + mu        # mean return per regime
    stds        = np.sqrt(model.covars_.flatten()) * std   # std per regime
    transmat    = model.transmat_                          # transition matrix
    startprob   = model.startprob_                         # starting probabilities

    # Identify bull and bear regimes by mean return
    # (just for printing — simulation uses all regimes)
    bull = np.argmax(means)
    bear = np.argmin(means)
    print(f"Bull regime (state {bull}): mu={means[bull]*252:.2%}, "
          f"sigma={stds[bull]*np.sqrt(252):.2%}")
    print(f"Bear regime (state {bear}): mu={means[bear]*252:.2%}, "
          f"sigma={stds[bear]*np.sqrt(252):.2%}")

    # Simulate paths
    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = initial_value

    # Each path starts in a regime drawn from startprob
    current_regimes = np.random.choice(n_regimes, size=n_paths, p=startprob)

    for t in range(1, n_steps + 1):
        # Draw returns based on current regime for each path
        Z = np.random.standard_normal(n_paths)
        mu_t    = means[current_regimes]
        sigma_t = stds[current_regimes]
        daily_return = np.exp(mu_t + sigma_t * Z)
        paths[t] = paths[t - 1] * daily_return

        # Transition to next regime for each path
        new_regimes = np.empty(n_paths, dtype=int)
        for regime in range(n_regimes):
            mask = current_regimes == regime
            if mask.any():
                new_regimes[mask] = np.random.choice(
                    n_regimes,
                    size=mask.sum(),
                    p=transmat[regime],
                )
        current_regimes = new_regimes

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

def test_distribution(returns: pd.Series, method: str = "gbm",) -> dict:
    """
    Test which distribution best fits the model residuals.

    For GBM, residuals are the standardised returns.
    For GARCH, residuals are the standardised innovations from the fitted GARCH(1,1) model.

    Parameters
    returns : daily log returns of the portfolio
    method  : 'gbm' or 'garch'

    Returns
    dict containing test statistics, p-values, fitted df, and recommendation
    """

    if method == "garch":
        # Fit GARCH and extract standardised residuals
        scaled_returns = returns * 100
        model  = arch_model(scaled_returns, vol="Garch", p=1, q=1, dist="normal")
        result = model.fit(disp="off")
        residuals = pd.Series(result.std_resid.dropna().values)
    else:
        # Standardise raw returns for GBM
        residuals = (returns - returns.mean()) / returns.std()

    # Shapiro-Wilk test (null: data is normally distributed)
    sw_stat, sw_pval = stats.shapiro(residuals)

    # Jarque-Bera test (null: data is normally distributed)
    jb_stat, jb_pval = stats.jarque_bera(residuals)

    # Kolmogorov-Smirnov test against normal distribution
    ks_stat, ks_pval = stats.kstest(residuals, "norm")

    # Fit Student-t distribution to get degrees of freedom
    df, loc, scale = stats.t.fit(residuals, floc=0, fscale=1)

    # Compute skewness and kurtosis
    skewness = float(stats.skew(residuals))
    kurtosis = float(stats.kurtosis(residuals))

    # If all tests fail normality, recommend t or EDF based on df
    normal = sw_pval > 0.05 and jb_pval > 0.05 and ks_pval > 0.05
    if normal:
        recommendation = "normal"
    elif df < 10:
        recommendation = "student-t"
    else:
        recommendation = "edf"

    return {
        "residuals"     : residuals,
        "sw_stat"       : float(sw_stat),
        "sw_pval"       : float(sw_pval),
        "jb_stat"       : float(jb_stat),
        "jb_pval"       : float(jb_pval),
        "ks_stat"       : float(ks_stat),
        "ks_pval"       : float(ks_pval),
        "df"            : float(df),
        "skewness"      : skewness,
        "kurtosis"      : kurtosis,
        "recommendation": recommendation,
    }