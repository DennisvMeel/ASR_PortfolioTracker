"""
controllers/portfolio_controller.py
Controller layer: orchestrates data flow between Model and View.

Manages user commands by fetching data from the model, retrieving
live prices via yfinance, and passing results to the view for display.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from rich.console import Console

from models.portfolio import Portfolio
from models.simulation import (
    run_gbm_simulation,
    run_garch_simulation,
    run_regime_simulation,
    simulation_stats,
    expected_shortfall,
    test_distribution,
)
from views.display import (
    show_portfolio_table,
    show_summary,
    show_weights_table,
    show_price_chart_matplotlib,
    show_simulation_stats,
    show_simulation_chart,
    show_portfolio_list,
    show_risk_table,
    show_distribution_test,
    show_optimal_weights
)
console = Console()


class PortfolioController:
    """
    Main controller class.
    
    Instantiated once at startup and shared across all CLI commands.
    Holds a reference to the active Portfolio model and delegates
    rendering to the view functions.
    """

    def __init__(self):
        active = Portfolio.get_active_name()
        self.portfolio = Portfolio(name=active)

    # Asset management
    def add_asset(self, ticker: str, sector: str, asset_class: str,
                  quantity: float, purchase_price: float):
        """
        Validate the ticker via yfinance, fetch its current price,
        then add it to the portfolio model.
        """
        ticker = ticker.upper()

        # Validate ticker against yfinance before adding
        with console.status(f"Validating ticker {ticker}..."):
            # Validate for the last 5 days in case of weekends/holidays
            data = yf.download(ticker, period="5d", auto_adjust=True, progress=False)

        if data.empty:
            console.print(f"'{ticker}' could not be found on yfinance. "
                          f"Please check the ticker and try again.")
            return

        # Get the current price from the most recent close
        current_price = float(data["Close"].iloc[-1])

        asset = self.portfolio.add_asset(ticker, sector, asset_class,
                                         quantity, purchase_price)
        asset.current_price = current_price
        self.portfolio.save()

        console.print(f"Added {ticker} — {quantity} units @ "
                      f"€{purchase_price:.2f} (current price: €{current_price:.2f})")

    def remove_asset(self, ticker: str, quantity: float = None):
        """Remove or reduce an asset from the portfolio by ticker."""
        result = self.portfolio.remove_asset(ticker, quantity)
        
        if result is None:
            console.print(f"Cannot remove {quantity} units of {ticker.upper()} "
                      f"— you only hold {self.portfolio.get_asset(ticker.upper()).quantity} units.")
        elif result:
            if quantity:
                console.print(f"Removed {quantity} units of {ticker.upper()} from portfolio.")
            else:
                console.print(f"Removed {ticker.upper()} from portfolio entirely.")
        else:
            console.print(f"Ticker {ticker.upper()} not found in portfolio.")

    # Price refreshing
    def refresh_prices(self):
        """Get current prices for all tickers via yfinance."""
        tickers = [a.ticker for a in self.portfolio.assets]
        if not tickers:
            console.print("No assets in portfolio.")
            return

        with console.status("Fetching live prices..."):
            data = yf.download(tickers, period="1d", auto_adjust=True, progress=False)

        if data.empty:
            console.print("Could not fetch prices. Check your internet connection.")
            return

        # Handle single vs multiple tickers — yfinance returns different shapes
        prices = {}
        if len(tickers) == 1:
            prices[tickers[0]] = float(data["Close"].iloc[-1])
        else:
            for t in tickers:
                try:
                    prices[t] = float(data["Close"][t].iloc[-1])
                except Exception:
                    console.print(f"Warning: could not get price for {t}")

        self.portfolio.update_prices(prices)
        
    # Portfolio edit commands
    def new_portfolio(self, name: str):
        """Create a new portfolio and switch to it."""
        if name in Portfolio.list_portfolios():
            console.print(f"Portfolio '{name}' already exists, please choose a new name.")
            return
        self.portfolio = Portfolio(name=name)
        Portfolio.set_active_name(name)
        self.portfolio.save()
        console.print(f"Created and switched to portfolio '{name}'")

    def switch_portfolio(self, name: str):
        """Switch to an existing portfolio."""
        if name not in Portfolio.list_portfolios():
            console.print(f"Portfolio '{name}' does not exist.")
            return
        self.portfolio = Portfolio(name=name)
        Portfolio.set_active_name(name)
        console.print(f"Switched to portfolio '{name}'")

    def list_portfolios(self):
        """List all saved portfolios."""
        portfolios = Portfolio.list_portfolios()
        active = Portfolio.get_active_name()
        show_portfolio_list(portfolios, active)

    def delete_portfolio(self, name: str):
        """Delete a portfolio by name."""
        if name not in Portfolio.list_portfolios():
            console.print(f"Portfolio '{name}' does not exist.")
            return
        if name == Portfolio.get_active_name():
            console.print("Cannot delete the active portfolio.")
            return
        os.remove(f"data/{name}.json")
        console.print(f"Deleted portfolio '{name}'")

    # View commands

    def show_portfolio(self, refresh: bool = True):
        """
        Display the full portfolio table and summary panel.
        Optionally refreshes live prices before rendering.
        """
        if refresh:
            self.refresh_prices()
        weights = self.portfolio.asset_weights()
        show_portfolio_table(self.portfolio.assets, weights)
        show_summary(self.portfolio.total_value, self.portfolio.total_cost)

    def show_weights(self, by: str = "asset"):
        """
        Display portfolio weights grouped by asset, sector, or asset class.
        
        Parameters
        by : 'asset', 'sector', or 'class'
        """
        self.refresh_prices()
        if by == "sector":
            show_weights_table("Sector", self.portfolio.weights_by_sector())
        elif by == "class":
            show_weights_table("Asset Class", self.portfolio.weights_by_asset_class())
        else:
            show_weights_table("Ticker", self.portfolio.asset_weights())
            
    def get_price_history(self, tickers: list[str], period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical closing prices for one or more tickers via yfinance.
        
        Parameters
        tickers : list of ticker symbols
        period  : history period, e.g. '1y', '2y', '5y'
    
        Returns
        pd.DataFrame with dates as index and tickers as columns
        """
        with console.status(f"Fetching price history ({period})..."):
            raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
            if raw.empty:
                console.print("No data returned.")
                return pd.DataFrame()
            
            # Flatten MultiIndex columns if present (newer yfinance versions)
            if isinstance(raw.columns, pd.MultiIndex):
                hist = raw["Close"]
                if isinstance(hist, pd.Series):
                    hist = hist.to_frame(name=tickers[0])
                else:
                    if len(tickers) == 1:
                        hist = raw[["Close"]].rename(columns={"Close": tickers[0]})
                    else:
                        hist = raw["Close"]

        # Ensure column names are uppercase strings
        hist.columns = [c[0].upper() if isinstance(c, tuple) else str(c).upper() for c in hist.columns]
        return hist

    def show_prices(self, tickers: list[str], period: str = "1y", save: str = None):
        """
        Fetch and display a price history chart for one or more tickers.
        
        Parameters
        tickers : list of ticker symbols
        period  : history period
        save    : optional file path to save the chart as PNG
        """
        hist = self.get_price_history(tickers, period)
        if hist.empty:
            return
        
        show_price_chart_matplotlib(hist, tickers, save_path=save)
        
    def _weighted_portfolio_returns(self, hist: pd.DataFrame) -> pd.Series:
        """Compute daily weighted log returns for the current portfolio."""
        weights = self.portfolio.asset_weights()
        daily_returns = hist.apply(lambda col: np.log(col / col.shift(1))).dropna()
        w_series = pd.Series(weights)
        common = daily_returns.columns.intersection(w_series.index)
        w_norm = w_series[common] / w_series[common].sum()
        return (daily_returns[common] * w_norm).sum(axis=1)
        
    def run_simulation(self, method : str = "gbm", dist : str = "normal", years: int = 15, n_paths: int = 100_000, save: str = None):
        """
        Run a Monte Carlo simulation on the current portfolio.
        
        Fetches 5 years of price history, computes weighted portfolio
        returns, and simulates future portfolio value over the given horizon.

        Parameters
        method  : simulation method, either 'gbm' or 'garch'
        dist    : distribution used for drawing shocks, either 'normal', 'student-t' or 'edf'
        years   : simulation horizon in years
        n_paths : number of simulated paths
        save    : optional file path to save the chart as PNG
        """
        self.refresh_prices()
        tickers = [a.ticker for a in self.portfolio.assets]
        if not tickers:
            console.print("No assets to simulate.")
            return

        # Fetch 5 years of history for return estimation
        hist = self.get_price_history(tickers, period="5y")
        if hist.empty:
            return

        # Compute daily weighted portfolio log returns
        portfolio_returns = self._weighted_portfolio_returns(hist)
        
        initial_value = self.portfolio.total_value
        console.print(f"Running {n_paths:,} Monte Carlo paths over {years} years...")

        with console.status("Simulating..."):
            if method == "garch":
                paths = run_garch_simulation(portfolio_returns, initial_value,
                                             years=years, n_paths=n_paths, dist=dist)
            elif method == "gbm":
                paths = run_gbm_simulation(portfolio_returns, initial_value,
                                        years=years, n_paths=n_paths, dist=dist)
            elif method == "regime":
                paths = run_regime_simulation(portfolio_returns, initial_value,
                                            years=years, n_paths=n_paths)
            else:
                console.print(f"Method is not recognized")

            
        stats = simulation_stats(paths)
        es = expected_shortfall(paths)
        show_simulation_stats(stats, initial_value, years, n_paths, es, method=method, dist=dist)
        show_simulation_chart(paths, initial_value, years, n_paths, save_path=save, method=method, dist=dist)
    
    def show_risk(self, period: str = "1y"):
        """
        Compute and display risk contribution and individual Sharpe ratios
        for each asset in the portfolio.
        
        Note: uses current portfolio weights as a static approximation.
        
        Parameters
        period : history period for return estimation
        """
        
        self.refresh_prices()
        tickers = [a.ticker for a in self.portfolio.assets]
        if len(tickers) < 2:
            console.print("Need at least 2 assets for risk analysis.")
            return

        hist = self.get_price_history(tickers, period=period)
        if hist.empty:
            return

        # Compute daily log returns per asset
        returns = hist.apply(lambda col: np.log(col / col.shift(1))).dropna()

        weights = self.portfolio.asset_weights()
        risk_contribs = self.portfolio.risk_contribution(returns)
        individual_sharpes = self.portfolio.individual_sharpe(returns)

        show_risk_table(weights, risk_contribs, individual_sharpes)
        
    def run_distribution_test(self, method: str = "gbm", period: str = "1y"):
        """
        Fetch historical returns, compute model residuals, and run
        distribution tests to recommend the best shock distribution.
        
        Parameters
        method : simulation method, 'gbm' or 'garch'
        period : history period for return estimation
        """
        
        self.refresh_prices()
        tickers = [a.ticker for a in self.portfolio.assets]
        if not tickers:
            console.print("No assets in portfolio.")
            return

        hist = self.get_price_history(tickers, period=period)
        if hist.empty:
            return

        # Compute daily weighted portfolio log returns
        portfolio_returns = self._weighted_portfolio_returns(hist)
        
        results = test_distribution(portfolio_returns, method=method)
        show_distribution_test(results, method=method)
        
    def show_optimized_weights(self, period: str = "5y",
                             risk_free: float = 0.0):
        """
        Fetch historical returns and compute the optimal portfolio
        weights that maximise the Sharpe ratio.

        Parameters
        period    : history period for return estimation
        risk_free : annual risk-free rate, default 0.0
        """
        tickers = [a.ticker for a in self.portfolio.assets]
        if len(tickers) < 2:
            console.print("Need at least 2 assets for optimisation.")
            return

        hist = self.get_price_history(tickers, period=period)
        if hist.empty:
            return
        
        returns = hist.apply(lambda col: np.log(col / col.shift(1))).dropna()
        
        with console.status("Optimising portfolio weights..."):
            result = self.portfolio.optimal_weights(returns, risk_free=risk_free)
        show_optimal_weights(result)