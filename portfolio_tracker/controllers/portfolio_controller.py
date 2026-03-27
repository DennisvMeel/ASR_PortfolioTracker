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
from models.simulation import run_gbm_simulation, run_garch_simulation, simulation_stats, expected_shortfall
from views.display import (
    show_portfolio_table,
    show_summary,
    show_weights_table,
    show_price_chart_matplotlib,
    show_simulation_stats,
    show_simulation_chart,
    show_portfolio_list
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

    def remove_asset(self, ticker: str):
        """Remove an asset from the portfolio by ticker."""
        if self.portfolio.remove_asset(ticker):
            console.print(f"Removed {ticker.upper()} from portfolio.")
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
        
    def run_simulation(self, method : str = "gbm", years: int = 15, n_paths: int = 100_000, save: str = None):
        """
        Run a Monte Carlo simulation on the current portfolio.
        
        Fetches 5 years of price history, computes weighted portfolio
        returns, and simulates future portfolio value over the given horizon.

        Parameters
        method  : simulation method, either 'gbm' or 'garch'
        years   : simulation horizon in years
        n_paths : number of simulated paths
        save    : optional file path to save the chart as PNG
        """
        self.refresh_prices()
        tickers = [a.ticker for a in self.portfolio.assets]
        if not tickers:
            console.print("No assets to simulate.")
            return

        # Build portfolio weights
        weights = self.portfolio.asset_weights()

        # Fetch 5 years of history for return estimation
        hist = self.get_price_history(tickers, period="5y")
        if hist.empty:
            return

        # Compute daily weighted portfolio log returns
        daily_returns = hist.apply(lambda col: np.log(col / col.shift(1))).dropna()
        w_series = pd.Series(weights)
        common = daily_returns.columns.intersection(w_series.index)
        w_norm = w_series[common] / w_series[common].sum()
        portfolio_returns = (daily_returns[common] * w_norm).sum(axis=1)

        initial_value = self.portfolio.total_value
        console.print(f"Running {n_paths:,} Monte Carlo paths over {years} years...")

        with console.status("Simulating..."):
            if method == "garch":
                paths = run_garch_simulation(portfolio_returns, initial_value,
                                             years=years, n_paths=n_paths)
            elif method == "gbm":
                paths = run_gbm_simulation(portfolio_returns, initial_value,
                                        years=years, n_paths=n_paths)
            else:
                console.print(f"Method is not recognized")

            
        stats = simulation_stats(paths)
        es = expected_shortfall(paths)
        show_simulation_stats(stats, initial_value, years, n_paths, es, method=method)
        show_simulation_chart(paths, initial_value, years, n_paths, save_path=save, method=method)