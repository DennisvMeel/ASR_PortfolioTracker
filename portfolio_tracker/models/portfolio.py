"""
models/portfolio.py
Model layer: stores asset data and performs all calculations.
"""

import json
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Asset:
    """
    Represents a single asset in the portfolio.
    
    Attributes
    ticker          : stock ticker symbol
    sector          : sector the asset belongs to
    asset_class     : asset class
    quantity        : number of units held
    purchase_price  : price per unit at time of purchase
    current_price   : latest market price per unit (updated through controller)
    """
    ticker: str
    sector: str
    asset_class: str
    quantity: float
    purchase_price: float
    current_price: float = 0.0

    @property
    def transaction_value(self) -> float:
        """Total cost of the transaction made at purchase."""
        return self.quantity * self.purchase_price

    @property
    def current_value(self) -> float:
        """Current market value."""
        return self.quantity * self.current_price

    @property
    def profit_loss(self) -> float:
        """Absolute difference of current vs purchase."""
        return self.current_value - self.transaction_value

    @property
    def profit_loss_pct(self) -> float:
        """Difference in value as a percentage of the purchase."""
        if self.transaction_value == 0:
            return 0.0
        return (self.profit_loss / self.transaction_value) * 100

ACTIVE_FILE = "data/active_portfolio.txt"

class Portfolio:
    """
    Represents the full investment portfolio.
    
    Responsible for:
    - Saving asset data to a JSON file
    - Adding and removing assets
    - Updating current prices
    - Computing metrics for the entire portfolio
    """

    def __init__(self, name: str = "data/default.json"):
        self.name = name
        self.data_file = f"data/{name}.json"
        self.assets: list[Asset] = []
        self._ensure_data_dir()
        self.load()
        
        
    # Static methods for loading/saving portfolios
    @staticmethod
    def get_active_name() -> str:
        """Read which portfolio is currently active."""
        if not os.path.exists(ACTIVE_FILE):
            return "default"
        with open(ACTIVE_FILE) as f:
            return f.read().strip()

    @staticmethod
    def set_active_name(name: str):
        """Save the active portfolio name."""
        with open(ACTIVE_FILE, "w") as f:
            f.write(name)

    @staticmethod
    def list_portfolios() -> list[str]:
        """Return all saved portfolio names."""
        if not os.path.exists("data"):
            return []
        return [
            f.replace(".json", "")
            for f in os.listdir("data")
            if f.endswith(".json")
        ]

    # Saving data 
    def _ensure_data_dir(self):
        """Create the data directory if it does not exist."""
        os.makedirs("data", exist_ok=True)

    def save(self):
        """Serialise all assets to JSON and write to disk."""
        with open(self.data_file, "w") as f:
            json.dump([asdict(a) for a in self.assets], f, indent=2)

    def load(self):
        """Load assets from JSON file if it exists."""
        if not os.path.exists(self.data_file):
            return
        with open(self.data_file) as f:
            data = json.load(f)
        self.assets = [Asset(**d) for d in data]

    # Managing assets
    def add_asset(self, ticker: str, sector: str, asset_class: str,
                  quantity: float, purchase_price: float) -> Asset:
        """
        Add a new asset or increase an existing position.
        
        If the asset already exists, quantity is increased and the
        purchase price is updated to the average of the old and new prices.
        """
        ticker = ticker.upper()
        for asset in self.assets:
            if asset.ticker == ticker:
                # Update existing position with weighted average price
                asset.quantity += quantity
                asset.purchase_price = (
                    (asset.purchase_price * (asset.quantity - quantity) +
                     purchase_price * quantity) / asset.quantity
                )
                self.save()
                return asset
        # If not existing, append to the asset list
        asset = Asset(ticker, sector, asset_class, quantity, purchase_price)
        self.assets.append(asset)
        self.save()
        return asset

    def remove_asset(self, ticker: str, quantity: float = None) -> bool:
        """
        Remove an asset or reduce its quantity
        
        If quantity is specified, reduces the position by that amount.
        If quantity is not specified or equals the total held, removes
        the asset entirely.
        """
        ticker = ticker.upper() # Ensure that it is uppercase
        asset = self.get_asset(ticker)

        # Check if asset is present
        if asset is None:
            return False
        
        # Prevent removing more than is held
        if quantity is not None and quantity > asset.quantity:
            return None
    
        # If no quantity specified or quantity = held, remove entirely
        if quantity is None or quantity == asset.quantity:
            self.assets = [a for a in self.assets if a.ticker != ticker]
        else:
            asset.quantity -= quantity
    
        self.save()
        return True

    def get_asset(self, ticker: str) -> Optional[Asset]:
        """Return the Asset object for a given ticker, or None if not found."""
        return next((a for a in self.assets if a.ticker == ticker.upper()), None)

    def update_prices(self, prices: dict[str, float]):
        """
        Update current prices for all assets.
        Called by the controller after fetching live prices from yfinance.
        """
        for asset in self.assets:
            if asset.ticker in prices:
                asset.current_price = prices[asset.ticker]
        self.save()

    # Simple calculations

    @property
    def total_value(self) -> float:
        """Sum of current market value across all assets."""
        return sum(a.current_value for a in self.assets)

    @property
    def total_cost(self) -> float:
        """Sum of purchase cost across all assets."""
        return sum(a.transaction_value for a in self.assets)

    def asset_weights(self) -> dict[str, float]:
        """
        Compute each asset's weight as a fraction of total portfolio value.
        Returns a dict mapping ticker-weight.
        """
        total = self.total_value
        if total == 0:
            return {a.ticker: 0.0 for a in self.assets} # Returns zero if the portfolio has no current value
        return {a.ticker: a.current_value / total for a in self.assets}

    def weights_by_sector(self) -> dict[str, float]:
        """
        Aggregate portfolio weights by sector.
        Returns a dict mapping sector-weight.
        """
        totals: dict[str, float] = {}
        for a in self.assets:
            totals[a.sector] = totals.get(a.sector, 0) + a.current_value
        total = self.total_value
        if total == 0:
            return {k: 0.0 for k in totals}
        return {k: v / total for k, v in totals.items()}

    def weights_by_asset_class(self) -> dict[str, float]:
        """
        Aggregate portfolio weights by asset class.
        Returns a dict mapping class-weight.
        """
        totals: dict[str, float] = {}
        for a in self.assets:
            totals[a.asset_class] = totals.get(a.asset_class, 0) + a.current_value
        total = self.total_value
        if total == 0:
            return {k: 0.0 for k in totals}
        return {k: v / total for k, v in totals.items()}
    
    def risk_contribution(self, returns: pd.DataFrame) -> dict[str, float]:
        """
        Compute each asset's contribution to total portfolio volatility.
        
        Uses current portfolio weights as a static approximation.
        Weights are assumed constant over the history period.

        Risk contribution is calculated as:
        RC_i = w_i * (Sigma * w)_i / sigma_portfolio
        
        Parameters
        returns : DataFrame of daily log returns per asset
        
        Returns
        dict mapping ticker-risk contribution
        """
        weights = self.asset_weights()
    
        # Only include tickers present in returns
        common  = [t for t in weights if t in returns.columns]
        w = np.array([weights[t] for t in common])

        # Covariance matrix of asset returns
        cov = returns[common].cov().values

        # Portfolio volatility
        portfolio_vol = np.sqrt(w @ cov @ w)

        if portfolio_vol == 0:
            return {t: 0.0 for t in common}

        # Marginal and total risk contribution
        marginal = cov @ w
        rc = w * marginal
        total_rc = rc.sum()

        return {t: float(rc[i] / total_rc) for i, t in enumerate(common)}

    def individual_sharpe(self, returns: pd.DataFrame) -> dict[str, float]:
        """
        Compute annualised Sharpe ratio for each asset individually.
            
        Parameters
        returns : DataFrame of daily log returns per asset

        Returns
        dict mapping ticker-annualised Sharpe ratio
        """
        result = {}
        for ticker in returns.columns:
            r = returns[ticker].dropna()
            if r.std() == 0:
                result[ticker] = 0.0
            else:
                result[ticker] = float((r.mean() / r.std()) * np.sqrt(252))
        return result
    
    