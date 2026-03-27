"""
views/display.py
View layer: all terminal output — tables, charts, and graphs.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

console = Console()


def show_portfolio_list(portfolios: list[str], active: str):
    """
    Render a table listing all saved portfolios.

    Parameters
    portfolios : list of portfolio names
    active     : name of the currently active portfolio
    """
    table = Table(
        title="Saved Portfolios",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("Name")
    table.add_column("Status", justify="right")

    for name in portfolios:
        status = "active" if name == active else ""
        table.add_row(name, status)

    console.print(table)

# Portfolio tables

def show_portfolio_table(assets, weights: dict[str, float]):
    """
    Render a rich table showing all assets in the portfolio.

    Displays ticker, sector, asset class, quantity, purchase price,
    current price, cost, current value, profit/loss, and weight.

    Parameters
    assets  : list of Asset objects from the portfolio model
    weights : dict mapping ticker-portfolio weight
    """
    if not assets:
        console.print("Portfolio is empty. Add assets with 'add'.")
        return

    table = Table(
        title="Current Portfolio",
        box=box.ROUNDED,
        show_lines=True,
    )

    cols = [
        "Ticker",
        "Sector",
        "Class",
        "Qty",
        "Buy Price",
        "Cur Price",
        "Cost",
        "Value",
        "P&L",
        "P&L %",
        "Weight",
    ]
    for name in cols:
        table.add_column(name)

    for a in assets:
        pl     = a.profit_loss
        pl_str     = f"{pl:+,.2f}"
        pl_pct_str = f"{a.profit_loss_pct:+.2f}%"

        table.add_row(
            a.ticker,
            a.sector,
            a.asset_class,
            f"{a.quantity:,.4f}",
            f"€{a.purchase_price:,.2f}",
            f"€{a.current_price:,.2f}" if a.current_price else "—",
            f"€{a.transaction_value:,.2f}",
            f"€{a.current_value:,.2f}"  if a.current_price else "—",
            pl_str     if a.current_price else "—",
            pl_pct_str if a.current_price else "—",
            f"{weights.get(a.ticker, 0):.2%}",
        )

    console.print(table)


def show_summary(total_value: float, total_cost: float):
    """
    Render a summary panel showing total cost, value, and profit/loss.

    Parameters
    total_value : current market value of the portfolio
    total_cost  : original purchase cost of the portfolio
    """
    pl = total_value - total_cost

    text = Text()
    text.append(f"  Total Cost:     €{total_cost:>12,.2f}\n")
    text.append(f"  Total Value:    €{total_value:>12,.2f}\n")
    text.append(f"  Total P&L:      ")
    text.append(
        f"€{pl:>+12,.2f}  ({pl/total_cost*100:+.2f}%)\n" if total_cost else "—\n"
    )

    console.print(Panel(text, title="Portfolio Summary"))


def show_weights_table(label: str, weights: dict[str, float]):
    """
    Render a table showing portfolio weights with a visual bar.

    Parameters
    label   : column header, (Ticker, Sector, Asset Class)
    weights : dict mapping label-weight
    """
    
    table = Table(
        title=f"Weights by {label}",
        box=box.SIMPLE_HEAD,
    )
    table.add_column(label)
    table.add_column("Weight", justify="right")

    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        table.add_row(name, f"{w:.2%}")

    console.print(table)
    
def show_price_chart_matplotlib(hist: pd.DataFrame, tickers: list[str], 
                                save_path: str = None):
    """
    Plot historical closing prices for one or more tickers.

    Parameters
    ----------
    hist      : DataFrame with dates as index and tickers as columns
    tickers   : list of ticker symbols to plot
    save_path : optional file path to save the chart as PNG
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for ticker in tickers:
        if ticker in hist.columns:
            ax.plot(hist.index, hist[ticker], label=ticker, linewidth=1.5)
            
    ax.set_title(f"Price History: {', '.join(tickers)}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        console.print(f"Chart saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    
def show_simulation_stats(stats: dict, initial_value: float, years: int, 
                          n_paths: int, es: float = None, method: str = "gbm"):
    """
    Render a table showing Monte Carlo simulation summary statistics.

    Parameters
    stats         : output from simulation_stats()
    initial_value : starting portfolio value
    years         : simulation horizon in years
    n_paths       : number of simulated paths
    """
    table = Table(
        title=f"Monte Carlo Simulation - {method.upper()} ({years}-year horizon, {n_paths} paths)",
        box=box.ROUNDED,
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("vs. Today", justify="right")

    rows = [
        ("5th Percentile",  stats["p5"]),
        ("25th Percentile", stats["p25"]),
        ("Median",          stats["median"]),
        ("Mean",            stats["mean"]),
        ("75th Percentile", stats["p75"]),
        ("95th Percentile", stats["p95"]),
    ]
    for label, val in rows:
        ratio = val / initial_value if initial_value else 0
        table.add_row(label, f"€{val:,.0f}", f"{ratio:.2f}x")
        
    if es is not None:
        table.add_section()
        ratio = es / initial_value if initial_value else 0
        table.add_row("Expected Shortfall (95%)", f"€{es:,.0f}", f"{ratio:.2f}x")

    console.print(table)
    
def show_simulation_chart(paths: np.ndarray, initial_value: float,
                          years: int, n_paths: int, save_path: str = None,
                          method: str = "gbm"):
    """
    Plot a sample of simulation paths with percentile bands.

    Parameters
    paths         : simulation output from run_monte_carlo
    initial_value : starting portfolio value
    years         : simulation horizon in years
    n_paths       : number of simulated paths
    save_path     : optional file path to save the chart as PNG
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.linspace(0, years, paths.shape[0])
    
    # Downsample to 5000 paths for plotting — no visible difference
    if paths.shape[1] > 5000:
        idx = np.random.choice(paths.shape[1], size=5000, replace=False)
        plot_paths = paths[:, idx]
    else:
        plot_paths = paths
        
    # Reduce time steps to monthly intervals for faster plotting
    total_steps = plot_paths.shape[0]
    target_steps = years * 12
    step_size = max(1, total_steps // target_steps)
    plot_paths = plot_paths[::step_size, :]
    
    x = np.linspace(0, years, plot_paths.shape[0])

    # Draw 200 random sample paths
    sample_idx = np.random.choice(plot_paths.shape[1],
                                  size=min(200, plot_paths.shape[1]),
                                  replace=False)
    ax.plot(x, plot_paths[:, sample_idx], color="steelblue",
            alpha=0.08, linewidth=0.5)

    # Percentile bands
    p5  = np.percentile(plot_paths, 5,  axis=1)
    p25 = np.percentile(plot_paths, 25, axis=1)
    p50 = np.percentile(plot_paths, 50, axis=1)
    p75 = np.percentile(plot_paths, 75, axis=1)
    p95 = np.percentile(plot_paths, 95, axis=1)

    ax.fill_between(x, p5,  p95, alpha=0.15, color="steelblue", label="5-95th pct")
    ax.fill_between(x, p25, p75, alpha=0.25, color="steelblue", label="25-75th pct")
    ax.plot(x, p50, color="white", linewidth=2, label="Median")
    ax.axhline(initial_value, color="orange", linestyle="--",
               linewidth=1.2, label="Initial value")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.set_xlabel("Years")
    ax.set_ylabel("Portfolio Value")
    ax.set_title(f"Monte Carlo Simulation — {method.upper()} ({years}-Year Horizon, {n_paths} paths)")
    ax.legend()
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        console.print(f"Simulation chart saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)