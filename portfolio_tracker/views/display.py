"""
views/display.py
View layer: all terminal output — tables, charts, and graphs.

"""
import pandas as pd
import matplotlib.pyplot as plt

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

console = Console()


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