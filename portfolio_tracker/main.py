"""
main.py
CLI entry point — wires Click commands to the controller.
"""

import click
from controllers.portfolio_controller import PortfolioController

controller = PortfolioController()


@click.group()
def cli():
    """Portfolio Tracker — manage and analyse your investment portfolio."""
    pass


# Asset management                                                    #
@cli.command()
@click.option("--ticker",         "-t", required=True,  help="Asset ticker, e.g. AAPL")
@click.option("--sector",         "-s", required=True,  help="Sector, e.g. Technology")
@click.option("--asset-class",    "-c", required=True,  help="Asset class, e.g. Equity")
@click.option("--quantity",       "-q", required=True,  type=float, help="Number of units")
@click.option("--purchase-price", "-p", required=True,  type=float, help="Purchase price per unit")
def add(ticker, sector, asset_class, quantity, purchase_price):
    """Add an asset to the portfolio."""
    controller.add_asset(ticker, sector, asset_class, quantity, purchase_price)


@cli.command()
@click.option("--ticker", "-t", required=True, help="Ticker to remove")
def remove(ticker):
    """Remove an asset from the portfolio."""
    controller.remove_asset(ticker)


# Portfolio view                                                      #
@cli.command()
@click.option("--no-refresh", is_flag=True, default=False,
              help="Skip live price refresh (use cached prices)")
def view(no_refresh):
    """View the full portfolio with current values and weights."""
    controller.show_portfolio(refresh=not no_refresh)


@cli.command()
@click.option("--by", "-b",
              type=click.Choice(["asset", "sector", "class"]),
              default="asset",
              show_default=True,
              help="Group weights by asset, sector, or asset class")
def weights(by):
    """Show portfolio weights."""
    controller.show_weights(by=by)


# Entry point                                                         #

if __name__ == "__main__":
    cli()