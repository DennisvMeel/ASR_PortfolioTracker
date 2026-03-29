"""
main.py
CLI entry point — wires Click commands to the controller.
"""

import click
from controllers.portfolio_controller import PortfolioController

controller = PortfolioController()

class OrderedGroup(click.Group):
    def list_commands(self, ctx):
        return list(self.commands)

@click.group(cls=OrderedGroup)
def cli():
    """Portfolio Tracker — manage and analyse your investment portfolio."""
    pass

# Portfolio Management
@cli.command()
def list_portfolios():
    """List all saved portfolios."""
    controller.list_portfolios()
    
@cli.command()
@click.option("--name", "-n", required=True, help="Portfolio name")
def new_portfolio(name):
    """Create and switch to a new portfolio."""
    controller.new_portfolio(name)

@cli.command()
@click.option("--name", "-n", required=True)
def switch_portfolio(name):
    """Switch to an existing portfolio."""
    controller.switch_portfolio(name)

@cli.command()
@click.option("--name", "-n", required=True)
def delete_portfolio(name):
    """Delete a portfolio."""
    controller.delete_portfolio(name)


# Asset Management                                                    #
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
@click.option("--ticker",   "-t", required=True, help="Ticker to remove")
@click.option("--quantity", "-q", default=None,  type=float,
              help="Number of units to remove (omit to remove entirely)")
def remove(ticker, quantity):
    """Remove an asset or reduce its quantity."""
    controller.remove_asset(ticker, quantity)


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
    
@cli.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--period", "-p",
              default="1y", show_default=True,
              type=click.Choice(["1mo","3mo","6mo","1y","2y","5y"]),
              help="History period")
@click.option("--save", default=None, metavar="FILE",
              help="Save chart to a PNG file instead of showing it")
def prices(tickers, period, save):
    """Show price history for one or more tickers."""
    controller.show_prices(list(tickers), period=period, save=save)

# Historic risk-analysis
@cli.command()
@click.option("--period", "-p",
              type=click.Choice(["1mo","3mo","6mo","1y","2y","5y"]),
              default="1y", show_default=True,
              help="History period for return estimation")
def risk(period):
    """Show risk contribution and Sharpe ratio per asset."""
    controller.show_risk(period=period)
    
@cli.command()
@click.option("--period", "-p",
              type=click.Choice(["1mo", "3mo", "6mo", "1y", "2y", "5y"]),
              default="1y", show_default=True,
              help="History period for return estimation")
@click.option("--save", default=None, metavar="FILE",
              help="Save heatmap to a PNG file instead of showing it")
def heatmap(period, save):
    """Show a pairwise correlation heatmap for all assets."""
    controller.show_correlation_heatmap(period=period, save=save)
    
# Testing
@cli.command()
@click.option("--method", "-m",
              type=click.Choice(["gbm", "garch"]),
              default="gbm", show_default=True,
              help="Simulation method to test residuals for")
@click.option("--period", "-p",
              type=click.Choice(["1mo","3mo","6mo","1y","2y","5y"]),
              default="1y", show_default=True,
              help="History period for return estimation")
def test_dist(method, period):
    """Test which distribution best fits the model residuals."""
    controller.run_distribution_test(method=method, period=period)
    
# Simulation Commands
@cli.command()
@click.option("--method", "-m",
              type=click.Choice(["gbm", "garch", "regime"]),
              default="gbm", show_default=True,
              help="Simulation method: gbm, garch or regime")
@click.option("--dist", "-d",
              type=click.Choice(["normal", "student-t", "edf"]),
              default="normal", show_default=True,
              help="Simulation distribution: normal, student-t or edf")
@click.option("--years",  "-y", default=15,      show_default=True, type=int,
              help="Simulation horizon in years")
@click.option("--paths",  "-n", default=100_000, show_default=True, type=int,
              help="Number of simulated paths")
@click.option("--save",   default=None, metavar="FILE",
              help="Save simulation chart to PNG")
def simulate(method, dist, years, paths, save):
    """Run a Monte Carlo simulation on the current portfolio."""
    controller.run_simulation(method=method, dist=dist, years=years, n_paths=paths, save=save)
    
# MV Optimization
@cli.command()
@click.option("--period", "-p",
              type=click.Choice(["1mo","3mo","6mo","1y","2y","5y"]),
              default="5y", show_default=True,
              help="History period for return estimation")
@click.option("--risk-free", "-r", default=0.0, show_default=True,
              type=float, help="Annual risk-free rate, e.g. 0.04 for 4%")
def optimize(period, risk_free):
    """Find optimal weights using mean-variance optimization."""
    controller.show_optimized_weights(period=period, risk_free=risk_free)

# Entry point
if __name__ == "__main__":
    cli()