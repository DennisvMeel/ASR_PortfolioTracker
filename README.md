# Portfolio Tracker

A command-line investment portfolio tracker built in Python, developed as part of the a.s.r. Vermogensbeheer assignment. The application follows the **Model-View-Controller (MVC)** pattern and lets you manage a portfolio, fetch live prices, analyse risk, run Monte Carlo simulations, and optimise weights — all from the terminal.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the application](#running-the-application)
- [Usage](#usage)
  - [Portfolio management](#portfolio-management)
  - [Asset management](#asset-management)
  - [Viewing the portfolio](#viewing-the-portfolio)
  - [Price history](#price-history)
  - [Risk analysis](#risk-analysis)
  - [Simulation](#simulation)
  - [Distribution testing](#distribution-testing)
  - [Mean-variance optimisation](#mean-variance-optimisation)
- [Full worked example](#full-worked-example)
- [Ticker reference](#ticker-reference)

---

## Features

- **Multiple portfolios** - create, switch between, and delete named portfolios
- **Live prices** - fetches current market prices via [yfinance](https://github.com/ranaroussi/yfinance)
- **Portfolio view** - full table with cost, current value, P&L, and weight per asset
- **Weights breakdown** - by individual asset, sector, or asset class
- **Price charts** - historical price chart for one or more tickers
- **Risk analysis** - risk contribution and Sharpe ratio per asset
- **Monte Carlo simulation** - three methods (GBM, GARCH, Regime-Switching) with three shock distributions (Normal, Student-t, EDF) over a configurable horizon
- **Distribution testing** - Shapiro-Wilk, Jarque-Bera, and KS tests to recommend the best distribution for simulation
- **Mean-variance optimisation** - finds Sharpe-maximising weights subject to long-only constraints

---

## Project Structure

```
portfolio-tracker/
	main.py   	# CLI entry point (Click commands)
	controllers/
		portfolio_controller.py   # Orchestrates model - view data flow
	models/
		portfolio.py   # Asset dataclass, Portfolio model, calculations
		simulation.py   # GBM, GARCH, Regime-Switching simulations
	views/
		display.py   # All terminal output: tables, charts, panels
	data/
        	default.JSON   # Default portfolio
	requirements.txt   # All required packages needed for the interface 
```

---

## Requirements

- Python **3.12** (3.13+ may have issues with `hmmlearn` on Windows)
- An active internet connection (for live price fetching)

> **Windows users (non-Anaconda only):** `hmmlearn` requires Python 3.12 and 
> Microsoft C++ Build Tools to install correctly. Download Python 3.12 from 
> https://www.python.org/downloads/release/python-3129/ and C++ Build Tools from
> https://visualstudio.microsoft.com/visual-cpp-build-tools/ (select 
> "Desktop development with C++" during installation).

---

## Installation

**1. Download the repository**

Click the green **Code** button on GitHub, **Download ZIP**, then unzip it. Or if you have Git:

```cmd
git clone https://github.com/DennisvMeel/ASR_PortfolioTracker.git
```

**2. Open Anaconda Prompt**

Search for "Anaconda Prompt" in the Windows start menu and open it.

**3. Navigate to the project folder**

```cmd
cd C:\path\to\portfolio-tracker
```

For example, if you unzipped to your Downloads folder:

```cmd
cd C:\Users\YourName\Downloads\ASR_PortfolioTracker\ASR_PortfolioTracker-main
```

**4. Install dependencies**

```cmd
pip install -r requirements.txt
```

All packages should install without errors since Anaconda already includes most of them.

**5. Verify the installation**

```cmd
python main.py --help
```

You should see a list of all available commands.

## Running the application

All commands follow this pattern:

```bash
python main.py <command> [options]
```

To see all available commands:

```bash
python main.py --help
```

To see the options for a specific command:

```bash
python main.py <command> --help
```

---

## Usage

### Portfolio management

The application supports multiple named portfolios. On first run a `default` portfolio is present.

| Command | Description |
|---|---|
| `list-portfolios` | Show all saved portfolios and mark the active one |
| `new-portfolio -n NAME` | Create a new portfolio and switch to it |
| `switch-portfolio -n NAME` | Switch to an existing portfolio |
| `delete-portfolio -n NAME` | Delete a portfolio (the active portfolio cannot be deleted) |

```bash
python main.py new-portfolio -n my_portfolio
python main.py list-portfolios
python main.py switch-portfolio -n my_portfolio
python main.py delete-portfolio -n old_portfolio
```

---

### Asset management

#### Add an asset

```bash
python main.py add -t TICKER -s SECTOR -c CLASS -q QUANTITY -p PRICE
```

| Option | Short | Description |
|---|---|---|
| `--ticker` | `-t` | Ticker symbol, e.g. `AAPL` |
| `--sector` | `-s` | Sector label, e.g. `Technology` |
| `--asset-class` | `-c` | Asset class label, e.g. `Equity` |
| `--quantity` | `-q` | Number of units purchased |
| `--purchase-price` | `-p` | Purchase price per unit |

The ticker is validated against Yahoo Finance before being added. If the ticker already exists in the portfolio, the quantity is increased and the average purchase price is updated automatically.

```bash
python main.py add -t AAPL    -s Technology      -c Equity          -q 10  -p 150.00
python main.py add -t MSFT    -s Technology      -c Equity          -q 5   -p 280.00
python main.py add -t VWRL.AS -s "Global Equity" -c ETF             -q 50  -p 95.00
python main.py add -t BTC-USD -s Crypto          -c "Digital Asset" -q 0.5 -p 30000.00
```

#### Remove an asset

```bash
python main.py remove -t TICKER              # remove the asset entirely
python main.py remove -t TICKER -q QUANTITY  # reduce the position by a number of units
```

---

### Viewing the portfolio

```bash
python main.py view               # fetch live prices, then display the portfolio
python main.py view --no-refresh  # use most recent fetched prices (useful offline or for speed)
```

Displays a table with ticker, sector, asset class, quantity, purchase price, current price, cost, current value, P&L (€ and %), and portfolio weight for each asset. Below the table a summary panel shows total cost, total value, and total P&L.

#### Weights breakdown

```bash
python main.py weights               # weights by individual asset (default)
python main.py weights --by sector   # weights aggregated by sector
python main.py weights --by class    # weights aggregated by asset class
```

---

### Price history

Plot historical closing prices for one or more tickers:

```bash
python main.py prices AAPL
python main.py prices AAPL MSFT GOOGL --period 2y
python main.py prices AAPL --period 6mo --save chart.png
```

| Option | Default | Choices |
|---|---|---|
| `--period` / `-p` | `1y` | `1mo`  `3mo`  `6mo`  `1y`  `2y`  `5y` |
| `--save` | — | File path to save the chart as a PNG |

A matplotlib window opens by default. Pass `--save filename.png` to write the chart to disk instead of displaying it.

---

### Risk analysis

```bash
python main.py risk
python main.py risk --period 2y
```

Displays a table showing each asset's portfolio weight, its percentage contribution to total portfolio volatility, and its individual annualised Sharpe ratio. Requires at least 2 assets in the portfolio.

| Option | Default | Choices |
|---|---|---|
| `--period` / `-p` | `1y` | `1mo`  `3mo`  `6mo`  `1y`  `2y`  `5y` |

---

### Simulation

Run a Monte Carlo simulation on the current portfolio value over a configurable horizon:

```bash
python main.py simulate
python main.py simulate --method garch --dist student-t --years 15 --paths 100000
python main.py simulate --method regime --years 20 --save simulation.png
```

| Option | Short | Default | Choices |
|---|---|---|---|
| `--method` | `-m` | `gbm` | `gbm`  `garch`  `regime` |
| `--dist` | `-d` | `normal` | `normal`  `student-t`  `edf` |
| `--years` | `-y` | `15` | any integer |
| `--paths` | `-n` | `100000` | any integer |
| `--save` | | — | file path to save the chart as PNG |

**Simulation methods:**

`gbm` — Geometric Brownian Motion. Assumes constant drift and volatility estimated from 5 years of historical returns. The simplest and fastest method.

`garch` — GARCH(1,1). Models time-varying volatility, capturing the volatility clustering seen in real financial returns. Slower than GBM.

`regime` — Hidden Markov Model with 2 regimes (bull and bear). Fits a regime-switching model to historical returns and simulates probabilistic transitions between states. Prints the fitted regime parameters before running.

**Shock distributions (`--dist`, applies to `gbm` and `garch`):**

`normal` — standard Gaussian shocks. Fast and simple.

`student-t` — heavier tails than normal. Degrees of freedom are fitted to the historical data. Use this if `test-dist` recommends it.

`edf` — Empirical Distribution Function. Resamples shocks directly from the historical standardised residuals with no distributional assumption.

The output shows a statistics table (5th–95th percentile, mean, median, Expected Shortfall at 95%) and a fan chart of simulated paths with percentile bands.

---

### Distribution testing

Before choosing a `--dist` for simulation, use this command to test which distribution best fits the model residuals:

```bash
python main.py test-dist
python main.py test-dist --method garch --period 2y
```

Runs three normality tests on the portfolio return residuals:

- **Shapiro-Wilk** — sensitive to departures from normality in smaller samples
- **Jarque-Bera** — tests for skewness and excess kurtosis
- **Kolmogorov-Smirnov** — tests the full distribution against a fitted normal

Also reports skewness, excess kurtosis, and the fitted Student-t degrees of freedom. The recommendation at the bottom (`normal`, `student-t`, or `edf`) can be passed directly to `simulate --dist`.

| Option | Short | Default | Choices |
|---|---|---|---|
| `--method` | `-m` | `gbm` | `gbm`  `garch` |
| `--period` | `-p` | `1y` | `1mo`  `3mo`  `6mo`  `1y`  `2y`  `5y` |

---

### Mean-variance optimisation

Find the portfolio weights that obtain the max Sharpe ratio, subject to long-only constraints (no short selling):

```bash
python main.py optimize
python main.py optimize --period 5y --risk-free 0.04
```

Displays two tables: a weight comparison (current vs optimal per ticker) and a metrics comparison (annualised return, volatility, and Sharpe ratio for both portfolios). Requires at least 2 assets.

| Option | Short | Default | Description |
|---|---|---|---|
| `--period` | `-p` | `5y` | History period used to estimate returns and covariances |
| `--risk-free` | `-r` | `0.0` | Annual risk-free rate, e.g. `0.04` for 4% |

---

## Full worked example

The following sequence builds a five-asset portfolio from scratch and runs a complete analysis:

```bash
# 1. Create a new portfolio
python main.py new-portfolio -n demo

# 2. Add assets
python main.py add -t AAPL  -s Technology  -c Equity -q 10  -p 150.00
python main.py add -t MSFT  -s Technology  -c Equity -q 8   -p 280.00
python main.py add -t GOOGL -s Technology  -c Equity -q 3   -p 2800.00
python main.py add -t JPM   -s Financials  -c Equity -q 15  -p 130.00
python main.py add -t GLD   -s Commodities -c ETF    -q 20  -p 170.00

# 3. View the portfolio with live prices
python main.py view

# 4. Check weights by sector
python main.py weights --by sector

# 5. Show a 2-year price chart for all tickers
python main.py prices AAPL MSFT GOOGL JPM GLD --period 2y

# 6. Check risk contribution per asset
python main.py risk --period 1y

# 7. Test which distribution fits the residuals best
python main.py test-dist --method gbm --period 1y

# 8. Run a 15-year simulation with 100,000 paths
python main.py simulate --method gbm --dist normal --years 15 --paths 100000

# 9. Find the Sharpe-optimal weights using a 4% risk-free rate
python main.py optimize --period 5y --risk-free 0.04
```

---

## Ticker reference

Tickers must match Yahoo Finance exactly. Some common formats:

| Asset type | Example tickers |
|---|---|
| US equities | `AAPL`, `MSFT`, `GOOGL`, `JPM`, `TSLA` |
| European equities | `ASML.AS`, `RDSA.AS`, `AIR.PA` |
| ETFs (US-listed) | `SPY`, `QQQ`, `GLD`, `TLT` |
| ETFs (European-listed) | `VWRL.AS`, `IWDA.AS`, `CSPX.AS` |
| Crypto | `BTC-USD`, `ETH-USD` |

If a ticker cannot be found on Yahoo Finance, yfinance will return no data and the asset will not be added.

---
