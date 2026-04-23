# High-Frequency Trading in a Limit Order Book

A Python implementation of the classical **Avellaneda-Stoikov (2008)**
market-making strategy, complete with a limit order book, a Monte-Carlo
market simulator, a backtesting engine, and a full test suite.

---

## Background

The Avellaneda-Stoikov model is a cornerstone of quantitative market-making.
A risk-averse market maker continuously posts bid and ask limit orders around
a *reservation price* that accounts for the cost of holding inventory.

### Key formulae

| Quantity | Formula |
|---|---|
| Reservation price | `r = s − q · γ · σ² · (T − t)` |
| Optimal full spread | `δ* = γ · σ² · (T − t) + (2/γ) · ln(1 + γ/κ)` |
| Bid quote | `bid = r − δ*/2` |
| Ask quote | `ask = r + δ*/2` |

**Parameters**

| Symbol | Meaning |
|---|---|
| `s` | Current mid-price |
| `q` | Market maker inventory (+ = long, − = short) |
| `γ` | Risk-aversion coefficient |
| `σ` | Mid-price volatility |
| `T−t` | Time remaining in the trading session |
| `κ` | Market-order arrival sensitivity to spread |

---

## Repository structure

```
order_book.py          — Limit order book (bids/asks heap, matching engine)
market_simulator.py    — Synthetic price + order-flow generator (Brownian motion + Poisson arrivals)
avellaneda_stoikov.py  — AS-2008 strategy: reservation price, optimal spread, quotes
backtest.py            — Backtesting engine + BacktestResult dataclass
main.py                — Entry point; prints summary, optionally saves chart
tests.py               — 33 pytest unit & integration tests
requirements.txt       — Python dependencies
```

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the backtest (prints summary)
python main.py

# Run with chart output (saves backtest_result.png)
python main.py --plot

# Run tests
python -m pytest tests.py -v
```

### Sample output

```
Running Avellaneda-Stoikov backtest …

=== Backtest Summary ===
  Steps recorded    : 200
  Initial mid-price : 100.00
  Final mid-price   : 99.33
  Final P&L (MtM)   : 227.14
  Max |inventory|   : 9.00
  Avg quoted spread : 1.49
  Sharpe ratio      : 0.74
```

---

## Strategy parameters (edit `main.py`)

| Parameter | Default | Description |
|---|---|---|
| `INITIAL_PRICE` | 100.0 | Starting mid-price |
| `SIGMA` | 2.0 | Price volatility |
| `GAMMA` | 0.1 | Risk-aversion coefficient |
| `KAPPA` | 1.5 | Market-order arrival sensitivity |
| `DT` | 0.005 | Simulation time step |
| `T` | 1.0 | Total horizon |
| `LAM` | 1.5 | Poisson arrival rate per side per step |
| `ORDER_SIZE` | 1.0 | Size of each limit quote |

---

## Reference

> Avellaneda, M. & Stoikov, S. (2008).
> *High-frequency trading in a limit order book.*
> **Quantitative Finance**, 8(3), 217-224.
