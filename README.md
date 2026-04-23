# High-Frequency Trading in a Limit Order Book

A Python implementation of the classical **Avellaneda-Stoikov (2008)**
market-making strategy, complete with:

* a limit order book data structure,
* a synthetic Monte-Carlo market simulator,
* a **real-data backtesting engine** driven by 4 days of 000931.SZ tick data,
* automatic parameter calibration from Level-2 quotes and transaction data,
* full reproduction of all paper experiments (Sections 3 & 4 of AS-2008), and
* a 33-test pytest suite.

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
| Arrival intensity | `λ(δ) = A · exp(−κ · δ)` |

**Parameters**

| Symbol | Meaning |
|---|---|
| `s` | Current mid-price |
| `q` | Market maker inventory in lots (+ = long, − = short) |
| `γ` | Risk-aversion coefficient (per lot) |
| `σ` | Mid-price realised volatility (CNY / √(norm. trading day)) |
| `T−t` | Normalised time remaining (1 = full trading day) |
| `κ` | Market-order arrival sensitivity to spread (CNY⁻¹) |
| `A` | Base arrival intensity at zero spread |

---

## Repository structure

```
order_book.py          — Limit order book (bids/asks heap, matching engine)
market_simulator.py    — Synthetic price + order-flow generator (BM + Poisson)
avellaneda_stoikov.py  — AS-2008 strategy: reservation price, optimal spread
backtest.py            — Synthetic-data backtesting engine
main.py                — Entry point for synthetic backtest; optional chart

tick_data_loader.py    — Load & preprocess 000931.SZ Level-2 tick CSV files
param_estimation.py    — Estimate σ, A, κ from real market + transaction data
tick_backtest.py       — Real-data backtester (fill-if-touched with real transactions)
paper_experiments.py   — Reproduce all AS-2008 paper experiments with real data

000931.SZ_YYYYMMDD_market.csv      — L2 quote snapshots (10-level bid/ask)
000931.SZ_YYYYMMDD_order.csv       — Limit order submissions
000931.SZ_YYYYMMDD_transaction.csv — Executed trades

tests.py               — 33 pytest unit & integration tests
requirements.txt       — Python dependencies
```

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# ── Synthetic backtest (original) ──────────────────────────────────────────
python main.py            # prints summary
python main.py --plot     # also saves backtest_result.png

# ── Paper experiments with real tick data ──────────────────────────────────
python paper_experiments.py                  # text output only
python paper_experiments.py --save-plots     # also saves fig_experiment*.png

# ── Tests ──────────────────────────────────────────────────────────────────
python -m pytest tests.py -v
```

---

## Real-data experiments — results summary

All experiments use 000931.SZ tick data for **2026-04-20 through 2026-04-23**
(4 trading days, ~7 000–16 000 transactions per day).

### Experiment 1 — Calibrated parameters

| Date | σ (CNY/√day) | A (orders/day) | κ (CNY⁻¹) |
|---|---|---|---|
| 2026-04-20 | 0.1116 | 91 195 | 197.3 |
| 2026-04-21 | 0.0962 | 43 307 | 195.0 |
| 2026-04-22 | 0.1213 | 48 781 | 196.2 |
| 2026-04-23 | 0.0918 | 43 000 | 191.6 |
| **Average** | **0.1052** | **56 571** | **195.0** |

> **Calibration notes**
> * σ is the realised volatility of the mid-price over continuous-trading
>   hours (09:30–11:30, 13:00–15:00; T = 1 normalised day = 14 400 s).
> * κ is calibrated by matching the model's γ → 0 spread (= 2/κ) to the
>   observed mean bid-ask spread (~0.010 CNY), giving κ ≈ 195 CNY⁻¹.
> * A is the transaction arrival rate corrected upward for the κ decay at
>   the minimum quote distance (half-spread ≈ 0.005 CNY).

---

### Experiment 2 — Single-day strategy visualisation (2026-04-20, γ = 0.1)

Reproduces Figures 1–3 of AS-2008 with real data.

| Metric | Value |
|---|---|
| Final P&L | **+23.6 CNY** |
| Max \|inventory\| | 3.0 lots (300 shares) |
| Average quoted spread | 0.0111 CNY |
| Sharpe ratio | 0.063 |

*(Run with `--save-plots` to generate `fig_experiment2_20260420_gamma0.1.png`)*

---

### Experiment 3 — Sensitivity to risk-aversion γ (averaged over 4 days)

Reproduces Table 1 / Section 4 of AS-2008.

| γ | Mean P&L (CNY) | Sharpe | Max\|q\| (lots) |
|---|---|---|---|
| 0.001 | −28.5 | −0.010 | 11.5 |
| 0.010 | −1.1 | −0.002 | 3.5 |
| **0.100** | **+13.6** | **+0.040** | **2.8** |
| 0.500 | +12.4 | +0.033 | 2.0 |
| 1.000 | +4.2 | +0.010 | 1.0 |

**Key finding** (matching the paper): too-low γ → excessive inventory
accumulation and negative P&L; too-high γ → over-cautious quoting and
reduced earnings.  The optimal γ ≈ 0.1 achieves the best risk-adjusted
return (Sharpe 0.04) with inventory capped at ~3 lots.

---

### Experiment 4 — AS (optimal) vs. Symmetric reference strategy (γ = 0.1)

| Strategy | Mean P&L | Sharpe | Mean Max\|q\| (lots) |
|---|---|---|---|
| **AS (optimal)** | **+13.6 CNY** | **+0.040** | **2.8** |
| Symmetric | −16.6 CNY | −0.008 | 16.5 |

**Key finding** (matching the paper): the inventory-adjusted AS strategy
dominates the symmetric reference on *both* P&L and inventory-risk dimensions.
The symmetric strategy quotes too aggressively (no spread adjustment), builds
up large unhedged positions, and suffers mark-to-market losses when prices
move adversely.

---

## Strategy parameters (synthetic `main.py`)

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
