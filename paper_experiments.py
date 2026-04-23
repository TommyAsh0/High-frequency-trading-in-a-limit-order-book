"""
Reproduction of the experiments in:

    Avellaneda, M. & Stoikov, S. (2008).
    "High-frequency trading in a limit order book."
    Quantitative Finance, 8(3), 217-224.

Using real tick data for 000931.SZ (Chinese A-share) over 4 trading days
(2026-04-20 through 2026-04-23).

Experiments reproduced
----------------------
1. Parameter estimation (σ, A, κ) per trading day — corresponds to
   Section 3 of the paper (calibration).

2. Single-day strategy visualisation — corresponds to Figures 1-3 in
   the paper:
   • Figure 1: mid-price S(t), reservation price r(t), bid b(t), ask a(t)
   • Figure 2: inventory q(t)
   • Figure 3: mark-to-market P&L = cash + q(t) · S(t)

3. Sensitivity to risk-aversion coefficient γ — corresponds to the
   comparative table in Section 4 of the paper:
   γ ∈ {0.001, 0.01, 0.1, 0.5, 1.0}
   Metrics: final P&L, P&L std, Sharpe, max|q|

4. AS (optimal) vs. Symmetric reference strategy — the paper argues
   that the inventory-adjusted strategy dominates the symmetric one
   on both P&L and inventory-risk dimensions.

Run
---
    python paper_experiments.py                  # text output only
    python paper_experiments.py --save-plots     # also save PNG figures
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from avellaneda_stoikov import AvellanedaStoikov
from param_estimation import estimate_all_days, estimate_parameters
from tick_backtest import TickBacktester, TickBacktestResult
from tick_data_loader import load_all_days

# ── Configuration ─────────────────────────────────────────────────────────────

DATES = ["20260420", "20260421", "20260422", "20260423"]

# γ values to sweep (matching AS-2008 Table 1 spirit)
GAMMA_SWEEP = [0.001, 0.01, 0.1, 0.5, 1.0]

# Default representative day for single-day figures (most liquid)
REFERENCE_DATE = "20260420"

# Lot size in shares
ORDER_SIZE = 100


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1 – Parameter estimation
# ══════════════════════════════════════════════════════════════════════════════

def experiment_1(days_data: Dict) -> pd.DataFrame:
    """
    Estimate AS-2008 model parameters (σ, A, κ) from each trading day.

    Corresponds to Section 3 (calibration) of the paper.

    Returns
    -------
    pd.DataFrame
        Rows = dates, columns = ['sigma', 'A', 'kappa'].
    """
    print("\n" + "=" * 65)
    print("EXPERIMENT 1 — Parameter Estimation  (Section 3 of AS-2008)")
    print("=" * 65)
    print(
        "  σ  : realised volatility of mid-price, CNY / √(norm. day)\n"
        "  A  : base order-arrival intensity  [orders / norm. time unit]\n"
        "  κ  : arrival-intensity sensitivity to spread  [CNY⁻¹]\n"
    )

    param_df = estimate_all_days(days_data)

    # Add average row
    avg = param_df.mean()
    avg.name = "Average"
    param_df = pd.concat([param_df, avg.to_frame().T])

    print(param_df.to_string(float_format="{:.4f}".format))
    print()

    return param_df


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2 – Single-day strategy visualisation
# ══════════════════════════════════════════════════════════════════════════════

def experiment_2(
    day_data: Dict,
    date: str,
    sigma: float,
    A: float,
    kappa: float,
    gamma: float = 0.1,
    save_plots: bool = False,
) -> TickBacktestResult:
    """
    Run the AS strategy on one day and produce Figures 1-3 from the paper.

    Figure 1 (top): S(t), r(t), bid(t), ask(t)
    Figure 2 (mid): inventory q(t)
    Figure 3 (bot): mark-to-market P&L
    """
    print("\n" + "=" * 65)
    print(f"EXPERIMENT 2 — Single-day visualisation  [{date}]  γ = {gamma}")
    print("=" * 65)

    strategy = AvellanedaStoikov(gamma=gamma, sigma=sigma, kappa=kappa, T=1.0)
    bt = TickBacktester(
        day_data["market"], day_data["transactions"],
        strategy, order_size=ORDER_SIZE, lot_size=ORDER_SIZE
    )
    result = bt.run()

    print(f"  Snapshots      : {len(result.times)}")
    print(f"  Final P&L      : {result.final_pnl():.2f} CNY")
    print(f"  Max |inventory|: {result.max_abs_inventory():.0f} shares  "
          f"({result.max_abs_inventory() / ORDER_SIZE:.1f} lots)")
    print(f"  Avg spread     : {result.avg_spread():.4f} CNY")
    print(f"  Sharpe ratio   : {result.sharpe_ratio():.4f}")

    if save_plots:
        _plot_experiment_2(result, date, gamma, sigma, kappa)

    return result


def _plot_experiment_2(
    result: TickBacktestResult,
    date: str,
    gamma: float,
    sigma: float,
    kappa: float,
    lot_size: int = 100,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = result.times

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"AS Market-Making Strategy — {date}  "
        f"(γ={gamma}, σ={sigma:.4f}, κ={kappa:.2f})",
        fontsize=13,
    )

    # ── Figure 1: prices ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(ts, result.mid_prices, "k-", lw=0.8, label="Mid-price $S(t)$")
    ax.plot(ts, result.reservation_prices, "b--", lw=0.8,
            label="Reservation price $r(t)$")
    ax.plot(ts, result.bid_quotes, color="green", lw=0.6, alpha=0.8,
            label="Bid quote $b(t)$")
    ax.plot(ts, result.ask_quotes, color="red", lw=0.6, alpha=0.8,
            label="Ask quote $a(t)$")
    ax.set_ylabel("Price (CNY)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Figure 1 — Mid-price, Reservation Price, and Quoted Prices")

    # ── Figure 2: inventory (in lots) ─────────────────────────────────────────
    ax = axes[1]
    inv_lots = [q / lot_size for q in result.inventories]
    ax.plot(ts, inv_lots, color="purple", lw=0.8)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("Inventory (lots)")
    ax.set_title("Figure 2 — Inventory Process $q(t)$  [1 lot = 100 shares]")

    # ── Figure 3: P&L ─────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(ts, result.pnl, color="darkorange", lw=0.8)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("P&L (CNY)")
    ax.set_xlabel("Normalised time $t/T$")
    ax.set_title("Figure 3 — Mark-to-Market P&L = Cash + $q(t) \\cdot S(t)$")

    plt.tight_layout()
    fname = f"fig_experiment2_{date}_gamma{gamma}.png"
    plt.savefig(fname, dpi=150)
    print(f"  → Saved {fname}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3 – Sensitivity to γ
# ══════════════════════════════════════════════════════════════════════════════

def experiment_3(
    days_data: Dict,
    sigma: float,
    kappa: float,
    save_plots: bool = False,
) -> pd.DataFrame:
    """
    Run the AS strategy for multiple γ values across all trading days.

    Corresponds to Table 1 / Section 4 of AS-2008.

    Metrics reported (averaged over all days):
    • Mean P&L (CNY)
    • Std of P&L increments × √N (comparable across days)
    • Sharpe ratio
    • Max |inventory| (shares)
    • Std of inventory (shares)
    • Average quoted spread (CNY)
    """
    print("\n" + "=" * 65)
    print("EXPERIMENT 3 — Sensitivity to γ  (Table 1 of AS-2008)")
    print("=" * 65)
    print(
        "  Columns: mean P&L (CNY),  P&L std (CNY),  Sharpe,\n"
        "           max|q| (lots),  σ_q (lots),  avg spread (CNY)\n"
    )

    rows = []
    for gamma in GAMMA_SWEEP:
        pnl_list, sharpe_list, max_q_list, std_q_list, spread_list = [], [], [], [], []
        for date in DATES:
            dd = days_data[date]
            strategy = AvellanedaStoikov(
                gamma=gamma, sigma=sigma, kappa=kappa, T=1.0
            )
            bt = TickBacktester(
                dd["market"], dd["transactions"],
                strategy, order_size=ORDER_SIZE, lot_size=ORDER_SIZE
            )
            r = bt.run()
            pnl_list.append(r.final_pnl())
            sharpe_list.append(r.sharpe_ratio())
            max_q_list.append(r.max_abs_inventory())
            std_q_list.append(r.std_inventory())
            spread_list.append(r.avg_spread())

        rows.append({
            "γ": gamma,
            "Mean P&L (CNY)": np.mean(pnl_list),
            "Std P&L (CNY)": np.std(pnl_list),
            "Sharpe": np.nanmean(sharpe_list),
            "Max|q| (lots)": np.mean(max_q_list) / ORDER_SIZE,
            "σ_q (lots)": np.mean(std_q_list) / ORDER_SIZE,
            "Avg spread (CNY)": np.mean(spread_list),
        })

    df = pd.DataFrame(rows).set_index("γ")
    print(df.to_string(float_format="{:.4f}".format))
    print()

    if save_plots:
        _plot_experiment_3(df, sigma, kappa)

    return df


def _plot_experiment_3(df: pd.DataFrame, sigma: float, kappa: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Sensitivity to Risk-Aversion γ  (σ={sigma:.4f}, κ={kappa:.2f})",
        fontsize=13,
    )

    gammas = [str(g) for g in df.index]

    ax = axes[0]
    ax.bar(gammas, df["Mean P&L (CNY)"], color="steelblue")
    ax.set_xlabel("γ")
    ax.set_ylabel("Mean P&L (CNY)")
    ax.set_title("Mean P&L across days")

    ax = axes[1]
    ax.bar(gammas, df["Sharpe"], color="seagreen")
    ax.set_xlabel("γ")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe ratio")

    ax = axes[2]
    ax.bar(gammas, df["Max|q| (lots)"], color="tomato")
    ax.set_xlabel("γ")
    ax.set_ylabel("Max |inventory| (lots)")
    ax.set_title("Max |inventory| (higher γ → better control)")

    plt.tight_layout()
    fname = "fig_experiment3_gamma_sensitivity.png"
    plt.savefig(fname, dpi=150)
    print(f"  → Saved {fname}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 4 – AS optimal vs. Symmetric reference strategy
# ══════════════════════════════════════════════════════════════════════════════

def experiment_4(
    days_data: Dict,
    sigma: float,
    kappa: float,
    gamma: float = 0.1,
    save_plots: bool = False,
) -> pd.DataFrame:
    """
    Compare the AS (inventory-adjusted) strategy against the symmetric
    reference strategy (no inventory adjustment, constant spread = 2/κ).

    Corresponds to the discussion in Section 4 of AS-2008 where the
    optimal strategy is shown to dominate the naive symmetric one.
    """
    print("\n" + "=" * 65)
    print(
        f"EXPERIMENT 4 — AS (γ={gamma}) vs. Symmetric strategy  "
        "(Section 4 of AS-2008)"
    )
    print("=" * 65)

    records = []
    for date in DATES:
        dd = days_data[date]
        strategy = AvellanedaStoikov(gamma=gamma, sigma=sigma, kappa=kappa, T=1.0)

        # AS strategy
        bt_as = TickBacktester(
            dd["market"], dd["transactions"],
            strategy, order_size=ORDER_SIZE, lot_size=ORDER_SIZE, symmetric=False
        )
        r_as = bt_as.run()

        # Symmetric strategy
        bt_sym = TickBacktester(
            dd["market"], dd["transactions"],
            strategy, order_size=ORDER_SIZE, lot_size=ORDER_SIZE, symmetric=True
        )
        r_sym = bt_sym.run()

        records.append({
            "Date": date,
            "AS P&L (CNY)": r_as.final_pnl(),
            "Sym P&L (CNY)": r_sym.final_pnl(),
            "AS Sharpe": r_as.sharpe_ratio(),
            "Sym Sharpe": r_sym.sharpe_ratio(),
            "AS Max|q| (lots)": r_as.max_abs_inventory() / ORDER_SIZE,
            "Sym Max|q| (lots)": r_sym.max_abs_inventory() / ORDER_SIZE,
            "AS Avg spread": r_as.avg_spread(),
            "Sym Avg spread": r_sym.avg_spread(),
        })

        if save_plots:
            _plot_experiment_4_day(r_as, r_sym, date, gamma, sigma, kappa)

    df = pd.DataFrame(records).set_index("Date")

    # Append mean row
    avg = df.mean(numeric_only=True)
    avg.name = "Average"
    df = pd.concat([df, avg.to_frame().T])

    print(df.to_string(float_format="{:.4f}".format))
    print()
    return df


def _plot_experiment_4_day(
    r_as: TickBacktestResult,
    r_sym: TickBacktestResult,
    date: str,
    gamma: float,
    sigma: float,
    kappa: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"AS vs. Symmetric Strategy — {date}  "
        f"(γ={gamma}, σ={sigma:.4f}, κ={kappa:.2f})",
        fontsize=13,
    )

    ts = r_as.times

    ax = axes[0]
    ax.plot(ts, r_as.pnl, color="steelblue", lw=0.8, label="AS (optimal)")
    ax.plot(ts, r_sym.pnl, color="tomato", lw=0.8, ls="--", label="Symmetric")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("P&L (CNY)")
    ax.legend(fontsize=9)
    ax.set_title("Mark-to-Market P&L")

    ax = axes[1]
    ax.plot(ts, [q / ORDER_SIZE for q in r_as.inventories], color="steelblue",
            lw=0.8, label="AS (optimal)")
    ax.plot(ts, [q / ORDER_SIZE for q in r_sym.inventories], color="tomato",
            lw=0.8, ls="--", label="Symmetric")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("Inventory (lots)")
    ax.set_xlabel("Normalised time $t/T$")
    ax.legend(fontsize=9)
    ax.set_title("Inventory Process  [1 lot = 100 shares]")

    plt.tight_layout()
    fname = f"fig_experiment4_comparison_{date}.png"
    plt.savefig(fname, dpi=150)
    print(f"  → Saved {fname}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(save_plots: bool = False) -> None:
    print("Loading tick data …")
    days_data = load_all_days(DATES)
    print(f"  Loaded {len(DATES)} trading days for {list(days_data.keys())}")

    # ── Experiment 1: parameter estimation ───────────────────────────────────
    param_df = experiment_1(days_data)

    # Use average parameters for downstream experiments
    sigma_avg = float(param_df.loc["Average", "sigma"])
    A_avg = float(param_df.loc["Average", "A"])
    kappa_avg = float(param_df.loc["Average", "kappa"])

    print(f"\nCalibrated parameters (average across all days):")
    print(f"  σ = {sigma_avg:.6f} CNY/√(norm.day)")
    print(f"  A = {A_avg:.4f}  orders / norm.time")
    print(f"  κ = {kappa_avg:.4f}  CNY⁻¹")

    # ── Experiment 2: single-day strategy visualisation ───────────────────────
    result_exp2 = experiment_2(
        days_data[REFERENCE_DATE],
        date=REFERENCE_DATE,
        sigma=sigma_avg,
        A=A_avg,
        kappa=kappa_avg,
        gamma=0.1,
        save_plots=save_plots,
    )

    # ── Experiment 3: sensitivity to γ ────────────────────────────────────────
    df_gamma = experiment_3(
        days_data, sigma=sigma_avg, kappa=kappa_avg, save_plots=save_plots
    )

    # ── Experiment 4: AS vs. symmetric ────────────────────────────────────────
    df_comparison = experiment_4(
        days_data,
        sigma=sigma_avg,
        kappa=kappa_avg,
        gamma=0.1,
        save_plots=save_plots,
    )

    print("=" * 65)
    print("All experiments complete.")
    if save_plots:
        print("Figures saved to: fig_experiment*.png")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce AS-2008 paper experiments with real tick data"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save experiment figures as PNG files",
    )
    args = parser.parse_args()
    main(save_plots=args.save_plots)
