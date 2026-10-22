"""
Main entry point.

Runs a backtest of the Avellaneda-Stoikov market-making strategy and
prints a performance summary.  Pass --plot to generate charts.

Usage
-----
    python main.py
    python main.py --plot
"""

from __future__ import annotations

import argparse
import sys


def main(plot: bool = False) -> None:
    from avellaneda_stoikov import AvellanedaStoikov
    from backtest import Backtester
    from market_simulator import MarketSimulator

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    INITIAL_PRICE = 100.0
    SIGMA = 2.0          # price volatility
    GAMMA = 0.1          # risk-aversion
    KAPPA = 1.5          # market-order arrival sensitivity
    DT = 0.005           # time step (fraction of T)
    T = 1.0              # total horizon
    LAM = 1.5            # Poisson arrival rate per side per step
    ORDER_SIZE = 1.0     # size of each quote posted
    SEED = 42

    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------
    simulator = MarketSimulator(
        initial_price=INITIAL_PRICE,
        sigma=SIGMA,
        dt=DT,
        T=T,
        lam=LAM,
        mean_order_size=ORDER_SIZE,
        seed=SEED,
    )

    strategy = AvellanedaStoikov(
        gamma=GAMMA,
        sigma=SIGMA,
        kappa=KAPPA,
        T=T,
    )

    backtester = Backtester(simulator, strategy, order_size=ORDER_SIZE)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    print("Running Avellaneda-Stoikov backtest …")
    result = backtester.run()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Backtest Summary ===")
    print(f"  Steps recorded    : {len(result.times)}")
    print(f"  Initial mid-price : {INITIAL_PRICE:.2f}")
    print(f"  Final mid-price   : {result.mid_prices[-1]:.4f}")
    print(f"  Final P&L (MtM)   : {result.final_pnl():.4f}")
    print(f"  Max |inventory|   : {result.max_inventory():.4f}")
    print(f"  Avg quoted spread : {result.avg_spread():.4f}")
    print(f"  Sharpe ratio      : {result.sharpe_ratio():.4f}")

    if not plot:
        return

    # ------------------------------------------------------------------
    # Optional charts
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed – skipping plots.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Avellaneda-Stoikov Market-Making Strategy", fontsize=14)

    ts = result.times

    # Panel 1 – prices
    ax1 = axes[0]
    ax1.plot(ts, result.mid_prices, label="Mid price", color="black", linewidth=0.8)
    ax1.plot(ts, result.reservation_prices, label="Reservation price", color="blue",
             linewidth=0.8, linestyle="--")
    ax1.plot(ts, result.bid_quotes, label="Bid quote", color="green", linewidth=0.6,
             alpha=0.7)
    ax1.plot(ts, result.ask_quotes, label="Ask quote", color="red", linewidth=0.6,
             alpha=0.7)
    ax1.set_ylabel("Price")
    ax1.legend(fontsize=8)
    ax1.set_title("Quoted Prices vs Mid-Price")

    # Panel 2 – inventory
    ax2 = axes[1]
    ax2.plot(ts, result.inventories, color="purple", linewidth=0.8)
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.6)
    ax2.set_ylabel("Inventory")
    ax2.set_title("Market Maker Inventory")

    # Panel 3 – P&L
    ax3 = axes[2]
    ax3.plot(ts, result.pnl, color="darkorange", linewidth=0.8)
    ax3.axhline(0, color="gray", linestyle=":", linewidth=0.6)
    ax3.set_ylabel("P&L")
    ax3.set_xlabel("Time")
    ax3.set_title("Mark-to-Market P&L")

    plt.tight_layout()
    plt.savefig("backtest_result.png", dpi=150)
    print("\nChart saved to backtest_result.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AS market-making backtest")
    parser.add_argument("--plot", action="store_true", help="Generate charts")
    args = parser.parse_args()
    main(plot=args.plot)
