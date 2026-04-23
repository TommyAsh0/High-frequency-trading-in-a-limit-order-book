"""
Backtesting engine.

Drives the MarketSimulator events through the AvellanedaStoikov strategy,
keeps track of the market maker's P&L, inventory, and quoted prices over
time, and returns a BacktestResult for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from avellaneda_stoikov import AvellanedaStoikov
from market_simulator import MarketEvent, MarketSimulator
from order_book import LimitOrderBook


@dataclass
class BacktestResult:
    """Stores per-step snapshots produced by the backtest."""

    times: List[float] = field(default_factory=list)
    mid_prices: List[float] = field(default_factory=list)
    reservation_prices: List[float] = field(default_factory=list)
    bid_quotes: List[float] = field(default_factory=list)
    ask_quotes: List[float] = field(default_factory=list)
    inventories: List[float] = field(default_factory=list)
    cash: List[float] = field(default_factory=list)
    pnl: List[float] = field(default_factory=list)          # mark-to-market P&L
    spreads: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived statistics (computed after the run)
    # ------------------------------------------------------------------

    def final_pnl(self) -> float:
        return self.pnl[-1] if self.pnl else 0.0

    def max_inventory(self) -> float:
        return max(abs(q) for q in self.inventories) if self.inventories else 0.0

    def avg_spread(self) -> float:
        return sum(self.spreads) / len(self.spreads) if self.spreads else 0.0

    def sharpe_ratio(self) -> float:
        """Approximate Sharpe using per-step P&L changes."""
        if len(self.pnl) < 2:
            return float("nan")
        diffs = [self.pnl[i] - self.pnl[i - 1] for i in range(1, len(self.pnl))]
        mean = sum(diffs) / len(diffs)
        variance = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        std = variance**0.5
        if std == 0:
            return float("nan")
        return mean / std


class Backtester:
    """
    Runs the Avellaneda-Stoikov strategy against a MarketSimulator.

    At each price-update step the strategy:
    1. Cancels any existing outstanding quotes.
    2. Computes new optimal bid/ask via the AS model.
    3. Posts new limit orders in the LOB.

    When a market order arrives it is matched against the LOB; fills
    update the market maker's cash and inventory position.
    """

    def __init__(
        self,
        simulator: MarketSimulator,
        strategy: AvellanedaStoikov,
        order_size: float = 1.0,
    ) -> None:
        self.simulator = simulator
        self.strategy = strategy
        self.order_size = order_size   # size of each quote

    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        events = self.simulator.generate()
        lob = LimitOrderBook()
        result = BacktestResult()

        cash: float = 0.0
        inventory: float = 0.0
        current_bid_id: Optional[int] = None
        current_ask_id: Optional[int] = None

        for event in events:
            t = event.time
            mid = event.mid_price

            if event.event_type == "price_update":
                # ---- cancel previous quotes ----------------------------
                if current_bid_id is not None:
                    lob.cancel_order(current_bid_id)
                    current_bid_id = None
                if current_ask_id is not None:
                    lob.cancel_order(current_ask_id)
                    current_ask_id = None

                # ---- compute new quotes --------------------------------
                bid_price, ask_price = self.strategy.quotes(mid, inventory, t)
                bid_price = max(bid_price, 0.01)   # price floor

                # ---- post new limit orders -----------------------------
                current_bid_id = lob.add_limit_order("buy", bid_price, self.order_size, t)
                current_ask_id = lob.add_limit_order("sell", ask_price, self.order_size, t)

                # ---- record state -------------------------------------
                r = self.strategy.reservation_price(mid, inventory, t)
                spread = ask_price - bid_price
                mtm_pnl = cash + inventory * mid

                result.times.append(t)
                result.mid_prices.append(mid)
                result.reservation_prices.append(r)
                result.bid_quotes.append(bid_price)
                result.ask_quotes.append(ask_price)
                result.inventories.append(inventory)
                result.cash.append(cash)
                result.pnl.append(mtm_pnl)
                result.spreads.append(spread)

            elif event.event_type == "market_buy":
                # Incoming buy market order → hits our ask
                fills = lob.match_market_order("buy", event.quantity, t)
                for fill in fills:
                    # We sold at fill.price
                    cash += fill.price * fill.quantity
                    inventory -= fill.quantity
                    current_ask_id = None   # partially or fully consumed

            elif event.event_type == "market_sell":
                # Incoming sell market order → hits our bid
                fills = lob.match_market_order("sell", event.quantity, t)
                for fill in fills:
                    # We bought at fill.price
                    cash -= fill.price * fill.quantity
                    inventory += fill.quantity
                    current_bid_id = None   # partially or fully consumed

        return result
