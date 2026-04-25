"""
Real-data backtester for the Avellaneda-Stoikov strategy.

Replaces the synthetic MarketSimulator with real L2 quote snapshots
and real transaction data from 000931.SZ tick files.

Fill model (§ 3 of AS-2008, adapted for real data)
----------------------------------------------------
At each market snapshot the strategy posts:
    bid_price  = r − δ*/2
    ask_price  = r + δ*/2

Between consecutive snapshots at [t_i, t_{i+1}):
- A buyer-initiated transaction (bsflag='B') hit the ask side of the LOB.
  If our ask_price ≤ transaction_price  ⟹  our ask was filled
  (the market was willing to pay at least our ask price).
- A seller-initiated transaction (bsflag='S') hit the bid side.
  If our bid_price ≥ transaction_price  ⟹  our bid was filled.

Each fill is for min(order_size, transaction_volume) shares.
A quote is removed from the LOB once it is fully consumed; a new quote
is re-posted at the next snapshot.

Two strategy variants are compared:
  • AS (optimal): uses inventory-adjusted reservation price r(s, q, t)
  • Symmetric   : quotes at mid ± δ_sym/2  (no inventory adjustment)
                  δ_sym = 2/κ  (the γ→0 limit of the AS spread)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from avellaneda_stoikov import AvellanedaStoikov


# ── Trade event dataclass ────────────────────────────────────────────────────

@dataclass
class TradeEvent:
    """A single strategy action recorded during the backtest.

    event_type values
    -----------------
    PLACE_BID   – new bid limit order posted
    PLACE_ASK   – new ask limit order posted
    CANCEL_BID  – unfilled bid cancelled before re-quote
    CANCEL_ASK  – unfilled ask cancelled before re-quote
    FILL_BID    – our bid was hit (we bought)
    FILL_ASK    – our ask was hit (we sold)
    """

    time_str: str      # wall-clock time 'HH:MM:SS.mmm'
    time_norm: float   # normalised trading-day time [0, 1]
    event_type: str    # one of the six types above
    price: float       # order/fill price (CNY)
    volume: float      # order size or fill size (shares)
    mid_price: float   # current market mid-price
    inventory: float   # inventory in shares *after* this event
    cash: float        # cash balance *after* this event
    pnl: float         # mark-to-market P&L *after* this event


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class TickBacktestResult:
    """Per-snapshot state snapshots produced by TickBacktester."""

    times: List[float] = field(default_factory=list)
    mid_prices: List[float] = field(default_factory=list)
    reservation_prices: List[float] = field(default_factory=list)
    bid_quotes: List[float] = field(default_factory=list)
    ask_quotes: List[float] = field(default_factory=list)
    inventories: List[float] = field(default_factory=list)  # in shares
    cash: List[float] = field(default_factory=list)          # in CNY
    pnl: List[float] = field(default_factory=list)           # cash + q*mid (CNY)
    spreads: List[float] = field(default_factory=list)
    trade_log: List[TradeEvent] = field(default_factory=list)

    # ── Derived statistics ────────────────────────────────────────────────

    def final_pnl(self) -> float:
        return self.pnl[-1] if self.pnl else 0.0

    def max_abs_inventory(self) -> float:
        return max(abs(q) for q in self.inventories) if self.inventories else 0.0

    def std_inventory(self) -> float:
        return float(np.std(self.inventories)) if self.inventories else 0.0

    def avg_spread(self) -> float:
        return float(np.mean(self.spreads)) if self.spreads else 0.0

    def sharpe_ratio(self) -> float:
        """Per-snapshot P&L change Sharpe ratio."""
        if len(self.pnl) < 2:
            return float("nan")
        diffs = np.diff(self.pnl)
        std = float(np.std(diffs))
        if std == 0:
            return float("nan")
        return float(np.mean(diffs) / std)

    def total_fills(self) -> int:
        """Number of snapshots at which the inventory changed."""
        if not self.inventories:
            return 0
        return int(sum(1 for i in range(1, len(self.inventories))
                       if self.inventories[i] != self.inventories[i - 1]))

    def to_trade_log_df(self) -> pd.DataFrame:
        """Return the trade log as a tidy DataFrame (one row per event)."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "time_str": e.time_str,
                    "time_norm": e.time_norm,
                    "event_type": e.event_type,
                    "price": e.price,
                    "volume": e.volume,
                    "mid_price": e.mid_price,
                    "inventory": e.inventory,
                    "cash": e.cash,
                    "pnl": e.pnl,
                }
                for e in self.trade_log
            ]
        )


# ── Backtester ───────────────────────────────────────────────────────────────

class TickBacktester:
    """
    Run an AS market-making strategy on one day of real tick data.

    Parameters
    ----------
    market_df : pd.DataFrame
        Pre-processed L2 snapshot data from TickDataLoader.market().
    transaction_df : pd.DataFrame
        Pre-processed transaction data from TickDataLoader.transactions().
    strategy : AvellanedaStoikov
        Calibrated strategy instance.
    order_size : int
        Size of each limit-order quote in shares (default 100 = 1 lot).
    symmetric : bool
        If True, run the *symmetric* reference strategy (no inventory
        adjustment, constant spread = 2/κ) for comparison.
    """

    def __init__(
        self,
        market_df: pd.DataFrame,
        transaction_df: pd.DataFrame,
        strategy: AvellanedaStoikov,
        order_size: int = 100,
        symmetric: bool = False,
        lot_size: int = 100,
    ) -> None:
        self.market = market_df.sort_values("time_norm").reset_index(drop=True)
        self.txn = transaction_df.sort_values("time_norm").reset_index(drop=True)
        self.strategy = strategy
        self.order_size = order_size
        self.symmetric = symmetric
        # lot_size: inventory passed to the strategy is expressed in units of
        # lot_size shares.  Chinese A-shares trade in 100-share lots, so γ in
        # the AS model is per lot, not per share.  This prevents the inventory
        # adjustment from being 100× too large when order_size is in shares.
        self.lot_size = lot_size

        # Pre-index transactions by snapshot for fast lookup
        self._txn_index = self._build_txn_index()

    # ── Main run method ───────────────────────────────────────────────────────

    def run(self) -> TickBacktestResult:
        result = TickBacktestResult()

        cash: float = 0.0
        inventory: float = 0.0

        # Current quote state
        bid_price: Optional[float] = None
        ask_price: Optional[float] = None
        bid_remaining: float = 0.0
        ask_remaining: float = 0.0

        n = len(self.market)

        for i, row in self.market.iterrows():
            t = row["time_norm"]
            t_str = str(row.get("time_str", t))
            mid = row["mid_price"]

            # ── Cancel previous unfilled quotes before re-quoting ─────────────
            if bid_price is not None and bid_remaining > 0:
                result.trade_log.append(TradeEvent(
                    time_str=t_str, time_norm=t, event_type="CANCEL_BID",
                    price=bid_price, volume=bid_remaining, mid_price=mid,
                    inventory=inventory, cash=cash, pnl=cash + inventory * mid,
                ))
            if ask_price is not None and ask_remaining > 0:
                result.trade_log.append(TradeEvent(
                    time_str=t_str, time_norm=t, event_type="CANCEL_ASK",
                    price=ask_price, volume=ask_remaining, mid_price=mid,
                    inventory=inventory, cash=cash, pnl=cash + inventory * mid,
                ))

            # ── Re-quote ─────────────────────────────────────────────────────
            # Express inventory in lots for the strategy so that γ is in
            # "per lot" units, matching the paper's calibration.
            inv_lots = inventory / self.lot_size

            if self.symmetric:
                half_spread = 1.0 / self.strategy.kappa   # δ_sym/2 = 1/κ
                bid_price = mid - half_spread
                ask_price = mid + half_spread
            else:
                bid_price, ask_price = self.strategy.quotes(mid, inv_lots, t)
                bid_price = max(bid_price, 0.01)

            bid_remaining = float(self.order_size)
            ask_remaining = float(self.order_size)

            # Record new quote placements
            result.trade_log.append(TradeEvent(
                time_str=t_str, time_norm=t, event_type="PLACE_BID",
                price=bid_price, volume=bid_remaining, mid_price=mid,
                inventory=inventory, cash=cash, pnl=cash + inventory * mid,
            ))
            result.trade_log.append(TradeEvent(
                time_str=t_str, time_norm=t, event_type="PLACE_ASK",
                price=ask_price, volume=ask_remaining, mid_price=mid,
                inventory=inventory, cash=cash, pnl=cash + inventory * mid,
            ))

            # ── Process transactions in (t_i, t_{i+1}) ───────────────────────
            txn_slice = self._txn_index.get(i, pd.DataFrame())

            for _, txn in txn_slice.iterrows():
                tp = txn["price"]
                tv = float(txn["volume"])
                flag = txn["bsflag"]
                txn_t_str = str(txn.get("time_str", t_str))
                txn_t_norm = float(txn.get("time_norm", t))

                if flag == "B" and ask_remaining > 0 and ask_price is not None:
                    # Buyer-initiated: hits ask side → we sell
                    if tp >= ask_price:
                        fill = min(ask_remaining, tv)
                        cash += fill * ask_price
                        inventory -= fill
                        ask_remaining -= fill
                        result.trade_log.append(TradeEvent(
                            time_str=txn_t_str, time_norm=txn_t_norm,
                            event_type="FILL_ASK",
                            price=ask_price, volume=fill, mid_price=mid,
                            inventory=inventory, cash=cash,
                            pnl=cash + inventory * mid,
                        ))

                elif flag == "S" and bid_remaining > 0 and bid_price is not None:
                    # Seller-initiated: hits bid side → we buy
                    if tp <= bid_price:
                        fill = min(bid_remaining, tv)
                        cash -= fill * bid_price
                        inventory += fill
                        bid_remaining -= fill
                        result.trade_log.append(TradeEvent(
                            time_str=txn_t_str, time_norm=txn_t_norm,
                            event_type="FILL_BID",
                            price=bid_price, volume=fill, mid_price=mid,
                            inventory=inventory, cash=cash,
                            pnl=cash + inventory * mid,
                        ))

            # ── Record state ──────────────────────────────────────────────────
            r = (mid if self.symmetric
                 else self.strategy.reservation_price(mid, inv_lots, t))
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

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_txn_index(self) -> Dict[int, pd.DataFrame]:
        """
        Pre-assign each transaction to the market snapshot i such that
        market.time_norm[i] ≤ txn.time_norm < market.time_norm[i+1].
        """
        index: Dict[int, List] = {}
        mkt_times = self.market["time_norm"].values
        n = len(mkt_times)

        for _, txn_row in self.txn.iterrows():
            tn = txn_row["time_norm"]
            # Find latest snapshot ≤ tn
            idx = int(np.searchsorted(mkt_times, tn, side="right")) - 1
            idx = max(0, min(idx, n - 1))
            if idx not in index:
                index[idx] = []
            index[idx].append(txn_row)

        return {k: pd.DataFrame(v) for k, v in index.items()}
