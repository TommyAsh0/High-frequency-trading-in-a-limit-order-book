"""
Limit Order Book implementation.

Maintains a sorted book of bids and asks, supports adding/cancelling orders,
and matching incoming market orders against resting limit orders.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Order:
    """Represents a single limit order."""

    order_id: int
    side: str          # 'buy' or 'sell'
    price: float
    quantity: float
    timestamp: float   # simulation time of order placement

    def __lt__(self, other: "Order") -> bool:
        # Used by heapq; ordering by price is handled at the book level.
        return self.timestamp < other.timestamp


@dataclass
class Trade:
    """Records an executed trade."""

    buy_order_id: int
    sell_order_id: int
    price: float
    quantity: float
    timestamp: float


class LimitOrderBook:
    """
    A minimal limit order book that supports:
    - Adding limit orders (bid / ask)
    - Cancelling orders by ID
    - Matching a market order against the resting book
    - Reporting best bid / ask and mid-price
    """

    def __init__(self) -> None:
        # Bids: max-heap (negate prices so Python's min-heap gives max)
        self._bids: List[Tuple[float, int, Order]] = []   # (-price, ts, order)
        # Asks: min-heap (natural price ordering)
        self._asks: List[Tuple[float, int, Order]] = []   # (+price, ts, order)

        self._orders: Dict[int, Order] = {}   # id -> order (active)
        self._cancelled: set = set()
        self._trades: List[Trade] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def new_order_id(self) -> int:
        oid = self._next_id
        self._next_id += 1
        return oid

    def add_limit_order(
        self, side: str, price: float, quantity: float, timestamp: float
    ) -> int:
        """Add a resting limit order; returns the assigned order ID."""
        oid = self.new_order_id()
        order = Order(oid, side, price, quantity, timestamp)
        self._orders[oid] = order
        if side == "buy":
            heapq.heappush(self._bids, (-price, timestamp, order))
        else:
            heapq.heappush(self._asks, (price, timestamp, order))
        return oid

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a resting limit order. Returns True if it existed."""
        if order_id in self._orders:
            self._cancelled.add(order_id)
            del self._orders[order_id]
            return True
        return False

    def match_market_order(
        self, side: str, quantity: float, timestamp: float
    ) -> List[Trade]:
        """
        Execute a market order against the resting book.
        Returns a list of Trade objects for each fill.
        """
        fills: List[Trade] = []
        remaining = quantity

        book = self._asks if side == "buy" else self._bids

        while remaining > 0 and book:
            key, ts, resting = book[0]
            if resting.order_id in self._cancelled:
                heapq.heappop(book)
                continue
            if resting.order_id not in self._orders:
                heapq.heappop(book)
                continue

            fill_qty = min(remaining, resting.quantity)
            trade = Trade(
                buy_order_id=resting.order_id if side == "sell" else 0,
                sell_order_id=resting.order_id if side == "buy" else 0,
                price=resting.price,
                quantity=fill_qty,
                timestamp=timestamp,
            )
            fills.append(trade)
            self._trades.append(trade)

            resting.quantity -= fill_qty
            remaining -= fill_qty

            if resting.quantity <= 1e-12:
                heapq.heappop(book)
                del self._orders[resting.order_id]

        return fills

    # ------------------------------------------------------------------
    # Book state queries
    # ------------------------------------------------------------------

    def best_bid(self) -> Optional[float]:
        """Return the best (highest) bid price, or None."""
        self._clean_heap(self._bids, is_bid=True)
        if self._bids:
            return -self._bids[0][0]
        return None

    def best_ask(self) -> Optional[float]:
        """Return the best (lowest) ask price, or None."""
        self._clean_heap(self._asks, is_bid=False)
        if self._asks:
            return self._asks[0][0]
        return None

    def mid_price(self) -> Optional[float]:
        """Return the mid-price, or None if one side is empty."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None

    def spread(self) -> Optional[float]:
        """Return the bid-ask spread, or None if one side is empty."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return None

    @property
    def trades(self) -> List[Trade]:
        return list(self._trades)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_heap(self, heap: list, is_bid: bool) -> None:
        """Remove stale (cancelled / fully-filled) entries from heap top."""
        while heap:
            _, _, order = heap[0]
            if order.order_id in self._cancelled or order.order_id not in self._orders:
                heapq.heappop(heap)
            else:
                break
