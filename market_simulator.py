"""
Market simulator for the limit order book environment.

Generates a synthetic mid-price path (arithmetic Brownian motion) and
simulates random market-order arrivals on both sides of the book.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from order_book import LimitOrderBook, Trade


@dataclass
class MarketEvent:
    """Represents a discrete market event in the simulation."""

    time: float
    event_type: str       # 'price_update' | 'market_buy' | 'market_sell'
    mid_price: float
    quantity: float = 0.0


class MarketSimulator:
    """
    Generates synthetic price and order-flow data for backtesting.

    Mid-price follows arithmetic Brownian motion:
        dS = σ · dW

    Market order arrivals on each side follow independent Poisson processes
    with intensity λ (orders per unit time).  The quantity of each market
    order is exponentially distributed with mean ``mean_order_size``.

    Parameters
    ----------
    initial_price : float
        Starting mid-price.
    sigma : float
        Annualised volatility (expressed in price units per sqrt(time)).
    dt : float
        Simulation time step.
    T : float
        Total simulation horizon.
    lam : float
        Poisson arrival rate of market orders on each side (per dt step).
    mean_order_size : float
        Mean quantity of each arriving market order.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        sigma: float = 2.0,
        dt: float = 0.005,
        T: float = 1.0,
        lam: float = 1.5,
        mean_order_size: float = 1.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.initial_price = initial_price
        self.sigma = sigma
        self.dt = dt
        self.T = T
        self.lam = lam
        self.mean_order_size = mean_order_size
        self.rng = random.Random(seed)

        self._price_path: List[float] = []
        self._events: List[MarketEvent] = []

    # ------------------------------------------------------------------

    def generate(self) -> List[MarketEvent]:
        """
        Run the simulation and return the list of market events in
        chronological order.
        """
        self._price_path = []
        self._events = []

        steps = int(self.T / self.dt)
        price = self.initial_price

        sqrt_dt = math.sqrt(self.dt)

        for i in range(steps):
            t = i * self.dt

            # Brownian increment
            dW = self.rng.gauss(0.0, 1.0)
            price += self.sigma * sqrt_dt * dW
            price = max(price, 1.0)   # floor at 1 to avoid negative prices

            self._price_path.append(price)
            self._events.append(
                MarketEvent(time=t, event_type="price_update", mid_price=price)
            )

            # Poisson market order arrivals (buy side)
            n_buy = self._poisson(self.lam)
            for _ in range(n_buy):
                qty = self.rng.expovariate(1.0 / self.mean_order_size)
                self._events.append(
                    MarketEvent(
                        time=t,
                        event_type="market_buy",
                        mid_price=price,
                        quantity=qty,
                    )
                )

            # Poisson market order arrivals (sell side)
            n_sell = self._poisson(self.lam)
            for _ in range(n_sell):
                qty = self.rng.expovariate(1.0 / self.mean_order_size)
                self._events.append(
                    MarketEvent(
                        time=t,
                        event_type="market_sell",
                        mid_price=price,
                        quantity=qty,
                    )
                )

        return self._events

    # ------------------------------------------------------------------

    @property
    def price_path(self) -> List[float]:
        return list(self._price_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _poisson(self, lam: float) -> int:
        """Draw a Poisson-distributed random integer."""
        # Knuth algorithm for small λ
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return k - 1
