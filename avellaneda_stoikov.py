"""
Avellaneda-Stoikov (2008) market-making strategy.

Reference:
    Avellaneda, M. & Stoikov, S. (2008).
    "High-frequency trading in a limit order book."
    Quantitative Finance, 8(3), 217-224.

The strategy continuously posts a bid and an ask around the *reservation
price* — the mid-price adjusted for the market maker's inventory exposure.

Key formulas
------------
Reservation price:
    r(s, q, t) = s - q * γ * σ² * (T - t)

Optimal half-spread (distance from reservation price to each quote):
    δ* = (γ * σ² * (T - t)) / 2  +  (1/γ) * ln(1 + γ/κ)

Quoted bid and ask:
    bid = r - δ*
    ask = r + δ*

Parameters
----------
gamma : float
    Risk-aversion coefficient (γ).  Higher → tighter inventory control.
sigma : float
    Volatility of the mid-price.
kappa : float
    Market-order arrival sensitivity to spread (κ).
T : float
    Total time horizon of the simulation (normalises time remaining).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


class AvellanedaStoikov:
    """
    Classical Avellaneda-Stoikov market-making strategy.

    Usage::

        strategy = AvellanedaStoikov(gamma=0.1, sigma=2.0, kappa=1.5, T=1.0)
        bid, ask = strategy.quotes(mid_price=100.0, inventory=0, time=0.5)
    """

    def __init__(
        self,
        gamma: float = 0.1,
        sigma: float = 2.0,
        kappa: float = 1.5,
        T: float = 1.0,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if T <= 0:
            raise ValueError("T must be positive")

        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def reservation_price(
        self, mid_price: float, inventory: float, time: float
    ) -> float:
        """
        Compute the reservation (indifference) price.

        Parameters
        ----------
        mid_price : float
            Current mid-price of the asset.
        inventory : float
            Current inventory held by the market maker
            (positive = long, negative = short).
        time : float
            Current simulation time (0 ≤ time ≤ T).

        Returns
        -------
        float
            Reservation price.
        """
        time_remaining = max(self.T - time, 1e-9)
        return mid_price - inventory * self.gamma * self.sigma**2 * time_remaining

    def optimal_spread(self, time: float) -> float:
        """
        Compute the full optimal spread 2·δ*.

        Parameters
        ----------
        time : float
            Current simulation time.

        Returns
        -------
        float
            Full bid-ask spread around the reservation price.
        """
        time_remaining = max(self.T - time, 1e-9)
        term1 = self.gamma * self.sigma**2 * time_remaining
        term2 = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        return term1 + term2

    def quotes(
        self, mid_price: float, inventory: float, time: float
    ) -> Tuple[float, float]:
        """
        Return the optimal (bid_price, ask_price) pair.

        Parameters
        ----------
        mid_price : float
            Current mid-price.
        inventory : float
            Current inventory.
        time : float
            Current simulation time.

        Returns
        -------
        (bid_price, ask_price) : Tuple[float, float]
        """
        r = self.reservation_price(mid_price, inventory, time)
        half_spread = self.optimal_spread(time) / 2.0
        bid = r - half_spread
        ask = r + half_spread
        return bid, ask

    # ------------------------------------------------------------------
    # Arrival intensity helpers  (Equation 3 in AS-2008)
    # ------------------------------------------------------------------

    def arrival_intensity(self, delta: float) -> float:
        """
        Expected market-order arrival intensity at a quoted distance *delta*
        from the mid-price, modelled as:

            λ(δ) = A · exp(-κ · δ)

        We normalise A = 1 here; the relative ordering of intensities
        is what matters for the optimal spread derivation.

        Parameters
        ----------
        delta : float
            Distance of the quoted price from the mid-price (half-spread).

        Returns
        -------
        float
            Arrival intensity (in the same units as the Poisson arrival
            parameter of the simulator).
        """
        return math.exp(-self.kappa * delta)
