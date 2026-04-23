"""
Parameter estimation for the Avellaneda-Stoikov model
from real tick data (000931.SZ).

Two sets of parameters are estimated:

1. **Volatility σ** (Eq. 1 in AS-2008)
   -------------------------------
   Estimated as the realised standard deviation of mid-price changes
   over the continuous trading session, expressed in CNY per
   sqrt(normalised-day).

       σ = sqrt( Σ_i (ΔS_i)²  )          (T = 1 by construction)

2. **Arrival-intensity parameters A and κ** (Eq. 3 in AS-2008)
   -----------------------------------------------------------
   The paper assumes the rate at which market orders arrive at a
   market-maker quote placed at distance δ from the mid-price is:

       λ(δ) = A · exp(−κ · δ)

   For Chinese A-share tick data where the bid-ask spread is almost
   always at the minimum tick (0.01 CNY), the spread barely varies, so
   the OLS regression approach from the original paper is unreliable.
   We therefore use a two-step calibration:

   ① **κ** is calibrated by matching the model's symmetric optimal
      spread to the observed average bid-ask spread:
          δ_sym* = 2/κ  (γ → 0 limit)
          κ = 2 / observed_mean_spread

   ② **A** is the base arrival rate estimated from the observed
      transaction rate at the prevailing best-quote level δ_min = 0.005:
          A = rate_at_min_spread / exp(−κ · δ_min)
      where  rate_at_min_spread = total_CT_transactions / total_CT_time.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ── σ estimation ─────────────────────────────────────────────────────────────

def estimate_sigma(market_df: pd.DataFrame) -> float:
    """
    Estimate mid-price volatility σ (CNY per sqrt(normalised-day)).

    Uses the realised-variance formula:
        σ² = Σ (S_{i+1} − S_i)²  / T     [T = 1]

    Parameters
    ----------
    market_df : pd.DataFrame
        Pre-processed market data from TickDataLoader (must contain
        'mid_price' and 'time_norm' columns).

    Returns
    -------
    float
        σ in CNY / sqrt(normalised trading day).
    """
    prices = market_df["mid_price"].values
    dS = np.diff(prices)
    # Realised variance = Σ dS² (T=1 by normalisation)
    sigma = float(np.sqrt(np.sum(dS ** 2)))
    return sigma


# ── A and κ estimation ────────────────────────────────────────────────────────

def estimate_kappa_A(
    market_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Estimate order-arrival intensity parameters A and κ.

    The model is  λ(δ) = A · exp(−κ · δ)  where δ is the half-spread.

    For Chinese A-share data, the bid-ask spread is almost always at the
    minimum tick (0.01 CNY), leaving very little variation to fit the
    exponential model directly.  We therefore use a two-step calibration:

    Step 1 — κ from observed spread
        Match the model's zero-inventory optimal spread to the observed
        average bid-ask spread using the γ → 0 limit:
            κ = 2 / E[spread_observed]

    Step 2 — A from transaction rate
        Correct the observed transaction rate at the best quote level
        (δ_min = E[half_spread]) upward to get the intercept A:
            A = rate_observed / exp(−κ · δ_min)

    Parameters
    ----------
    market_df : pd.DataFrame
        Pre-processed market data (columns: 'time_norm', 'half_spread', 'spread').
    transaction_df : pd.DataFrame
        Pre-processed transaction data (columns: 'time_norm', 'bsflag').

    Returns
    -------
    (A, kappa) : Tuple[float, float]
    """
    # Step 1: κ from mean observed spread
    mean_spread = float(market_df["spread"].mean())
    kappa = 2.0 / mean_spread

    # Step 2: A from observed transaction rate corrected for κ decay
    total_time = (
        market_df["time_norm"].iloc[-1] - market_df["time_norm"].iloc[0]
    )
    if total_time <= 0:
        return 1.0, kappa

    # Raw arrival rate at the prevailing best-quote level
    n_transactions = len(transaction_df)
    rate_at_min_spread = n_transactions / total_time

    # Adjust back to δ = 0 using the minimum half-spread
    delta_min = float(market_df["half_spread"].min())
    A = rate_at_min_spread / np.exp(-kappa * delta_min)

    return float(A), float(kappa)


# ── Combined estimation ───────────────────────────────────────────────────────

def estimate_parameters(
    market_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Estimate all AS-2008 model parameters for one trading day.

    Returns a dict with keys: 'sigma', 'A', 'kappa'.
    """
    sigma = estimate_sigma(market_df)
    A, kappa = estimate_kappa_A(market_df, transaction_df)
    return {"sigma": sigma, "A": A, "kappa": kappa}


def estimate_all_days(
    days_data: Dict[str, Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Estimate parameters for all days and return a summary DataFrame.

    Parameters
    ----------
    days_data : dict
        { date_str: {'market': df, 'transactions': df} }

    Returns
    -------
    pd.DataFrame
        Index: date strings.  Columns: 'sigma', 'A', 'kappa'.
    """
    records = {}
    for date, data in sorted(days_data.items()):
        params = estimate_parameters(data["market"], data["transactions"])
        records[date] = params
    return pd.DataFrame(records).T
