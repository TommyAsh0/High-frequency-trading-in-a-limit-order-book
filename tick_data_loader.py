"""
Tick data loader for 000931.SZ (Chinese A-share).

Loads market (L2 quote snapshot), order, and transaction CSV files,
filters to continuous-trading hours, computes mid-price, and normalises
timestamps to the unit interval [0, 1] where:
    0  = 09:30:00 (start of continuous trading)
    1  = 15:00:00 (market close)

Chinese A-share continuous-trading session:
    Morning  : 09:30 – 11:30  (2 hours = 7 200 s)
    Afternoon: 13:00 – 15:00  (2 hours = 7 200 s)
    Total    : 14 400 seconds → 1 normalised time unit

Data files expected (in data/limit_order_data/ by default):
    000931.SZ_{date}_market.csv
    000931.SZ_{date}_order.csv
    000931.SZ_{date}_transaction.csv
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

TICKER = "000931.SZ"
TRADING_SECONDS = 14_400   # total continuous-trading seconds per day

# Continuous-trading windows as (open, close) tuples of (HH, MM) inclusive/exclusive
_WINDOWS: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [
    ((9, 30), (11, 30)),    # morning
    ((13, 0), (15, 0)),     # afternoon  (15:00:00.000 included as closing tick)
]


# ── Time helpers ─────────────────────────────────────────────────────────────

def _time_str_to_seconds(time_str: str) -> float:
    """Convert 'HH:MM:SS.mmm' to seconds from midnight."""
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _is_continuous_trading(time_str: str) -> bool:
    """Return True if the timestamp falls within continuous-trading hours."""
    sec = _time_str_to_seconds(time_str)
    for (oh, om), (ch, cm) in _WINDOWS:
        open_sec = oh * 3600 + om * 60
        close_sec = ch * 3600 + cm * 60
        if open_sec <= sec <= close_sec:
            return True
    return False


def _seconds_to_normalised(sec: float) -> float:
    """
    Map wall-clock seconds-from-midnight to a normalised trading-day time in [0, 1].
    Skips the lunch break (11:30 – 13:00).
    """
    morning_open = 9 * 3600 + 30 * 60    # 34 200 s
    morning_close = 11 * 3600 + 30 * 60  # 41 400 s
    afternoon_open = 13 * 3600            # 46 800 s

    if sec <= morning_close:
        elapsed = sec - morning_open
    else:
        elapsed = (morning_close - morning_open) + (sec - afternoon_open)

    return elapsed / TRADING_SECONDS


# ── Loader ───────────────────────────────────────────────────────────────────

class TickDataLoader:
    """
    Loads and pre-processes one trading day of 000931.SZ tick data.

    Parameters
    ----------
    date : str
        Date string in 'YYYYMMDD' format (e.g. '20260420').
    data_dir : str
        Directory containing the CSV files.  Defaults to 'data/limit_order_data'.
    """

    def __init__(self, date: str, data_dir: str = "data/limit_order_data") -> None:
        self.date = date
        self.data_dir = data_dir
        self._market: Optional[pd.DataFrame] = None
        self._transactions: Optional[pd.DataFrame] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def market(self) -> pd.DataFrame:
        """
        Return pre-processed market (L2 quote snapshot) data.

        Columns
        -------
        time_raw    : original time-raw integer
        time_str    : 'HH:MM:SS.mmm'
        time_norm   : normalised time in [0, 1]
        mid_price   : (BidPrice1 + AskPrice1) / 2
        bid1        : BidPrice1
        ask1        : AskPrice1
        spread      : AskPrice1 − BidPrice1
        half_spread : spread / 2
        """
        if self._market is None:
            self._market = self._load_market()
        return self._market

    def transactions(self) -> pd.DataFrame:
        """
        Return pre-processed transaction data filtered to
        continuous-trading hours and B/S-flagged rows only.

        Columns
        -------
        time_raw  : original time-raw integer
        time_str  : 'HH:MM:SS.mmm'
        time_norm : normalised time in [0, 1]
        price     : transaction price (CNY)
        volume    : transaction volume (shares)
        bsflag    : 'B' (buyer-initiated) or 'S' (seller-initiated)
        """
        if self._transactions is None:
            self._transactions = self._load_transactions()
        return self._transactions

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_market(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, f"{TICKER}_{self.date}_market.csv")
        df = pd.read_csv(path)

        # Filter continuous-trading rows with valid quotes
        mask = df["time"].apply(_is_continuous_trading) & (df["BidPrice1"] > 0)
        df = df[mask].copy()

        df["time_str"] = df["time"]
        df["time_norm"] = df["time"].apply(
            lambda t: _seconds_to_normalised(_time_str_to_seconds(t))
        )
        df["mid_price"] = (df["BidPrice1"] + df["AskPrice1"]) / 2
        df["bid1"] = df["BidPrice1"]
        df["ask1"] = df["AskPrice1"]
        df["spread"] = df["AskPrice1"] - df["BidPrice1"]
        df["half_spread"] = df["spread"] / 2

        return df[
            ["time_raw", "time_str", "time_norm", "mid_price",
             "bid1", "ask1", "spread", "half_spread"]
        ].reset_index(drop=True)

    def _load_transactions(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, f"{TICKER}_{self.date}_transaction.csv")
        df = pd.read_csv(path)

        # Keep only B/S-flagged transactions during continuous trading
        mask = (
            df["time"].apply(_is_continuous_trading)
            & df["bsflag"].isin(["B", "S"])
        )
        df = df[mask].copy()

        df["time_str"] = df["time"]
        df["time_norm"] = df["time"].apply(
            lambda t: _seconds_to_normalised(_time_str_to_seconds(t))
        )
        return df[["time_raw", "time_str", "time_norm", "price", "volume", "bsflag"]
                  ].reset_index(drop=True)


# ── Convenience function ─────────────────────────────────────────────────────

def load_all_days(
    dates: List[str],
    data_dir: str = "data/limit_order_data",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all trading days and return a dict:
        { date_str: {'market': DataFrame, 'transactions': DataFrame} }
    """
    result: Dict[str, Dict[str, pd.DataFrame]] = {}
    for date in dates:
        loader = TickDataLoader(date, data_dir)
        result[date] = {
            "market": loader.market(),
            "transactions": loader.transactions(),
        }
    return result
