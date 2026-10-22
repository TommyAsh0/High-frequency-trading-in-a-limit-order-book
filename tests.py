"""
Unit tests for the HFT limit order book implementation.

Run with:
    python -m pytest tests.py -v
"""

from __future__ import annotations

import math

import pytest

from avellaneda_stoikov import AvellanedaStoikov
from backtest import Backtester, BacktestResult
from market_simulator import MarketSimulator
from order_book import LimitOrderBook, Order, Trade


# ======================================================================
# LimitOrderBook tests
# ======================================================================


class TestLimitOrderBook:
    def test_add_bid_ask_and_mid_price(self):
        lob = LimitOrderBook()
        lob.add_limit_order("buy", 99.0, 1.0, 0.0)
        lob.add_limit_order("sell", 101.0, 1.0, 0.0)
        assert lob.best_bid() == pytest.approx(99.0)
        assert lob.best_ask() == pytest.approx(101.0)
        assert lob.mid_price() == pytest.approx(100.0)
        assert lob.spread() == pytest.approx(2.0)

    def test_cancel_order_removes_from_book(self):
        lob = LimitOrderBook()
        oid = lob.add_limit_order("buy", 99.0, 1.0, 0.0)
        assert lob.best_bid() == pytest.approx(99.0)
        lob.cancel_order(oid)
        assert lob.best_bid() is None

    def test_cancel_nonexistent_order_returns_false(self):
        lob = LimitOrderBook()
        assert lob.cancel_order(9999) is False

    def test_best_bid_returns_highest(self):
        lob = LimitOrderBook()
        lob.add_limit_order("buy", 98.0, 1.0, 0.0)
        lob.add_limit_order("buy", 100.0, 1.0, 0.1)
        lob.add_limit_order("buy", 99.0, 1.0, 0.2)
        assert lob.best_bid() == pytest.approx(100.0)

    def test_best_ask_returns_lowest(self):
        lob = LimitOrderBook()
        lob.add_limit_order("sell", 102.0, 1.0, 0.0)
        lob.add_limit_order("sell", 100.5, 1.0, 0.1)
        lob.add_limit_order("sell", 101.0, 1.0, 0.2)
        assert lob.best_ask() == pytest.approx(100.5)

    def test_market_buy_fills_against_ask(self):
        lob = LimitOrderBook()
        lob.add_limit_order("sell", 101.0, 2.0, 0.0)
        fills = lob.match_market_order("buy", 1.5, 0.1)
        assert len(fills) == 1
        assert fills[0].price == pytest.approx(101.0)
        assert fills[0].quantity == pytest.approx(1.5)

    def test_market_sell_fills_against_bid(self):
        lob = LimitOrderBook()
        lob.add_limit_order("buy", 99.0, 2.0, 0.0)
        fills = lob.match_market_order("sell", 1.0, 0.1)
        assert len(fills) == 1
        assert fills[0].price == pytest.approx(99.0)
        assert fills[0].quantity == pytest.approx(1.0)

    def test_full_fill_removes_order(self):
        lob = LimitOrderBook()
        lob.add_limit_order("sell", 101.0, 1.0, 0.0)
        lob.match_market_order("buy", 1.0, 0.1)
        assert lob.best_ask() is None

    def test_empty_book_returns_none(self):
        lob = LimitOrderBook()
        assert lob.best_bid() is None
        assert lob.best_ask() is None
        assert lob.mid_price() is None
        assert lob.spread() is None

    def test_market_order_on_empty_book(self):
        lob = LimitOrderBook()
        fills = lob.match_market_order("buy", 1.0, 0.0)
        assert fills == []


# ======================================================================
# AvellanedaStoikov tests
# ======================================================================


class TestAvellanedaStoikov:
    def setup_method(self):
        self.strategy = AvellanedaStoikov(gamma=0.1, sigma=2.0, kappa=1.5, T=1.0)

    def test_reservation_price_no_inventory(self):
        r = self.strategy.reservation_price(100.0, 0, 0.5)
        assert r == pytest.approx(100.0)

    def test_reservation_price_long_inventory(self):
        # Long inventory → reservation price below mid
        r = self.strategy.reservation_price(100.0, 5.0, 0.5)
        assert r < 100.0

    def test_reservation_price_short_inventory(self):
        # Short inventory → reservation price above mid
        r = self.strategy.reservation_price(100.0, -5.0, 0.5)
        assert r > 100.0

    def test_optimal_spread_is_positive(self):
        spread = self.strategy.optimal_spread(0.5)
        assert spread > 0.0

    def test_spread_decreases_toward_horizon(self):
        spread_early = self.strategy.optimal_spread(0.0)
        spread_late = self.strategy.optimal_spread(0.9)
        assert spread_early > spread_late

    def test_quotes_bid_below_ask(self):
        bid, ask = self.strategy.quotes(100.0, 0, 0.5)
        assert bid < ask

    def test_quotes_symmetric_at_zero_inventory(self):
        bid, ask = self.strategy.quotes(100.0, 0, 0.5)
        mid = (bid + ask) / 2.0
        r = self.strategy.reservation_price(100.0, 0, 0.5)
        assert mid == pytest.approx(r, rel=1e-9)

    def test_quotes_shift_with_long_inventory(self):
        bid0, ask0 = self.strategy.quotes(100.0, 0, 0.5)
        bid_long, ask_long = self.strategy.quotes(100.0, 5.0, 0.5)
        # Long inventory → quotes skewed downward
        assert ask_long < ask0
        assert bid_long < bid0

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError):
            AvellanedaStoikov(gamma=0, sigma=2.0, kappa=1.5, T=1.0)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            AvellanedaStoikov(gamma=0.1, sigma=0, kappa=1.5, T=1.0)

    def test_arrival_intensity_decreases_with_delta(self):
        i1 = self.strategy.arrival_intensity(0.1)
        i2 = self.strategy.arrival_intensity(0.5)
        assert i1 > i2


# ======================================================================
# MarketSimulator tests
# ======================================================================


class TestMarketSimulator:
    def test_generates_events(self):
        sim = MarketSimulator(T=0.1, dt=0.01, seed=1)
        events = sim.generate()
        assert len(events) > 0

    def test_price_path_length(self):
        sim = MarketSimulator(T=0.5, dt=0.05, seed=2)
        sim.generate()
        # steps = T / dt = 10
        assert len(sim.price_path) == 10

    def test_prices_positive(self):
        sim = MarketSimulator(T=1.0, dt=0.01, seed=3, sigma=5.0)
        sim.generate()
        assert all(p > 0 for p in sim.price_path)

    def test_reproducibility(self):
        events_a = MarketSimulator(seed=99).generate()
        events_b = MarketSimulator(seed=99).generate()
        prices_a = [e.mid_price for e in events_a if e.event_type == "price_update"]
        prices_b = [e.mid_price for e in events_b if e.event_type == "price_update"]
        assert prices_a == prices_b

    def test_event_types(self):
        sim = MarketSimulator(T=0.1, dt=0.01, seed=5)
        events = sim.generate()
        types = {e.event_type for e in events}
        assert "price_update" in types


# ======================================================================
# Backtester integration tests
# ======================================================================


class TestBacktester:
    def _run_short(self, seed=42):
        sim = MarketSimulator(T=0.1, dt=0.01, seed=seed)
        strategy = AvellanedaStoikov(gamma=0.1, sigma=2.0, kappa=1.5, T=0.1)
        bt = Backtester(sim, strategy, order_size=1.0)
        return bt.run()

    def test_result_has_records(self):
        result = self._run_short()
        assert len(result.times) > 0

    def test_pnl_series_length_matches_times(self):
        result = self._run_short()
        assert len(result.pnl) == len(result.times)

    def test_spread_is_positive(self):
        result = self._run_short()
        assert all(s > 0 for s in result.spreads)

    def test_avg_spread_positive(self):
        result = self._run_short()
        assert result.avg_spread() > 0

    def test_max_inventory_non_negative(self):
        result = self._run_short()
        assert result.max_inventory() >= 0

    def test_sharpe_is_finite_or_nan(self):
        result = self._run_short()
        sharpe = result.sharpe_ratio()
        # Should be a real number (possibly nan if no P&L change)
        assert isinstance(sharpe, float)

    def test_bid_always_below_ask(self):
        result = self._run_short()
        for b, a in zip(result.bid_quotes, result.ask_quotes):
            assert b < a
