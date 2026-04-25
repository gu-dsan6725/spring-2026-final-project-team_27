"""
Tests for pure helper functions in collector.py.
No network calls, no WebSocket, no API keys.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from polysignal.collector import _compute_rsi, _pct_change, _parse_updown_prices_event


# ── _pct_change ───────────────────────────────────────────────────────────────

class TestPctChange:
    def test_one_percent_up(self):
        assert _pct_change(101.0, 100.0) == pytest.approx(1.0, abs=0.001)

    def test_one_percent_down(self):
        assert _pct_change(99.0, 100.0) == pytest.approx(-1.0, abs=0.001)

    def test_flat(self):
        assert _pct_change(100.0, 100.0) == pytest.approx(0.0, abs=0.001)

    def test_none_prev_returns_none(self):
        assert _pct_change(100.0, None) is None

    def test_zero_prev_returns_none(self):
        assert _pct_change(100.0, 0.0) is None


# ── _compute_rsi ──────────────────────────────────────────────────────────────

class TestComputeRSI:
    def _flat_closes(self, n: int = 20, price: float = 100.0) -> list:
        return [price] * n

    def _trending_up(self, n: int = 20) -> list:
        return [100.0 + i for i in range(n)]

    def _trending_down(self, n: int = 20) -> list:
        return [100.0 - i for i in range(n)]

    def test_insufficient_data_returns_none(self):
        # Need at least period+1 = 15 closes for period=14
        closes = [100.0] * 14
        assert _compute_rsi(closes) is None

    def test_exactly_minimum_data(self):
        closes = [100.0 + i for i in range(15)]
        result = _compute_rsi(closes)
        assert result is not None

    def test_flat_market_returns_100_or_0(self):
        # All flat (no gains, no losses) → RSI = 100 (no avg_loss)
        closes = self._flat_closes(20)
        result = _compute_rsi(closes)
        assert result in (100.0, 0.0) or result is None

    def test_strong_uptrend_gives_high_rsi(self):
        closes = self._trending_up(20)
        result = _compute_rsi(closes)
        assert result is not None
        assert result > 70  # consistent gains → overbought

    def test_strong_downtrend_gives_low_rsi(self):
        closes = self._trending_down(20)
        result = _compute_rsi(closes)
        assert result is not None
        assert result < 30  # consistent losses → oversold

    def test_rsi_in_valid_range(self):
        for closes in [
            self._trending_up(20),
            self._trending_down(20),
            [100 + (i % 3 - 1) * 0.5 for i in range(20)],
        ]:
            result = _compute_rsi(closes)
            if result is not None:
                assert 0.0 <= result <= 100.0

    def test_all_gains_no_loss_returns_100(self):
        closes = [100.0 + i * 0.5 for i in range(20)]
        result = _compute_rsi(closes)
        assert result == pytest.approx(100.0, abs=0.01)

    def test_all_losses_returns_0(self):
        closes = [100.0 - i * 0.5 for i in range(20)]
        result = _compute_rsi(closes)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_uses_only_recent_period(self):
        # Long history trending up, but last 14 candles trending down
        # RSI should reflect the recent downtrend
        closes = [100.0 + i for i in range(20)]   # first 20 trending up
        closes += [120.0 - i for i in range(1, 16)]  # last 15 trending down
        result = _compute_rsi(closes)
        assert result is not None
        assert result < 50  # recent downtrend should dominate

    def test_default_period_14(self):
        closes = self._trending_up(20)
        result_default = _compute_rsi(closes)
        result_explicit = _compute_rsi(closes, period=14)
        assert result_default == result_explicit


# ── _parse_updown_prices_event ────────────────────────────────────────────────

class TestParseUpDownPrices:
    def _make_market(self, outcomes, prices):
        import json
        return {
            "outcomes":      json.dumps(outcomes),
            "outcomePrices": json.dumps([str(p) for p in prices]),
        }

    def test_standard_up_down(self):
        m = self._make_market(["Up", "Down"], [0.55, 0.45])
        up, down = _parse_updown_prices_event(m)
        assert up == pytest.approx(0.55, abs=0.001)
        assert down == pytest.approx(0.45, abs=0.001)

    def test_case_insensitive(self):
        m = self._make_market(["UP", "DOWN"], [0.52, 0.48])
        up, down = _parse_updown_prices_event(m)
        assert up == pytest.approx(0.52, abs=0.001)
        assert down == pytest.approx(0.48, abs=0.001)

    def test_reversed_order(self):
        m = self._make_market(["Down", "Up"], [0.48, 0.52])
        up, down = _parse_updown_prices_event(m)
        assert up == pytest.approx(0.52, abs=0.001)
        assert down == pytest.approx(0.48, abs=0.001)

    def test_missing_data_returns_default(self):
        up, down = _parse_updown_prices_event({})
        assert up == 0.5
        assert down == 0.5

    def test_even_split(self):
        m = self._make_market(["Up", "Down"], [0.50, 0.50])
        up, down = _parse_updown_prices_event(m)
        assert up == pytest.approx(0.50, abs=0.001)
        assert down == pytest.approx(0.50, abs=0.001)
