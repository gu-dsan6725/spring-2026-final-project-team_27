"""
Tests for the payout math in simulator.py.

All tests are pure — no database, no filesystem, no API calls.
The settlement math is extracted and tested in isolation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

BET_SIZE = 500.0  # matches simulator.BET_SIZE


# ── Replicate payout math from TradingSimulator ───────────────────────────────

def _calc_tokens(odds_price: float) -> tuple[float, float]:
    """Mirrors TradingSimulator.place_trade token/payout calculation."""
    tokens          = BET_SIZE / odds_price
    potential_payout = tokens * 1.0
    return tokens, potential_payout


def _settle(direction: str, actual_outcome: str, tokens: float) -> tuple[bool, float, float]:
    """Mirrors TradingSimulator.settle payout logic."""
    won           = (direction == actual_outcome)
    actual_payout = tokens if won else 0.0
    pnl           = actual_payout - BET_SIZE
    return won, actual_payout, pnl


# ── Token calculation ─────────────────────────────────────────────────────────

class TestTokenCalculation:
    def test_even_odds(self):
        tokens, payout = _calc_tokens(0.5)
        assert tokens == pytest.approx(1000.0, abs=0.01)

    def test_payout_equals_tokens(self):
        for odds in [0.48, 0.50, 0.52, 0.55, 0.60, 0.65]:
            tokens, payout = _calc_tokens(odds)
            assert tokens == pytest.approx(payout, abs=1e-6)

    def test_higher_odds_fewer_tokens(self):
        # Favourite (0.60) → fewer tokens than underdog (0.48)
        tokens_fav, _  = _calc_tokens(0.60)
        tokens_dog, _  = _calc_tokens(0.48)
        assert tokens_fav < tokens_dog

    def test_at_52_odds(self):
        tokens, _ = _calc_tokens(0.52)
        assert tokens == pytest.approx(500.0 / 0.52, abs=0.001)


# ── Win / loss settlement ─────────────────────────────────────────────────────

class TestSettlement:
    def test_win_returns_tokens(self):
        tokens, _ = _calc_tokens(0.52)
        won, payout, _ = _settle("UP", "UP", tokens)
        assert won is True
        assert payout == pytest.approx(tokens, abs=1e-6)

    def test_loss_returns_zero(self):
        tokens, _ = _calc_tokens(0.52)
        won, payout, _ = _settle("UP", "DOWN", tokens)
        assert won is False
        assert payout == 0.0

    def test_loss_pnl_always_minus_bet(self):
        for odds in [0.48, 0.50, 0.52, 0.60]:
            tokens, _ = _calc_tokens(odds)
            _, _, pnl = _settle("UP", "DOWN", tokens)
            assert pnl == pytest.approx(-BET_SIZE, abs=1e-6)

    def test_win_pnl_positive(self):
        tokens, _ = _calc_tokens(0.52)
        _, _, pnl = _settle("UP", "UP", tokens)
        assert pnl > 0

    def test_win_pnl_at_even_odds(self):
        tokens, _ = _calc_tokens(0.5)
        _, _, pnl = _settle("UP", "UP", tokens)
        assert pnl == pytest.approx(500.0, abs=0.01)


# ── Payout asymmetry ──────────────────────────────────────────────────────────

class TestPayoutAsymmetry:
    def test_favourite_win_pays_less_than_even(self):
        # 0.60 odds → winning pays less than $500 profit
        tokens, _ = _calc_tokens(0.60)
        _, _, pnl = _settle("UP", "UP", tokens)
        assert pnl < 500.0

    def test_underdog_win_pays_more_than_even(self):
        # 0.40 odds → winning pays more than $500 profit
        tokens, _ = _calc_tokens(0.40)
        _, _, pnl = _settle("UP", "UP", tokens)
        assert pnl > 500.0

    def test_slight_favourite_marginal_edge(self):
        # Real Polymarket scenario: UP at 0.52 odds
        # Win: pnl slightly below $500. Loss: pnl = -$500
        tokens, _ = _calc_tokens(0.52)
        _, _, win_pnl  = _settle("UP", "UP",   tokens)
        _, _, loss_pnl = _settle("UP", "DOWN", tokens)
        assert win_pnl  < 500.0
        assert loss_pnl == pytest.approx(-500.0, abs=1e-6)

    def test_expected_value_at_50_pct_accuracy(self):
        # At 0.52 odds with 50% win rate: E[pnl] < 0 (house edge)
        tokens, _ = _calc_tokens(0.52)
        _, _, win_pnl  = _settle("UP", "UP",   tokens)
        _, _, loss_pnl = _settle("UP", "DOWN", tokens)
        ev = 0.5 * win_pnl + 0.5 * loss_pnl
        assert ev < 0

    def test_expected_value_at_55_pct_accuracy(self):
        # At 0.52 odds with 55% win rate: E[pnl] > 0 (edge over market)
        tokens, _ = _calc_tokens(0.52)
        _, _, win_pnl  = _settle("UP", "UP",   tokens)
        _, _, loss_pnl = _settle("UP", "DOWN", tokens)
        ev = 0.55 * win_pnl + 0.45 * loss_pnl
        assert ev > 0
