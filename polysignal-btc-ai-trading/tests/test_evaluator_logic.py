"""
Tests for the core scoring logic in EvaluatorAgent (evaluator.py).

All tests are pure — no network calls, no database, no API keys required.
The scoring logic is extracted and tested in isolation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ── Replicate core scoring logic from EvaluatorAgent._score_one ──────────────

def _score(direction: str, confidence: float, open_price: float, close_price: float) -> dict:
    """Mirror of the evaluator's scoring math for isolated unit testing."""
    actual  = "UP" if close_price >= open_price else "DOWN"
    correct = (direction == actual)
    brier   = round((confidence - (1.0 if correct else 0.0)) ** 2, 4)
    pct     = round((close_price - open_price) / open_price * 100, 4)
    return {"actual": actual, "correct": correct, "brier": brier, "pct": pct}


# ── Outcome determination ─────────────────────────────────────────────────────

class TestOutcomeDetermination:
    def test_price_rises_resolves_up(self):
        assert _score("UP", 0.6, 84000.0, 84001.0)["actual"] == "UP"

    def test_price_falls_resolves_down(self):
        assert _score("DOWN", 0.6, 84000.0, 83999.0)["actual"] == "DOWN"

    def test_equal_price_resolves_up(self):
        # Polymarket rule: close >= open → UP
        assert _score("UP", 0.6, 84000.0, 84000.0)["actual"] == "UP"

    def test_tiny_move_up(self):
        assert _score("UP", 0.6, 84000.0, 84000.01)["actual"] == "UP"

    def test_tiny_move_down(self):
        assert _score("DOWN", 0.6, 84000.0, 83999.99)["actual"] == "DOWN"


# ── Correctness flag ──────────────────────────────────────────────────────────

class TestCorrectnessFlag:
    def test_correct_up_call(self):
        assert _score("UP", 0.6, 84000.0, 84100.0)["correct"] is True

    def test_wrong_up_call(self):
        assert _score("UP", 0.6, 84000.0, 83900.0)["correct"] is False

    def test_correct_down_call(self):
        assert _score("DOWN", 0.6, 84000.0, 83900.0)["correct"] is True

    def test_wrong_down_call(self):
        assert _score("DOWN", 0.6, 84000.0, 84100.0)["correct"] is False


# ── Brier score ───────────────────────────────────────────────────────────────

class TestBrierScore:
    def test_correct_call_formula(self):
        # correct: BS = (conf - 1)^2
        r = _score("UP", 0.6, 84000.0, 84100.0)
        assert r["brier"] == pytest.approx((0.6 - 1.0) ** 2, abs=1e-4)

    def test_wrong_call_formula(self):
        # wrong: BS = (conf - 0)^2
        r = _score("UP", 0.6, 84000.0, 83900.0)
        assert r["brier"] == pytest.approx((0.6 - 0.0) ** 2, abs=1e-4)

    def test_perfect_correct(self):
        # 100% confidence, correct → BS = 0
        assert _score("UP", 1.0, 84000.0, 84100.0)["brier"] == 0.0

    def test_perfect_wrong(self):
        # 100% confidence, wrong → BS = 1 (worst possible)
        assert _score("UP", 1.0, 84000.0, 83900.0)["brier"] == 1.0

    def test_coin_flip_baseline(self):
        # 50% confidence → BS = 0.25 regardless of outcome
        r_correct = _score("UP", 0.5, 84000.0, 84100.0)
        r_wrong   = _score("UP", 0.5, 84000.0, 83900.0)
        assert r_correct["brier"] == pytest.approx(0.25, abs=1e-4)
        assert r_wrong["brier"]   == pytest.approx(0.25, abs=1e-4)

    def test_overconfident_correct_penalized(self):
        # 90% confidence, correct → BS = 0.01
        # vs 60% confidence, correct → BS = 0.16
        # Lower is better — 90% correct should score better than 60% correct
        r_90 = _score("UP", 0.9, 84000.0, 84100.0)
        r_60 = _score("UP", 0.6, 84000.0, 84100.0)
        assert r_90["brier"] < r_60["brier"]

    def test_overconfident_wrong_heavily_penalized(self):
        # 90% confidence, wrong → BS = 0.81 (much worse than 60% wrong = 0.36)
        r_90 = _score("UP", 0.9, 84000.0, 83900.0)
        r_60 = _score("UP", 0.6, 84000.0, 83900.0)
        assert r_90["brier"] > r_60["brier"]


# ── Price change percentage ───────────────────────────────────────────────────

class TestPriceChangePct:
    def test_one_percent_up(self):
        r = _score("UP", 0.6, 100.0, 101.0)
        assert r["pct"] == pytest.approx(1.0, abs=1e-3)

    def test_one_percent_down(self):
        r = _score("DOWN", 0.6, 100.0, 99.0)
        assert r["pct"] == pytest.approx(-1.0, abs=1e-3)

    def test_flat_market(self):
        r = _score("UP", 0.6, 100.0, 100.0)
        assert r["pct"] == 0.0

    def test_large_price(self):
        # Typical BTC price — ensure no overflow
        r = _score("UP", 0.6, 84000.0, 84042.0)
        assert r["pct"] == pytest.approx(0.05, abs=0.01)
