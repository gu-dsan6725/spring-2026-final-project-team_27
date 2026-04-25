"""
Tests for compute_baselines, compute_per_model_accuracy, and _ci95 (reporter.py).
All tests are pure — no database, no network, no API keys.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
from polysignal.reporter import compute_baselines, compute_per_model_accuracy, _ci95


# ── _ci95 ─────────────────────────────────────────────────────────────────────

class TestCI95:
    def test_zero_n_returns_zero(self):
        assert _ci95(0.5, 0) == 0.0

    def test_large_n_gives_tight_interval(self):
        # 1000 samples → CI should be < 4% (actual ≈ 3.1%)
        assert _ci95(0.5, 1000) < 0.04

    def test_small_n_gives_wide_interval(self):
        # 10 samples → CI should be > 25%
        assert _ci95(0.5, 10) > 0.25

    def test_extreme_probability_narrows_ci(self):
        # p=0 or p=1 → CI = 0 (no variance)
        assert _ci95(0.0, 100) == 0.0
        assert _ci95(1.0, 100) == 0.0

    def test_midpoint_widest(self):
        # p=0.5 gives widest CI for given n
        assert _ci95(0.5, 100) > _ci95(0.3, 100)
        assert _ci95(0.5, 100) > _ci95(0.7, 100)


# ── compute_baselines ─────────────────────────────────────────────────────────

def _pred(actual: str, market_up_odds: float = 0.52) -> dict:
    return {"actual_outcome": actual, "market_up_odds": market_up_odds}


class TestComputeBaselines:
    def test_empty_returns_empty(self):
        assert compute_baselines([]) == {}

    def test_returns_required_keys(self):
        preds = [_pred("UP")] * 5 + [_pred("DOWN")] * 5
        b = compute_baselines(preds)
        assert "n" in b
        assert "coin_flip" in b
        assert "always_up" in b
        assert "always_down" in b
        assert "crowd" in b
        assert "up_rate" in b

    def test_n_matches_input(self):
        preds = [_pred("UP")] * 7 + [_pred("DOWN")] * 3
        b = compute_baselines(preds)
        assert b["n"] == 10

    def test_up_rate(self):
        preds = [_pred("UP")] * 6 + [_pred("DOWN")] * 4
        b = compute_baselines(preds)
        assert b["up_rate"] == pytest.approx(0.6, abs=0.01)

    def test_coin_flip_fixed(self):
        preds = [_pred("UP")] * 5 + [_pred("DOWN")] * 5
        b = compute_baselines(preds)
        assert b["coin_flip"]["accuracy"] == 0.5
        assert b["coin_flip"]["brier"] == pytest.approx(0.25, abs=0.001)

    # ── always_up ─────────────────────────────────────────────────────────────

    def test_always_up_accuracy_equals_up_rate(self):
        preds = [_pred("UP")] * 6 + [_pred("DOWN")] * 4
        b = compute_baselines(preds)
        assert b["always_up"]["accuracy"] == pytest.approx(0.6, abs=0.01)

    def test_always_up_perfect_when_all_up(self):
        preds = [_pred("UP")] * 10
        b = compute_baselines(preds)
        assert b["always_up"]["accuracy"] == pytest.approx(1.0, abs=0.001)

    def test_always_up_zero_when_all_down(self):
        preds = [_pred("DOWN")] * 10
        b = compute_baselines(preds)
        assert b["always_up"]["accuracy"] == pytest.approx(0.0, abs=0.001)

    def test_always_up_brier_when_all_down(self):
        # Always UP with confidence 1.0, all DOWN → BS = 1.0 per prediction
        preds = [_pred("DOWN")] * 5
        b = compute_baselines(preds)
        assert b["always_up"]["brier"] == pytest.approx(1.0, abs=0.001)

    def test_always_up_brier_when_all_up(self):
        # Always UP with confidence 1.0, all UP → BS = 0.0 per prediction
        preds = [_pred("UP")] * 5
        b = compute_baselines(preds)
        assert b["always_up"]["brier"] == pytest.approx(0.0, abs=0.001)

    # ── always_down ───────────────────────────────────────────────────────────

    def test_always_down_plus_always_up_equals_one(self):
        preds = [_pred("UP")] * 7 + [_pred("DOWN")] * 3
        b = compute_baselines(preds)
        assert b["always_up"]["accuracy"] + b["always_down"]["accuracy"] == pytest.approx(1.0, abs=0.001)

    # ── crowd ─────────────────────────────────────────────────────────────────

    def test_crowd_perfect_when_always_correct(self):
        # Market odds > 0.5 for UP, all outcomes UP → 100% crowd accuracy
        preds = [_pred("UP", market_up_odds=0.60)] * 10
        b = compute_baselines(preds)
        assert b["crowd"]["accuracy"] == pytest.approx(1.0, abs=0.001)

    def test_crowd_zero_when_always_wrong(self):
        # Market odds > 0.5 for UP, all outcomes DOWN → 0% crowd accuracy
        preds = [_pred("DOWN", market_up_odds=0.60)] * 10
        b = compute_baselines(preds)
        assert b["crowd"]["accuracy"] == pytest.approx(0.0, abs=0.001)

    def test_crowd_uses_odds_for_brier(self):
        # Crowd UP odds = 0.6, actual = UP → BS = (0.6 - 1)^2 = 0.16
        preds = [_pred("UP", market_up_odds=0.60)] * 4
        b = compute_baselines(preds)
        assert b["crowd"]["brier"] == pytest.approx(0.16, abs=0.001)

    def test_crowd_50_50_odds(self):
        # Exactly 0.5 odds → crowd predicts DOWN (<=0.5 branch)
        preds = [_pred("UP", market_up_odds=0.50)] * 5 + [_pred("DOWN", market_up_odds=0.50)] * 5
        b = compute_baselines(preds)
        # crowd predicts DOWN for all (odds = 0.50 <= 0.5)
        # crowd correct for the 5 DOWN outcomes
        assert b["crowd"]["accuracy"] == pytest.approx(0.5, abs=0.01)

    def test_ci_fields_present(self):
        preds = [_pred("UP")] * 5 + [_pred("DOWN")] * 5
        b = compute_baselines(preds)
        assert "ci" in b["always_up"]
        assert "ci" in b["always_down"]
        assert "ci" in b["crowd"]

    def test_ci_non_negative(self):
        preds = [_pred("UP")] * 5 + [_pred("DOWN")] * 5
        b = compute_baselines(preds)
        assert b["always_up"]["ci"] >= 0
        assert b["crowd"]["ci"] >= 0


# ── compute_per_model_accuracy ────────────────────────────────────────────────

def _resolved_pred(actual: str, votes: list) -> dict:
    """Build a mock resolved prediction with ensemble reasoning."""
    import json
    votes_json = json.dumps(votes)
    reasoning  = f"[ENSEMBLE — {len(votes)}/3 models]\nVotes: {votes_json}\n\nMajority: {actual}"
    return {
        "actual_outcome": actual,
        "reasoning":      reasoning,
        "market_up_odds": 0.52,
    }


class TestPerModelAccuracy:
    def test_empty_returns_empty(self):
        assert compute_per_model_accuracy([]) == {}

    def test_non_ensemble_reasoning_skipped(self):
        preds = [{"actual_outcome": "UP", "reasoning": "just a regular prediction", "market_up_odds": 0.52}]
        result = compute_per_model_accuracy(preds)
        assert result == {}

    def test_single_vote_correct(self):
        preds = [_resolved_pred("UP", [
            {"model": "Haiku (Momentum)", "direction": "UP", "confidence": 0.55}
        ])]
        result = compute_per_model_accuracy(preds)
        assert "Haiku (Momentum)" in result
        assert result["Haiku (Momentum)"]["correct"] == 1
        assert result["Haiku (Momentum)"]["accuracy"] == 1.0

    def test_single_vote_wrong(self):
        preds = [_resolved_pred("DOWN", [
            {"model": "Haiku (Momentum)", "direction": "UP", "confidence": 0.55}
        ])]
        result = compute_per_model_accuracy(preds)
        assert result["Haiku (Momentum)"]["correct"] == 0
        assert result["Haiku (Momentum)"]["accuracy"] == 0.0

    def test_multiple_models_tracked_separately(self):
        preds = [_resolved_pred("UP", [
            {"model": "Haiku (Momentum)",       "direction": "UP",   "confidence": 0.55},
            {"model": "Sonnet (Technical)",      "direction": "DOWN", "confidence": 0.58},
            {"model": "Sonnet 4.6 (Contrarian)", "direction": "UP",   "confidence": 0.60},
        ])]
        result = compute_per_model_accuracy(preds)
        assert result["Haiku (Momentum)"]["correct"] == 1        # UP → correct
        assert result["Sonnet (Technical)"]["correct"] == 0      # DOWN → wrong
        assert result["Sonnet 4.6 (Contrarian)"]["correct"] == 1

    def test_accumulates_across_predictions(self):
        preds = [
            _resolved_pred("UP",   [{"model": "Haiku (Momentum)", "direction": "UP",   "confidence": 0.55}]),
            _resolved_pred("DOWN", [{"model": "Haiku (Momentum)", "direction": "DOWN", "confidence": 0.58}]),
            _resolved_pred("UP",   [{"model": "Haiku (Momentum)", "direction": "DOWN", "confidence": 0.56}]),
        ]
        result = compute_per_model_accuracy(preds)
        h = result["Haiku (Momentum)"]
        assert h["total"]   == 3
        assert h["correct"] == 2   # pred 1 correct, pred 2 correct, pred 3 wrong

    def test_brier_score_correct_call(self):
        preds = [_resolved_pred("UP", [
            {"model": "Haiku (Momentum)", "direction": "UP", "confidence": 0.60}
        ])]
        result = compute_per_model_accuracy(preds)
        expected_brier = (0.60 - 1.0) ** 2  # 0.16
        assert result["Haiku (Momentum)"]["avg_brier"] == pytest.approx(expected_brier, abs=0.001)

    def test_ci_present(self):
        preds = [_resolved_pred("UP", [
            {"model": "Haiku (Momentum)", "direction": "UP", "confidence": 0.55}
        ])]
        result = compute_per_model_accuracy(preds)
        assert "ci" in result["Haiku (Momentum)"]

    def test_total_counts_correct(self):
        preds = [
            _resolved_pred("UP",   [{"model": "M", "direction": "UP",   "confidence": 0.55}]),
            _resolved_pred("UP",   [{"model": "M", "direction": "DOWN", "confidence": 0.55}]),
            _resolved_pred("DOWN", [{"model": "M", "direction": "DOWN", "confidence": 0.55}]),
        ]
        result = compute_per_model_accuracy(preds)
        assert result["M"]["total"] == 3
