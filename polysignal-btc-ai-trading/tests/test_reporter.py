"""
Tests for reporter.py metric computation (compute_metrics).

All tests are pure — no database, no filesystem, no external calls.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from polysignal.reporter import compute_metrics


# ── Test helpers ──────────────────────────────────────────────────────────────

def _eval(direction: str, confidence: float, was_correct: bool, price_change_pct: float = 0.05):
    brier = round((confidence - (1.0 if was_correct else 0.0)) ** 2, 4)
    return {
        "direction":        direction,
        "confidence":       confidence,
        "was_correct":      was_correct,
        "brier_score":      brier,
        "price_change_pct": price_change_pct,
    }


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_list_returns_empty_dict(self):
        assert compute_metrics([]) == {}

    def test_single_correct_prediction(self):
        m = compute_metrics([_eval("UP", 0.55, True)])
        assert m["total"] == 1
        assert m["correct"] == 1
        assert m["accuracy"] == 1.0

    def test_single_wrong_prediction(self):
        m = compute_metrics([_eval("UP", 0.55, False)])
        assert m["total"] == 1
        assert m["correct"] == 0
        assert m["accuracy"] == 0.0


# ── Accuracy ──────────────────────────────────────────────────────────────────

class TestAccuracy:
    def test_perfect_accuracy(self):
        evals = [_eval("UP", 0.6, True) for _ in range(10)]
        m = compute_metrics(evals)
        assert m["accuracy"] == 1.0
        assert m["correct"] == 10

    def test_zero_accuracy(self):
        evals = [_eval("UP", 0.6, False) for _ in range(10)]
        m = compute_metrics(evals)
        assert m["accuracy"] == 0.0
        assert m["correct"] == 0

    def test_fifty_percent_accuracy(self):
        evals = [_eval("UP", 0.55, True)] * 5 + [_eval("DOWN", 0.55, False)] * 5
        m = compute_metrics(evals)
        assert m["accuracy"] == 0.5

    def test_total_count_correct(self):
        evals = [_eval("UP", 0.55, True)] * 7 + [_eval("DOWN", 0.55, False)] * 3
        m = compute_metrics(evals)
        assert m["total"] == 10


# ── Direction distribution ────────────────────────────────────────────────────

class TestDirectionCounts:
    def test_up_down_split(self):
        evals = [_eval("UP", 0.55, True)] * 7 + [_eval("DOWN", 0.55, False)] * 3
        m = compute_metrics(evals)
        assert m["up_calls"] == 7
        assert m["down_calls"] == 3

    def test_all_up(self):
        evals = [_eval("UP", 0.55, True) for _ in range(5)]
        m = compute_metrics(evals)
        assert m["up_calls"] == 5
        assert m["down_calls"] == 0

    def test_all_down(self):
        evals = [_eval("DOWN", 0.55, False) for _ in range(5)]
        m = compute_metrics(evals)
        assert m["up_calls"] == 0
        assert m["down_calls"] == 5


# ── Brier score ───────────────────────────────────────────────────────────────

class TestBrierScore:
    def test_all_correct_at_60_pct(self):
        evals = [_eval("UP", 0.6, True) for _ in range(4)]
        m = compute_metrics(evals)
        expected = (0.6 - 1.0) ** 2  # 0.16
        assert m["avg_brier"] == pytest.approx(expected, abs=0.001)

    def test_coin_flip_brier(self):
        evals = [_eval("UP", 0.5, True) for _ in range(10)]
        m = compute_metrics(evals)
        assert m["avg_brier"] == pytest.approx(0.25, abs=0.001)

    def test_mixed_brier(self):
        # 1 correct at 0.6: (0.6-1)^2 = 0.16
        # 1 wrong  at 0.6: (0.6-0)^2 = 0.36
        # avg = 0.26
        evals = [_eval("UP", 0.6, True), _eval("UP", 0.6, False)]
        m = compute_metrics(evals)
        assert m["avg_brier"] == pytest.approx(0.26, abs=0.001)


# ── Confidence calibration bins ───────────────────────────────────────────────

class TestCalibrationBins:
    def test_50_55_bin_populated(self):
        evals = [_eval("UP", 0.52, True) for _ in range(5)]
        m = compute_metrics(evals)
        assert "50–55%" in m["bins"]
        assert m["bins"]["50–55%"]["total"] == 5

    def test_55_60_bin_populated(self):
        evals = [_eval("UP", 0.57, False) for _ in range(3)]
        m = compute_metrics(evals)
        assert "55–60%" in m["bins"]

    def test_65_plus_bin_populated(self):
        evals = [_eval("UP", 0.70, True) for _ in range(4)]
        m = compute_metrics(evals)
        assert "65%+" in m["bins"]

    def test_empty_bins_absent(self):
        evals = [_eval("UP", 0.52, True) for _ in range(3)]
        m = compute_metrics(evals)
        assert "65%+" not in m["bins"]
        assert "55–60%" not in m["bins"]

    def test_bin_accuracy_all_correct(self):
        evals = [_eval("UP", 0.52, True) for _ in range(6)]
        m = compute_metrics(evals)
        assert m["bins"]["50–55%"]["accuracy"] == 1.0
        assert m["bins"]["50–55%"]["correct"] == 6

    def test_bin_accuracy_all_wrong(self):
        evals = [_eval("UP", 0.52, False) for _ in range(4)]
        m = compute_metrics(evals)
        assert m["bins"]["50–55%"]["accuracy"] == 0.0

    def test_bin_gap_well_calibrated(self):
        # accuracy = 0.52 exactly and avg_conf ≈ 0.52 → gap ≈ 0
        evals = (
            [_eval("UP", 0.52, True)] * 52 +
            [_eval("UP", 0.52, False)] * 48
        )
        m = compute_metrics(evals)
        gap = m["bins"]["50–55%"]["gap"]
        assert gap < 0.05  # well calibrated: < 10%

    def test_multiple_bins_simultaneously(self):
        evals = (
            [_eval("UP", 0.52, True)] * 5 +   # 50-55%
            [_eval("UP", 0.57, False)] * 3 +   # 55-60%
            [_eval("UP", 0.62, True)] * 2      # 60-65%
        )
        m = compute_metrics(evals)
        assert len(m["bins"]) == 3

    def test_avg_price_move(self):
        evals = [_eval("UP", 0.55, True, price_change_pct=0.1) for _ in range(4)]
        m = compute_metrics(evals)
        assert m["avg_price_move_pct"] == pytest.approx(0.1, abs=0.001)
