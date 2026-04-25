"""Tests for Pydantic data models (models.py)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timezone
from polysignal.models import MarketWindow, Prediction, EvalRecord, ModelVote, PriceCandle, Trade


def _now():
    return datetime.now(timezone.utc)


def _make_window(**kwargs):
    defaults = dict(
        market_id="btc-updown-5m-1745000000",
        market_question="Will BTC be higher in 5 minutes?",
        window_start=_now(),
        window_end=_now(),
        up_price=0.52,
        down_price=0.48,
        btc_price_now=84000.0,
    )
    defaults.update(kwargs)
    return MarketWindow(**defaults)


def _make_prediction(**kwargs):
    defaults = dict(
        prediction_id="pred-abc-123",
        market_id="btc-updown-5m-1745000000",
        market_question="Will BTC be higher in 5 minutes?",
        window_start=_now(),
        window_end=_now(),
        direction="UP",
        confidence=0.58,
        btc_price_at_call=84000.0,
        market_up_odds=0.52,
        reasoning="Strong upward momentum on 1m candle.",
    )
    defaults.update(kwargs)
    return Prediction(**defaults)


# ── MarketWindow ──────────────────────────────────────────────────────────────

class TestMarketWindow:
    def test_basic_creation(self):
        w = _make_window()
        assert w.btc_price_now == 84000.0
        assert w.up_price == 0.52
        assert w.down_price == 0.48

    def test_optional_fields_default_none(self):
        w = _make_window()
        assert w.rsi_14 is None
        assert w.volatility_5m is None
        assert w.fear_greed_score is None
        assert w.btc_price_1m_ago is None

    def test_candles_default_empty(self):
        w = _make_window()
        assert w.candles_1m == []

    def test_with_optional_fields(self):
        w = _make_window(rsi_14=65.2, volatility_5m=45.3, fear_greed_score=72, fear_greed_label="Greed")
        assert w.rsi_14 == 65.2
        assert w.fear_greed_label == "Greed"

    def test_market_id_stored(self):
        w = _make_window(market_id="btc-updown-5m-9999999")
        assert w.market_id == "btc-updown-5m-9999999"


# ── Prediction ────────────────────────────────────────────────────────────────

class TestPrediction:
    def test_basic_creation(self):
        p = _make_prediction()
        assert p.direction == "UP"
        assert p.confidence == 0.58

    def test_defaults_unresolved(self):
        p = _make_prediction()
        assert p.resolved is False
        assert p.actual_outcome is None
        assert p.btc_price_close is None

    def test_down_direction(self):
        p = _make_prediction(direction="DOWN")
        assert p.direction == "DOWN"

    def test_invalid_direction_raises(self):
        with pytest.raises(Exception):
            _make_prediction(direction="SIDEWAYS")

    def test_key_factors_default_empty(self):
        p = _make_prediction()
        assert p.key_factors == []

    def test_key_factors_stored(self):
        p = _make_prediction(key_factors=["RSI overbought", "Negative momentum"])
        assert len(p.key_factors) == 2


# ── EvalRecord ────────────────────────────────────────────────────────────────

class TestEvalRecord:
    def test_correct_up_call(self):
        r = EvalRecord(
            eval_id="eval-1",
            prediction_id="pred-1",
            market_id="market-1",
            direction="UP",
            confidence=0.60,
            actual_outcome="UP",
            btc_open=84000.0,
            btc_close=84050.0,
            price_change_pct=0.0595,
            was_correct=True,
            brier_score=round((0.60 - 1.0) ** 2, 4),
        )
        assert r.was_correct is True
        assert r.brier_score == pytest.approx(0.16, abs=0.001)

    def test_wrong_down_call(self):
        r = EvalRecord(
            eval_id="eval-2",
            prediction_id="pred-2",
            market_id="market-1",
            direction="DOWN",
            confidence=0.60,
            actual_outcome="UP",
            btc_open=84000.0,
            btc_close=84050.0,
            price_change_pct=0.0595,
            was_correct=False,
            brier_score=round((0.60 - 0.0) ** 2, 4),
        )
        assert r.was_correct is False
        assert r.brier_score == pytest.approx(0.36, abs=0.001)

    def test_invalid_direction_raises(self):
        with pytest.raises(Exception):
            EvalRecord(
                eval_id="e", prediction_id="p", market_id="m",
                direction="FLAT",
                confidence=0.5,
                actual_outcome="UP",
                btc_open=84000.0, btc_close=84000.0,
                price_change_pct=0.0,
                was_correct=False,
                brier_score=0.25,
            )


# ── ModelVote ─────────────────────────────────────────────────────────────────

class TestModelVote:
    def test_creation(self):
        v = ModelVote(
            model_id="claude-haiku-4-5",
            model_label="Haiku (Momentum)",
            direction="DOWN",
            confidence=0.55,
            reasoning="Negative 1m momentum",
        )
        assert v.direction == "DOWN"
        assert v.confidence == 0.55

    def test_invalid_direction_raises(self):
        with pytest.raises(Exception):
            ModelVote(
                model_id="m", model_label="test",
                direction="NEUTRAL",
                confidence=0.5,
                reasoning="",
            )


# ── PriceCandle ───────────────────────────────────────────────────────────────

class TestPriceCandle:
    def test_creation(self):
        c = PriceCandle(
            timestamp=_now(),
            open=84000.0, high=84100.0,
            low=83900.0,  close=84050.0,
            volume=12.5,
        )
        assert c.close == 84050.0
        assert c.volume == 12.5

    def test_high_gte_low(self):
        c = PriceCandle(
            timestamp=_now(),
            open=100.0, high=105.0,
            low=95.0,   close=102.0,
            volume=1.0,
        )
        assert c.high >= c.low


# ── Trade ─────────────────────────────────────────────────────────────────────

class TestTrade:
    def test_defaults(self):
        t = Trade(
            trade_id="t-1",
            prediction_id="p-1",
            market_question="Will BTC go up?",
            direction="UP",
            odds_price=0.52,
            bet_amount=500.0,
            tokens=round(500.0 / 0.52, 4),
            potential_payout=round(500.0 / 0.52, 4),
        )
        assert t.resolved is False
        assert t.pnl == 0.0
        assert t.actual_payout == 0.0

    def test_invalid_direction_raises(self):
        with pytest.raises(Exception):
            Trade(
                trade_id="t-1", prediction_id="p-1",
                market_question="test",
                direction="BOTH",
                odds_price=0.52, bet_amount=500.0,
                tokens=961.5, potential_payout=961.5,
            )
