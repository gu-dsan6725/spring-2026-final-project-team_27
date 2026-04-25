"""
Tests for ensemble.py — JSON parsing and vote aggregation.

All tests are pure — no API calls, no network, no API keys required.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from polysignal.ensemble import _parse_json, EnsembleAgent
from polysignal.models import ModelVote


# ── Test helpers ──────────────────────────────────────────────────────────────

def _vote(direction: str, confidence: float, label: str = "Test") -> ModelVote:
    return ModelVote(
        model_id=f"model-{label.lower()}",
        model_label=label,
        direction=direction,
        confidence=confidence,
        reasoning="test reasoning",
    )


class _EnsembleNoAPI(EnsembleAgent):
    """Subclass that skips __init__ to test _aggregate without an API key."""
    def __init__(self):
        pass  # intentionally skip super().__init__


# ── JSON parsing ──────────────────────────────────────────────────────────────

class TestParseJson:
    def test_clean_json(self):
        text = '{"direction": "UP", "confidence": 0.55, "reasoning": "momentum", "key_factors": []}'
        r = _parse_json(text)
        assert r["direction"] == "UP"
        assert r["confidence"] == 0.55

    def test_markdown_json_fence(self):
        text = '```json\n{"direction": "DOWN", "confidence": 0.58}\n```'
        r = _parse_json(text)
        assert r is not None
        assert r["direction"] == "DOWN"

    def test_plain_code_fence(self):
        text = '```\n{"direction": "UP", "confidence": 0.60}\n```'
        r = _parse_json(text)
        assert r is not None
        assert r["direction"] == "UP"

    def test_json_embedded_in_prose(self):
        text = 'Here is my analysis.\n{"direction": "UP", "confidence": 0.55}\nEnd.'
        r = _parse_json(text)
        assert r is not None
        assert r["direction"] == "UP"

    def test_invalid_json_returns_none(self):
        assert _parse_json("this is not json") is None

    def test_empty_string_returns_none(self):
        assert _parse_json("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_json("   \n\t  ") is None

    def test_partial_json_returns_none(self):
        assert _parse_json('{"direction": "UP"') is None

    def test_preserves_all_fields(self):
        text = '{"direction": "DOWN", "confidence": 0.62, "reasoning": "RSI overbought", "key_factors": ["rsi", "momentum"]}'
        r = _parse_json(text)
        assert r["key_factors"] == ["rsi", "momentum"]
        assert r["reasoning"] == "RSI overbought"


# ── Vote aggregation ──────────────────────────────────────────────────────────

class TestAggregate:
    def test_unanimous_up(self):
        agent = _EnsembleNoAPI()
        votes = [_vote("UP", 0.55), _vote("UP", 0.58), _vote("UP", 0.60)]
        direction, _, _ = agent._aggregate(votes)
        assert direction == "UP"

    def test_unanimous_down(self):
        agent = _EnsembleNoAPI()
        votes = [_vote("DOWN", 0.55), _vote("DOWN", 0.58), _vote("DOWN", 0.60)]
        direction, _, _ = agent._aggregate(votes)
        assert direction == "DOWN"

    def test_majority_wins(self):
        agent = _EnsembleNoAPI()
        votes = [_vote("DOWN", 0.60), _vote("DOWN", 0.58), _vote("UP", 0.55)]
        direction, _, _ = agent._aggregate(votes)
        assert direction == "DOWN"

    def test_unanimous_boost(self):
        # Unanimous 3-0 → avg_conf + 0.02 (capped at 0.65)
        agent = _EnsembleNoAPI()
        votes = [_vote("UP", 0.55), _vote("UP", 0.55), _vote("UP", 0.55)]
        _, confidence, _ = agent._aggregate(votes)
        assert confidence == pytest.approx(min(0.55 + 0.02, 0.65), abs=1e-4)

    def test_majority_discount(self):
        # 2-1 vote → avg of agreeing - 0.02
        agent = _EnsembleNoAPI()
        votes = [_vote("UP", 0.60), _vote("UP", 0.60), _vote("DOWN", 0.55)]
        _, confidence, _ = agent._aggregate(votes)
        assert confidence == pytest.approx(max(0.60 - 0.02, 0.50), abs=1e-4)

    def test_confidence_capped_at_65(self):
        agent = _EnsembleNoAPI()
        votes = [_vote("UP", 0.64), _vote("UP", 0.64), _vote("UP", 0.64)]
        _, confidence, _ = agent._aggregate(votes)
        assert confidence <= 0.65

    def test_confidence_floor_at_50(self):
        agent = _EnsembleNoAPI()
        votes = [_vote("UP", 0.50), _vote("UP", 0.50), _vote("DOWN", 0.50)]
        _, confidence, _ = agent._aggregate(votes)
        assert confidence >= 0.50

    def test_tie_broken_by_avg_confidence(self):
        # 1 UP @ 0.63 vs 1 DOWN @ 0.55 → UP wins (higher avg conf)
        agent = _EnsembleNoAPI()
        votes = [_vote("UP", 0.63), _vote("DOWN", 0.55)]
        direction, _, _ = agent._aggregate(votes)
        assert direction == "UP"

    def test_single_vote_resolves(self):
        agent = _EnsembleNoAPI()
        votes = [_vote("DOWN", 0.58)]
        direction, _, _ = agent._aggregate(votes)
        assert direction == "DOWN"

    def test_reasoning_contains_direction(self):
        agent = _EnsembleNoAPI()
        votes = [_vote("UP", 0.55), _vote("UP", 0.58), _vote("DOWN", 0.60)]
        direction, _, reasoning = agent._aggregate(votes)
        assert "UP" in reasoning

    def test_confidence_is_float(self):
        agent = _EnsembleNoAPI()
        votes = [_vote("UP", 0.55), _vote("UP", 0.58), _vote("UP", 0.60)]
        _, confidence, _ = agent._aggregate(votes)
        assert isinstance(confidence, float)
