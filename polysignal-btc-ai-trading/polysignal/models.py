from __future__ import annotations

"""
models.py — PolySignal BTC 5-Min data contracts.

Pipeline:
  Collector  ->  MarketWindow  (live BTC price + market state)
  Analyzer   ->  Prediction    (UP/DOWN call with confidence)
  Evaluator  ->  EvalRecord    (scored after 5-min window closes)
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# ── Collector output ──────────────────────────────────────────────────────────

class PriceCandle(BaseModel):
    """One OHLCV candle from Binance (1-min resolution)."""
    timestamp:  datetime
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float   # BTC volume


class MarketWindow(BaseModel):
    """
    Everything the Analyzer needs for one 5-min prediction window.
    Collected right before a new Polymarket 5-min market opens.
    """
    collected_at:   datetime = Field(default_factory=datetime.utcnow)

    # Polymarket market info
    market_id:      str
    market_question: str
    window_start:   datetime   # when the 5-min window begins
    window_end:     datetime   # when it resolves
    up_price:       float      # Polymarket crowd odds for UP  (0–1)
    down_price:     float      # Polymarket crowd odds for DOWN

    # Live BTC price (from Polymarket RTDS / Chainlink)
    btc_price_now:  float      # current price = opening reference
    btc_price_1m_ago: Optional[float] = None
    btc_price_5m_ago: Optional[float] = None

    # Binance derived features
    candles_1m:     list[PriceCandle] = []     # last 10 x 1-min candles
    volume_5m:      Optional[float] = None     # total BTC volume last 5 min
    price_change_1m: Optional[float] = None   # % change last 1 min
    price_change_5m: Optional[float] = None   # % change last 5 min
    price_change_15m: Optional[float] = None  # % change last 15 min
    volatility_5m:  Optional[float] = None    # std dev of last 5 closes
    rsi_14:         Optional[float] = None    # 14-period RSI

    # Macro sentiment
    fear_greed_score:  Optional[int] = None    # 0–100
    fear_greed_label:  Optional[str] = None    # "Fear", "Greed" etc.


# ── Analyzer output ───────────────────────────────────────────────────────────

class Prediction(BaseModel):
    """
    Analyzer's UP/DOWN call for one 5-min window.
    Saved immediately; scored by Evaluator after resolution.
    """
    prediction_id:  str
    market_id:      str
    market_question: str
    window_start:   datetime
    window_end:     datetime

    # What we called
    direction:      Literal["UP", "DOWN"]
    confidence:     float           # 0.0 – 1.0
    btc_price_at_call: float        # opening reference price
    market_up_odds: float           # crowd's UP probability at call time
    reasoning:      str
    key_factors:    list[str] = []

    # Resolution (filled by Evaluator)
    resolved:       bool = False
    actual_outcome: Optional[Literal["UP", "DOWN"]] = None
    btc_price_close: Optional[float] = None
    resolved_at:    Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Ensemble output ───────────────────────────────────────────────────────────

class ModelVote(BaseModel):
    """A single LLM model's vote within an ensemble call."""
    model_id:   str
    model_label: str          # human-readable name e.g. "Haiku", "Sonnet"
    direction:  Literal["UP", "DOWN"]
    confidence: float
    reasoning:  str


# ── Evaluator output ──────────────────────────────────────────────────────────

class EvalRecord(BaseModel):
    """Scored prediction after the 5-min window closes."""
    eval_id:        str
    prediction_id:  str
    market_id:      str

    direction:      Literal["UP", "DOWN"]
    confidence:     float
    actual_outcome: Literal["UP", "DOWN"]
    btc_open:       float
    btc_close:      float
    price_change_pct: float    # actual % move

    was_correct:    bool
    brier_score:    float      # (confidence - actual_binary)^2

    scored_at: datetime = Field(default_factory=datetime.utcnow)


# ── Simulator output ───────────────────────────────────────────────────────────

class Trade(BaseModel):
    """One simulated trade placed on a Polymarket 5-min market."""
    trade_id:       str
    prediction_id:  str
    market_question: str
    direction:      Literal["UP", "DOWN"]   # which token we bought
    odds_price:     float   # Polymarket price per token at entry (e.g. 0.52)
    bet_amount:     float   # dollars wagered
    tokens:         float   # bet_amount / odds_price
    potential_payout: float # tokens * $1.00 (if correct)

    # Filled at settlement
    resolved:       bool = False
    actual_outcome: Optional[Literal["UP", "DOWN"]] = None
    actual_payout:  float = 0.0
    pnl:            float = 0.0     # actual_payout - bet_amount
    balance_after:  float = 0.0
    resolved_at:    Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
