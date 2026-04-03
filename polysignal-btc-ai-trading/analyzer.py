"""
analyzer.py — BTC 5-Min Analyzer Agent.
"""
from __future__ import annotations

import json
import uuid
import re
import anthropic
from typing import Optional
from loguru import logger
from models import MarketWindow, Prediction

SYSTEM_PROMPT = """\
You are a quantitative analyst specializing in ultra-short-term Bitcoin price prediction.
Your task: given live technical indicators and market microstructure data, predict whether
Bitcoin will be HIGHER (UP) or LOWER (DOWN) than its current price at the end of a
5-minute window on Polymarket.

## Signal hierarchy
1. Momentum (1m and 5m price change) — primary driver at this timeframe
2. RSI extremes (>70 overbought → bearish lean, <30 oversold → bullish lean) — secondary
3. Crowd odds — tertiary; only meaningful if skewed >55/45 in one direction
4. Fear & Greed — background context only, ignore for direction

## Confidence rules (STRICT)
- Confidence 0.60+ requires AT LEAST TWO strong signals clearly agreeing:
  e.g. RSI >70 AND negative momentum AND crowd >55% DOWN
- Confidence 0.50–0.58 is correct when signals are mixed, weak, or contradictory
- NEVER exceed 0.65 unless three signals strongly agree
- If RSI is 40–60 AND 1m change is within ±0.05% AND crowd odds are 48–52%,
  this is a NO-EDGE situation — cap confidence at 0.52 and note it explicitly

## Directional bias guard
Historically ~50% of 5-minute BTC windows close UP and ~50% close DOWN.
If you find yourself consistently calling DOWN without strong signals, that is
a bias error — not an insight. When signals are weak or neutral, default to
the direction that is currently underrepresented in your reasoning.

## Calibration target
Your stated confidence should match your actual accuracy:
- 55% confidence calls should be right ~55% of the time
- 65% confidence calls should be right ~65% of the time
Overconfident wrong calls are penalised heavily by the Brier score.

Output ONLY valid JSON — no markdown, no extra text.\
"""


def _build_prompt(w: MarketWindow) -> str:
    candle_summary = ""
    if w.candles_1m:
        last5 = w.candles_1m[-5:]
        candle_summary = "Last 5 closes: " + ", ".join(
            f"${c.close:,.2f}" for c in last5
        )

    rsi_str  = f"{w.rsi_14:.1f}"  if w.rsi_14  is not None else "N/A"
    vol_str  = f"${w.volatility_5m:,.2f} std dev" if w.volatility_5m is not None else "N/A"
    volm_str = f"{w.volume_5m:.2f} BTC" if w.volume_5m is not None else "N/A"
    fg_str   = f"{w.fear_greed_score} ({w.fear_greed_label})" if w.fear_greed_score else "N/A"
    c1m      = f"{w.price_change_1m:+.3f}%"  if w.price_change_1m  is not None else "N/A"
    c5m      = f"{w.price_change_5m:+.3f}%"  if w.price_change_5m  is not None else "N/A"
    c15m     = f"{w.price_change_15m:+.3f}%" if w.price_change_15m is not None else "N/A"

    return f"""
Predict Bitcoin's 5-minute direction for the Polymarket market below.

=== MARKET ===================================================
Question:    {w.market_question}
Window:      {w.window_start.strftime('%H:%M')} -> {w.window_end.strftime('%H:%M')} UTC
Crowd odds:  UP={w.up_price:.1%}  DOWN={w.down_price:.1%}

=== PRICE DATA ===============================================
Current BTC:     ${w.btc_price_now:,.2f}
1-min change:    {c1m}
5-min change:    {c5m}
15-min change:   {c15m}
{candle_summary}

=== TECHNICAL INDICATORS =====================================
RSI (14):        {rsi_str}
Volatility 5m:   {vol_str}
Volume 5m:       {volm_str}

=== SENTIMENT ================================================
Fear & Greed:    {fg_str}

=== INSTRUCTIONS =============================================
1. Momentum check: is 1m or 5m change clearly positive, negative, or flat?
2. RSI check: is it in extreme territory (>70 or <30) or neutral (40-60)?
3. Crowd check: are odds skewed >55/45 or effectively 50/50?
4. Count strong signals: how many clearly point the same direction?
   - 0 strong signals → NO EDGE, confidence 0.50–0.52, note "no edge"
   - 1 strong signal  → weak edge, confidence 0.53–0.58
   - 2 strong signals → moderate edge, confidence 0.59–0.63
   - 3 strong signals → strong edge, confidence 0.64–0.65 (max)
5. Directional balance: if you called DOWN last time on similar flat data,
   lean UP this time — prevent systematic DOWN bias on no-edge windows.

Return ONLY this JSON:
{{
  "direction": "UP",
  "confidence": 0.63,
  "reasoning": "Concise step-by-step analysis...",
  "key_factors": ["factor 1", "factor 2", "factor 3"]
}}
"""


class AnalyzerAgent:
    def __init__(self, api_key: str):
        self.claude  = anthropic.AsyncAnthropic(api_key=api_key)
        self._total  = 0

    async def analyze(self, window: MarketWindow) -> Optional[Prediction]:
        logger.info(
            f"[Analyzer] Calling Claude — "
            f"BTC=${window.btc_price_now:,.2f}  "
            f"UP={window.up_price:.0%}"
        )
        try:
            resp = await self.claude.messages.create(
                model      = "claude-sonnet-4-5",
                max_tokens = 600,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": _build_prompt(window)}],
            )
            raw  = resp.content[0].text.strip()
            data = _parse_json(raw)

            if not data:
                logger.error("[Analyzer] Could not parse Claude response")
                return None

            direction  = str(data.get("direction", "UP")).upper()
            confidence = float(data.get("confidence", 0.5))

            if direction not in ("UP", "DOWN"):
                logger.error(f"[Analyzer] Invalid direction: {direction}")
                return None

            pred = Prediction(
                prediction_id      = str(uuid.uuid4()),
                market_id          = window.market_id,
                market_question    = window.market_question,
                window_start       = window.window_start,
                window_end         = window.window_end,
                direction          = direction,
                confidence         = round(confidence, 3),
                btc_price_at_call  = window.btc_price_now,
                market_up_odds     = window.up_price,
                reasoning          = data.get("reasoning", ""),
                key_factors        = data.get("key_factors", []),
            )

            self._total += 1
            logger.success(
                f"[Analyzer] -> {direction} @ {confidence:.0%} confidence  "
                f"(crowd UP={window.up_price:.0%})"
            )
            return pred

        except anthropic.APIError as e:
            logger.error(f"[Analyzer] Claude API error: {e}")
            return None
        except Exception as e:
            logger.error(f"[Analyzer] Unexpected error: {e}")
            return None

    @property
    def stats(self) -> dict:
        return {"total_predictions": self._total}


def _parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None
