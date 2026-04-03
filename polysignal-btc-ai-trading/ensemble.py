"""
ensemble.py — Multi-LLM Ensemble Analyzer.

Calls three models in parallel, each with a different analytical lens:
  1. Haiku       — pure momentum trader  (fast, cheap)
  2. Sonnet 4.5  — balanced technical analyst
  3. Sonnet 4.6  — contrarian / market microstructure

Aggregates votes into a single Prediction via majority vote + confidence weighting.
"""
from __future__ import annotations

import asyncio
import json
import uuid
import re
import anthropic
from typing import Optional
from loguru import logger
from models import MarketWindow, Prediction, ModelVote

# ── Model roster ──────────────────────────────────────────────────────────────

MODELS = [
    {
        "id":    "claude-haiku-4-5-20251001",
        "label": "Haiku (Momentum)",
        "system": """\
You are a momentum trader focused purely on Bitcoin price action.
Your only job: read short-term price momentum and call UP or DOWN for the next 5 minutes.

Rules:
- If 1m or 5m change is clearly positive (>+0.03%): lean UP
- If 1m or 5m change is clearly negative (<-0.03%): lean DOWN
- If both are flat (within ±0.03%): NO EDGE — confidence max 0.52
- RSI >70 with negative momentum = DOWN. RSI <30 with positive momentum = UP.
- Ignore crowd odds and sentiment — you are a pure price action trader.
- Confidence 0.50–0.58 for weak signals, up to 0.63 for clear momentum.

Output ONLY valid JSON — no markdown, no extra text.""",
    },
    {
        "id":    "claude-sonnet-4-5",
        "label": "Sonnet (Technical)",
        "system": """\
You are a quantitative analyst specializing in ultra-short-term Bitcoin price prediction.

Signal hierarchy:
1. Momentum (1m/5m change) — primary
2. RSI extremes (>70 overbought, <30 oversold) — secondary
3. Crowd odds (only meaningful if >55/45 skew) — tertiary

Confidence rules:
- 0 strong signals → NO EDGE, confidence 0.50–0.52
- 1 strong signal  → 0.53–0.58
- 2 strong signals → 0.59–0.63
- 3 strong signals → 0.64–0.65 max
- Never exceed 0.65

~50% of 5-min windows close UP and ~50% DOWN. Persistent DOWN bias is a red flag.
Output ONLY valid JSON — no markdown, no extra text.""",
    },
    {
        "id":    "claude-sonnet-4-6",
        "label": "Sonnet 4.6 (Contrarian)",
        "system": """\
You are a contrarian market analyst focused on crowd mispricing in Bitcoin 5-minute markets.

Your edge: the crowd (Polymarket odds) often overreacts to recent price moves.
Core strategy:
- If crowd heavily favors DOWN (>56%) BUT RSI is not extreme and momentum is flat → consider UP
- If crowd heavily favors UP (>56%) BUT RSI is not extreme and momentum is flat → consider DOWN
- If RSI is deeply oversold (<30): strong UP signal regardless of crowd
- If RSI is deeply overbought (>70): strong DOWN signal regardless of crowd
- If crowd is 50/50 and all signals are neutral: NO EDGE, confidence 0.50–0.52

You are skeptical of momentum continuation — you look for mean reversion setups.
Confidence 0.50–0.63 range. Never exceed 0.65.
Output ONLY valid JSON — no markdown, no extra text.""",
    },
]

PROMPT_TEMPLATE = """\
Predict Bitcoin's 5-minute direction for the Polymarket market below.

=== MARKET ===================================================
Question:    {question}
Window:      {window_start} -> {window_end} UTC
Crowd odds:  UP={up_price:.1%}  DOWN={down_price:.1%}

=== PRICE DATA ===============================================
Current BTC:     ${btc_now:,.2f}
1-min change:    {c1m}
5-min change:    {c5m}
15-min change:   {c15m}
{candle_summary}

=== TECHNICAL INDICATORS =====================================
RSI (14):        {rsi}
Volatility 5m:   {vol}
Volume 5m:       {volm}

=== SENTIMENT ================================================
Fear & Greed:    {fg}

Return ONLY this JSON:
{{
  "direction": "UP",
  "confidence": 0.55,
  "reasoning": "Step-by-step analysis...",
  "key_factors": ["factor 1", "factor 2"]
}}"""


def _build_prompt(w: MarketWindow) -> str:
    last5 = w.candles_1m[-5:] if w.candles_1m else []
    candle_summary = ("Last 5 closes: " + ", ".join(f"${c.close:,.2f}" for c in last5)) if last5 else ""
    return PROMPT_TEMPLATE.format(
        question      = w.market_question,
        window_start  = w.window_start.strftime("%H:%M"),
        window_end    = w.window_end.strftime("%H:%M"),
        up_price      = w.up_price,
        down_price    = w.down_price,
        btc_now       = w.btc_price_now,
        c1m           = f"{w.price_change_1m:+.3f}%"  if w.price_change_1m  is not None else "N/A",
        c5m           = f"{w.price_change_5m:+.3f}%"  if w.price_change_5m  is not None else "N/A",
        c15m          = f"{w.price_change_15m:+.3f}%" if w.price_change_15m is not None else "N/A",
        candle_summary = candle_summary,
        rsi           = f"{w.rsi_14:.1f}"                    if w.rsi_14          is not None else "N/A",
        vol           = f"${w.volatility_5m:,.2f} std dev"   if w.volatility_5m   is not None else "N/A",
        volm          = f"{w.volume_5m:.2f} BTC"             if w.volume_5m       is not None else "N/A",
        fg            = f"{w.fear_greed_score} ({w.fear_greed_label})" if w.fear_greed_score else "N/A",
    )


def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r'^```[a-z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Non-greedy match to get the first complete JSON object
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# ── Ensemble Agent ────────────────────────────────────────────────────────────

class EnsembleAgent:
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def analyze(self, window: MarketWindow) -> Optional[Prediction]:
        prompt = _build_prompt(window)
        tasks  = [self._call_model(m, prompt) for m in MODELS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        votes: list[ModelVote] = [r for r in results if isinstance(r, ModelVote)]

        if not votes:
            logger.error("[Ensemble] All model calls failed")
            return None

        logger.info(
            f"[Ensemble] Votes: " +
            "  |  ".join(f"{v.model_label}: {v.direction} {v.confidence:.0%}" for v in votes)
        )

        direction, confidence, reasoning = self._aggregate(votes)

        # Compile model votes as JSON for storage
        votes_json = json.dumps([
            {"model": v.model_label, "direction": v.direction,
             "confidence": round(v.confidence, 3)}
            for v in votes
        ])
        full_reasoning = (
            f"[ENSEMBLE — {len(votes)}/3 models]\n"
            f"Votes: {votes_json}\n\n"
            + reasoning
        )

        key_factors = [
            f"{v.model_label}: {v.direction} @ {v.confidence:.0%}"
            for v in votes
        ]

        return Prediction(
            prediction_id     = str(uuid.uuid4()),
            market_id         = window.market_id,
            market_question   = window.market_question,
            window_start      = window.window_start,
            window_end        = window.window_end,
            direction         = direction,
            confidence        = round(confidence, 3),
            btc_price_at_call = window.btc_price_now,
            market_up_odds    = window.up_price,
            reasoning         = full_reasoning,
            key_factors       = key_factors,
        )

    async def _call_model(self, model_cfg: dict, prompt: str) -> Optional[ModelVote]:
        try:
            resp = await self.client.messages.create(
                model      = model_cfg["id"],
                max_tokens = 400,
                system     = model_cfg["system"],
                messages   = [{"role": "user", "content": prompt}],
            )
            data = _parse_json(resp.content[0].text.strip())
            if not data:
                logger.warning(f"[Ensemble] {model_cfg['label']} returned unparseable response")
                return None

            direction  = str(data.get("direction", "")).upper()
            confidence = float(data.get("confidence", 0.5))
            if direction not in ("UP", "DOWN"):
                return None

            return ModelVote(
                model_id    = model_cfg["id"],
                model_label = model_cfg["label"],
                direction   = direction,
                confidence  = min(max(confidence, 0.5), 0.65),
                reasoning   = data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"[Ensemble] {model_cfg['label']} error: {e}")
            return None

    def _aggregate(self, votes: list[ModelVote]) -> tuple[str, float, str]:
        """Majority vote direction, confidence weighted by agreement."""
        up_votes   = [v for v in votes if v.direction == "UP"]
        down_votes = [v for v in votes if v.direction == "DOWN"]

        if len(up_votes) > len(down_votes):
            majority   = "UP"
            agreeing   = up_votes
            dissenting = down_votes
        elif len(down_votes) > len(up_votes):
            majority   = "DOWN"
            agreeing   = down_votes
            dissenting = up_votes
        else:
            # True tie (e.g. 1-1 when a model fails) — pick by avg confidence
            up_conf   = sum(v.confidence for v in up_votes)   / len(up_votes)
            down_conf = sum(v.confidence for v in down_votes) / len(down_votes)
            if up_conf >= down_conf:
                majority, agreeing, dissenting = "UP",   up_votes,   down_votes
            else:
                majority, agreeing, dissenting = "DOWN", down_votes, up_votes
            logger.warning(
                f"[Ensemble] Tied vote — breaking by avg confidence "
                f"(UP={up_conf:.2f} DOWN={down_conf:.2f}) → {majority}"
            )

        # Confidence: average of agreeing models
        # Unanimous: slight boost (+0.02). Split: slight discount (-0.02)
        avg_conf = sum(v.confidence for v in agreeing) / len(agreeing)
        if len(dissenting) == 0:
            confidence = min(avg_conf + 0.02, 0.65)   # unanimous boost
        elif len(agreeing) > len(dissenting):
            confidence = max(avg_conf - 0.02, 0.50)   # majority discount
        else:
            confidence = 0.51                          # tied — near coin flip

        # Reasoning = winning side summaries
        reasoning = f"Majority: {majority} ({len(agreeing)}/{len(votes)} models agree)\n\n"
        for v in agreeing:
            reasoning += f"[{v.model_label}] {v.reasoning[:200]}\n\n"
        if dissenting:
            reasoning += f"Dissent ({dissenting[0].model_label}): {dissenting[0].reasoning[:100]}"

        return majority, confidence, reasoning
