"""
debate.py — Adversarial Debate Prediction Agent.

Two LLMs (Groq and OpenAI) independently analyze the market, then read each
other's reasoning and write a rebuttal. A Claude judge reads all four outputs
and delivers a final UP/DOWN prediction with calibrated confidence.

Activation: both GROQ_API_KEY and OPENAI_API_KEY must be set in .env.
Fallback:   if either key is missing, is_available() returns False and the
            orchestrator falls back to EnsembleAgent silently.

Debate protocol (3 rounds):
  Round 1 — Independent analysis (Groq + OpenAI in parallel)
  Round 2 — Cross-examination  (each sees the other's Round 1, parallel)
  Round 3 — Judgment           (Claude reads all 4 outputs, decides)

Estimated latency: 10–15 seconds total — well within the 5-min window.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Optional

from loguru import logger

from models import MarketWindow, Prediction

# ── Optional SDK imports (graceful if packages not installed) ─────────────────

try:
    from groq import AsyncGroq          # pip install groq
    _GROQ_SDK = True
except ImportError:
    _GROQ_SDK = False

try:
    from openai import AsyncOpenAI      # pip install openai
    _OPENAI_SDK = True
except ImportError:
    _OPENAI_SDK = False

try:
    import anthropic                    # always available (core dep)
    _ANTHROPIC_SDK = True
except ImportError:
    _ANTHROPIC_SDK = False


# ── Model config ──────────────────────────────────────────────────────────────

GROQ_MODEL   = "llama-3.3-70b-versatile"   # fast, capable, free tier available
OPENAI_MODEL = "gpt-4o-mini"               # cost-efficient, strong reasoning
JUDGE_MODEL  = "claude-sonnet-4-6"         # judge — Claude stays in control


# ── System prompts ────────────────────────────────────────────────────────────

_GROQ_SYSTEM = """\
You are a quantitative momentum trader specializing in ultra-short-term Bitcoin forecasting.
Your edge is reading price velocity and volume signals. You are fast, direct, and data-driven.

For a 5-minute BTC window, weight signals in this order:
1. Short-term momentum (1m/5m price change) — primary signal
2. RSI extremes (>70 overbought, <30 oversold) — secondary signal
3. Volume confirmation — tertiary signal
4. Crowd odds and sentiment — background only

Confidence rules:
- No clear signal: 0.50–0.52
- One moderate signal: 0.53–0.58
- Two aligned signals: 0.59–0.63
- Strong confluence: 0.64–0.65 max

Return ONLY valid JSON — no markdown, no extra text:
{"direction": "UP", "confidence": 0.55, "reasoning": "...", "key_factors": ["...", "..."]}"""

_OPENAI_SYSTEM = """\
You are a contrarian options analyst specializing in Bitcoin microstructure and crowd behavior.
Your edge is spotting when the crowd (Polymarket odds) is overreacting to recent price moves.

For a 5-minute BTC window, weight signals in this order:
1. Crowd odds skew (>55/45) vs actual momentum alignment — primary signal
2. RSI mean-reversion setups — secondary signal
3. Volatility regime (high vol = uncertainty, widen confidence bands) — tertiary
4. Momentum — use only as confirmation, not primary driver

Confidence rules:
- No contrarian setup: 0.50–0.52
- Mild divergence: 0.53–0.58
- Clear crowd-vs-momentum divergence: 0.59–0.63
- Strong mean-reversion signal: 0.64–0.65 max

Return ONLY valid JSON — no markdown, no extra text:
{"direction": "UP", "confidence": 0.55, "reasoning": "...", "key_factors": ["...", "..."]}"""

_REBUTTAL_TEMPLATE = """\
Your opponent has made the following argument:

--- OPPONENT ARGUMENT ---
Direction: {opp_direction}
Confidence: {opp_confidence:.0%}
Reasoning: {opp_reasoning}
Key factors: {opp_factors}
------------------------

Your original position was: {my_direction} at {my_confidence:.0%}

Now write a rebuttal. Either:
  (a) Defend your original position and explain why your opponent is wrong, OR
  (b) Concede partially or fully if their argument reveals something you missed.

Be specific. Reference the actual market data from the original prompt.
Keep your rebuttal under 150 words.

After your rebuttal, output your FINAL position as JSON on the last line:
{"direction": "UP", "confidence": 0.55, "reasoning": "brief updated reasoning"}"""

_JUDGE_SYSTEM = """\
You are a senior quantitative researcher adjudicating a forecasting debate.
Two analysts have argued about the direction of Bitcoin's price over the next 5 minutes.
Your job is to read both positions and rebuttals, weigh the evidence, and deliver a final verdict.

Rules:
- Do not simply pick the majority — evaluate argument quality, not vote count
- Penalize overconfidence: if an analyst claims >60% confidence without multiple aligned signals, discount them
- If both analysts agree after rebuttals, the case is strong — boost confidence slightly
- If they still disagree, the edge is weak — cap confidence at 0.55
- Final confidence range: 0.50–0.65 max

Return ONLY valid JSON:
{"direction": "UP", "confidence": 0.55, "reasoning": "...", "winning_argument": "groq|openai|synthesis"}"""

_JUDGE_PROMPT_TEMPLATE = """\
=== MARKET CONTEXT ===
{market_prompt}

=== GROQ ANALYST (Momentum Trader) ===
Round 1 — Initial position:
  Direction: {groq_r1_dir} | Confidence: {groq_r1_conf:.0%}
  Reasoning: {groq_r1_reasoning}

Round 2 — Rebuttal:
  {groq_r2}
  Final position: {groq_r2_dir} | Confidence: {groq_r2_conf:.0%}

=== OPENAI ANALYST (Contrarian) ===
Round 1 — Initial position:
  Direction: {oai_r1_dir} | Confidence: {oai_r1_conf:.0%}
  Reasoning: {oai_r1_reasoning}

Round 2 — Rebuttal:
  {oai_r2}
  Final position: {oai_r2_dir} | Confidence: {oai_r2_conf:.0%}

=== YOUR VERDICT ===
Deliver the final prediction as JSON."""


# ── Market prompt builder (shared with ensemble) ──────────────────────────────

_MARKET_PROMPT_TEMPLATE = """\
Question:     {question}
Window:       {window_start} -> {window_end} UTC
Crowd odds:   UP={up_price:.1%}  DOWN={down_price:.1%}

Price data:
  Current BTC:    ${btc_now:,.2f}
  1-min change:   {c1m}
  5-min change:   {c5m}
  15-min change:  {c15m}
  {candle_summary}

Technical indicators:
  RSI (14):       {rsi}
  Volatility 5m:  {vol}
  Volume 5m:      {volm}

Sentiment:
  Fear & Greed:   {fg}"""


def _build_market_prompt(w: MarketWindow) -> str:
    last5 = w.candles_1m[-5:] if w.candles_1m else []
    candle_summary = ("Last 5 closes: " + ", ".join(f"${c.close:,.2f}" for c in last5)) if last5 else ""
    return _MARKET_PROMPT_TEMPLATE.format(
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


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```[a-z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()
    # Try last line first (rebuttal format ends with JSON)
    last_line = text.strip().splitlines()[-1].strip()
    try:
        return json.loads(last_line)
    except Exception:
        pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# ── Availability check ────────────────────────────────────────────────────────

def is_available() -> bool:
    """Returns True only when both Groq and OpenAI keys are configured."""
    groq_key  = os.getenv("GROQ_API_KEY", "").strip()
    oai_key   = os.getenv("OPENAI_API_KEY", "").strip()
    anth_key  = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if not groq_key:
        logger.debug("[Debate] GROQ_API_KEY not set — DebateAgent unavailable")
        return False
    if not oai_key:
        logger.debug("[Debate] OPENAI_API_KEY not set — DebateAgent unavailable")
        return False
    if not anth_key:
        logger.debug("[Debate] ANTHROPIC_API_KEY not set — DebateAgent unavailable")
        return False
    if not _GROQ_SDK:
        logger.debug("[Debate] groq package not installed — run: pip install groq")
        return False
    if not _OPENAI_SDK:
        logger.debug("[Debate] openai package not installed — run: pip install openai")
        return False
    return True


# ── Debate Agent ──────────────────────────────────────────────────────────────

class DebateAgent:
    """
    Adversarial debate between Groq (momentum) and OpenAI (contrarian),
    judged by Claude Sonnet.

    Usage:
        agent = DebateAgent()
        prediction = await agent.debate(window)
    """

    def __init__(self):
        if not is_available():
            raise RuntimeError(
                "DebateAgent requires GROQ_API_KEY, OPENAI_API_KEY, and ANTHROPIC_API_KEY. "
                "Check is_available() before instantiating."
            )
        self._groq   = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self._openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._claude = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def debate(self, window: MarketWindow) -> Optional[Prediction]:
        market_prompt = _build_market_prompt(window)

        # ── Round 1: Independent analysis (parallel) ──────────────────────────
        logger.info("[Debate] Round 1 — independent analysis...")
        groq_r1, oai_r1 = await asyncio.gather(
            self._groq_call(_GROQ_SYSTEM, market_prompt),
            self._openai_call(_OPENAI_SYSTEM, market_prompt),
            return_exceptions=True,
        )

        if isinstance(groq_r1, Exception) or groq_r1 is None:
            logger.error(f"[Debate] Groq Round 1 failed: {groq_r1}")
            return None
        if isinstance(oai_r1, Exception) or oai_r1 is None:
            logger.error(f"[Debate] OpenAI Round 1 failed: {oai_r1}")
            return None

        logger.info(
            f"[Debate] R1 — Groq: {groq_r1['direction']} {groq_r1['confidence']:.0%} | "
            f"OpenAI: {oai_r1['direction']} {oai_r1['confidence']:.0%}"
        )

        # ── Round 2: Cross-examination (parallel) ─────────────────────────────
        logger.info("[Debate] Round 2 — cross-examination...")
        groq_rebuttal_prompt = (
            market_prompt + "\n\n" +
            _REBUTTAL_TEMPLATE.format(
                opp_direction  = oai_r1["direction"],
                opp_confidence = oai_r1["confidence"],
                opp_reasoning  = oai_r1.get("reasoning", ""),
                opp_factors    = ", ".join(oai_r1.get("key_factors", [])),
                my_direction   = groq_r1["direction"],
                my_confidence  = groq_r1["confidence"],
            )
        )
        oai_rebuttal_prompt = (
            market_prompt + "\n\n" +
            _REBUTTAL_TEMPLATE.format(
                opp_direction  = groq_r1["direction"],
                opp_confidence = groq_r1["confidence"],
                opp_reasoning  = groq_r1.get("reasoning", ""),
                opp_factors    = ", ".join(groq_r1.get("key_factors", [])),
                my_direction   = oai_r1["direction"],
                my_confidence  = oai_r1["confidence"],
            )
        )

        groq_r2, oai_r2 = await asyncio.gather(
            self._groq_call(_GROQ_SYSTEM, groq_rebuttal_prompt),
            self._openai_call(_OPENAI_SYSTEM, oai_rebuttal_prompt),
            return_exceptions=True,
        )

        # Fallback to Round 1 positions if rebuttals fail
        if isinstance(groq_r2, Exception) or groq_r2 is None:
            logger.warning("[Debate] Groq rebuttal failed — using Round 1 position")
            groq_r2 = groq_r1
        if isinstance(oai_r2, Exception) or oai_r2 is None:
            logger.warning("[Debate] OpenAI rebuttal failed — using Round 1 position")
            oai_r2 = oai_r1

        logger.info(
            f"[Debate] R2 — Groq: {groq_r2['direction']} {groq_r2['confidence']:.0%} | "
            f"OpenAI: {oai_r2['direction']} {oai_r2['confidence']:.0%}"
        )

        # ── Round 3: Judge (Claude) ───────────────────────────────────────────
        logger.info("[Debate] Round 3 — Claude judge deliberating...")
        judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(
            market_prompt   = market_prompt,
            groq_r1_dir     = groq_r1["direction"],
            groq_r1_conf    = groq_r1["confidence"],
            groq_r1_reasoning = groq_r1.get("reasoning", ""),
            groq_r2         = groq_r2.get("reasoning", ""),
            groq_r2_dir     = groq_r2["direction"],
            groq_r2_conf    = groq_r2["confidence"],
            oai_r1_dir      = oai_r1["direction"],
            oai_r1_conf     = oai_r1["confidence"],
            oai_r1_reasoning = oai_r1.get("reasoning", ""),
            oai_r2          = oai_r2.get("reasoning", ""),
            oai_r2_dir      = oai_r2["direction"],
            oai_r2_conf     = oai_r2["confidence"],
        )

        verdict = await self._claude_call(judge_prompt)
        if verdict is None:
            logger.error("[Debate] Claude judge failed")
            return None

        direction  = str(verdict.get("direction", "")).upper()
        confidence = float(verdict.get("confidence", 0.5))
        if direction not in ("UP", "DOWN"):
            logger.error(f"[Debate] Judge returned invalid direction: {direction}")
            return None

        confidence = min(max(confidence, 0.5), 0.65)

        winning_arg = verdict.get("winning_argument", "synthesis")
        reasoning = (
            f"[DEBATE — Judge: {winning_arg}]\n"
            f"Groq ({GROQ_MODEL}): {groq_r1['direction']} R1 → {groq_r2['direction']} R2\n"
            f"OpenAI ({OPENAI_MODEL}): {oai_r1['direction']} R1 → {oai_r2['direction']} R2\n\n"
            f"Judge reasoning: {verdict.get('reasoning', '')}"
        )
        key_factors = [
            f"Groq final: {groq_r2['direction']} @ {groq_r2['confidence']:.0%}",
            f"OpenAI final: {oai_r2['direction']} @ {oai_r2['confidence']:.0%}",
            f"Winning argument: {winning_arg}",
        ]

        logger.info(
            f"[Debate] Verdict → {direction} {confidence:.0%} "
            f"(winning: {winning_arg})"
        )

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
            reasoning         = reasoning,
            key_factors       = key_factors,
        )

    # ── Private API call helpers ──────────────────────────────────────────────

    async def _groq_call(self, system: str, user: str) -> Optional[dict]:
        """Call Groq API and parse JSON response."""
        try:
            resp = await self._groq.chat.completions.create(
                model    = GROQ_MODEL,
                messages = [
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": user},
                ],
                max_tokens  = 500,
                temperature = 0.3,
            )
            text = resp.choices[0].message.content or ""
            data = _parse_json(text)
            if not data:
                logger.warning(f"[Debate/Groq] Unparseable response: {text[:200]}")
            return data
        except Exception as e:
            logger.error(f"[Debate/Groq] API error: {e}")
            return None

    async def _openai_call(self, system: str, user: str) -> Optional[dict]:
        """Call OpenAI API and parse JSON response."""
        try:
            resp = await self._openai.chat.completions.create(
                model    = OPENAI_MODEL,
                messages = [
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": user},
                ],
                max_tokens  = 500,
                temperature = 0.3,
            )
            text = resp.choices[0].message.content or ""
            data = _parse_json(text)
            if not data:
                logger.warning(f"[Debate/OpenAI] Unparseable response: {text[:200]}")
            return data
        except Exception as e:
            logger.error(f"[Debate/OpenAI] API error: {e}")
            return None

    async def _claude_call(self, user: str) -> Optional[dict]:
        """Call Claude as judge and parse JSON verdict."""
        try:
            resp = await self._claude.messages.create(
                model    = JUDGE_MODEL,
                messages = [{"role": "user", "content": user}],
                system   = _JUDGE_SYSTEM,
                max_tokens = 400,
            )
            text = resp.content[0].text.strip()
            data = _parse_json(text)
            if not data:
                logger.warning(f"[Debate/Judge] Unparseable response: {text[:200]}")
            return data
        except Exception as e:
            logger.error(f"[Debate/Judge] API error: {e}")
            return None
