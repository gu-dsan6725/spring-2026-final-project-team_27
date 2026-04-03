"""
evaluator.py — BTC 5-Min Evaluator Agent.
"""
from __future__ import annotations

import uuid
import httpx
from datetime import datetime, timezone
from typing import Optional
from loguru import logger
from models import EvalRecord
from storage import get_pending_predictions, save_eval, mark_resolved

KRAKEN = "https://api.kraken.com"


class EvaluatorAgent:
    async def score_pending(self) -> list:
        pending = get_pending_predictions()
        now     = datetime.now(tz=timezone.utc)
        scored  = []

        if not pending:
            return []

        async with httpx.AsyncClient(timeout=10.0) as client:
            for row in pending:
                try:
                    window_end = datetime.fromisoformat(row["window_end"])
                    if window_end.tzinfo is None:
                        window_end = window_end.replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                if now < window_end:
                    logger.debug(f"[Evaluator] Window still open: {row['market_question'][:50]}")
                    continue

                ev = await self._score_one(client, row, window_end)
                if ev:
                    scored.append(ev)

        logger.info(f"[Evaluator] Scored {len(scored)} predictions")
        return scored

    async def _score_one(
        self,
        client: httpx.AsyncClient,
        row: dict,
        window_end: datetime,
    ) -> Optional[EvalRecord]:
        try:
            open_price  = float(row["btc_price_at_call"])
            close_price = await self._get_close_price(client, window_end)

            if close_price is None:
                logger.warning(f"[Evaluator] Could not get close price for {row['prediction_id'][:8]}")
                return None

            actual  = "UP" if close_price >= open_price else "DOWN"
            correct = (row["direction"] == actual)
            conf    = float(row["confidence"])
            brier   = round((conf - (1.0 if correct else 0.0)) ** 2, 4)
            pct     = round((close_price - open_price) / open_price * 100, 4)

            ev = EvalRecord(
                eval_id          = str(uuid.uuid4()),
                prediction_id    = row["prediction_id"],
                market_id        = row["market_id"],
                direction        = row["direction"],
                confidence       = conf,
                actual_outcome   = actual,
                btc_open         = open_price,
                btc_close        = close_price,
                price_change_pct = pct,
                was_correct      = correct,
                brier_score      = brier,
            )

            save_eval(ev)
            mark_resolved(row["prediction_id"], actual, close_price)

            logger.success(
                f"[Evaluator] {'checkmark' if correct else 'x'}  "
                f"{row['direction']} call  actual={actual}  "
                f"delta={pct:+.3f}%  Brier={brier:.3f}"
            )
            return ev

        except Exception as e:
            logger.error(f"[Evaluator] Scoring error: {e}")
            return None

    async def _get_close_price(
        self,
        client: httpx.AsyncClient,
        window_end: datetime,
    ) -> Optional[float]:
        try:
            # Kraken OHLC: fetch 1-min candles, since= window_end minus a few minutes
            since = int(window_end.timestamp()) - 3 * 60

            r = await client.get(f"{KRAKEN}/0/public/OHLC", params={
                "pair":     "XBTUSD",
                "interval": 1,
                "since":    since,
            })
            r.raise_for_status()
            result  = r.json().get("result", {})
            candles = [v for k, v in result.items() if k != "last"]
            if not candles:
                return None
            candles = candles[0]  # list of [time, open, high, low, close, vwap, volume, count]

            end_ts   = window_end.timestamp()
            best     = None
            min_diff = float("inf")
            for c in candles:
                diff = abs(c[0] - end_ts)
                if diff < min_diff:
                    min_diff = diff
                    best     = float(c[4])  # close price

            return best

        except Exception as e:
            logger.warning(f"[Evaluator] Kraken kline error: {e}")
            return None
