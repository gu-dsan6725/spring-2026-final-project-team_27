"""
collector.py — BTC 5-Min Collector Agent.

Data sources (all free, no auth):
  1. Kraken WebSocket            — live BTC/USD price stream
  2. Polymarket Gamma API        — active 5-min markets + Up/Down odds
  3. Kraken REST API             — 1-min OHLCV candles for technical features
  4. Alternative.me API         — Fear & Greed Index (daily)
"""
from __future__ import annotations

import asyncio
import collections
import json
import math
import httpx
import websockets
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger
from models import MarketWindow, PriceCandle

GAMMA      = "https://gamma-api.polymarket.com"
KRAKEN     = "https://api.kraken.com"
KRAKEN_WS  = "wss://ws.kraken.com"
FEAR_GREED = "https://api.alternative.me/fng/"


class BTCCollector:
    """
    Builds a MarketWindow snapshot right before each 5-min Polymarket
    window opens. Called once per window (~every 5 minutes).
    """

    def __init__(self):
        self._btc_price: Optional[float] = None
        # 20 min * ~60 ticks/min = 1200 max entries
        self._price_history: collections.deque = collections.deque(maxlen=1200)
        self._ws_task: Optional[asyncio.Task] = None
        self._http = httpx.AsyncClient(timeout=10.0)
        self._fear_greed: Optional[tuple] = None

    async def start(self):
        """Start background WebSocket price stream."""
        self._ws_task = asyncio.create_task(self._stream_prices())
        await asyncio.sleep(2)
        logger.info("[Collector] BTC price stream started")

    async def close(self):
        if self._ws_task:
            self._ws_task.cancel()
        await self._http.aclose()

    async def build_window(self) -> Optional[MarketWindow]:
        """
        Fetch everything needed for one analysis window.
        Returns None if no active 5-min market found.
        """
        market = await self._get_active_5m_market()
        if not market:
            logger.warning("[Collector] No active 5-min BTC market found")
            return None

        btc_now = self._btc_price or await self._rest_btc_price()
        if not btc_now:
            logger.error("[Collector] Could not get BTC price")
            return None

        candles = await self._fetch_candles(limit=20)
        fg      = await self._get_fear_greed()
        window  = self._build_window(market, btc_now, candles, fg)

        logger.info(
            f"[Collector] Window built — BTC=${btc_now:,.2f}  "
            f"UP={market['up_price']:.0%}  DOWN={market['down_price']:.0%}"
        )
        return window

    # ── WebSocket price stream (Kraken) ───────────────────────────────────────

    async def _stream_prices(self):
        subscribe_msg = json.dumps({
            "event": "subscribe",
            "pair": ["XBT/USD"],
            "subscription": {"name": "ticker"},
        })

        while True:
            try:
                async with websockets.connect(
                    KRAKEN_WS,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    await ws.send(subscribe_msg)
                    logger.debug("[Collector] Kraken WebSocket connected")
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            # Kraken ticker: [channelID, {"c": ["price", ...], ...}, "ticker", "XBT/USD"]
                            if isinstance(msg, list) and len(msg) == 4 and msg[2] == "ticker":
                                price = float(msg[1]["c"][0])
                                ts    = datetime.now(tz=timezone.utc)
                                self._btc_price = price
                                self._price_history.append((ts, price))
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"[Collector] WebSocket error: {e} — reconnecting in 5s")
                await asyncio.sleep(5)

    async def _rest_btc_price(self) -> Optional[float]:
        """Fallback: get BTC price from Kraken REST if WebSocket not ready."""
        try:
            r = await self._http.get(
                f"{KRAKEN}/0/public/Ticker",
                params={"pair": "XBTUSD"}
            )
            r.raise_for_status()
            result = r.json().get("result", {})
            price = float(list(result.values())[0]["c"][0])
            return price
        except Exception as e:
            logger.error(f"[Collector] Kraken REST price failed: {e}")
            return None

    # ── Polymarket 5-min market ───────────────────────────────────────────────

    async def _get_active_5m_market(self) -> Optional[dict]:
        """
        The Gamma API returns a geo-filtered default list that excludes the
        restricted BTC 5-min series.  Instead we compute the current window's
        slug directly:  btc-updown-5m-{window_start_unix}
        and query the events endpoint by slug — which always works.
        We try the current window and, if it isn't live yet, the next one.
        """
        now = datetime.now(tz=timezone.utc)
        floored_min = (now.minute // 5) * 5
        current_start = now.replace(minute=floored_min, second=0, microsecond=0)

        for window_start in (current_start, current_start + timedelta(minutes=5)):
            slug = f"btc-updown-5m-{int(window_start.timestamp())}"
            try:
                r = await self._http.get(f"{GAMMA}/events", params={"slug": slug})
                r.raise_for_status()
                events = r.json()
                if not events:
                    continue
                event = events[0]
                if event.get("closed") or not event.get("active"):
                    continue
                markets = event.get("markets", [])
                if not markets:
                    continue
                m = markets[0]
                up_price, down_price = _parse_updown_prices_event(m)
                return {
                    "market_id":  m["conditionId"],
                    "question":   m.get("question", ""),
                    "up_price":   up_price,
                    "down_price": down_price,
                    "closes_at":  m.get("endDate", ""),
                }
            except Exception as e:
                logger.error(f"[Collector] Gamma API error for slug {slug}: {e}")

        return None

    # ── Kraken candles ────────────────────────────────────────────────────────

    async def _fetch_candles(self, limit: int = 20) -> list:
        """Fetch 1-min OHLCV candles from Kraken."""
        try:
            r = await self._http.get(f"{KRAKEN}/0/public/OHLC", params={
                "pair":     "XBTUSD",
                "interval": 1,
            })
            r.raise_for_status()
            result  = r.json().get("result", {})
            raw     = list(result.values())[0]  # drop the "last" key entry
            # Kraken format: [time, open, high, low, close, vwap, volume, count]
            candles = []
            for c in raw[-limit:]:
                candles.append(PriceCandle(
                    timestamp = datetime.fromtimestamp(c[0], tz=timezone.utc),
                    open      = float(c[1]),
                    high      = float(c[2]),
                    low       = float(c[3]),
                    close     = float(c[4]),
                    volume    = float(c[6]),
                ))
            return candles
        except Exception as e:
            logger.warning(f"[Collector] Kraken candles failed: {e}")
            return []

    # ── Fear & Greed ──────────────────────────────────────────────────────────

    async def _get_fear_greed(self) -> Optional[tuple]:
        if self._fear_greed:
            return self._fear_greed
        try:
            r = await self._http.get(FEAR_GREED, params={"limit": 1})
            r.raise_for_status()
            d = r.json()["data"][0]
            self._fear_greed = (int(d["value"]), d["value_classification"])
            return self._fear_greed
        except Exception as e:
            logger.warning(f"[Collector] Fear & Greed failed: {e}")
            return None

    # ── Window builder ────────────────────────────────────────────────────────

    def _build_window(
        self,
        market: dict,
        btc_now: float,
        candles: list,
        fg: Optional[tuple],
    ) -> MarketWindow:
        now = datetime.now(tz=timezone.utc)

        closes_at_str = market.get("closes_at", "")
        try:
            window_end = datetime.fromisoformat(
                closes_at_str.replace("Z", "+00:00")
            )
        except Exception:
            window_end = now + timedelta(minutes=5)
        window_start = window_end - timedelta(minutes=5)

        p1m = _price_at_offset(self._price_history, minutes=1)
        p5m = _price_at_offset(self._price_history, minutes=5)

        closes  = [c.close for c in candles]
        vol_5m  = sum(c.volume for c in candles[-5:]) if len(candles) >= 5 else None
        chg_1m  = _pct_change(btc_now, p1m)
        chg_5m  = _pct_change(btc_now, p5m)
        chg_15m = _pct_change(btc_now, closes[0] if closes else None)

        vol5 = None
        if len(closes) >= 5:
            mean     = sum(closes[-5:]) / 5
            variance = sum((x - mean) ** 2 for x in closes[-5:]) / 5
            vol5     = round(math.sqrt(variance), 2)

        rsi = _compute_rsi(closes) if len(closes) >= 14 else None

        return MarketWindow(
            collected_at     = now,
            market_id        = market["market_id"],
            market_question  = market["question"],
            window_start     = window_start,
            window_end       = window_end,
            up_price         = market["up_price"],
            down_price       = market["down_price"],
            btc_price_now    = btc_now,
            btc_price_1m_ago = p1m,
            btc_price_5m_ago = p5m,
            candles_1m       = candles[-10:],
            volume_5m        = vol_5m,
            price_change_1m  = chg_1m,
            price_change_5m  = chg_5m,
            price_change_15m = chg_15m,
            volatility_5m    = vol5,
            rsi_14           = rsi,
            fear_greed_score = fg[0] if fg else None,
            fear_greed_label = fg[1] if fg else None,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_updown_prices_event(m: dict) -> tuple:
    """Parse Up/Down prices from events API market (outcomes + outcomePrices JSON strings)."""
    up, down = 0.5, 0.5
    try:
        outcomes = json.loads(m.get("outcomes", "[]"))
        prices   = json.loads(m.get("outcomePrices", "[]"))
        for label, price in zip(outcomes, prices):
            label = str(label).lower()
            price = float(price)
            if "up" in label:
                up = price
            elif "down" in label:
                down = price
    except Exception:
        pass
    return round(up, 4), round(down, 4)


def _price_at_offset(history: list, minutes: int) -> Optional[float]:
    target  = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
    closest = None
    min_diff = float("inf")
    for ts, price in history:
        diff = abs((ts - target).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest  = price
    return closest if min_diff < 90 else None


def _pct_change(now: float, prev: Optional[float]) -> Optional[float]:
    if prev is None or prev == 0:
        return None
    return round((now - prev) / prev * 100, 4)


def _compute_rsi(closes: list, period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    if not gains or not losses:
        return 100.0 if not losses else 0.0
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)
