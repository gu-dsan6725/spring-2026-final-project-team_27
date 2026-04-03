"""
simulator.py — Paper trading simulator for PolySignal predictions.

Simulates a $100,000 account trading Polymarket BTC 5-min Up/Down markets.

Mechanics (mirrors real Polymarket):
  - You buy tokens for the direction you predict (UP or DOWN)
  - Each token costs the current Polymarket odds price (e.g. $0.52 for UP)
  - If correct: each token pays $1.00  → profit = tokens - bet = bet*(1/price - 1)
  - If wrong:   tokens worth $0.00    → loss = -bet_amount
  - Bet size:   $500 flat per trade (0.5% of $100k starting capital)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional
from loguru import logger
from models import Prediction, MarketWindow, Trade
from storage import (
    save_trade, settle_trade, get_trade_by_prediction,
    get_account_balance, get_all_trades, INITIAL_BALANCE
)

BET_SIZE = 500.0   # dollars per trade


class TradingSimulator:

    def place_trade(self, prediction: Prediction, window: MarketWindow) -> Optional[Trade]:
        """Open a new simulated trade based on the ensemble prediction."""
        balance = get_account_balance()
        if balance < BET_SIZE:
            logger.warning(f"[Simulator] Insufficient balance: ${balance:,.2f}")
            return None

        # Buy the token corresponding to our predicted direction
        odds_price = window.up_price if prediction.direction == "UP" else window.down_price
        if odds_price <= 0:
            logger.warning("[Simulator] Invalid odds price")
            return None

        tokens          = BET_SIZE / odds_price
        potential_payout = tokens * 1.0   # each winning token pays $1

        trade = Trade(
            trade_id         = str(uuid.uuid4()),
            prediction_id    = prediction.prediction_id,
            market_question  = prediction.market_question,
            direction        = prediction.direction,
            odds_price       = round(odds_price, 4),
            bet_amount       = BET_SIZE,
            tokens           = round(tokens, 4),
            potential_payout = round(potential_payout, 4),
        )
        save_trade(trade)

        logger.info(
            f"[Simulator] Trade placed — {prediction.direction} @ {odds_price:.2%}  "
            f"bet=${BET_SIZE:.0f}  tokens={tokens:.1f}  "
            f"potential=${potential_payout:.2f}"
        )
        return trade

    def settle(self, prediction_id: str, actual_outcome: str) -> Optional[dict]:
        """Settle a trade after the market resolves. Returns P&L summary."""
        row = get_trade_by_prediction(prediction_id)
        if not row or row.get("resolved"):
            return None

        won           = (row["direction"] == actual_outcome)
        actual_payout = row["tokens"] if won else 0.0
        pnl           = actual_payout - row["bet_amount"]
        balance_after = get_account_balance() + pnl

        settle_trade(
            trade_id       = row["trade_id"],
            actual_outcome = actual_outcome,
            actual_payout  = round(actual_payout, 4),
            pnl            = round(pnl, 4),
            balance_after  = round(balance_after, 4),
        )

        logger.info(
            f"[Simulator] Settled — {'WIN' if won else 'LOSS'}  "
            f"P&L={pnl:+.2f}  balance=${balance_after:,.2f}"
        )

        return {
            "won":           won,
            "direction":     row["direction"],
            "actual":        actual_outcome,
            "odds_price":    row["odds_price"],
            "bet_amount":    row["bet_amount"],
            "actual_payout": actual_payout,
            "pnl":           pnl,
            "balance_after": balance_after,
        }


def compute_sim_metrics() -> dict:
    """Aggregate statistics across all settled trades."""
    trades   = get_all_trades()
    balance  = get_account_balance()

    if not trades:
        return {}

    total     = len(trades)
    wins      = sum(1 for t in trades if t["direction"] == t["actual_outcome"])
    total_pnl = sum(t["pnl"] for t in trades)
    total_bet = sum(t["bet_amount"] for t in trades)
    roi       = total_pnl / INITIAL_BALANCE * 100

    avg_win_pnl  = sum(t["pnl"] for t in trades if t["direction"] == t["actual_outcome"]) / max(wins, 1)
    losses       = total - wins
    avg_loss_pnl = sum(t["pnl"] for t in trades if t["direction"] != t["actual_outcome"]) / max(losses, 1)

    return {
        "balance":       round(balance, 2),
        "initial":       INITIAL_BALANCE,
        "total_pnl":     round(total_pnl, 2),
        "roi_pct":       round(roi, 3),
        "total_trades":  total,
        "wins":          wins,
        "losses":        losses,
        "win_rate":      round(wins / total, 3),
        "avg_win":       round(avg_win_pnl, 2),
        "avg_loss":      round(avg_loss_pnl, 2),
        "total_wagered": round(total_bet, 2),
        "trades":        trades,
    }
