"""
storage.py — SQLite persistence for BTC 5-Min PolySignal.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from models import Prediction, EvalRecord, Trade

DB = Path("polysignal_btc.db")


def init():
    conn = sqlite3.connect(DB)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id   TEXT PRIMARY KEY,
        market_id       TEXT,
        market_question TEXT,
        window_start    TEXT,
        window_end      TEXT,
        direction       TEXT,
        confidence      REAL,
        btc_price_at_call REAL,
        market_up_odds  REAL,
        reasoning       TEXT,
        key_factors     TEXT,
        resolved        INTEGER DEFAULT 0,
        actual_outcome  TEXT,
        btc_price_close REAL,
        resolved_at     TEXT,
        created_at      TEXT
    );

    CREATE TABLE IF NOT EXISTS eval_records (
        eval_id          TEXT PRIMARY KEY,
        prediction_id    TEXT,
        market_id        TEXT,
        direction        TEXT,
        confidence       REAL,
        actual_outcome   TEXT,
        btc_open         REAL,
        btc_close        REAL,
        price_change_pct REAL,
        was_correct      INTEGER,
        brier_score      REAL,
        scored_at        TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_pred_resolved
        ON predictions(resolved);
    CREATE INDEX IF NOT EXISTS idx_eval_scored
        ON eval_records(scored_at);

    CREATE TABLE IF NOT EXISTS trades (
        trade_id        TEXT PRIMARY KEY,
        prediction_id   TEXT,
        market_question TEXT,
        direction       TEXT,
        odds_price      REAL,
        bet_amount      REAL,
        tokens          REAL,
        potential_payout REAL,
        resolved        INTEGER DEFAULT 0,
        actual_outcome  TEXT,
        actual_payout   REAL DEFAULT 0,
        pnl             REAL DEFAULT 0,
        balance_after   REAL DEFAULT 0,
        resolved_at     TEXT,
        created_at      TEXT
    );

    CREATE TABLE IF NOT EXISTS account (
        id              INTEGER PRIMARY KEY CHECK (id = 1),
        balance         REAL,
        initial_balance REAL,
        updated_at      TEXT
    );
    """)
    # Add model_votes column to existing predictions if missing
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN model_votes TEXT")
        conn.commit()
    except Exception:
        pass  # column already exists
    conn.close()


def save_prediction(p: Prediction):
    conn = sqlite3.connect(DB)
    conn.execute("""
        INSERT OR REPLACE INTO predictions (
            prediction_id, market_id, market_question,
            window_start, window_end,
            direction, confidence, btc_price_at_call,
            market_up_odds, reasoning, key_factors,
            resolved, actual_outcome, btc_price_close,
            resolved_at, created_at, model_votes
        ) VALUES (
            :prediction_id, :market_id, :market_question,
            :window_start, :window_end,
            :direction, :confidence, :btc_price_at_call,
            :market_up_odds, :reasoning, :key_factors,
            :resolved, :actual_outcome, :btc_price_close,
            :resolved_at, :created_at, :model_votes
        )
    """, {
        "prediction_id":    p.prediction_id,
        "market_id":        p.market_id,
        "market_question":  p.market_question,
        "window_start":     p.window_start.isoformat(),
        "window_end":       p.window_end.isoformat(),
        "direction":        p.direction,
        "confidence":       p.confidence,
        "btc_price_at_call": p.btc_price_at_call,
        "market_up_odds":   p.market_up_odds,
        "reasoning":        p.reasoning,
        "key_factors":      json.dumps(p.key_factors),
        "resolved":         int(p.resolved),
        "actual_outcome":   p.actual_outcome,
        "btc_price_close":  p.btc_price_close,
        "resolved_at":      p.resolved_at.isoformat() if p.resolved_at else None,
        "created_at":       p.created_at.isoformat(),
        "model_votes":      None,  # populated by ensemble via reasoning field
    })
    conn.commit()
    conn.close()


def save_eval(ev: EvalRecord):
    conn = sqlite3.connect(DB)
    conn.execute("""
        INSERT OR REPLACE INTO eval_records VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        ev.eval_id, ev.prediction_id, ev.market_id,
        ev.direction, ev.confidence, ev.actual_outcome,
        ev.btc_open, ev.btc_close, ev.price_change_pct,
        int(ev.was_correct), ev.brier_score,
        ev.scored_at.isoformat()
    ))
    conn.commit()
    conn.close()


def mark_resolved(prediction_id: str, outcome: str, close_price: float):
    conn = sqlite3.connect(DB)
    conn.execute("""
        UPDATE predictions
        SET resolved=1, actual_outcome=?, btc_price_close=?, resolved_at=?
        WHERE prediction_id=?
    """, (outcome, close_price, datetime.utcnow().isoformat(), prediction_id))
    conn.commit()
    conn.close()


def get_pending_predictions() -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions WHERE resolved=0 ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_evals() -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM eval_records ORDER BY scored_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Simulator ─────────────────────────────────────────────────────────────────

INITIAL_BALANCE = 100_000.0


def get_account_balance() -> float:
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT balance FROM account WHERE id=1").fetchone()
    conn.close()
    if row:
        return float(row[0])
    # First time — initialise
    _init_account()
    return INITIAL_BALANCE


def _init_account():
    conn = sqlite3.connect(DB)
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO account VALUES (1, ?, ?, ?)",
        (INITIAL_BALANCE, INITIAL_BALANCE, now)
    )
    conn.commit()
    conn.close()


def save_trade(t: Trade):
    conn = sqlite3.connect(DB)
    conn.execute("""
        INSERT OR REPLACE INTO trades VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        t.trade_id, t.prediction_id, t.market_question,
        t.direction, t.odds_price, t.bet_amount, t.tokens, t.potential_payout,
        int(t.resolved), t.actual_outcome, t.actual_payout,
        t.pnl, t.balance_after,
        t.resolved_at.isoformat() if t.resolved_at else None,
        t.created_at.isoformat()
    ))
    conn.commit()
    conn.close()


def settle_trade(trade_id: str, actual_outcome: str, actual_payout: float,
                 pnl: float, balance_after: float):
    conn = sqlite3.connect(DB)
    now = datetime.utcnow().isoformat()
    conn.execute("""
        UPDATE trades
        SET resolved=1, actual_outcome=?, actual_payout=?, pnl=?,
            balance_after=?, resolved_at=?
        WHERE trade_id=?
    """, (actual_outcome, actual_payout, pnl, balance_after, now, trade_id))
    conn.execute(
        "UPDATE account SET balance=?, updated_at=? WHERE id=1",
        (balance_after, now)
    )
    conn.commit()
    conn.close()


def get_pending_trades() -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM trades WHERE resolved=0 ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trade_by_prediction(prediction_id: str) -> Optional[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM trades WHERE prediction_id=?", (prediction_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_trades() -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM trades WHERE resolved=1 ORDER BY resolved_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_predictions(limit: int = 12) -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_summary() -> dict:
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    result = {
        "total_predictions": c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
        "pending":           c.execute("SELECT COUNT(*) FROM predictions WHERE resolved=0").fetchone()[0],
        "scored":            c.execute("SELECT COUNT(*) FROM eval_records").fetchone()[0],
        "correct":           c.execute("SELECT COUNT(*) FROM eval_records WHERE was_correct=1").fetchone()[0],
    }
    conn.close()
    return result
