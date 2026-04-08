# 3. System Architecture

## 3.1 Overview

PolySignal is a modular multi-agent pipeline that operates on a fixed 5-minute clock synchronized to Polymarket's BTC Up/Down market windows. Each cycle consists of four sequential stages — evaluation, collection, prediction, and reporting — orchestrated by a central coordinator (`run.py`). The system is designed to be stateless between cycles: all intermediate results are persisted to a SQLite database, allowing the pipeline to resume cleanly after interruption.

Figure 1 shows the high-level architecture.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Orchestrator (run.py)                         │
│         Triggered every 5 minutes, or once in one-shot mode          │
└────────┬──────────────┬───────────────┬──────────────────┬───────────┘
         │              │               │                  │
         ▼              ▼               ▼                  ▼
  ┌────────────┐ ┌────────────┐ ┌─────────────────┐ ┌──────────────┐
  │ Evaluator  │ │ Collector  │ │  EnsembleAgent  │ │   Reporter   │
  │  (Step 1)  │ │  (Step 2)  │ │    (Step 3a)    │ │   (Step 4)   │
  └─────┬──────┘ └─────┬──────┘ └────────┬────────┘ └──────┬───────┘
        │              │                 │                  │
        │   ┌──────────┴──────────┐  ┌───┴─────────────┐   │
        │   │    External APIs    │  │   DebateAgent   │   │
        │   │  • Kraken WS (live) │  │   (Step 3b)     │   │
        │   │  • Kraken REST OHLC │  │  [optional]     │   │
        │   │  • Polymarket Gamma │  └───────┬─────────┘   │
        │   │  • Alternative.me   │          │              │
        │   └─────────────────────┘   ┌──────┴──────────┐  │
        │                             │  LLM APIs        │  │
        │                             │  • Anthropic     │  │
        │                             │  • Groq (opt.)   │  │
        │                             │  • OpenAI (opt.) │  │
        │                             └─────────────────┘  │
        │                                                    │
        └──────────────────┬─────────────────────────────────┘
                           ▼
                  ┌─────────────────┐     ┌──────────────────┐
                  │   SQLite DB     │     │  reports/*.md    │
                  │  predictions    │     │  (per-cycle      │
                  │  eval_records   │     │   snapshots)     │
                  │  trades         │     └──────────────────┘
                  │  account        │
                  └─────────────────┘
```
*Figure 1: PolySignal system architecture. Step 3 supports two prediction modes: EnsembleAgent (always available) and DebateAgent (activated when Groq and OpenAI API keys are present). Arrows indicate data flow within a single 5-minute cycle.*

---

## 3.2 Orchestrator

The orchestrator (`run.py`) drives the pipeline and exposes a CLI interface with two operating modes:

- **One-shot mode** (`python run.py`): executes a single full cycle and exits. Used for testing and debugging.
- **Live mode** (`python run.py --live --hours N`): runs continuously, sleeping between cycles to align with the next 5-minute Polymarket window boundary.

The orchestrator enforces a strict execution order within each cycle to ensure that predictions from prior cycles are scored before new ones are made, and that reports always reflect the latest evaluation state.

---

## 3.3 Agent Descriptions

### 3.3.1 BTCCollector (Step 1: Data Collection)

The BTCCollector constructs a `MarketWindow` — a structured snapshot of market conditions at the moment each 5-minute window opens. It draws from four external sources:

| Source | Data | Latency |
|--------|------|---------|
| Kraken WebSocket (`wss://ws.kraken.com`) | Live BTC/USD tick price | ~100ms |
| Kraken REST (`/0/public/OHLC`) | 1-minute OHLCV candles | ~1s |
| Polymarket Gamma API | Active 5-min market slug, UP/DOWN odds | ~2s |
| Alternative.me | Fear & Greed Index (0–100) | ~1s |

The agent maintains a background WebSocket thread to track a 20-minute rolling BTC price history, from which it derives momentum features at 1-minute, 5-minute, and 15-minute horizons. Technical indicators — RSI(14) and 5-minute realized volatility — are computed from the Kraken OHLCV candles. The Polymarket market slug is derived dynamically from the current window's Unix timestamp, enabling the agent to fetch live crowd odds without hardcoding market identifiers.

The `MarketWindow` output contains the following features passed to the prediction stage:

```
btc_price_now        Current BTC/USD spot price
price_change_1m      % price change over past 1 minute
price_change_5m      % price change over past 5 minutes
price_change_15m     % price change over past 15 minutes
volatility_5m        Realized volatility (StdDev of last 5 closes)
rsi_14               RSI computed over last 14 one-minute candles
volume_5m            Aggregate BTC volume over last 5 minutes
up_price             Polymarket implied probability of UP outcome
down_price           Polymarket implied probability of DOWN outcome
fear_greed_score     Daily sentiment index (0 = Extreme Fear, 100 = Extreme Greed)
```

### 3.3.2 EnsembleAgent (Step 2: Prediction)

Rather than relying on a single model, the EnsembleAgent queries three Claude instances in parallel, each configured with a distinct analytical persona designed to capture different signal patterns:

| Agent Role | Model | Primary Focus |
|------------|-------|---------------|
| Momentum Trader | `claude-haiku-4-5` | Short-term price action and directional persistence |
| Technical Analyst | `claude-sonnet-4-5` | Balanced multi-signal weighting (RSI, momentum, sentiment) |
| Contrarian | `claude-sonnet-4-6` | Market microstructure and mean-reversion signals |

Each model independently returns a direction (`UP` or `DOWN`) and a confidence score in [0.5, 1.0]. The ensemble aggregates these votes via majority rule:

- **Direction**: whichever direction receives ≥ 2 of 3 votes is selected
- **Confidence**: the mean confidence of all models, adjusted by agreement level (+0.02 if unanimous, −0.02 if split two-to-one)

This design reduces the variance of any single model's idiosyncratic reasoning errors while preserving diverse analytical perspectives. All three API calls are issued in parallel to minimize latency.

The final `Prediction` object contains:
```
direction       "UP" | "DOWN"
confidence      float in [0.5, 1.0]
reasoning       Free-text step-by-step analysis (from winning majority model)
key_factors     Top 2–3 decision drivers cited by the model
btc_price_at_call  Spot price at time of prediction
window_start    Unix timestamp of the market window start
window_end      Unix timestamp of the market window end
```

### 3.3.3 EvaluatorAgent (Step 3: Outcome Scoring)

The EvaluatorAgent runs at the start of each cycle (before new predictions are made) to score all pending predictions from prior windows that have since closed. For each unscored prediction:

1. The actual BTC close price is fetched from the Kraken REST API at `window_end`
2. The true outcome is determined: `UP` if `close ≥ open`, else `DOWN`
3. Directional correctness is recorded as a binary flag
4. A Brier score is computed: `BS = (confidence − outcome)²`, where `outcome ∈ {0, 1}`

Results are stored as `EvalRecord` objects in the database. This design ensures that no prediction is scored until its window has fully closed, preventing look-ahead bias.

### 3.3.4 Reporter (Step 4: Metrics & Reporting)

The Reporter aggregates all `EvalRecord`s and emits a structured accuracy report at the end of each cycle. It computes:

- **Overall directional accuracy**: proportion of correct UP/DOWN calls
- **Brier score**: mean squared error of probabilistic predictions (lower is better; random baseline = 0.25)
- **Confidence calibration by bin**: actual accuracy versus stated confidence within four bins (50–55%, 55–60%, 60–65%, 65%+)
- **Calibration gap**: `|actual_accuracy − avg_confidence|` per bin

Each report is saved as a timestamped Markdown file in `reports/` and rendered to the terminal using a rich-formatted table display.

### 3.3.5 DebateAgent (Step 3b: Adversarial Prediction — Planned)

The DebateAgent is an alternative prediction mode that replaces the parallel-vote ensemble with a structured adversarial debate between two heterogeneous LLMs, mediated by a third judge model. This design is motivated by the hypothesis that explicit disagreement and rebuttal between models with different training distributions may surface reasoning errors that a voting ensemble would silently average away.

**Debate protocol (three rounds):**

```
Round 1 — Independent Analysis (parallel)
  Groq (Llama 3)   →  Position A: direction + confidence + reasoning
  OpenAI (GPT-4o)  →  Position B: direction + confidence + reasoning

Round 2 — Cross-Examination (parallel)
  Groq sees Position B  →  Rebuttal A: defend or revise position
  OpenAI sees Position A →  Rebuttal B: defend or revise position

Round 3 — Judgment (single call)
  Claude (Sonnet)  reads all four outputs
                   →  Final: direction + confidence + winning argument summary
```

**Timing estimate:**

| Step | Latency |
|------|---------|
| Round 1 (parallel) | ~2–4s (Groq LPU is sub-second) |
| Round 2 (parallel) | ~3–5s |
| Round 3 (judge) | ~3–5s |
| **Total** | **~8–14s** |

This is well within the 5-minute prediction window. Groq's LPU inference hardware makes it particularly suited for latency-sensitive applications.

**Activation condition:** The DebateAgent activates automatically when both `GROQ_API_KEY` and `OPENAI_API_KEY` are present in the environment. If either key is missing, the system falls back to the EnsembleAgent without interruption.

**Research value:** Running both modes in parallel during the same data collection period allows a direct controlled comparison — same market windows, same features, different prediction architectures — isolating the effect of debate versus voting on directional accuracy and Brier score.

---

### 3.3.6 Simulator (Paper Trading)

A paper trading simulator tracks a $100,000 simulated account, placing a flat $500 bet on each prediction using live Polymarket odds. Winning positions pay $1.00 per token; losing positions forfeit the bet. The simulator tracks running balance, per-trade P&L, win rate, and ROI. It is strictly read-only with respect to actual markets — no wallet or trading infrastructure is required.

---

## 3.4 Data Persistence

All agent outputs are persisted to a local SQLite database with the following schema:

| Table | Contents |
|-------|----------|
| `predictions` | All predictions: direction, confidence, reasoning, window timestamps |
| `eval_records` | Scored outcomes: correctness, Brier score, actual BTC prices |
| `trades` | Simulated trades: bet size, odds, payout, P&L, running balance |
| `account` | Current simulated account balance |

The database is the single source of truth across sessions. When running on ephemeral infrastructure (e.g., AWS Academy lab environments with 4-hour session limits), the database is downloaded at the end of each session and re-uploaded at the start of the next, preserving continuity across lab boundaries.

---

## 3.5 Design Decisions

**Why SQLite over a hosted database?** The system is designed for a single-node research deployment. SQLite eliminates infrastructure dependencies, reduces cost to zero, and allows the entire experiment state to be backed up as a single file.

**Why three models instead of one?** Ensemble methods reduce variance. A single model's reasoning can be unduly influenced by salient but uninformative features in a particular market snapshot. Assigning distinct analytical personas encourages the ensemble to explore different regions of the reasoning space, analogous to analyst diversity in human prediction panels.

**Why Brier score over accuracy alone?** Directional accuracy rewards overconfident correct calls equally with calibrated ones. The Brier score penalizes overconfidence quadratically, incentivizing the system to express genuine uncertainty rather than inflate confidence to appear decisive.

**Why Kraken instead of Binance?** Binance returns HTTP 451 (legal restriction) for US-based IP addresses. Kraken provides equivalent OHLCV data with no geographic restrictions and no authentication requirements.

**Why adversarial debate instead of (or alongside) voting?** Ensemble voting averages out disagreements silently — a 2-to-1 majority simply overrules the dissenting model. Debate forces the minority position to be articulated and challenged, surfacing whether disagreement stems from genuine signal differences or noise. If the dissenting model cannot rebut the majority's argument, the judge discards it with justification; if it can, the final prediction reflects a higher-quality synthesis. This is structurally analogous to red-teaming in security research or adversarial collaboration in scientific peer review.

**Why Groq for the debate?** Groq's LPU (Language Processing Unit) hardware delivers sub-second inference for Llama 3-class models, making it uniquely suited for latency-sensitive pipelines where multiple sequential LLM calls must complete within a fixed time budget. Pairing Groq's speed with OpenAI's reasoning depth creates a natural cost-quality tradeoff within a single debate round.
