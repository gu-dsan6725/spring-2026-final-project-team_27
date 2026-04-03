# PolySignal — BTC 5-Min Forecaster

> A multi-agent AI system that predicts Bitcoin price direction on Polymarket's 5-minute derivative markets. No wallet. No trading. Pure accuracy measurement.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Method](#method)
4. [The Math](#the-math)
5. [Data Sources](#data-sources)
6. [Agents](#agents)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Preliminary Results](#preliminary-results)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Project Structure](#project-structure)

---

## Overview

PolySignal is a research-grade multi-agent pipeline that:

1. **Collects** real-time BTC/USD price data, technical indicators, market sentiment, and live crowd odds from Polymarket's 5-minute up/down markets
2. **Analyzes** the data using Claude Sonnet (Anthropic) to generate a directional prediction (UP or DOWN) with a calibrated confidence score
3. **Evaluates** each prediction automatically once the 5-minute window closes, scoring it against the actual Bitcoin price movement
4. **Reports** cumulative accuracy metrics, Brier scores, and confidence calibration statistics

The system is designed as a scientific instrument — it measures how accurately a large language model can forecast ultra-short-term crypto price direction when given structured market data, technical signals, and sentiment context.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        run.py  (Orchestrator)                   │
│   One-shot or --live mode (every 5 min, configurable duration)  │
└────────┬──────────────┬──────────────┬──────────────┬───────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
  ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────────┐
  │ Evaluator  │ │ Collector  │ │   Analyzer   │ │   Reporter   │
  │  (Step 1)  │ │  (Step 2)  │ │   (Step 3)   │ │   (Step 4)   │
  └─────┬──────┘ └─────┬──────┘ └──────┬───────┘ └──────┬───────┘
        │              │               │                 │
        │    ┌─────────┴──────┐        │                 │
        │    │  External APIs  │        │                 │
        │    │ • Kraken WS    │        │                 │
        │    │ • Kraken REST  │        │                 │
        │    │ • Polymarket   │        │                 │
        │    │ • Alt.me F&G   │        │                 │
        │    └────────────────┘        │                 │
        │                              │                 │
        └──────────────┬───────────────┘                 │
                       │                                 │
              ┌────────▼────────┐                        │
              │   SQLite DB     │◄───────────────────────┘
              │ predictions     │
              │ eval_records    │
              └─────────────────┘
```

### Data Flow

Each cycle follows four sequential steps:

| Step | Agent | Input | Output |
|------|-------|-------|--------|
| 1 | Evaluator | Pending predictions in DB | Scores for closed windows |
| 2 | Collector | Live APIs | `MarketWindow` snapshot |
| 3 | Analyzer | `MarketWindow` | `Prediction` (direction + confidence) |
| 4 | Reporter | All `EvalRecord`s in DB | Accuracy report (console + markdown) |

---

## Method

### Why Polymarket 5-Minute Markets?

Polymarket's BTC Up/Down 5-minute markets resolve to "Up" if the Chainlink BTC/USD oracle price at the end of the window is ≥ the price at the start, and "Down" otherwise. This creates a clean, objective binary outcome with:

- **No ambiguity**: Resolution is deterministic via Chainlink oracle
- **High frequency**: A new market opens every 5 minutes, 24/7
- **Crowd signal**: The market's Up/Down token prices encode the crowd's implied probability of each outcome
- **Measurability**: Outcomes are verifiable against actual BTC price data

### Prediction Strategy

The system uses a structured prompt to guide Claude through a four-factor analysis:

1. **Momentum** (primary signal): 1-minute and 5-minute price change percentages. Short-term momentum tends to persist over 5-minute horizons.

2. **Technical Indicators** (secondary signal): RSI(14) identifies overbought (>70) or oversold (<30) conditions that often precede mean reversion.

3. **Crowd Odds** (contrarian signal): If the market is pricing Up at 60%+, the crowd may be overweighting recency bias — the model uses this as a weak contrarian indicator.

4. **Sentiment** (background context): Fear & Greed Index provides macro context but is given low weight at the 5-minute timescale.

The model is explicitly calibrated: a 70% confidence call should succeed approximately 70% of the time. Overconfidence is penalised by the Brier score.

---

## The Math

### Brier Score

The Brier score measures the accuracy of probabilistic predictions:

```
BS = (1/N) × Σ (fᵢ - oᵢ)²
```

Where:
- `fᵢ` = predicted probability (confidence) for prediction `i`
- `oᵢ` = actual outcome (1 if correct direction, 0 if wrong)
- `N` = total number of predictions

| Brier Score | Interpretation |
|-------------|---------------|
| 0.00 | Perfect — always right with 100% confidence |
| 0.25 | Random — equivalent to always saying 50% |
| 1.00 | Worst possible — always wrong with 100% confidence |

A model with 70% accuracy but only 55% average confidence will score **better** than a model with 70% accuracy and 90% confidence — because the Brier score penalises overconfidence heavily.

### Confidence Calibration

A well-calibrated model satisfies:

```
P(correct | confidence = c) ≈ c   for all c
```

The system bins predictions by stated confidence and computes the calibration gap:

```
gap = | actual_accuracy_in_bin - avg_confidence_in_bin |
```

| Gap | Rating |
|-----|--------|
| < 10% | Well calibrated |
| 10–20% | Slightly off |
| > 20% | Poorly calibrated |

### RSI (Relative Strength Index)

Computed over the last 14 one-minute candles:

```
RSI = 100 - (100 / (1 + RS))

RS = avg_gain_over_period / avg_loss_over_period
```

- RSI > 70 → overbought (bearish signal for mean reversion)
- RSI < 30 → oversold (bullish signal for mean reversion)
- RSI 45–55 → neutral

### 5-Minute Volatility

Standard deviation of the last 5 closing prices:

```
σ = sqrt( (1/5) × Σ(xᵢ - μ)² )
```

High volatility windows are flagged as lower-confidence prediction environments.

### Price Change

```
Δ% = ((price_now - price_t) / price_t) × 100
```

Computed at 1-minute, 5-minute, and 15-minute lookback horizons.

---

## Data Sources

| Source | Data | Latency | Auth |
|--------|------|---------|------|
| **Kraken WebSocket** `wss://ws.kraken.com` | Live BTC/USD tick price | ~100ms | None |
| **Kraken REST** `api.kraken.com/0/public/OHLC` | 1-min OHLCV candles | ~1s | None |
| **Polymarket Gamma API** `gamma-api.polymarket.com/events` | Active 5-min market + Up/Down odds | ~2s | None |
| **Alternative.me** `api.alternative.me/fng/` | Fear & Greed Index (0–100) | ~1s | None |
| **Anthropic API** | Claude Sonnet 4.5 inference | ~3–5s | API Key |

All market data sources are free and require no authentication. Only the Anthropic API requires a key.

> **Note**: Binance is intentionally excluded — it returns HTTP 451 (geo-blocked) for US users. Kraken provides equivalent OHLCV data with no restrictions.

---

## Agents

### BTCCollector

Responsible for building a `MarketWindow` — a complete snapshot of market conditions at the moment a new 5-minute window opens.

**Key behaviour:**
- Maintains a background Kraken WebSocket to track a 20-minute rolling BTC price history
- Computes the current market's Polymarket slug dynamically: `btc-updown-5m-{window_start_unix_timestamp}`
- Falls back to Kraken REST for BTC price if WebSocket hasn't received data yet
- Caches the Fear & Greed index for the session (daily data, no need to re-fetch)

**Output: `MarketWindow`**
```
btc_price_now        float     Current BTC/USD price
btc_price_1m_ago     float     Price 1 minute ago (from WebSocket history)
btc_price_5m_ago     float     Price 5 minutes ago
candles_1m           list[10]  Last 10 x 1-min OHLCV candles
price_change_1m      float     % change over 1 minute
price_change_5m      float     % change over 5 minutes
price_change_15m     float     % change over 15 minutes
volatility_5m        float     StdDev of last 5 closes
rsi_14               float     RSI over last 14 candles
volume_5m            float     Total BTC volume last 5 minutes
up_price             float     Polymarket implied P(Up)
down_price           float     Polymarket implied P(Down)
fear_greed_score     int       Fear & Greed Index (0–100)
fear_greed_label     str       e.g. "Greed", "Fear", "Neutral"
```

---

### AnalyzerAgent

Sends the `MarketWindow` to Claude Sonnet and parses a structured JSON prediction.

**System prompt instructs the model to:**
- Act as a quantitative analyst specialising in ultra-short-term BTC prediction
- Weight momentum > RSI > crowd odds > sentiment
- Express honest, calibrated confidence (not inflated)
- Return structured JSON with direction, confidence, reasoning, and key factors

**Output: `Prediction`**
```
direction       "UP" | "DOWN"
confidence      float (0.5 – 1.0)
reasoning       str   step-by-step analysis
key_factors     list  top 2–3 decision drivers
btc_price_at_call  float
market_id       str
```

---

### EvaluatorAgent

Runs at the start of every cycle. Finds all predictions where the window has closed, fetches the actual close price from Kraken, and scores each one.

**Scoring logic:**
```python
actual   = "UP" if close_price >= open_price else "DOWN"
correct  = (prediction.direction == actual)
brier    = (confidence - (1.0 if correct else 0.0)) ** 2
```

**Output: `EvalRecord`**
```
was_correct        bool
actual_outcome     "UP" | "DOWN"
btc_open           float
btc_close          float
price_change_pct   float
brier_score        float
```

---

### Reporter

Aggregates all `EvalRecord`s and produces a formatted accuracy report. Runs at the end of every cycle.

**Produces:**
- Overall accuracy table
- Confidence calibration breakdown by bin
- Pipeline summary (total / pending / scored / correct)
- Timestamped markdown file saved to `reports/`

---

## Evaluation Metrics

| Metric | Formula | Baseline | Interpretation |
|--------|---------|----------|----------------|
| **Directional Accuracy** | correct / scored | 50% (coin flip) | Higher is better |
| **Brier Score** | mean[(conf - actual)²] | 0.25 (random) | Lower is better |
| **Calibration Gap** | \|actual% - avg_conf\| per bin | 0% (perfect) | Lower is better |
| **DOWN Bias** | DOWN_calls / total_calls | 50% | Should approach 50% over time |
| **Avg Price Move** | mean(\|Δ%\|) per window | ~0.05% | Context for signal difficulty |

### Why Brier Score Over Accuracy?

Pure accuracy (win rate) rewards a system that says "DOWN" with 99% confidence on every call — if it's right 60% of the time, accuracy looks fine but the model is badly miscalibrated and would be dangerous to rely on.

The Brier score penalises exactly this. A correct call at 99% confidence earns a score of `(0.99 - 1.0)² = 0.0001` (near-perfect). A wrong call at 99% confidence earns `(0.99 - 0.0)² = 0.9801` (near-worst). This forces the model to be honest about uncertainty.

---

## Preliminary Results

Results from **15 scored predictions** (initial session, 2026-03-26):

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Directional accuracy | **66.7%** | +16.7% above 50% coin flip |
| Brier score | **0.2596** | ~random (0.25) — calibration needs work |
| UP calls | 3 (20%) | — |
| DOWN calls | 12 (80%) | Heavy bias |

**Confidence Calibration:**

| Bin | Predictions | Correct | Actual % | Gap | Rating |
|-----|-------------|---------|----------|-----|--------|
| 50–60% | 12 | 10 | 83.3% | 28.3% | Poorly calibrated |
| 60–70% | 3 | 0 | 0.0% | 64.0% | Poorly calibrated |

**Key observations:**
- The model is **underconfident** in the 50–60% bin (actually correct 83% of the time)
- The model **overperforms on DOWN calls** during a bearish session but shows UP call bias failure
- 15 samples is insufficient for statistical conclusions — target 100+ for stable calibration estimates
- Calibration should improve as the model accumulates more signal diversity (UP markets, neutral RSI windows)

---

## Installation

```bash
# Clone or download the project
cd polysignal-btc-2

# Install dependencies
pip3 install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...
```

**requirements.txt** covers: `anthropic`, `httpx`, `websockets`, `loguru`, `rich`, `python-dotenv`

---

## Usage

```bash
# Run one prediction cycle
python3 run.py

# Run automatically every 5 minutes for 1 hour (~$0.07)
python3 run.py --live --hours 1

# Run for 2 hours (~$0.14)
python3 run.py --live --hours 2

# View accuracy report without running a new cycle
python3 run.py --report

# View pending (unscored) predictions
python3 run.py --pending
```

**Cost estimate**: ~$0.006 per cycle (Claude Sonnet API call). One hour = 12 cycles ≈ $0.07.

---

## Project Structure

```
polysignal-btc-2/
├── run.py            # Orchestrator — entry point, CLI, live mode
├── collector.py      # BTCCollector — real-time data gathering
├── analyzer.py       # AnalyzerAgent — Claude LLM prediction
├── evaluator.py      # EvaluatorAgent — outcome scoring
├── reporter.py       # Reporter — accuracy metrics & reports
├── storage.py        # SQLite helpers (init, save, query)
├── models.py         # Dataclasses: MarketWindow, Prediction, EvalRecord
├── requirements.txt  # Python dependencies
├── .env.example      # API key template
├── .env              # Your API key (gitignored)
├── polysignal_btc.db # SQLite database (auto-created)
├── polysignal_btc.log# Debug log (auto-created, 10MB rotation)
└── reports/          # Timestamped markdown accuracy reports
```

---

## Limitations & Future Work

- **Small sample size**: 15 predictions is not statistically meaningful. Run for several days to accumulate 200+ scored predictions.
- **Direction bias**: The model currently over-calls DOWN. A longer session with diverse market conditions should correct this.
- **Single model**: Only Claude Sonnet is tested. Comparing against a momentum baseline (always predict the direction of the last 1-minute candle) would establish a stronger benchmark.
- **No feature selection**: All features are passed to the model equally. Ablation studies (removing Fear & Greed, removing RSI, etc.) could identify which signals actually contribute.
- **Polymarket geo-restriction**: The BTC 5-min series is restricted for US users but the data API is publicly accessible. This system reads data only — no trading functionality.

---

*PolySignal — built for research. No financial advice. No wallet required.*
