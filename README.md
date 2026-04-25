# PolySignal-BTC5 — Adversarial Multi-Agent LLM Debate for Bitcoin Prediction Markets

> Can three AI models arguing with each other beat the crowd on Bitcoin's fastest prediction market?


---

## Results — 16-Hour Evaluation Session 

| Metric | Value |
|--------|-------|
| Directional accuracy | **57.6% ± 7.0%** (95% CI lower bound: 50.6%) |
| Brier score | **0.2474** (vs 0.2500 random baseline) |
| Crowd baseline (Polymarket odds) | 55.0% — PolySignal beats it by 2.6% |
| xAI Grok individual accuracy | **59.6%** — strongest single model |
| Simulated ROI | **+$13,321 (+13.32%)** on $100,000 over 191 trades |

---

## Deliverables

| Item | Location |
|------|----------|
| Paper | [`polysignal-btc-ai-trading/docs/report.md`](polysignal-btc-ai-trading/docs/report.md) |
| Paper(pdf) | [`paper.pdf`](paper.pdf) |
| Code | [`polysignal-btc-ai-trading/`](polysignal-btc-ai-trading/) |
| Results & charts | [`results.qmd`](results.qmd) |
| System architecture | [`architecture.qmd`](architecture.qmd) |
| Presentation slides | [Google Slides](https://drive.google.com/drive/folders/1ugB_92PO_5v2iLdjXO1Fv46pA_j6f_FG?usp=share_link) |
| Poster | [Posters-pdf version has some issue, please refer to html or png version](poster.png) |
| Demo (video) | [Google Drive — Video Demo](https://drive.google.com/drive/folders/1ugB_92PO_5v2iLdjXO1Fv46pA_j6f_FG?usp=share_link) |
| Demo (static) | [Results page with live charts](results.qmd) — interactive charts built from the 16-hour session data |

---

## Setup and Installation

### 1. Clone and enter the project

```bash
git clone https://github.com/gu-dsan6725/spring-2026-final-project-team_27.git
cd ur_file_name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
ANTHROPIC_API_KEY=sk-ant-...
(Optional but Optimized :))
XAI_API_KEY=...        
OPENAI_API_KEY=...     
```

> **Note:** Without `XAI_API_KEY` and `OPENAI_API_KEY`, the system automatically falls back to the EnsembleAgent (Claude-only voting). The DebateAgent requires all three keys.

---

## Running the System

### Run one prediction cycle

```bash
python3 run.py
```

Evaluates any pending predictions, collects live market data, runs the 3-round debate, and saves the result to the SQLite database.

### Run continuously for N hours

```bash
python3 run.py --live --hours 16
```

Runs automatically every 5 minutes, synchronized to Polymarket's BTC Up/Down market windows.

### View the accuracy report

```bash
python3 run.py --report
```

Prints directional accuracy, Brier score, calibration breakdown, per-model accuracy, and baseline comparison.

### View pending (unscored) predictions

```bash
python3 run.py --pending
```

### View the last 12 full debate reasonings

```bash
python3 run.py --analysis
```

---

## Dashboard

Launch the interactive performance dashboard locally:

```bash
python3 -m streamlit run dashboard.py
```

Opens at `http://localhost:8501` and auto-refreshes every 30 seconds. Displays:

- Portfolio balance over time with win/loss markers
- P&L per trade
- Rolling directional accuracy vs baselines
- Confidence calibration chart
- Per-model accuracy breakdown
- Trade history table

---

## Running Tests

```bash
pytest tests/ -v
```

147 unit tests covering Brier score logic, RSI calculation, vote aggregation, payout math, calibration bins, and Polymarket odds parsing. No API calls required — all tests run locally.

---

## Project Structure

```
polysignal-btc-ai-trading/
├── run.py                    # Entry point — CLI orchestrator
├── dashboard.py              # Streamlit performance dashboard
│
├── polysignal/               # Core package
│   ├── collector.py          # BTCCollector — live data from Kraken + Polymarket
│   ├── debate.py             # DebateAgent — xAI × OpenAI × Claude (3-round debate)
│   ├── ensemble.py           # EnsembleAgent — fallback if XAI/OpenAI keys missing
│   ├── evaluator.py          # EvaluatorAgent — scores predictions vs oracle outcomes
│   ├── reporter.py           # Accuracy report — Brier, calibration, baselines
│   ├── simulator.py          # Paper trading simulator ($100k account)
│   ├── storage.py            # SQLite persistence
│   ├── models.py             # Pydantic data contracts
│   └── analyzer.py           # Single-model analyzer (legacy)
│
├── tests/                    # 147 unit tests (no API calls)
├── docs/
│   ├── report.md             # Conference paper
│   └── architecture.md      # System architecture writeup
├── scripts/
│   ├── session_start.sh      # AWS EC2 session setup
│   └── session_end.sh        # Emergency DB download
│
├── requirements.txt
├── .env.example
└── polysignal_btc.db         # SQLite database (auto-created)
```

---

## Cloud Deployment (Planned — AWS)

The 16-hour evaluation session in this project was run on a local machine with sleep prevention (`caffeinate -i`). For longer or continuous data collection, the system is designed to run on AWS EC2. The `scripts/` folder includes two shell scripts that fully automate the cloud session workflow for anyone who wants to use this setup.

### For cloud users — session workflow



**Start a session** — run at the beginning of each lab window:

```bash
cd polysignal-btc-ai-trading
./scripts/session_start.sh <EC2-IP> [path-to-key.pem]
```


**End a session early** — run if you need to stop before the script finishes (e.g., lab window is about to close):

```bash
./scripts/session_end.sh <EC2-IP> [path-to-key.pem]
```

This script:
1. Backs up your existing local database with a timestamp
2. Downloads the current database from EC2
3. Prints a quick stats summary (predictions scored, accuracy, balance)


## Cost Estimate

| Mode | Cost |
|------|------|
| Single prediction cycle | ~$0.006 |
| 1 hour (12 cycles) | ~$0.07 |
| 16-hour session (192 cycles) | ~$1.15 |

