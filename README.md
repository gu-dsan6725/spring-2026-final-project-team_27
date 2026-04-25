# PolySignal-BTC5 — Adversarial Multi-Agent LLM Debate for Bitcoin Prediction Markets

> Can three AI models arguing with each other beat the crowd on Bitcoin's fastest prediction market?


---

## Results (16-Hour Evaluation Session · April 24, 2026)

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
| Code | [`polysignal-btc-ai-trading/`](polysignal-btc-ai-trading/) |
| Results & charts | [`results.qmd`](results.qmd) |
| System architecture | [`architecture.qmd`](architecture.qmd) |
| Presentation slides | [Google Slides](https://YOUR_GOOGLE_SLIDES_LINK_HERE) |
| Poster | [Google Slides Poster](https://YOUR_POSTER_LINK_HERE) |

---

## Quick Start

```bash
cd polysignal-btc-ai-trading
pip install -r requirements.txt
cp .env.example .env   # add API keys
python3 run.py         # one prediction cycle
python3 -m streamlit run dashboard.py  # view results dashboard
```

---

## Team

Lance HDY — Georgetown University DSAN 6725, Spring 2026
