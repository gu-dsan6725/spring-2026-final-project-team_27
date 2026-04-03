#### 1. Project Title

PolySignal: A Multi-Agent AI System for Bitcoin 5-Minute Prediction Market Forecasting

---

#### 2. Abstract

Prediction markets offer an ideal environment for evaluating AI forecasting accuracy, using binary outcomes, transparent resolution criteria, and high-frequency feedback loops that are difficult to replicate in traditional financial market settings. This project builds a tool called PolySignal, a multi-agent AI system that forecasts the direction of Bitcoin's price over 5-minute windows on Polymarket. This prediction market is currently one of the platform's most active market, with reported single-day trading volumes exceeding $60 million which proivdes enough data for training and validation.

The system implements three collaborating AI agents. A Collector agent continuously streams live BTC/USD price data via Polymarket's Real-Time Data Socket (powered by Chainlink oracles) and retrieves 1-minute OHLCV candles from Binance to compute technical indicators including RSI, short-term momentum across 1-minute, 5-minute, and 15-minute windows, volatility, and volume. A second Analyzer agent uses LLMs like Claude, ChatGPT, and Grok to reason over this structured market snapshot and produce a calibrated UP or DOWN directional forecast with an explicit confidence score. An Evaluator agent automatically scores each prediction after the 5-minute window closes by fetching the actual BTC close price from Binance, then computes directional accuracy and Brier scores per confidence bin.

The primary research contribution is the design and empirical evaluation of an LLM-based calibration pipeline applied to a high-frequency prediction market. The evaluation framework measures not just whether the system is directionally correct, but whether its stated confidence levels are trustworthy. All data sources used are publicly accessible with no authentication requirements except the APIs for LLM usage.

---

#### 3. Data Sources

| Source | Data | Auth Required |
|--------|------|---------------|
| Polymarket RTDS WebSocket | Live BTC/USD price (Chainlink oracle) | None |
| Polymarket Gamma API | Active 5-min markets, UP/DOWN odds | None |
| Binance REST API (`/api/v3/klines`) | 1-min OHLCV candles, close prices | None |
| Alternative.me Fear & Greed API | Daily crypto sentiment score (0–100) | None |
| Claude/ChatGPT/Gemini/Grok | LLM inference for Analyzer agent | API key |