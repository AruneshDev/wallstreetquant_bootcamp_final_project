Arunesh — this is already very strong. I’m going to clean it further and make it:

* More concise
* More institutional
* Less repetitive
* Cleaner structure
* Sharper positioning (especially around alpha claims and ML results)

This version reads like something a Point72 IAC reviewer would skim in 3–4 minutes and immediately understand your rigor.

---

# Semiconductor Alpha Research

**Point72 Investor Analyst Competition — Arunesh Lal | Feb 2026**

A fully reproducible quantitative research pipeline on **12 U.S. semiconductor stocks + 5 mega-cap tech names** over **Jan 2020 – Feb 2026 (~1,541 trading days)**.

This project follows a strict scientific workflow:

> **Hypothesis → Falsification → Pivot → Robust Strategy Design → Alpha Attribution → ML Diagnostics → Interactive Dashboard**

Deliverables:

* Cross-sectional momentum strategy (semi universe)
* Cointegration-based QCOM/MRVL pairs trade
* 50/50 combined portfolio with improved Sharpe and reduced drawdown
* Diagnostic ML framework (RF / GBM / Transformer / GNN)
* 7-page Streamlit research dashboard

---

# 1. Data & Universe

### Universe

**Semiconductors (12)**
`NVDA AMD AVGO TSM QCOM AMAT LRCX MU KLAC TXN ASML MRVL`

**Big Tech (5)**
`AAPL MSFT GOOGL META AMZN`

### Benchmarks

* `SOXX` — Semiconductor ETF
* `SPY` — S&P 500 ETF

### Data

* Period: 2020-01-03 → 2026-02-20
* Frequency: Daily OHLCV
* Source: Yahoo Finance (`yfinance`)
* Local Parquet caching for reproducibility

---

# 2. Research Narrative

## 2.1 Hypothesis — NVDA Leads the Sector

> “NVDA predicts other semiconductor stocks at 1–5 day lags.”

### Test

* Directed lead–lag analysis across all semi pairs
* Lags 1–5 days
* Multi-year sample

### Result

* No economically meaningful positive lift
* Strong same-day correlation, but no predictive lag

### Conclusion

Daily lead–lag alpha is **rejected**.
Observed structure is contemporaneous beta, not exploitable signal.

---

## 2.2 Pivot — Cross-Sectional Momentum

After falsification, a robustness sweep across lookback windows reveals a regime transition:

* 3–5d → reversal
* ~20d → transition
* 30–60d → persistent momentum
* Peak around 45d

### Strategy Design

* Rank 12 semis by 45-day trailing return (lagged 1 day)
* Long top 3, short bottom 3
* Equal-weight legs
* Daily rebalance
* Transaction costs included

---

# 3. Strategy Results

## 3.1 Cross-Sectional Momentum (2020–2026)

```
Annual Return  : 11.38%
Annual Vol     : 20.30%
Sharpe         : 0.561
Sortino        : 0.881
Max Drawdown   : -26.24%
Total Return   : 69.82%
Win Rate       : 50.5%
N days         : 1431
```

### Yearly Behavior

Strong years: 2021, 2024
Weak years: 2022, 2025

Momentum shows **clear regime dependence**.

---

## 3.2 QCOM/MRVL Cointegration Pairs Trade

### Pair Selection

* Engle–Granger sweep across all semi pairs
* Best pair: **QCOM/MRVL**
* Cointegration p-value ≈ 0.015

### Trading Rule

* Log spread z-score (120d rolling)
* Entry |z| > 1.5
* Exit |z| < 0.3
* Market-neutral positioning
* Transaction costs included

### Performance

```
Annual Return  : 12.17%
Annual Vol     : 22.73%
Sharpe         : 0.535
Sortino        : 0.708
Max Drawdown   : -37.20%
Total Return   : 37.20%
Trades         : 39
```

### Properties

* Beta vs SOXX ≈ 0
* Beta vs SPY ≈ 0
* R² ≈ 0

Returns driven by **relative value**, not market direction.

---

## 3.3 Combined Portfolio (50/50)

Strategy correlation ≈ –0.40

Equal-weight combination:

```
Annual Return :  8.60%
Annual Vol    : 11.79%
Sharpe        : 0.729
Max Drawdown  : -18.10%
```

Combining decorrelated trend and mean-reversion edges:

* Improves Sharpe
* Reduces drawdown
* Stabilizes regime sensitivity

---

# 4. Alpha Attribution

Regression framework:

[
R_p - R_f = \alpha + \beta (R_m - R_f) + \epsilon
]

### Full-Period Results

**CS Momentum vs SOXX**

* Alpha ≈ +5.6% / yr
* Beta ≈ 0.02
* R² ≈ 0.002

**Pairs vs SOXX/SPY**

* Alpha ≈ +5–7% / yr
* Beta ≈ 0
* R² ≈ 0

Interpretation:

* Both strategies are approximately market-neutral.
* Statistical power is limited by daily noise and finite sample.
* Stronger conclusion: **low beta and diversification benefit**, not guaranteed alpha.

Yearly alpha tables are generated via `src/alpha.py`.

---

# 5. ML Diagnostic Layer

ML is used as a diagnostic overlay — not primary alpha source.

## 5.1 Features

Per stock-day:

* Momentum: 1d, 5d, 10d, 20d, 60d
* Volatility: 5d, 20d
* Reversal (1d)
* Volume regime ratio
* Distance to 52w high/low
* RSI (normalized)
* MACD histogram
* Cross-sectional rank

Target: **5-day forward return**

---

## 5.2 Models

### Baselines

* Random Forest
* Gradient Boosting

Result: negative or near-zero IC.

### Transformer

* 20-day sequence
* 2 encoder layers
* d_model=32
* MPS acceleration

Result: no robust IC.

### Graph Neural Network

* Nodes: 12 semis
* Edges: 60-day rolling correlation (threshold 0.3)
* 2 GCN layers
* ~1.6k parameters

Result: small, unstable IC.

Conclusion:

With this feature set and daily horizon, ML does not uncover a robust additional edge beyond explicit strategies.

---

# 6. Streamlit Dashboard

Run:

```
streamlit run app.py
```

Pages:

1. Overview
2. EDA & Correlations
3. Lead–Lag Study
4. Cross-Sectional Momentum
5. Pairs Trade
6. Alpha Attribution
7. ML Diagnostics

Designed for transparent research review.

---

# 7. Repository Structure

```
semiconductor_quant_research/
├── app.py
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── backtest.py
│   ├── alpha.py
│   ├── model_baseline.py
│   ├── model_transformer.py
│   └── model_gnn.py
├── data/
├── results/
├── models/
├── charts/
├── requirements.txt
└── README.md
```

---

# 8. Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/data_loader.py
python src/features.py
python src/backtest.py
python src/model_baseline.py
python src/model_transformer.py
python src/model_gnn.py
python src/alpha.py

streamlit run app.py
```

---

# 9. Limitations

1. Six-year daily sample limits statistical power.
2. Simplified transaction cost model.
3. Momentum regime dependence.
4. ML IC near zero at this horizon.

---

# 10. Next Steps

* Explicit regime filters (rolling Sharpe / volatility states)
* Alternative labels (residual vs SOXX, longer horizon)
* Expanded universe
* Liquidity-aware cost model
* Residual momentum instead of raw returns

---

**Arunesh Lal**
M.S. Computer Science — Boston University
Quantitative Research | Systematic Trading | Machine Learning

---