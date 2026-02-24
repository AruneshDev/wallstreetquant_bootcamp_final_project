
# Semiconductor Alpha Research

### Point72 Investor Analyst Competition

**Arunesh Lal | February 2026**

A full end-to-end quantitative research pipeline applied to U.S. semiconductor equities over a live 535-day market window (Jan 2024 – Feb 2026).

The project follows the scientific method:

> **Hypothesis → Falsification → Regime Discovery → Strategy Design → ML Enhancement → Evaluation**

The result is:

* Two live, testable systematic strategies
* A Graph Neural Network alpha signal crossing institutional IC tradability thresholds
* A reproducible research and backtesting framework

---

## Research Summary

### Initial Hypothesis

**NVDA leads semiconductor equities by 1–5 trading days.**

Motivation: NVDA acts as the sector benchmark for AI capex and sentiment shocks.

### Falsification

* 17 tickers
* 272 directed lead–lag pairs
* 785 trading days
* Lags tested: 1–5 days

**Result:**

* Zero positive lift pairs
* Maximum observed lift: –0.20
* Same-day correlation (0.60–0.75) dominates

**Conclusion:**
Daily lead–lag alpha does not exist.
Hypothesis rejected.

This contradicts common practitioner intuition.

---

## Regime Discovery — Reversal to Momentum Transition

After rejecting lead–lag structure, a robustness sweep across momentum horizons revealed a clean regime crossover:

| Lookback | Sharpe    | Interpretation        |
| -------- | --------- | --------------------- |
| 3–5d     | Negative  | Short-term reversal   |
| 10–15d   | Near zero | Transition zone       |
| 20d      | Positive  | Momentum onset        |
| 45d      | Best      | Strongest persistence |
| 60d      | Positive  | Stabilized momentum   |

Momentum emerges at approximately **20 trading days**, consistent with microstructure research on sector persistence.

---

## Strategy Results

### 1️⃣ Cross-Sectional Momentum (45d)

* Long top 3 / short bottom 3 semiconductor stocks
* Ranked by trailing 45-day return
* Daily rebalance
* Equal-weight legs

**Performance (425 trading days)**

* Annual Return: 13.7%
* Annual Volatility: 23.3%
* Sharpe: 0.59
* Sortino: 0.92
* Max Drawdown: –18.9%
* Win Rate: 50.1%

**Yearly Behavior**

| Year | Sharpe | Notes                          |
| ---- | ------ | ------------------------------ |
| 2024 | 1.15   | Strong AI narrative regime     |
| 2025 | –0.52  | Sector rotation / regime break |
| 2026 | 4.57*  | Short sample                   |

Momentum is regime-dependent.

---

### 2️⃣ NVDA / TXN Mean-Reversion Pairs Trade

* Log spread z-score (120d rolling)
* Entry ±1.5σ
* Exit ±0.3σ

**Performance**

* Annual Return: 14.1%
* Annual Volatility: 25.8%
* Sharpe: 0.55
* Max Drawdown: –22.9%
* Trades: 11

More stable across regimes than cross-sectional momentum.

---

### 3️⃣ Combined Portfolio (50/50)

Reduces 2025 drawdown while preserving positive year-on-year Sharpe.

Demonstrates diversification between momentum and relative value structures.

---

## Machine Learning Layer

The ML objective:

> Predict next-day cross-sectional returns and evaluate Information Coefficient (IC).

Out-of-sample test period:
Apr 2025 – Feb 2026 (214 trading days)

### Model Comparison

| Model                    | IC        | ICIR      | RankIC    | IC Positive % |
| ------------------------ | --------- | --------- | --------- | ------------- |
| Random Forest            | –0.014    | –0.039    | –0.022    | 51.6%         |
| Gradient Boosting        | –0.009    | –0.030    | –0.003    | 50.8%         |
| Transformer              | 0.006     | 0.019     | 0.017     | 50.9%         |
| **Graph Neural Network** | **0.051** | **0.148** | **0.050** | **56.1%**     |

### Key Result

The **GNN** is the only model exceeding the institutional IC tradability threshold (≈0.05).

Why?

Because semiconductor stocks behave as a correlated contagion network — not independent time series.

Flat-feature and sequence-only models cannot encode:

* Sector co-movement
* Dynamic correlation structure
* Cross-stock propagation effects

The GNN explicitly models this structure.

---

## Feature Set (11 Per Stock Per Day)

* Momentum: 1d, 5d, 10d, 20d, 60d
* Volatility: 5d, 20d
* Reversal: 1d
* Volatility ratio regime indicator
* Distance to 52-week high
* Cross-sectional rank signal

Baseline model importance:

* `reversal_1d`
* `dist_52w_high`

Short-term microstructure signals dominate naive momentum ranking.

---

## GNN Architecture

* Nodes: 12 stocks
* Features: 11 per node
* Edges: Correlation-weighted (threshold 0.3)
* Rolling window: 60 days
* Normalized adjacency: D⁻¹/²(A + I)D⁻¹/²
* 2 GCN layers
* 1,601 parameters
* Pure PyTorch implementation
* Apple M-series MPS acceleration

Lightweight. Fully reproducible. No torch_geometric dependency.

---

## Project Structure

```
semiconductor_quant_research/
├── app.py
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── backtest.py
│   ├── model_baseline.py
│   ├── model_transformer.py
│   └── model_gnn.py
├── results/
├── models/
├── charts/
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/AruneshDev/wallstreetquant_bootcamp_final_project.git
cd semiconductor_quant_research

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python src/data_loader.py
python src/features.py
python src/backtest.py
python src/model_gnn.py

streamlit run app.py
```

---

## Data

* Source: Yahoo Finance via yfinance
* Universe: 12 semiconductor equities + 5 mega-cap tech
* Frequency: Daily adjusted close
* Period: Jan 2024 – Feb 2026

---

## Key Limitations

1. Limited 2-year sample
2. Transaction costs not modeled
3. Regime dependency in momentum strategy
4. Correlation window lag in GNN
5. Small universe (12 stocks)

This is a research prototype — not production deployment.

---

## Roadmap

* Regime detection overlay (rolling Sharpe filter / HMM)
* Expand universe to 50+ semiconductor-related equities
* Full transaction cost + slippage simulation
* Cointegration scan across broader pairs universe
* Alternative data integration (earnings revisions, SOXX flows)

---

## Author

**Arunesh Lal**
MS Computer Science — Boston University
Quantitative Research | Machine Learning | Systematic Trading
