```markdown
# Semiconductor Alpha Research

### Point72 Investor Analyst Competition — Arunesh Lal | Feb 2026

A full quantitative research pipeline applied to 12 US-listed semiconductor
equities over a 535-day live market period (Jan 2024 – Feb 2026). The project
follows the scientific method: hypothesis → falsification → pivot → signal
extraction → ML layer, producing two live strategies and a GNN-based alpha
signal that crosses the institutional IC tradability threshold.

**Live demo**: `streamlit run app.py`

---

## Research Narrative

### Step 1 — Hypothesis

*NVDA leads other semiconductor stocks by 1–5 days at daily frequency.*

Motivated by NVDA's outsized weight in AI capex announcements and its role as
the de-facto benchmark for sector sentiment.

### Step 2 — Falsification

Tested all 272 directed pairs across 17 tickers over 785 trading days.

    Result : Zero pairs with positive lift at any lag 1–5d
    Max lift: –0.20
    Verdict : Daily lead-lag alpha REJECTED

Most practitioners assume NVDA leads — the data says it does not at daily
resolution. Same-day correlation (r ≈ 0.60–0.75) is the dominant signal and
is not exploitable after transaction costs.

### Step 3 — Pivot to Momentum

A robustness sweep across 8 lookback windows revealed a clean
reversal → momentum regime transition:

| Window | Sharpe | Zone             |
|--------|--------|------------------|
| 3d     | –0.666 | Reversal         |
| 5d     | –1.448 | Reversal         |
| 10d    | –0.168 | Transition       |
| 15d    | –0.304 | Transition       |
| **20d**| **+0.520** | **Momentum starts** |
| **45d**| **+0.590** | **Best window** |
| 60d    | +0.478 | Momentum         |

The reversal → momentum crossover occurs at ~20 days, consistent with
microstructure literature on semiconductor sector dynamics.

### Step 4 — ML Layer

Four models evaluated on an identical OOS test set (Apr 2025 – Feb 2026, 214 days):

| Model             | IC       | ICIR   | RankIC  | IC pos% |
|-------------------|----------|--------|---------|---------|
| Random Forest     | –0.01390 | –0.039 | –0.0215 | 51.6%   |
| Gradient Boosting | –0.00900 | –0.030 | –0.0030 | 50.8%   |
| Transformer       | +0.00633 | +0.019 | +0.0170 | 50.9%   |
| **GNN**           | **+0.05081** | **+0.148** | **+0.0496** | **56.1%** |

The GNN is the only model crossing the institutional tradability threshold
(IC > 0.05). The correlation-weighted graph structure captures semiconductor
sector contagion dynamics that flat-feature and sequential models cannot encode.

---

## Strategy Results

### Strategy 1 — Cross-Sectional Momentum (45d)

Long top 3 / short bottom 3 semis ranked by trailing 45-day return.
Daily rebalance, equal-weight legs.

```text
Annual Return  :  13.72%
Annual Vol     :  23.25%
Sharpe         :   0.590
Sortino        :   0.923
Max Drawdown   : -18.87%
Total Return   :  20.44%
Win Rate       :  50.1%
Period         :  425 days
```

**Annual breakdown**

| Year | Total Ret | Sharpe | Notes |
|------|-----------|--------|-------|
| 2024 | +14.0%    | 1.145  | Strong AI narrative momentum |
| 2025 | –12.8%    | –0.522 | Regime break — DeepSeek shock, sector rotation |
| 2026 | +21.2%    | 4.576  | Only 34 days — not statistically meaningful |

### Strategy 2 — NVDA / TXN Pairs Trade

Log-spread mean reversion. Entry ±1.5σ, exit ±0.3σ, 120-day rolling
z-score normalisation.

```text
Annual Return  :  14.09%
Annual Vol     :  25.82%
Sharpe         :   0.546
Sortino        :   0.836
Max Drawdown   : -22.88%
Total Return   :  10.05%
Win Rate       :  49.1%
Trades         :  11
```

More regime-stable than CS Momentum — positive Sharpe in both full calendar
years (2024: 1.007, 2025: 0.530).

### Combined Portfolio (50/50 Equal Weight)

Combining both strategies reduces the 2025 momentum drawdown while preserving
the pairs trade's year-on-year consistency.

---

## ML Architecture

### Features (11 per stock per day)

```text
mom_1d, mom_5d, mom_10d, mom_20d, mom_60d   momentum at 5 horizons
vol_5d, vol_20d                              short / medium volatility
reversal_1d                                  overnight reversal
vol_ratio                                    vol_5d / vol_20d regime indicator
dist_52w_high                                proximity to 52-week high
cs_rank_mom10                                cross-sectional rank signal
```

Feature importance (RF): `reversal_1d` (10.1%) and `dist_52w_high` (8.2%)
dominate — microstructure and trend signals outweigh momentum rank.

### GNN Architecture

```text
Input  : node features  (12 stocks × 11 features)
Adj    : D^{-1/2} (A + I) D^{-1/2}
         correlation-weighted edges, threshold = 0.3
         60-day rolling window, recomputed daily
Conv 1 : GCNLayer(11 → 32) + ELU + LayerNorm + Dropout(0.1)
Conv 2 : GCNLayer(32 → 32) + ELU + LayerNorm
Head   : Linear(32 → 1) — predicted next-day return per stock
Params : 1,601
Device : Apple MPS (M-series)
```

Pure PyTorch — no torch_geometric dependency.

### Transformer Architecture

```text
Input proj : Linear(11 → 32)
Pos embed  : nn.Embedding(100, 32)
Encoder    : 2 × TransformerEncoderLayer
             d_model=32 | heads=4 | ff=128 | pre-norm
Head       : LayerNorm → Dropout(0.1) → Linear(32 → 1)
Seq len    : 20 days lookback
Params     : 29,089
Device     : Apple MPS (M-series)
```

---

## Repository Structure

```text
semiconductor_quant_research/
│
├── app.py                         7-page Streamlit demo
│
├── src/
│   ├── data_loader.py             yfinance download + cache
│   ├── features.py                feature engineering — 6,420 × 12 panel
│   ├── backtest.py                CS momentum + pairs trade + reporting
│   ├── plots.py                   13 static charts → charts/
│   ├── model_baseline.py          Random Forest + Gradient Boosting
│   ├── model_transformer.py       Temporal Transformer (MPS-accelerated)
│   └── model_gnn.py               Graph Neural Network (pure PyTorch)
│
├── results/
│   ├── cs_momentum_robustness.csv
│   ├── cs_momentum_annual.csv
│   ├── cs_momentum_monthly.csv
│   ├── pairs_annual.csv
│   ├── pairs_monthly.csv
│   ├── rf_ic.csv
│   ├── rf_rank_ic.csv
│   ├── gbm_ic.csv
│   ├── gbm_rank_ic.csv
│   ├── transformer_ic.csv
│   ├── transformer_rank_ic.csv
│   ├── gnn_ic.csv
│   ├── gnn_rank_ic.csv
│   ├── rf_feature_importance.csv
│   └── ml_baseline_comparison.csv
│
├── models/
│   ├── transformer.pt
│   └── gnn.pt
│
├── charts/                        13 PNG charts
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and create environment
git clone https://github.com/AruneshDev/wallstreetquant_bootcamp_final_project.git
cd semiconductor_quant_research
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline in order
python src/data_loader.py         # download + cache price data
python src/features.py            # build feature panel
python src/backtest.py            # run strategies + save results
python src/plots.py               # generate 13 PNG charts
python src/model_baseline.py      # RF + GBM IC evaluation
python src/model_transformer.py   # Transformer IC evaluation
python src/model_gnn.py           # GNN IC evaluation (best model)

# 4. Launch Streamlit app
streamlit run app.py
```

---

## Requirements

```text
pandas>=2.0
numpy>=1.24
torch>=2.1
scikit-learn>=1.3
scipy>=1.11
yfinance>=0.2
plotly>=5.18
streamlit>=1.32
kaleido>=1.0.0
```

---

## Data

| Field    | Detail |
|----------|--------|
| Source   | Yahoo Finance via `yfinance` |
| Universe | 12 semiconductors: NVDA AMD AVGO TSM QCOM AMAT LRCX MU KLAC TXN ASML MRVL + 5 big tech: AAPL MSFT GOOGL META AMZN |
| Period   | 2024-01-03 → 2026-02-20 (535 trading days) |
| Frequency| Daily OHLCV, adjusted close |

---

## Key Limitations

1. **Short history (2 years)** — Walk-forward IC estimates have wide confidence
   intervals. A 5–10 year backtest is required before live deployment.

2. **Transaction costs not modelled** — CS Momentum rebalances daily across
   6 legs. At realistic bid-ask spreads (~5–10 bps per leg for mid-cap semis),
   net Sharpe would decline meaningfully.

3. **Train/test sample imbalance (Transformer)** — 321 training days with a
   20-day sequence length gave limited samples. Transfer learning from a
   broader equity universe would improve IC.

4. **GNN correlation window** — The 60-day rolling adjacency matrix is
   look-ahead-free but lagged. During fast regime shifts (e.g. DeepSeek shock,
   Feb 2025), the graph reflects stale correlations for up to 60 days.

5. **2025 regime break** — CS Momentum Sharpe dropped from +1.15 (2024) to
   –0.52 (2025). Momentum strategies are regime-dependent; an HMM or rolling
   Sharpe filter would improve capital allocation.

---

## Next Steps

- [ ] Regime detection overlay — cut CS Momentum exposure when rolling 63d Sharpe < 0
- [ ] GNN signal as position-sizing multiplier on CS Momentum
- [ ] Expand universe to 50+ semis for statistically reliable IC estimates
- [ ] Add alternative data: earnings surprise, analyst revision momentum, SOXX flow
- [ ] Full transaction cost model with realistic slippage simulation
- [ ] Cointegration test across broader pairs universe beyond NVDA/TXN

---

*Author: Arunesh Lal | MS Computer Science, Boston University*
```
***
