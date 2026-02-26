# Semiconductor Alpha Research

**Point72 Investor Analyst Competition — Arunesh Lal | Feb 2026**

A fully reproducible quantitative research pipeline on **12 U.S. semiconductor stocks + 5 mega-cap tech names** over **Jan 2020 – Feb 2026 (~1,541 trading days)**.

This project follows a strict scientific workflow:

> **Hypothesis → Falsification → Pivot → Robust Strategy Design → Alpha Attribution → Alt-Data → NLP → ML Signal Combiner → Interactive Dashboard**

**Deliverables**

- Cross-sectional momentum strategy (semi universe)
- Cointegration-based QCOM/MRVL pairs trade
- 50/50 combined portfolio with improved Sharpe and reduced drawdown
- **Three universe tiers** (12 → ~80 → ~150 tickers) for cross-sectional IC research
- **Alternative data signals**: Earnings surprise (SUE), analyst revision proxy, short interest proxy
- **NLP / LLM signal**: SEC EDGAR 8-K earnings sentiment (sentence-transformers)
- **GBM signal combiner**: ML re-positioned as meta-model over IC-positive base signals
- Explicit **gross vs net** performance reporting (7 bps one-way + 2 bps slippage)
- 13-page Streamlit research dashboard

---

## 1. Data & Universe

### Universe Tiers (`src/universe.py`)

| Tier | N | Contents |
|------|---|----------|
| `semi_core` | 17 | 12 semis + 5 big tech (original strategy backtests) |
| `sp_tech_semi` | ~80 | S&P 500 Tech + Semi names (IC breadth research) |
| `r1000_tech` | ~150 | Russell-1000 tech proxy (broad alpha evaluation) |

**Core tickers**

- Semiconductors (12): `NVDA AMD AVGO TSM QCOM AMAT LRCX MU KLAC TXN ASML MRVL`
- Big Tech (5): `AAPL MSFT GOOGL META AMZN`

**Benchmarks**: `SOXX` (sector), `SPY` (market)

**Data**: Daily OHLCV from Yahoo Finance, cached as Parquet for reproducibility.

---

## 2. Research Narrative

### 2.1 Hypothesis — NVDA leads the sector (REJECTED)

> "NVDA predicts other semiconductor stocks at 1–5 day lags."

Directed lead–lag analysis across all semi pairs at lags 1–5: no economically meaningful positive lift. Observed structure is same-day beta, not exploitable signal.

**Conclusion**: Daily lead–lag alpha rejected. Pivot required.

---

### 2.2 Pivot — Cross-Sectional Momentum

After falsification a robustness sweep reveals a clear lookback structure:

- 3–5d → reversal zone
- ~20d → transition
- 30–60d → persistent momentum (peak ~45d)

**Strategy design**: Rank 12 semis by 45-day trailing return (lagged 1 day), long top-3 / short bottom-3, equal-weight, daily rebalance.

Regime dependence: strong in trend regimes (2021, 2024), suffers in choppy rotation (2022, 2025).

---

### 2.3 Pairs — cointegration-selected

Best pair by Engle–Granger sweep: **QCOM/MRVL**, p ≈ **0.0154**.

Log-spread z-score (120d rolling), entry |z| > 1.5, exit |z| < 0.3.

---

### 2.4 Alternative Data Signals (`src/features_alt.py`)

Three non-price/volume signals evaluated against 10-day forward cross-sectional returns. All signals use `.shift(1)` for leakage-free alignment.

| Signal | Alpha Hypothesis | Expected IC | Key Failure Mode |
|--------|-----------------|------------|-----------------|
| **SUE** (Earnings Surprise) | Post-earnings announcement drift (PEAD): beats on EPS consensus → continued outperformance 1–60d (Ball & Brown 1968; Bernard & Thomas 1989) | +0.03–0.08 | Crowding; stale signal decay |
| **ARM** (Analyst Revision Proxy) | Cumulative EPS surprise proxies upward estimate revisions → forward return momentum (Hawkins et al. 1984) | +0.01–0.04 | Low-quality proxy vs real IBES data |
| **SI Proxy** (Short Interest) | Low short interest (inverted) → less informed bearish conviction → outperformance (Asquith, Pathak & Ritter 2005) | +0.01–0.03 | Short squeezes; borrow cost not modelled |

Outputs: `data/features_alt.parquet`, `results/alt_signal_ic.csv`

---

### 2.5 NLP / LLM Signal (`src/nlp_signal.py`)

**Architecture**

1. Fetch quarterly 8-K filings from SEC EDGAR (CIK lookup for 12 semi tickers).
2. Embed with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 22M params, runs fully locally — no API key).
3. Project onto a positive/negative polarity axis calibrated with reference sentences → `nlp_sent` ∈ [-1, 1].
4. Compare embedding to year-ago filing → `nlp_drift` (tone-change signal).
5. Forward-fill from each earnings date; `.shift(1)` for leakage-free alignment.

**Alpha hypotheses**

- `nlp_sent`: Positive tone signals management confidence and accompanies estimate revisions (Loughran & McDonald 2011). Expected IC: +0.02–0.05 on 10d.
- `nlp_drift`: Tone *improvement* vs year-ago is incremental information beyond level. Expected IC: +0.01–0.03.

**Failure modes**: Management cheerleading; 8-K ≠ full transcript quality; IC t-stat is low on a 12-ticker universe.

Outputs: `data/features_nlp.parquet`, `results/nlp_ic.csv`

---

### 2.6 ML Signal Combiner (`src/model_signal_combiner.py`)

**Why original ML failed**: Predicting raw 5-day returns from OHLCV features on 12 tickers is near-impossible — IC was negative or near-zero across all models (RF, GBM, Transformer, GNN). The signal-to-noise ratio is too low.

**Re-positioning — ML as meta-model**

1. Evaluate each base signal individually (CS momentum rank, SUE, NLP sentiment, vol ratio, etc.).
2. A shallow GBM (depth=2, walk-forward retrained every 63 days, trained on 252-day windows) learns optimal combination weights.
3. Target: cross-sectional *rank* of 10-day forward return (more robust than raw level prediction).

**Theoretical justification** (Grinold & Kahn — Fundamental Law of Active Management):

$$\text{ICIR}_{\text{combined}} \approx \text{ICIR}_{\text{individual}} \times \sqrt{N_{\text{signals}}}$$

Combining 5 partially-decorrelated signals can improve ICIR by up to √5 ≈ 2.2× vs any single signal.

**Base signals used**: `cs_rank_mom45`, `reversal_1d`, `sue_decay`, `nlp_sent`, `vol_ratio`, `mom_20d`, `dist_52w_high`

Outputs: `results/signal_combiner_summary.csv`, `results/signal_weights.csv`, `results/individual_signal_ic.csv`, `results/signal_combiner_folds.csv`

---

### 2.7 ML Diagnostic Layer (original models)

Models evaluated on OHLCV feature panel (14 features, 5-day target):

| Model | IC | RankIC | Verdict |
|-------|----|----|---------|
| Random Forest | < 0 | < 0 | Anti-predictive |
| Gradient Boosting | < 0 | < 0 | Anti-predictive |
| Transformer | < 0 | ≈ 0 | No robust edge |
| GNN | ≈ 0.05 | ≈ 0.04 | Borderline (correlation graph helps) |

**Conclusion**: No exploitable OHLCV structure at daily horizon on 12 tickers. Correctly reported; ML is not asserted as a source of alpha.

---

## 3. Strategy Results

### 3.1 Cross-Sectional Momentum (45d) — NET of 7+2 bps

```
Annual Return  : 11.38%
Annual Vol     : 20.30%
Sharpe         : 0.561
Sortino        : 0.881
Max Drawdown   : -26.24%
Total Return   : 69.82%
Win Rate       : 50.5%
N days         : 1,431
```

Strong years: 2021, 2024. Weak years: 2022, 2025.

### 3.2 QCOM/MRVL Pairs Trade — NET

```
Annual Return  : 12.17%
Annual Vol     : 22.73%
Sharpe         : 0.535
Sortino        : 0.708
Max Drawdown   : -37.20%
Trades         : 39
Beta (SOXX/SPY): ≈ 0   R² ≈ 0.0002
```

Returns are pure relative-value — not exposed to market direction.

### 3.3 Combined Portfolio (50/50)

Correlation(CS, Pairs) ≈ **-0.40** → diversification benefit.

```
Annual Return  :  8.60%
Annual Vol     : 11.79%
Sharpe         :  0.729
Max Drawdown   : -18.10%
```

---

## 4. Alpha Attribution

Both strategies are effectively market-neutral (beta ≈ 0, R² ≈ 0.001–0.002 vs SOXX/SPY). Alpha t-stats are modest (< 1.65) due to finite sample and 12-ticker universe — flagged with ⚠️ in all outputs where p > 0.10. Alpha is never asserted without statistical backing.

---

## 5. Transaction Costs & Risk Realism

All backtests report **both gross and net** performance:

- `tcost_bps = 7` (one-way — covers bid-ask + market impact for liquid semis)
- `slippage_bps = 2` (additional market impact slippage)
- Both are function arguments, easily adjusted.

`run_capacity_check()` in `backtest.py` verifies positions stay within 10% of 10-day ADV (the threshold consistent with ≤7 bps cost).

---

## 6. Streamlit Dashboard

```bash
streamlit run app.py
```

**13 pages:**

| # | Page | Description |
|---|------|-------------|
| 1 | Overview | Universe, period, key metrics, combined stats |
| 2 | EDA & Correlations | Correlation matrices, rolling correlations |
| 3 | Lead–Lag Study | NVDA hypothesis and falsification |
| 4 | CS Momentum | Equity curve, drawdown, annual/monthly tables, robustness |
| 5 | Pairs Trade | Log-spread, z-score, trade annotations, PnL |
| 6 | Strategy Comparison | Side-by-side metrics |
| 7 | Alpha Attribution | Full-period and yearly alpha vs SOXX/SPY |
| 8 | Market Impact | Beta to sectors, shock days, risk contribution |
| 9 | ML Signal Analysis | IC/RankIC tables, feature importance, GNN diagnostics |
| 10 | 🌐 Universe Expansion | Tier comparison, IC breadth benefit, download guide |
| 11 | 📡 Alt-Data Signals | SUE/ARM/SI IC tables, hypotheses |
| 12 | 💬 NLP Signal | EDGAR sentiment IC, architecture description |
| 13 | 🔗 Signal Combiner | GBM meta-model IC, fold diagnostics, signal weights |

---

## 7. Repository Structure

```
semiconductor_quant_research/
├── app.py
├── src/
│   ├── universe.py                 ← NEW: universe tier definitions
│   ├── data_loader.py              ← UPDATED: multi-universe download/load
│   ├── features.py
│   ├── features_alt.py             ← NEW: SUE, ARM, SI alt-data signals
│   ├── nlp_signal.py               ← NEW: NLP/LLM sentence-transformer signal
│   ├── backtest.py                 ← UPDATED: gross/net costs, capacity check
│   ├── alpha.py
│   ├── evaluate.py
│   ├── industrial_correlation.py
│   ├── model_baseline.py
│   ├── model_transformer.py
│   ├── model_gnn.py
│   └── model_signal_combiner.py    ← NEW: GBM signal meta-model
├── data/
│   ├── prices.parquet                   (legacy semi_core)
│   ├── prices_semi_core.parquet
│   ├── prices_sp_tech_semi.parquet      (expanded ~80-ticker universe)
│   ├── prices_r1000_tech.parquet        (broad ~150-ticker universe)
│   ├── features.parquet
│   ├── features_alt.parquet             ← NEW: alt-data features
│   └── features_nlp.parquet             ← NEW: NLP features
├── results/
│   ├── alt_signal_ic.csv                ← NEW: alt-data IC evaluation
│   ├── nlp_ic.csv                       ← NEW: NLP signal IC
│   ├── individual_signal_ic.csv         ← NEW: per-signal IC benchmark
│   ├── signal_combiner_summary.csv      ← NEW: combiner vs individual IC
│   ├── signal_weights.csv               ← NEW: GBM signal importance
│   ├── signal_combiner_folds.csv        ← NEW: walk-forward fold metrics
│   └── ... (existing CSVs)
├── models/
├── charts/
├── requirements.txt
└── Readme.md
```

---

## 8. Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Core pipeline (original)
python src/data_loader.py
python src/features.py
python src/backtest.py
python src/alpha.py

# 2. Alt-data signal pipeline (NEW)
python src/features_alt.py

# 3. NLP signal pipeline (NEW — requires sentence-transformers)
python src/nlp_signal.py

# 4. ML signal combiner (NEW)
python src/model_signal_combiner.py

# 5. Original ML diagnostics
python src/model_baseline.py
python src/model_transformer.py
python src/model_gnn.py

# 6. Launch Streamlit app (13 pages)
streamlit run app.py
```

---

## 9. Requirements

```
pandas>=2.0
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3
torch>=2.1
statsmodels>=0.14
yfinance>=0.2
plotly>=5.18
streamlit>=1.32
kaleido>=0.2.1
sentence-transformers       # NLP signal
transformers                # NLP backbone
```

---

## 10. Limitations & Next Steps

**Current limitations**

1. **Scale**: Strategy backtests use 12–17 tickers; IC statistics require the 80-ticker `sp_tech_semi` universe to be statistically significant.
2. **Costs**: The 7+2 bps model is simplified; a full execution model (spreads, slippage, borrow, liquidity) would further reduce Sharpe.
3. **Regime sensitivity**: CS momentum shows clear regime dependence; no overlay filter yet.
4. **NLP data quality**: EDGAR 8-K is a proxy for full earnings call transcripts; IC would be higher with Refinitiv/FactSet data.
5. **Alpha t-stats**: Modest (< 1.65) for all strategies; correctly flagged as insignificant in outputs.

**Next steps**

- Run signal combiner on `sp_tech_semi` (80 tickers) for statistically significant IC.
- Replace SUE proxy with Compustat IBES actuals for institutional-grade signal quality.
- Replace EDGAR 8-K with full earnings call transcripts for the NLP signal.
- Add a **regime overlay** for CS momentum (rolling Sharpe gate, volatility/trend state).
- Upgrade execution modelling: spreads, slippage, borrow costs, liquidity-aware sizing.
- Explore binary classification labels ("top vs bottom quartile") for improved ML IC framing.

---

**Arunesh Lal**
M.S. Computer Science — Boston University
Quantitative Research | Systematic Trading | Machine Learning
