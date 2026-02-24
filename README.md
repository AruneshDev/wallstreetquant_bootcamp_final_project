# Semiconductor Alpha Research

A fully reproducible quantitative research pipeline on **12 U.S.-listed semiconductor equities + 5 mega-cap tech names** over **2020-01-03 → 2026-02-20 (~1,541 trading days)**. The project follows a strict research workflow:

> **Hypothesis → Falsification → Pivot → Robust Strategies → Alpha Attribution → ML Diagnostics → Streamlit App**

**Deliverables**
- **Cross-sectional momentum (CS Mom)** strategy on semiconductors (45d lookback).
- **Cointegration-based QCOM/MRVL pairs trade** (Engle–Granger-selected).
- **50/50 combined portfolio** with improved Sharpe and reduced drawdown.
- Diagnostic **ML layer** (RF / GBM / Transformer / GNN) to test predictability and avoid false positives.

---

## 1) Data & Universe

**Universe**
- Semiconductors (12): `NVDA AMD AVGO TSM QCOM AMAT LRCX MU KLAC TXN ASML MRVL`
- Big Tech (5): `AAPL MSFT GOOGL META AMZN`

**Benchmarks**
- `SOXX` — semiconductor ETF (sector benchmark).
- `SPY`  — S&P 500 ETF (market benchmark).

**Data**
- Period: `2020-01-03 → 2026-02-20` (~1,541 trading days).
- Source: Yahoo Finance via `yfinance` (adjusted close + volume).
- Frequency: Daily OHLCV.
- Local caching: Parquet (`data/prices.parquet`, `data/returns.parquet`, `data/features.parquet`) for reproducibility and fast reruns.

---

## 2) Research Narrative (Scientific Workflow)

### 2.1 Hypothesis — NVDA as a sector leader

**Hypothesis**

> NVDA leads other semiconductor stocks by **1–5 trading days** at daily frequency.

**Test**

- Lead–lag study across directed semi pairs at lags **1–5 days**.
- Check whether yesterday’s NVDA return predicts today’s returns of others.

**Result**

- No economically meaningful positive lift at daily resolution.
- Strong same-day correlation across semis, but **lagged predictability is not exploitable**.

**Conclusion**

> Daily lead–lag alpha is **rejected**. The signal is same-day beta, not tradable lag.

---

### 2.2 Pivot — cross-sectional momentum (regime-dependent)

After falsifying lead–lag, the project pivots to **cross-sectional momentum** within the semi universe.

**Robustness sweep (lookbacks: 3, 5, 10, 15, 20, 30, 45, 60d)**

- 3–5d: **reversal zone** (negative Sharpe).
- ~20d: reversal → momentum transition.
- 30–60d: **momentum zone**, peak near **45d**.

**Core design**

> Long top-3 / short bottom-3 semis by **45d trailing return** (lagged 1 day), equal-weight, daily rebalance.

**Regime dependence**

- CS momentum is **strong in persistent trend regimes** (sector in “beta-on” mode).
- It **suffers in choppy rotation** regimes where winners mean-revert and leadership flips.
- This shows up cleanly in the year-by-year Sharpe and alpha (see 3.1).

Practical interpretation:

- CS momentum is best used as a **conditional allocation** (exposure gating) signal, not a constant-leverage strategy.
- A regime overlay (rolling Sharpe, volatility state, trend filter) is a natural next step (see section 10).

---

### 2.3 Pairs — stop guessing, use cointegration

Rather than hand-picking pairs, the project performs an **Engle–Granger cointegration sweep**:

- Test all unordered semi pairs over 2020–2026.
- Rank by p-value; choose the strongest cointegrated pair.

**Result**

- Best pair: **QCOM/MRVL**.
- Engle–Granger p-value ≈ **0.0154** (strong cointegration signal).

The strategy trades **log-spread z-score mean reversion** on QCOM/MRVL.

---

### 2.4 ML layer — diagnostic, not production

Models evaluated on the same engineered feature panel:

- Random Forest.
- Gradient Boosting.
- Transformer (temporal).
- GNN (correlation-weighted graph).

**Target**

- `fwd_ret_5d` (5-day forward return), chosen to reduce daily noise vs 1d prediction.

**Outcome (full-period run)**

- RF / GBM / Transformer: **negative IC, negative RankIC**.
- GNN: IC ≈ 0, RankIC slightly positive but small.

**Interpretation**

> With this feature set and horizon, ML does **not** reveal a robust incremental alpha beyond the transparent strategies. The ML stack is used as a **sanity check** against overfitting and “false ML alpha.”

---

## 3) Strategy Results

### 3.1 Strategy 1 — Cross-Sectional Momentum (45d)

**Definition**

- Universe: 12 semis.
- Daily:
  - Compute trailing **45d cumulative return** (lagged by 1 day).
  - Rank semis by 45d return.
  - **Long top 3 / Short bottom 3**, equal-weight per leg.
- Daily rebalance.
- Volatility targeting (after warm-up).
- Transaction costs included via simple turnover-based cost.

**Full-period performance (2020–2026)**

```text
Annual Return  : 11.38%
Annual Vol     : 20.30%
Sharpe         : 0.561
Sortino        : 0.881
Max Drawdown   : -26.24%
Total Return   : 69.82%
Win Rate       : 50.5%
N days         : 1431
```

**Year-by-year vs SOXX (Jensen’s alpha)**

- **2020:**  –5.0% return, Sharpe –0.55, alpha ≈ **–12.9%**
- **2021:** +43.3% return, Sharpe ≈ **2.0**, alpha ≈ **+32.3%**
- **2022:** –12.1% return, Sharpe ≈ –0.97, alpha ≈ **–23.5%**
- **2023:**  –1.0% return, Sharpe ≈ –0.32, alpha ≈ **–10.5%**
- **2024:** +28.7% return, Sharpe ≈ **1.09**, alpha ≈ **+22.7%**
- **2025:** –11.4% return, Sharpe ≈ –0.76, alpha ≈ **–18.0%**
- **2026:** +147.7% annualised return (partial year), Sharpe ≈ **4.4**, alpha ≈ **+70.4%**

**Takeaways (CS momentum)**

- Strong momentum regimes: **2021, 2024, early 2026**.
- Difficult regimes: **2022, 2025** (rotation/chop).
- Supports applying a **regime filter** rather than running full exposure at all times.

---

### 3.2 Strategy 2 — QCOM/MRVL Pairs Trade

**Pair selection**

- Engle–Granger cointegration across all semi pairs.
- Best pair: **QCOM/MRVL** (p ≈ 0.0154).

**Trading rule**

- Log-spread: `log(QCOM) - log(MRVL)`.
- 120d rolling mean/std → z-score.
- Entry: `|z| > 1.5`.
- Exit:  `|z| < 0.3`.
- Positions:
  - z > +1.5: short QCOM, long MRVL.
  - z < –1.5: long QCOM, short MRVL.
- Transaction costs applied per position change.

**Full-period performance**

```text
Annual Return  : 12.17%
Annual Vol     : 22.73%
Sharpe         : 0.535
Sortino        : 0.708
Max Drawdown   : -37.20%
Total Return   : 37.20%
Win Rate       : 51.2%
N days         : 832   (active days)
Trades         : 39
```

**Year-by-year vs SOXX (Jensen’s alpha)**

- **2020:**  +73.3% return, Sharpe ≈ 2.8, alpha ≈ **+70.1%**
- **2021:**   +3.6% return, alpha ≈ **+2.3%**
- **2022:**  +23.2% return, alpha ≈ **+15.9%**
- **2023:** –14.2% return, alpha ≈ **–2.5%**
- **2024:** –31.3% return, alpha ≈ **–37.3%**
- **2025:**  +68.6% return, alpha ≈ **+62.7%**
- **2026:**  +67.4% annualised (9 days), alpha ≈ **+66.4%** (partial year)

**Properties**

- Full-period vs SOXX:
  - Annual return ≈ **12.2%**
  - Sharpe (excess vs cash) ≈ **0.31**
  - Jensen’s alpha ≈ **+7.35%/yr**
  - Beta ≈ **–0.01**
  - R² ≈ **0.0002**
- Similar profile vs SPY: alpha ≈ +5.8%/yr, beta ≈ 0.05, R² ≈ 0.0016.

Interpretation:

- The QCOM/MRVL spread is a **true relative-value trade**:
  - **Near-zero beta** to SOXX and SPY.
  - Returns are driven by spread mean reversion, not market direction.
  - Regime-sensitive: performs very well in 2020, 2022, 2025; poorly in 2023–2024 when the spread relationship breaks during the AI melt-up.

---

### 3.3 Combined Portfolio — 50/50 CS + Pairs

The two strategies have a **correlation ≈ –0.40** on overlapping days:

- Corr(CS Momentum, Pairs Trade) ≈ **–0.404**

Define:

- \(R_{\text{combined}} = 0.5 \times R_{\text{CS}} + 0.5 \times R_{\text{Pairs}}\)

**Full-period combined metrics**

```text
Annual Return :  8.60%
Annual Vol    : 11.79%
Sharpe        : 0.729
Max Drawdown  : -18.10%
Corr(CS,Pairs): -0.404
```

**Year-by-year vs SOXX (Jensen’s alpha)**

- **2020:**  +29.4% return, Sharpe ≈ 1.61, alpha ≈ **+23.2%**
- **2021:**  +31.6% return, Sharpe ≈ 2.54, alpha ≈ **+27.1%**
- **2022:**   –5.2% return, Sharpe ≈ –0.95, alpha ≈ **–10.5%**
- **2023:**   –7.2% return, Sharpe ≈ –1.12, alpha ≈ **–8.7%**
- **2024:**   –6.7% return, Sharpe ≈ –1.05, alpha ≈ **–12.2%**
- **2025:**  +19.2% return, Sharpe ≈ 1.06, alpha ≈ **+13.3%**
- **2026:**  +74.7% annualised (9 days), Sharpe ≈ 4.68, alpha ≈ **+60.7%**

Result:

> The **combined portfolio** offers a better risk-return profile (Sharpe ≈ 0.73, max drawdown ≈ –18%) than running either CS momentum or pairs in isolation, thanks to the negative correlation between the two legs.

---

## 4) Alpha Attribution (Full-period)

For each strategy vs each benchmark (`SOXX`, `SPY`), run:

\[
R_p - R_f = \alpha + \beta (R_m - R_f) + \epsilon
\]

**CS Momentum (45d)**

- vs **SOXX**
  - Annual return: **11.4%**
  - Jensen’s alpha: **+5.6%/yr**
  - Sharpe (excess vs cash): **0.31**
  - Beta: **0.024**
  - R²: **0.0017**
- vs **SPY**
  - Annual return: **11.4%**
  - Jensen’s alpha: **+6.7%/yr**
  - Sharpe (excess vs cash): **0.31**
  - Beta: **–0.043**
  - R²: **0.0013**

**QCOM/MRVL Pairs**

- vs **SOXX**
  - Annual return: **12.2%**
  - Jensen’s alpha: **+7.35%/yr**
  - Sharpe (excess vs cash): **0.31**
  - Beta: **–0.0095**
  - R²: **0.0002**
- vs **SPY**
  - Annual return: **12.2%**
  - Jensen’s alpha: **+5.77%/yr**
  - Sharpe (excess vs cash): **0.31**
  - Beta: **0.055**
  - R²: **0.0016**

Interpretation:

- Both strategies are **effectively market-neutral** with respect to SOXX and SPY (beta and R² both near zero).
- Alpha t-stats (< 1) reflect limited sample and high daily noise; the stronger conclusion is:
  - The strategies generate **decorrelated returns** versus market and sector.
  - A **risk-budgeted combination** (50/50) is more attractive than simply adding more beta.

Key outputs:

- `results/alpha_decomposition.csv`
- `results/*yearly_alpha*.csv`

---

## 5) ML Diagnostics (Non-Production)

### 5.1 Features (`src/features.py`)

Features per (date, ticker) for semis:

- **Momentum**: `mom_1d, mom_5d, mom_10d, mom_20d, mom_60d`
- **Volatility**: `vol_5d, vol_20d`
- **Reversal**: `reversal_1d`
- **Volume regime**: `vol_ratio` (5d / 20d average volume)
- **52-week location**: `dist_52w_high, dist_52w_low`
- **Oscillators**: `rsi_norm, macd_hist`
- **Cross-sectional**: `cs_rank_mom10` (percentile rank of 10d momentum across semis)

Target:

- `fwd_ret_5d` — next 5-day cumulative return.

---

### 5.2 Models

**Random Forest / Gradient Boosting**

- Walk-forward splits (rolling train/test).
- Metrics: R², IC, RankIC, ICIR.
- Both show **negative IC and RankIC** in OOS → no edge.

**Transformer**

- Input: 14-dim feature vector projected to 32-dim.
- 2× TransformerEncoderLayer (d_model=32, heads=4, ff=128).
- Sequence length: 20 days.
- Device: Apple MPS.
- Result: **negative IC**; no reliable signal.

**GNN**

- Nodes: 12 semis.
- Edges: 60d rolling correlation (> 0.3), weighted adjacency.
- GCN architecture: 14 → 32 → 32 → 1 (~1.6k params).
- Result: IC ≈ 0, RankIC slightly positive but small and unstable.

Conclusion:

> On this universe and horizon, the ML layer is useful as a **diagnostic** that confirms there is limited predictable structure beyond the explicit CS momentum and pairs trade.

---

## 6) Streamlit App (`app.py`)

Launch:

```bash
streamlit run app.py
```

Pages:

1. **Overview** — universe, period, key metrics, best pair, combined portfolio stats.
2. **EDA & Correlations** — correlation matrices, rolling correlations.
3. **Lead–Lag Study** — NVDA-leads hypothesis and its falsification.
4. **CS Momentum** — equity curve, drawdown, annual and monthly tables, robustness sweep.
5. **Pairs Trade** — log-spread, z-score, trades, PnL breakdowns.
6. **Alpha Attribution** — full-period and yearly alpha vs SOXX/SPY; combined portfolio.
7. **ML Diagnostics** — IC/RankIC tables, feature importance, narrative on why ML doesn’t add here.

---

## 7) Repository Structure

```text
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
│   ├── prices.parquet
│   ├── volume.parquet
│   ├── returns.parquet
│   └── features.parquet
├── results/
│   ├── cs_momentum_robustness.csv
│   ├── cs_momentum_annual.csv
│   ├── cs_momentum_monthly.csv
│   ├── pairs_annual.csv
│   ├── pairs_monthly.csv
│   ├── alpha_decomposition.csv
│   ├── strategy_correlation.csv
│   ├── cs_rolling_alpha.csv
│   ├── pairs_rolling_alpha.csv
│   ├── cs_yearly_alpha_soxx.csv
│   ├── cs_yearly_alpha_spy.csv
│   ├── pairs_yearly_alpha_soxx.csv
│   ├── pairs_yearly_alpha_spy.csv
│   ├── combined_yearly_alpha.csv
│   ├── rf_ic.csv / gbm_ic.csv / transformer_ic.csv / gnn_ic.csv
│   ├── *_rank_ic.csv
│   └── rf_feature_importance.csv
├── models/
│   ├── transformer.pt
│   └── gnn.pt
├── charts/
├── requirements.txt
└── README.md
```

---

## 8) Quickstart

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

## 9) Requirements

```text
pandas>=2.0
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3
torch>=2.1
statsmodels>=0.14
yfinance>=0.2
plotly>=5.18
streamlit>=1.32
kaleido>=1.0.0
```

---

## 10) Limitations & Next Steps

**Limitations**

1. **Scale**: 12 semis over ~6 years is solid but not institutional breadth; alpha t-stats are modest.
2. **Costs**: transaction costs are modeled simply; a full execution model (spreads, slippage, borrow, liquidity) would reduce Sharpe.
3. **Regime sensitivity**: CS momentum and the QCOM/MRVL spread both show strong regime dependence (e.g., 2021/2024/2025 vs 2022/2023/2024).
4. **ML**: Near-zero IC suggests limited incremental predictability with the current features/labels on a 5-day horizon.

**Next steps**

- Add a **regime overlay** for CS momentum (rolling Sharpe gate, volatility/trend state).
- Explore alternative labels: residual returns vs SOXX, 10-day horizon, or binary “top vs bottom decile” classification.
- Expand the universe (more semis + supply-chain neighbours) for more robust cross-sectional IC.
- Upgrade execution modeling: spreads, slippage, borrow costs, and liquidity-aware position sizing.

---

*Author: Arunesh Lal — M.S. Computer Science, Boston University*
```
