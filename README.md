````markdown
# Semiconductor Alpha Research

**Point72 Investor Analyst Competition — Arunesh Lal | Feb 2026**

A fully reproducible quantitative research pipeline on **12 U.S.-listed semiconductor equities + 5 mega-cap tech names** over **2020-01-03 → 2026-02-20 (~1,541 trading days)**. The project follows a strict research workflow:

> **Hypothesis → Falsification → Pivot → Robust Strategies → Alpha Attribution → ML Diagnostics → Streamlit App**

**Deliverables**
- **Cross-sectional momentum (CS Mom)** strategy on semiconductors (45d lookback).
- **Cointegration-based QCOM/MRVL pairs trade** (Engle–Granger selected).
- **50/50 combined portfolio** with improved Sharpe and reduced drawdown.
- Diagnostic **ML layer** (RF / GBM / Transformer / GNN) to test predictability and avoid false positives.

---

## 1) Data & Universe

**Universe**
- Semiconductors (12): `NVDA AMD AVGO TSM QCOM AMAT LRCX MU KLAC TXN ASML MRVL`
- Big Tech (5): `AAPL MSFT GOOGL META AMZN`

**Benchmarks**
- `SOXX` — Semiconductor ETF (sector benchmark)
- `SPY` — S&P 500 ETF (market benchmark)

**Data**
- Period: `2020-01-03 → 2026-02-20` (~1,541 trading days)
- Source: Yahoo Finance via `yfinance` (adjusted close + volume)
- Frequency: Daily OHLCV
- Local caching: Parquet (`data/prices.parquet`, `data/returns.parquet`, `data/features.parquet`) for reproducibility and fast reruns

---

## 2) Research Narrative (Scientific Workflow)

### 2.1 Hypothesis — NVDA as a sector leader

**Hypothesis**
> NVDA leads other semiconductor stocks by **1–5 trading days** at daily frequency.

**Test**
- Lead–lag study across directed semi pairs at lags **1–5 days**
- Check whether yesterday’s NVDA return predicts today’s returns of others

**Result**
- No economically meaningful positive lift at daily resolution
- Strong same-day correlation across semis, but **lagged predictability is not exploitable**

**Conclusion**
> Daily lead–lag alpha is **rejected**. The signal is same-day beta, not tradable lag.

---

### 2.2 Pivot — Cross-sectional momentum (regime-dependent)

After falsifying lead–lag, the project pivots to **cross-sectional momentum** within the semi universe.

**Robustness sweep (lookbacks: 3, 5, 10, 15, 20, 30, 45, 60d)**
- 3–5d: **reversal zone** (negative Sharpe)
- ~20d: reversal → momentum transition
- 30–60d: **momentum zone**, peak near **45d**

**Core design**
> Long top-3 / short bottom-3 semis by **45d trailing return** (lagged 1 day), equal-weight, daily rebalance.

**Regime dependence (what actually happens in the tape)**
- The CS momentum edge is **strong in persistent trend regimes** (e.g., semiconductor “beta-on” / narrative-driven leadership).
- The edge degrades during **choppy mean-reversion / rotation regimes** where winners mean-revert and leadership flips.
- This shows up clearly year-by-year: strong positive Sharpe in trend years and negative Sharpe in rotation years (see §3.1).

Practical interpretation:
- CS momentum should be treated as a **conditional allocation strategy** (exposure gating), not a constant-leverage “set and forget” signal.
- A regime overlay (rolling Sharpe / volatility state / trend filter) is a natural next step (see §10).

---

### 2.3 Pairs — stop guessing, use cointegration

Instead of hand-picking pairs, the project runs an **Engle–Granger cointegration sweep**:

- All unordered semi pairs tested over 2020–2026
- Rank by p-value and select the strongest pair

**Result**
- Best pair: **QCOM/MRVL**
- Engle–Granger p-value ≈ **0.0154** (strong cointegration evidence)

The strategy trades **log-spread z-score mean reversion** on QCOM/MRVL.

---

### 2.4 ML layer — diagnostic, not production

Models evaluated on the same engineered feature panel:

- Random Forest
- Gradient Boosting
- Transformer (temporal)
- GNN (correlation-weighted graph)

**Target**
- `fwd_ret_5d` (5-day forward return), chosen to reduce daily noise vs 1d prediction.

**Outcome**
- RF / GBM / Transformer: **negative → near-zero IC**
- GNN: **near-zero IC** in full-period runs

**Interpretation**
> With this feature set and horizon, ML does **not** reveal a robust incremental alpha beyond transparent strategies. The ML stack primarily serves as a **sanity-check** against overfitting and “false ML alpha.”

---

## 3) Strategy Results

### 3.1 Strategy 1 — Cross-Sectional Momentum (45d)

**Definition**
- Universe: 12 semis
- Daily:
  - Compute trailing **45d cumulative return** (lagged by 1 day)
  - Rank semis by 45d return
  - **Long top 3 / Short bottom 3**, equal-weight per leg
- Daily rebalance
- Volatility targeting (warm-up)
- Transaction costs included as a per-turnover cost

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
````

**Regime dependence — year-by-year behavior (returns, Sharpe, SOXX alpha)**
Per-year alpha vs SOXX (Jensen’s alpha):

* **2020:** –5.0% return, Sharpe –0.55, alpha ≈ **–12.9%**
* **2021:** +43.3% return, Sharpe ≈ **2.0**, alpha ≈ **+32.3%**
* **2022:** –12.1% return, Sharpe ≈ –0.97, alpha ≈ **–23.5%**
* **2023:** –1.0% return, Sharpe ≈ –0.32, alpha ≈ **–10.5%**
* **2024:** +28.7% return, Sharpe ≈ **1.09**, alpha ≈ **+22.7%**
* **2025:** –11.4% return, Sharpe ≈ –0.76, alpha ≈ **–18.0%**
* **2026:** partial year, very strong but not statistically stable

**Takeaways (CS momentum)**

* Strong momentum regimes: **2021, 2024** (high Sharpe, positive alpha)
* Rotation / chop regimes: **2022, 2025** (negative Sharpe, negative alpha)
* This supports a **regime filter overlay** (e.g., rolling Sharpe gate) rather than constant exposure.

---

### 3.2 Strategy 2 — QCOM/MRVL Pairs Trade

**Pair selection**

* Engle–Granger cointegration sweep over all semi pairs
* Best pair: **QCOM/MRVL** (p ≈ **0.0154**)

**Trading rule**

* Log-spread:  `log(QCOM) - log(MRVL)`
* 120d rolling mean/std → z-score
* Entry: `|z| > 1.5`
* Exit:  `|z| < 0.3`
* Positioning:

  * z > +1.5: short QCOM, long MRVL
  * z < –1.5: long QCOM, short MRVL
* Transaction costs modeled per position change

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

**Year-by-year (vs SOXX alpha)**

* **2020:** +73.3% return, Sharpe ≈ 2.8, alpha ≈ **+70.1%**
* **2021:** +3.6% return, alpha ≈ **+2.3%**
* **2022:** +23.2% return, alpha ≈ **+15.9%**
* **2023:** –14.2% return, alpha ≈ **–2.5%**
* **2024:** –31.3% return, alpha ≈ **–37.3%**
* **2025:** +68.6% return, alpha ≈ **+62.7%**

**Properties**

* Beta vs SOXX/SPY ~ 0 over full period
* R² vs SOXX/SPY ≈ 0.0–0.002
* Performance driven by **relative value**, not market direction

---

### 3.3 Combined Portfolio — 50/50 CS + Pairs

The two strategies exhibit **negative correlation ≈ –0.40** on common active days.

Portfolio:

* `R_port = 0.5 * R_cs_mom + 0.5 * R_pairs`

**Full-period combined metrics**

```text
Annual Return :  8.60%
Annual Vol    : 11.79%
Sharpe        : 0.729
Max Drawdown  : -18.10%
Corr(CS,Pairs): -0.404
```

Result:

> Combining decorrelated edges (trend-following CS momentum + mean-reversion spreads) **improves risk-adjusted returns** and **reduces drawdowns** vs either standalone strategy.

---

## 4) Alpha Attribution

For each strategy vs each benchmark (`SOXX`, `SPY`):

[
R_p - R_f = \alpha + \beta(R_m - R_f) + \epsilon
]

Reported:

* Annualized return & Sharpe
* Jensen’s alpha (annualized)
* t-stat, p-value
* beta & R²
* tracking error & information ratio

**Key results**

* CS Momentum vs SOXX: alpha ≈ **+5.6%/yr**, beta ≈ **0.02**, R² ≈ **0.002**
* CS Momentum vs SPY:  alpha ≈ **+6.7%/yr**, beta ≈ **–0.04**, R² ≈ **0.001**
* Pairs vs SOXX/SPY:   alpha ≈ **+5–7%/yr**, beta ~ **0**, R² ~ **0**

Interpretation:

* Both strategies are **close to market-neutral** vs SOXX and SPY.
* Alpha significance is limited (t-stats < 2) due to finite horizon + daily noise.
* Stronger claim: **decorrelation + risk-adjusted portfolio construction**, not “guaranteed alpha.”

**Outputs**

* `results/*yearly_alpha*.csv`
* `results/alpha_decomposition.csv`

---

## 5) ML Diagnostics (Non-Production)

### 5.1 Features (`src/features.py`)

Per (date, ticker) for semis:

* Momentum: `mom_1d, mom_5d, mom_10d, mom_20d, mom_60d`
* Volatility: `vol_5d, vol_20d`
* Reversal: `reversal_1d`
* Volume regime: `vol_ratio` (5d / 20d avg volume)
* 52w position: `dist_52w_high, dist_52w_low`
* Oscillators: `rsi_norm, macd_hist`
* Cross-sectional: `cs_rank_mom10`

Target:

* `fwd_ret_5d`

---

### 5.2 Models

**Random Forest / Gradient Boosting**

* Walk-forward splits (rolling train/test)
* Metrics: R², IC, RankIC, ICIR
* Result: negative IC → limited predictability at this horizon

**Transformer**

* Input projection: Linear(14 → 32)
* 2× TransformerEncoderLayer (d_model=32, heads=4, ff=128)
* Sequence length: 20
* Device: Apple MPS
* Result: no robust IC

**GNN**

* Nodes: 12 semis
* Edges: 60d rolling correlation (threshold > 0.3), weighted adjacency
* GCN: (14 → 32 → 32 → 1), ~1.6k params
* Result: small/flat IC; useful mainly as structure diagnostic

Conclusion:

> With this universe and feature/label design, ML does not contribute robust incremental alpha beyond explicit strategies.

---

## 6) Streamlit App (`app.py`)

Run:

```bash
streamlit run app.py
```

Pages:

1. Overview
2. EDA & Correlations
3. Lead–Lag Study (falsification)
4. CS Momentum (equity, DD, yearly/monthly, robustness sweep)
5. Pairs Trade (spread, z-score, trades, breakdowns)
6. Alpha Attribution (SOXX/SPY, yearly alpha tables)
7. ML Diagnostics (IC/RankIC, feature importance, interpretation)

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
├── models/
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

### Limitations

1. **Scale**: 12 semis over ~6 years is solid but not institutional breadth; alpha t-stats remain modest.
2. **Costs**: costs are modeled simply; a full slippage/liquidity model would reduce Sharpe.
3. **Regime sensitivity**: CS momentum meaningfully underperforms in rotation/chop regimes (2022, 2025).
4. **ML**: near-zero IC suggests limited predictability with this feature/label choice.

### Next Steps

* Add a **regime overlay** for CS momentum (e.g., rolling Sharpe gate / volatility state / trend filter).
* Explore alternative labels (residual returns vs SOXX, longer horizon like 10d).
* Expand universe (more semis + supply chain adjacencies) for more robust cross-sectional IC.
* Upgrade execution model: spreads, slippage, borrow, liquidity-aware sizing.

---

*Author: Arunesh Lal — M.S. Computer Science, Boston University*

```
```
