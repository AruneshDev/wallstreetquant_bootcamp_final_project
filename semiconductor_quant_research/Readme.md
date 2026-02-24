```markdown
# Semiconductor Alpha Research

**Point72 Investor Analyst Competition — Arunesh Lal | Feb 2026**

A reproducible quant-research pipeline on **12 US-listed semiconductor equities + 5 big-tech** from **Jan 2020 → Feb 2026** (daily data). The project follows the scientific method (hypothesis → falsification → pivot → strategy + attribution → ML diagnostics) and ships with a Streamlit app.

**Live demo**
```bash
streamlit run app.py
```

---

## What’s inside (TL;DR)

### Data
- **Universe**
  - Semis (12): `NVDA AMD AVGO TSM QCOM AMAT LRCX MU KLAC TXN ASML MRVL`
  - Big tech (5): `AAPL MSFT GOOGL META AMZN`
- **Period**: `2020-01-03 → 2026-02-20` (≈ 1,541 trading days)
- **Source**: Yahoo Finance via `yfinance` (adjusted close + volume)

### Core outputs
- **Strategy 1 — Cross-Sectional Momentum (45d)**: Long top-3 / short bottom-3 semis by trailing 45d return (lagged 1 day).
- **Strategy 2 — Cointegration-selected Pairs Trade**: Auto-select best semi pair via Engle–Granger test; trade log-spread mean reversion.

---

## Research narrative

### Step 1 — Hypothesis
*“NVDA leads other semiconductor stocks by 1–5 days at daily frequency.”*

Motivation: NVDA is often treated as the sector’s “sentiment leader” during AI capex regimes.

### Step 2 — Falsification
The lead-lag hypothesis was tested across the directed pair set and did **not** produce exploitable positive-lift pairs (daily resolution).  
**Verdict**: daily lead-lag alpha **rejected** → pivot required.

### Step 3 — Pivot to cross-sectional momentum
A robustness sweep across momentum lookback windows found a clear reversal→momentum transition around ~20 trading days, with a **peak around 45d**.

### Step 4 — Pairs via cointegration (stop guessing pairs)
Instead of hand-picking pairs (which breaks badly across regimes), the pipeline runs an **Engle–Granger cointegration sweep** across all semi pairs and chooses the lowest p-value.

**Best pair (2020–2026 run)**: `QCOM/MRVL` (p ≈ 0.015)

---

## Strategy results

### Strategy 1 — Cross-Sectional Momentum (45d)
**Rule**: Each day rank semis by trailing 45d cumulative return (lagged 1 day).  
**Portfolio**: Long top-3, short bottom-3, equal-weight legs, daily rebalance, costs applied.

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

**Regime behavior (annual)**
- Strong: 2021, 2024
- Weak: 2022, 2025
- Note: 2026 is a partial year (not statistically meaningful)

---

### Strategy 2 — Pairs Trade (auto-selected)
**Pair selection**: best cointegrated semi pair by Engle–Granger p-value (lowest).  
**Execution**: log-spread z-score mean reversion, entry ±1.5σ, exit ±0.3σ, 120d rolling stats, costs applied.

**Best pair (this run)**: `QCOM/MRVL`

```text
Annual Return  : 12.17%
Annual Vol     : 22.73%
Sharpe         : 0.535
Sortino        : 0.708
Max Drawdown   : -37.20%
Total Return   : 37.20%
Win Rate       : 51.2%
N days         : 832   (active trading days)
Trades         : 39
```

**Note on interpretability**: this is a true relative-value strategy; returns come from spread reversion rather than market direction.

---

## Alpha attribution (beta vs benchmarks)

Benchmarks:
- `SOXX` (semiconductor index ETF proxy)
- `SPY` (market proxy)

### CS Momentum (45d)
Regression: \(R_p - R_f = \alpha + \beta (R_m - R_f) + \epsilon\)

- vs SOXX: **α ≈ +5.59%/yr**, **β ≈ 0.024**, **R² ≈ 0.0017**
- vs SPY : **α ≈ +6.70%/yr**, **β ≈ -0.043**, **R² ≈ 0.0013**

**Interpretation**: near-zero beta and near-zero R² → returns are not explained by broad market or SOXX exposure (decorrelated).

### Pairs trade attribution
To keep attribution consistent with the auto-selected pair, update `src/alpha.py` to read the chosen pair (see “Reproducibility notes” below) and re-run:
```bash
python src/alpha.py
```

---

## ML signal analysis (diagnostic layer)

This repo includes baseline ML models (RF/GBM), a temporal Transformer, and a correlation-graph GNN.  
The ML layer is treated as **diagnostic research**, not a guaranteed tradable signal.

### Target
- Predict **5-day forward return** (`fwd_ret_5d`) to reduce noise vs 1-day targets.

### Feature set (14 features per stock-day)
- Momentum: `mom_1d, mom_5d, mom_10d, mom_20d, mom_60d`
- Volatility: `vol_5d, vol_20d`
- Reversal: `reversal_1d`
- Volume regime: `vol_ratio`
- 52w position: `dist_52w_high, dist_52w_low`
- Oscillators: `rsi_norm, macd_hist`
- Cross-sectional: `cs_rank_mom10`

### Latest OOS snapshot (example run)
- RF / GBM / Transformer: negative-to-flat IC
- **GNN**: small positive IC (best among tested models in this run)

---

## Streamlit app

Dashboard pages:
- Overview (universe, period, snapshots)
- EDA & correlations
- Lead-lag study
- CS Momentum
- Pairs trade
- Strategy comparison
- Alpha attribution
- ML signal analysis

---

## Repository structure

```text
semiconductor_quant_research/
│
├── app.py
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── backtest.py
│   ├── alpha.py
│   ├── model_baseline.py
│   ├── model_transformer.py
│   └── model_gnn.py
│
├── data/                  # cached parquet (prices/volume/returns/features)
├── results/               # CSV outputs (strategy + ML + attribution)
├── models/                # saved transformer / gnn weights
├── charts/                # optional generated PNG charts
├── requirements.txt
└── README.md
```

---

## Quickstart (reproduce end-to-end)

```bash
# 1) Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Run full pipeline
python src/data_loader.py
python src/features.py
python src/backtest.py
python src/alpha.py
python src/model_baseline.py
python src/model_transformer.py
python src/model_gnn.py

# 3) Launch app
streamlit run app.py
```

**One-liner**
```bash
python src/data_loader.py && \
python src/features.py && \
python src/backtest.py && \
python src/alpha.py && \
python src/model_baseline.py && \
python src/model_transformer.py && \
python src/model_gnn.py && \
streamlit run app.py
```

---

## Reproducibility notes

### Keep alpha.py consistent with backtest.py
`src/backtest.py` auto-selects the best pair (e.g., `QCOM/MRVL`). To ensure `src/alpha.py` uses the same pair:
- Option A (simple): manually set the pair tickers in `alpha.py`
- Option B (recommended): have `backtest.py` write `results/best_pair.txt` and `alpha.py` read it

Example (recommended):
- In `backtest.py`, after selecting `t1, t2`, write:
  ```python
  Path("results").mkdir(exist_ok=True)
  Path("results/best_pair.txt").write_text(f"{t1},{t2}\n")
  ```
- In `alpha.py`, read:
  ```python
  t1, t2 = Path("results/best_pair.txt").read_text().strip().split(",")
  ```

---

## Key limitations (important)

1. **Transaction costs & slippage**: costs are modeled simply; real execution (spreads, borrow, sizing) can materially reduce Sharpe.
2. **Multiple testing (pairs)**: cointegration sweep is a selection step; a true OOS validation for the chosen pair is required.
3. **Regime dependence**: momentum and spreads behave differently across 2020 crash, 2022 bear, and AI-led cycles.
4. **Data quality**: `yfinance` is convenient but not institutional-grade (splits/dividends adjustments are generally OK, but edge cases exist).
5. **Statistical significance**: alpha t-stats remain modest; “decorrelated” is stronger than “proven alpha.”

---

## Next steps

- Regime filter (rolling Sharpe or volatility state) for CS Momentum allocation
- Expand universe (more semis / hardware / supply-chain adjacencies)
- Better pairs engine: Kalman filter hedge ratio, half-life based exits, stopouts
- Stronger ML labeling: sector-neutral targets, residual returns vs SOXX, or 5–10 day horizon
- Full cost model: spreads + slippage + borrow fees, and liquidity-aware sizing

---

*Author: Arunesh Lal | MS Computer Science, Boston University*
```