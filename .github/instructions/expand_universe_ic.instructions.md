# Copilot Task: Expand Signal IC Studies to Broad Universe

## Objective
Move all signal IC evaluation and ML training from the 12-ticker `semi_core`
universe to the `sp_tech_semi` (71 tickers) and `r1000_tech` (108 tickers)
universes. The goal is to determine whether any existing signal — especially
the analyst revision proxy (ARM) and NLP sentiment — shows statistically
meaningful positive IC at scale.

## Context

Current state:
- All IC studies run on 12 semi tickers. With N=12 the minimum resolvable IC
  difference is ~0.12 (one rank swap). This makes all IC estimates unreliable.
- Analyst revision proxy (ARM): IC_mean=0.05, RankIC=0.033, IC_pos%=58.7%
  but only on 12 tickers and 259 days — statistically meaningless.
- NLP sentiment: IC≈0, failed IC threshold on 12 tickers.
- GBM signal combiner: IC_mean=-0.006, failed.
- Target: get IC studies running on sp_tech_semi (71 tickers) and r1000_tech
  (108 tickers). With N=71, minimum resolvable IC ≈ 0.025; with N=108 ≈ 0.019.
  IC t-stat improves by √(71/12) ≈ 2.4× and √(108/12) ≈ 3× respectively.

Data already cached:
- data/prices_sp_tech_semi.parquet  : (1542, 71)
- data/returns_sp_tech_semi.parquet : (1541, 71)
- data/prices_r1000_tech.parquet    : (1542, 108)
- data/returns_r1000_tech.parquet   : (1541, 108)

## Implementation Status

### ✅ Task 1 — features.py expanded to broad universes

`src/features.py` now accepts a `universe_name` parameter and runs on any
loaded universe.  All tickers present in the loaded DataFrames are used.

CLI:
```bash
./.venv/bin/python src/features.py --universe sp_tech_semi
./.venv/bin/python src/features.py --universe r1000_tech
```

Output files:
- `data/features.parquet`               (semi_core, legacy default)
- `data/features_sp_tech_semi.parquet`  (71 tickers)
- `data/features_r1000_tech.parquet`    (108 tickers)

### ✅ Task 2 — features_alt.py expanded to broad universes

`src/features_alt.py` now accepts `universe_name` parameter.  Loads `close`
and `ret` from the appropriate parquet files.  Cross-sectional z-scores use
the FULL universe on each date.  tqdm progress bars added to EPS fetch loops.

CLI:
```bash
./.venv/bin/python src/features_alt.py --universe sp_tech_semi
./.venv/bin/python src/features_alt.py --universe r1000_tech
```

Output files:
- `data/features_alt.parquet`               (semi_core, legacy default)
- `data/features_alt_sp_tech_semi.parquet`  (71 tickers)
- `data/features_alt_r1000_tech.parquet`    (108 tickers)

### ✅ Task 3 — nlp_signal.py expanded to broad universes

`src/nlp_signal.py` `build_nlp_signal()` now accepts `universe_name`.
Falls back to synthetic embeddings for tickers without a known CIK.

CLI:
```bash
./.venv/bin/python src/nlp_signal.py --universe sp_tech_semi
./.venv/bin/python src/nlp_signal.py --universe r1000_tech
```

Output files:
- `data/features_nlp.parquet`               (semi_core, legacy default)
- `data/features_nlp_sp_tech_semi.parquet`  (71 tickers)
- `data/features_nlp_r1000_tech.parquet`    (108 tickers)

### ✅ Task 4 — IC study module (src/ic_study.py)

New file `src/ic_study.py` implements `run_ic_study()` and
`compare_universes()`.

Signals evaluated:
- From features panel:     mom_20d, mom_60d, reversal_1d, vol_ratio,
                           dist_52w_high, cs_rank_mom10
- From features_alt panel: sue, sue_decay, arm_signal, si_proxy
- From features_nlp panel: nlp_sent, nlp_drift

Each signal reports: IC_mean, IC_std, ICIR, IC_tstat, RankIC_mean, RankICIR,
IC_pos_pct, N_days, N_tickers, flag (✅ / ~ / ❌).

IC t-stat included: t = IC_mean / (IC_std / √N_days).

Flag thresholds: IC_mean > 0.02 AND IC_pos_pct > 52% → ✅.

CLI:
```bash
# Single universe
./.venv/bin/python src/ic_study.py --universe sp_tech_semi --fwd-days 5
./.venv/bin/python src/ic_study.py --universe r1000_tech   --fwd-days 5

# All three universes + comparison table
./.venv/bin/python src/ic_study.py --compare --fwd-days 5
```

Output files:
- `results/ic_study_semi_core.csv`
- `results/ic_study_sp_tech_semi.csv`
- `results/ic_study_r1000_tech.csv`
- `results/ic_study_comparison.csv`  (side-by-side table)

### ✅ Task 5 — Signal combiner expanded to broad universes

`src/model_signal_combiner.py` now accepts `--universe` flag.  It reads the
corresponding `ic_study_{universe}.csv` to identify IC-positive signals and
only trains the GBM on those.  If no signals pass the IC threshold it exits
with a clear warning rather than training on failed signals.

CLI:
```bash
./.venv/bin/python src/model_signal_combiner.py --universe sp_tech_semi
./.venv/bin/python src/model_signal_combiner.py --universe r1000_tech
```

Output files (per universe):
- `results/signal_combiner_{universe}_ic.csv`
- `results/signal_combiner_{universe}_rank_ic.csv`
- `results/signal_combiner_{universe}_summary.csv`
- `results/signal_combiner_{universe}_weights.csv`
- `results/signal_combiner_{universe}_folds.csv`
- `results/signal_combiner_{universe}_individual_ic.csv`

## Recommended Run Order

```bash
# Step 1: Build OHLCV features for broad universes
./.venv/bin/python src/features.py --universe sp_tech_semi
./.venv/bin/python src/features.py --universe r1000_tech

# Step 2: Build alt-data features (slow: yfinance per ticker)
./.venv/bin/python src/features_alt.py --universe sp_tech_semi
./.venv/bin/python src/features_alt.py --universe r1000_tech

# Step 3: Build NLP features (optional; slow for large universes)
./.venv/bin/python src/nlp_signal.py --universe sp_tech_semi
./.venv/bin/python src/nlp_signal.py --universe r1000_tech

# Step 4: Run IC study on all universes and get comparison table
./.venv/bin/python src/ic_study.py --compare --fwd-days 5

# Step 5: Run signal combiner using IC-vetted signals only
./.venv/bin/python src/model_signal_combiner.py --universe sp_tech_semi
./.venv/bin/python src/model_signal_combiner.py --universe r1000_tech
```

## Acceptance Criteria

1. `data/features_sp_tech_semi.parquet` and `data/features_r1000_tech.parquet`
   exist with correct shapes (~109K and ~165K rows respectively).
2. `data/features_alt_sp_tech_semi.parquet` and
   `data/features_alt_r1000_tech.parquet` exist.
3. `results/ic_study_sp_tech_semi.csv` and `results/ic_study_r1000_tech.csv`
   exist with IC scores for all signals.
4. At least ONE signal shows IC_mean > 0.02 AND IC_pos_pct > 52% on either
   sp_tech_semi or r1000_tech.  If none do, the code prints:
   "No IC-positive signals found at scale. Feature set needs fundamental
    rethink (earnings quality, analyst coverage, sector-relative features)."
5. Signal combiner is re-run using only IC-positive signals as base inputs.

## Code Quality Constraints

- All functions accept `universe_name` parameter; no hardcoded "semi_core"
  or SEMI_CORE lists in IC evaluation logic.
- All scripts have `--universe` argparse flag.
- All output file names include the universe_name tag.
- N_tickers printed clearly in every IC report header.
- Failed signals always reported with ❌ — never suppressed.
- Strategy backtests (CS momentum, QCOM/MRVL pairs) remain on semi_core.
- Existing semi_core IC result files are NOT deleted or overwritten.
