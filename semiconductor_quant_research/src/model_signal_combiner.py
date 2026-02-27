"""
model_signal_combiner.py — ML as a signal combiner over IC-positive base signals.

Purpose
-------
Instead of using ML models as raw return predictors on OHLCV features (which
produced negative IC on the 12-ticker semi_core universe), this module
re-positions ML as a *meta-model* that learns how to combine several
IC-positive base signals into a single composite signal.

Theoretical motivation
----------------------
Each base signal has a positive-but-noisy IC individually.  The key insight
from the academic literature (Grinold & Kahn 1999, "The Fundamental Law of
Active Management") is:

    ICIR_combined ≈ ICIR_individual × √N

where N is the number of independent signals.  Combining even 3–5 imperfect
signals via ML can roughly double the ICIR if the signals are partially
decorrelated.

The ML model does NOT try to predict returns directly.  Instead:
  1. Each base signal is evaluated daily.
  2. The ML model predicts a *composite rank score* from the signal vector.
  3. The composite score is then used to build a long/short portfolio.

Base signals used (in order of expected IC reliability):
  1. CS momentum (45d rank)         — IC ~0.05–0.10 in trend regimes.
  2. Short-term reversal (1d)       — IC ~0.02–0.04 (mean reversion).
  3. SUE decay signal               — IC ~0.02–0.05 post-earnings.
  4. NLP sentiment (nlp_sent)       — IC ~0.01–0.03 (textual tone).
  5. Volatility regime (vol_ratio)  — IC ~0.01–0.02 (volume signal).

The combiner model
------------------
We use a Gradient Boosting tree ensemble as the combiner because:
  - Tree models handle non-linear interactions between signals naturally.
  - GBM with shallow trees (depth=2) acts as a feature-interaction model,
    not a complex return predictor, reducing overfitting risk.
  - Walk-forward retraining every 63 days ensures the model tracks regime
    changes in signal correlations.

Training target
---------------
  - *Rank* of 10-day forward return, cross-sectionally within the universe,
    normalised to [0, 1].  Using rank instead of raw return is more robust
    to outliers and means the model is optimising ordinal precision, not
    level prediction.

Evaluation
----------
  - Walk-forward OOS IC / RankIC / ICIR.
  - We compare the combined signal IC to each individual signal IC.
  - The combiner is successful if its ICIR > max(individual ICIRs).

Running this module:
  python src/model_signal_combiner.py

Outputs:
  results/signal_combiner_ic.csv       — IC series (OOS).
  results/signal_combiner_summary.csv  — Signal comparison table.
  results/signal_weights.csv           — GBM feature importances over folds.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

Path("results").mkdir(exist_ok=True)

# ── Base signal columns (must be present in the combined feature panel) ───────
BASE_SIGNALS: list[str] = [
    "cs_rank_mom45",   # 45d CS momentum rank (primary momentum signal)
    "reversal_1d",     # short-term reversal
    "sue_decay",       # SUE decay-weighted earnings surprise
    "nlp_sent",        # NLP sentiment score
    "vol_ratio",       # volume regime proxy
    "mom_20d",         # intermediate momentum
    "dist_52w_high",   # 52-week high distance (trend strength)
]

TARGET_COL   = "fwd_ret_10d"    # 10-day forward return rank (more stable than 5d)
TRAIN_DAYS   = 252              # ~1 year training window per fold
TEST_DAYS    = 63               # ~1 quarter test window per fold
STEP_DAYS    = 63               # roll forward 1 quarter per fold


# ══════════════════════════════════════════════════════════════════════════════
# PANEL BUILDER — merges OHLCV features, alt-data features, NLP features
# ══════════════════════════════════════════════════════════════════════════════

def build_combined_panel(
    features_ohlcv: pd.DataFrame,
    features_alt:   Optional[pd.DataFrame],
    features_nlp:   Optional[pd.DataFrame],
    ret:            pd.DataFrame,
    mom_win:        int = 45,
) -> pd.DataFrame:
    """
    Merge OHLCV features, alternative data features, and NLP features into a
    single flat panel with MultiIndex (date, ticker).

    Also adds:
      - cs_rank_mom{mom_win}: cross-sectional rank of mom_{mom_win}d.
      - fwd_ret_10d: 10-day forward log return rank (training target).

    Parameters
    ----------
    features_ohlcv : Panel from src/features.py with standard OHLCV signals.
    features_alt   : Panel from src/features_alt.py (SUE, ARM, SI).
    features_nlp   : Panel from src/nlp_signal.py (NLP sentiment / drift).
    ret            : Wide log-return DataFrame.
    mom_win        : Momentum window for cross-sectional rank feature.

    Returns
    -------
    panel : Merged DataFrame with all base signals and the 10-day target.
    """
    panel = features_ohlcv.copy()

    # ── Add 45d CS momentum rank (the key momentum signal) ───────────────────
    mom_col = f"mom_{mom_win}d"
    if mom_col in panel.columns:
        pivot_mom = panel[mom_col].unstack("ticker")
        cs_rank   = pivot_mom.rank(axis=1, pct=True)
        panel[f"cs_rank_mom{mom_win}"] = (
            cs_rank.stack().reindex(panel.index)
        )

    # ── Add 10-day forward return (TARGET) ────────────────────────────────────
    fwd_ret_wide = ret.rolling(10).sum().shift(-10)
    # Convert to cross-sectional rank [0,1] to remove time-series noise
    fwd_rank     = fwd_ret_wide.rank(axis=1, pct=True)
    fwd_long     = fwd_rank.stack().rename(TARGET_COL)
    fwd_long.index.names = ["date", "ticker"]
    panel = panel.join(fwd_long, how="left")

    # ── Merge alt-data features ───────────────────────────────────────────────
    if features_alt is not None and not features_alt.empty:
        alt_cols = [c for c in features_alt.columns
                    if c not in panel.columns]
        if alt_cols:
            panel = panel.join(features_alt[alt_cols], how="left")

    # ── Merge NLP features ────────────────────────────────────────────────────
    if features_nlp is not None and not features_nlp.empty:
        nlp_cols = [c for c in features_nlp.columns
                    if c not in panel.columns]
        if nlp_cols:
            panel = panel.join(features_nlp[nlp_cols], how="left")

    # ── Fill NaN in optional signals with 0 (neutral) ─────────────────────────
    optional_cols = ["sue_decay", "nlp_sent", "vol_ratio"]
    for col in optional_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)

    panel = panel.dropna(subset=[TARGET_COL])
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD SIGNAL COMBINER
# ══════════════════════════════════════════════════════════════════════════════

def _available_signals(panel: pd.DataFrame) -> list[str]:
    """Return the subset of BASE_SIGNALS present in *panel*."""
    return [s for s in BASE_SIGNALS if s in panel.columns]


def run_signal_combiner(
    panel:         pd.DataFrame,
    universe_name: str   = "semi_core",
    train_days:    int   = TRAIN_DAYS,
    test_days:     int   = TEST_DAYS,
    step_days:     int   = STEP_DAYS,
    n_estimators:  int   = 100,
    max_depth:     int   = 2,
    learning_rate: float = 0.05,
    verbose:       bool  = True,
) -> dict:
    """
    Walk-forward GBM signal combiner over IC-positive base signals.

    The model learns to combine base signals into a composite rank score.
    Evaluated OOS with IC / RankIC / ICIR.

    When fewer than 2 IC-positive signals are available this function returns
    gracefully with ``status='skipped'`` (0 signals) or
    ``status='single_signal_passthrough'`` (exactly 1 signal) instead of
    raising a ``ValueError``.  The caller / ``__main__`` block handles each
    status appropriately so the full pipeline never crashes on a valid
    low-signal research outcome.

    Parameters
    ----------
    panel         : Combined feature panel from build_combined_panel().
    universe_name : Universe identifier used in log messages.
    train_days    : Number of trading days per training window.
    test_days     : Number of trading days per test window.
    step_days     : Roll-forward step size.
    n_estimators  : Number of GBM trees.
    max_depth     : Tree depth (keep shallow to reduce overfit).
    learning_rate : GBM shrinkage.
    verbose       : Print fold-level diagnostics.

    Returns
    -------
    dict with keys:
      'ic_series'    : pd.Series of daily OOS IC values.
      'rank_ic_series': pd.Series of daily OOS RankIC values.
      'metrics'      : Summary dict (IC_mean, ICIR, etc.).
      'feature_importances': pd.DataFrame of per-fold GBM importances.
      'pred_df'      : Wide-format OOS predictions.
      'target_df'    : Wide-format OOS targets (raw rank).
    """
    signal_cols = _available_signals(panel)

    # ── Graceful degradation when too few IC-positive signals exist ───────────
    if len(signal_cols) == 0:
        msg = (
            f"[WARN] No IC-positive signals found for universe='{universe_name}'. "
            "Signal combiner requires at least 2 base signals; skipping training."
        )
        print(msg)
        return {
            "status":          "skipped",
            "reason":          "insufficient_base_signals",
            "universe":        universe_name,
            "base_signals":    signal_cols,
            "n_base_signals":  0,
        }

    if len(signal_cols) == 1:
        sig = signal_cols[0]
        print(
            f"[WARN] Only one IC-positive signal '{sig}' found for "
            f"universe='{universe_name}'. "
            "Signal combiner requires at least 2 base signals; "
            "returning the single signal as a pass-through (no GBM trained)."
        )
        return {
            "status":          "single_signal_passthrough",
            "reason":          "only_one_base_signal",
            "universe":        universe_name,
            "base_signals":    signal_cols,
            "n_base_signals":  1,
            "passthrough_signal": sig,
        }

    if verbose:
        print("\nRunning Signal Combiner (GBM meta-model)")
        print(f"  Base signals : {signal_cols}")
        print(f"  Target       : {TARGET_COL}")
        print(f"  Train/Test/Step : {train_days}/{test_days}/{step_days} days")
        print(f"  GBM          : n_est={n_estimators}, depth={max_depth}, "
              f"lr={learning_rate}")

    dates  = panel.index.get_level_values("date").unique().sort_values()
    n      = len(dates)
    start  = 0

    all_pred: dict  = {}
    all_true: dict  = {}
    feat_imps: list[pd.Series] = []
    fold_metrics: list[dict]   = []

    while start + train_days + test_days <= n:
        train_dates = dates[start : start + train_days]
        test_dates  = dates[start + train_days : start + train_days + test_days]

        train_df = panel.loc[
            panel.index.get_level_values("date").isin(train_dates),
            signal_cols + [TARGET_COL]
        ].dropna()

        test_df = panel.loc[
            panel.index.get_level_values("date").isin(test_dates),
            signal_cols + [TARGET_COL]
        ].dropna()

        if len(train_df) < 50 or len(test_df) < 5:
            start += step_days
            continue

        X_train = train_df[signal_cols].values
        y_train = train_df[TARGET_COL].values
        X_test  = test_df[signal_cols].values
        y_test  = test_df[TARGET_COL].values

        # ── Scale features (tree-based models benefit from similar scales) ────
        scaler  = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # ── Train GBM combiner ────────────────────────────────────────────────
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # ── Store predictions ─────────────────────────────────────────────────
        for (date, ticker), pred, true in zip(
                test_df.index, preds, y_test):
            if date not in all_pred:
                all_pred[date] = {}
                all_true[date] = {}
            all_pred[date][ticker] = float(pred)
            all_true[date][ticker] = float(true)

        # ── Fold IC ───────────────────────────────────────────────────────────
        fold_ics: list[float] = []
        for date in test_dates:
            if date not in all_pred:
                continue
            p = pd.Series(all_pred[date]).dropna()
            t = pd.Series(all_true[date]).reindex(p.index).dropna()
            both = pd.concat([p, t], axis=1).dropna()
            if len(both) < 4:
                continue
            ic, _ = pearsonr(both.iloc[:, 0], both.iloc[:, 1])
            fold_ics.append(ic)

        fold_ic_mean = float(np.mean(fold_ics)) if fold_ics else np.nan

        feat_imp = pd.Series(model.feature_importances_, index=signal_cols)
        feat_imps.append(feat_imp)

        fold_metrics.append({
            "train_start": train_dates[0].date(),
            "test_start":  test_dates[0].date(),
            "train_rows":  len(train_df),
            "test_rows":   len(test_df),
            "fold_IC":     round(fold_ic_mean, 5),
        })

        if verbose:
            print(f"  Fold [{train_dates[0].date()} | "
                  f"test {test_dates[0].date()}]  "
                  f"train={len(train_df)} test={len(test_df)}  "
                  f"fold_IC={fold_ic_mean:.4f}")

        start += step_days

    if not all_pred:
        raise RuntimeError("No OOS predictions generated.  "
                           "Check panel size and walk-forward parameters.")

    pred_df   = pd.DataFrame(all_pred).T.sort_index()
    target_df = pd.DataFrame(all_true).T.sort_index()

    # ── Full OOS IC series ────────────────────────────────────────────────────
    ics: dict  = {}
    rks: dict  = {}
    for date in pred_df.index:
        if date not in target_df.index:
            continue
        p    = pred_df.loc[date].dropna()
        t    = target_df.loc[date].reindex(p.index).dropna()
        both = pd.concat([p, t], axis=1).dropna()
        if len(both) < 4:
            continue
        ics[date] = pearsonr(both.iloc[:, 0], both.iloc[:, 1])[0]
        rks[date] = spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]

    ic_s  = pd.Series(ics).sort_index()
    ric_s = pd.Series(rks).sort_index()

    icir  = ic_s.mean()  / ic_s.std()  if ic_s.std()  > 0 else np.nan
    ricir = ric_s.mean() / ric_s.std() if ric_s.std() > 0 else np.nan

    # ── Average feature importances across folds ──────────────────────────────
    if feat_imps:
        fi_df = pd.concat(feat_imps, axis=1).T.mean()
        fi_df = fi_df.sort_values(ascending=False)
    else:
        fi_df = pd.Series(dtype=float)

    metrics = {
        "IC_mean":    round(float(ic_s.mean()),  5),
        "IC_std":     round(float(ic_s.std()),   5),
        "ICIR":       round(float(icir),         4),
        "RankIC_mean":round(float(ric_s.mean()), 5),
        "RankICIR":   round(float(ricir),        4),
        "IC_pos_pct": round(float((ic_s > 0).mean()), 4),
        "N_days":     len(ic_s),
        "N_folds":    len(fold_metrics),
    }

    if verbose:
        print(f"\n{'='*60}")
        print("  Signal Combiner — OOS IC Summary")
        print(f"{'='*60}")
        print(f"  IC mean     : {metrics['IC_mean']:.5f}  "
              f"{'✅' if metrics['IC_mean'] > 0.04 else '~ ' if metrics['IC_mean'] > 0 else '❌'}")
        print(f"  IC std      : {metrics['IC_std']:.5f}")
        print(f"  ICIR        : {metrics['ICIR']:.4f}")
        print(f"  RankIC mean : {metrics['RankIC_mean']:.5f}")
        print(f"  RankICIR    : {metrics['RankICIR']:.4f}")
        print(f"  IC pos%     : {metrics['IC_pos_pct']*100:.1f}%")
        print(f"  N days      : {metrics['N_days']}")
        print(f"  N folds     : {metrics['N_folds']}")
        print("\n  GBM Signal Weights (avg feature importance):")
        for sig, imp in fi_df.items():
            print(f"    {sig:<22}: {imp:.4f}")
        print(f"{'='*60}")

    return {
        "ic_series":           ic_s,
        "rank_ic_series":      ric_s,
        "metrics":             metrics,
        "feature_importances": fi_df,
        "fold_metrics":        pd.DataFrame(fold_metrics),
        "pred_df":             pred_df,
        "target_df":           target_df,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL SIGNAL IC BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_individual_signal_ic(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute IC and RankIC for each base signal individually (OOS on the
    full sample — no walk-forward split needed for this diagnostic).

    Parameters
    ----------
    panel : Combined panel from build_combined_panel().

    Returns
    -------
    DataFrame comparing IC statistics per signal.
    """
    signal_cols = _available_signals(panel)
    rows: list[dict] = []

    for sig in signal_cols:
        sub = panel[[sig, TARGET_COL]].dropna()
        if len(sub) < 20:
            continue

        sig_pivot = sub[sig].unstack("ticker")
        tgt_pivot = sub[TARGET_COL].unstack("ticker")

        ics: list[float] = []
        rks: list[float] = []

        for date in sig_pivot.index:
            s = sig_pivot.loc[date].dropna()
            t = tgt_pivot.loc[date].reindex(s.index).dropna()
            both = pd.concat([s, t], axis=1).dropna()
            if len(both) < 4:
                continue
            ics.append(pearsonr(both.iloc[:, 0], both.iloc[:, 1])[0])
            rks.append(spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0])

        if not ics:
            continue

        ic_s  = pd.Series(ics)
        ric_s = pd.Series(rks)
        icir  = ic_s.mean() / ic_s.std() if ic_s.std() > 0 else np.nan
        ricir = ric_s.mean() / ric_s.std() if ric_s.std() > 0 else np.nan

        rows.append({
            "signal":        sig,
            "IC_mean":       round(ic_s.mean(),  5),
            "ICIR":          round(icir,         4),
            "RankIC_mean":   round(ric_s.mean(), 5),
            "RankICIR":      round(ricir,        4),
            "IC_pos_pct":    round((ic_s > 0).mean(), 3),
            "N_days":        len(ic_s),
        })

    df = pd.DataFrame(rows).set_index("signal")
    print("\n  Individual Signal IC Benchmark:")
    print(f"  {'Signal':<22} {'IC_mean':>10} {'ICIR':>8} "
          f"{'RankIC':>10} {'IC_pos%':>9}")
    print(f"  {'─'*65}")
    for sig, row in df.iterrows():
        flag = "✅" if row["IC_mean"] > 0.04 else (
               "~" if row["IC_mean"] > 0 else "❌")
        print(f"  {sig:<22} {row['IC_mean']:>10.5f} {row['ICIR']:>8.4f} "
              f"{row['RankIC_mean']:>10.5f} {row['IC_pos_pct']*100:>8.1f}%  "
              f"{flag}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data_loader import load
    from src.features import load_features

    parser = argparse.ArgumentParser(
        description="Run GBM signal combiner on a universe tier."
    )
    parser.add_argument(
        "--universe", default="semi_core",
        choices=["semi_core", "sp_tech_semi", "r1000_tech"],
        help="Universe tier to run the signal combiner on.",
    )
    args = parser.parse_args()
    universe_name = args.universe

    # ── Resolve feature file paths ────────────────────────────────────────────
    if universe_name == "semi_core":
        ohlcv_path = Path("data/features.parquet")
        alt_path   = Path("data/features_alt.parquet")
        nlp_path   = Path("data/features_nlp.parquet")
        ic_path    = Path("results/ic_study_semi_core.csv")
        out_prefix = "results/signal_combiner"
    else:
        ohlcv_path = Path(f"data/features_{universe_name}.parquet")
        alt_path   = Path(f"data/features_alt_{universe_name}.parquet")
        nlp_path   = Path(f"data/features_nlp_{universe_name}.parquet")
        ic_path    = Path(f"results/ic_study_{universe_name}.csv")
        out_prefix = f"results/signal_combiner_{universe_name}"

    # ── Load base data ────────────────────────────────────────────────────────
    print(f"\nLoading data for universe={universe_name}...")
    close, volume, ret = load(universe_name=universe_name)
    panel_ohlcv        = load_features(str(ohlcv_path))

    # ── Load alt-data features (optional) ────────────────────────────────────
    panel_alt: Optional[pd.DataFrame] = None
    if alt_path.exists():
        panel_alt = pd.read_parquet(alt_path)
        print(f"  Alt-data features loaded: {panel_alt.shape}")
    else:
        print(f"  Alt-data features not found at {alt_path} — "
              "run features_alt.py --universe {universe_name} first.")

    # ── Load NLP features (optional) ─────────────────────────────────────────
    panel_nlp: Optional[pd.DataFrame] = None
    if nlp_path.exists():
        panel_nlp = pd.read_parquet(nlp_path)
        print(f"  NLP features loaded: {panel_nlp.shape}")
    else:
        print(f"  NLP features not found at {nlp_path} — "
              "run nlp_signal.py --universe {universe_name} first.")

    # ── Build combined panel ──────────────────────────────────────────────────
    print("\nBuilding combined signal panel...")
    combined = build_combined_panel(
        panel_ohlcv, panel_alt, panel_nlp, ret, mom_win=45
    )
    print(f"  Combined panel: {combined.shape}")
    available = _available_signals(combined)
    print(f"  Signals available: {available}")

    # ── Filter to IC-positive signals only ────────────────────────────────────
    # Load IC study results if available; otherwise use all available signals.
    ic_positive_signals: list[str] = []
    if ic_path.exists():
        ic_df = pd.read_csv(ic_path, index_col="signal")
        ic_positive_signals = ic_df[
            (ic_df["IC_mean"]    > 0.02) &
            (ic_df["IC_pos_pct"] > 0.52)
        ].index.tolist()
        # Keep only signals that also appear in the combined panel
        ic_positive_signals = [s for s in ic_positive_signals
                               if s in available]
        print(f"\n  IC-positive signals (from {ic_path.name}): "
              f"{ic_positive_signals}")
    else:
        print(f"\n  IC study file not found ({ic_path}).  "
              "Using all available signals as fallback.\n"
              "  → Run src/ic_study.py --universe {universe_name} first for "
              "cleaner combiner inputs.")
        ic_positive_signals = available

    if not ic_positive_signals:
        print(
            f"\n⚠️  No IC-positive base signals found for {universe_name}.\n"
            "   Expand feature set before training combiner.\n"
            "   (earnings quality, analyst coverage, sector-relative features)"
        )
        sys.exit(0)

    # ── Benchmark individual signals ──────────────────────────────────────────
    indiv_df = benchmark_individual_signal_ic(combined)
    indiv_df.to_csv(f"{out_prefix}_individual_ic.csv")
    print(f"\n✓ Saved {out_prefix}_individual_ic.csv")

    # ── Override BASE_SIGNALS with IC-positive subset for this run ───────────
    # Temporarily restrict the global list so run_signal_combiner uses only
    # the IC-vetted signals.
    original_base_signals = list(BASE_SIGNALS)
    BASE_SIGNALS.clear()
    BASE_SIGNALS.extend(ic_positive_signals)
    print(f"\n  Training combiner with IC-positive signals: {BASE_SIGNALS}")

    # ── Run signal combiner ───────────────────────────────────────────────────
    result = run_signal_combiner(combined, universe_name=universe_name,
                                 verbose=True)

    # Restore original list
    BASE_SIGNALS.clear()
    BASE_SIGNALS.extend(original_base_signals)

    # ── Handle graceful-skip statuses — no output files, no crash ────────────
    if result.get("status") in ("skipped", "single_signal_passthrough"):
        print(
            f"\n[INFO] Signal combiner exited early: "
            f"status={result['status']!r}, "
            f"reason={result.get('reason')!r}, "
            f"base_signals={result.get('base_signals')}"
        )
        if result["status"] == "single_signal_passthrough":
            print(
                f"[INFO] Pass-through signal '{result['passthrough_signal']}' "
                "is available in the combined panel but GBM was not trained. "
                "Collect at least one more IC-positive signal to enable combiner."
            )
        else:
            print(
                "[INFO] No IC-positive signals available. "
                "Feature set needs fundamental rethink "
                "(earnings quality, analyst coverage, sector-relative features)."
            )
        sys.exit(0)

    # ── Save results ──────────────────────────────────────────────────────────
    result["ic_series"].to_csv(
        f"{out_prefix}_ic.csv", header=["ic"])
    result["rank_ic_series"].to_csv(
        f"{out_prefix}_rank_ic.csv", header=["rank_ic"])

    # Compare combiner vs individual signals
    summary = indiv_df.copy()
    combiner_row = pd.Series({
        "IC_mean":    result["metrics"]["IC_mean"],
        "ICIR":       result["metrics"]["ICIR"],
        "RankIC_mean":result["metrics"]["RankIC_mean"],
        "RankICIR":   result["metrics"]["RankICIR"],
        "IC_pos_pct": result["metrics"]["IC_pos_pct"],
        "N_days":     result["metrics"]["N_days"],
    }, name="GBM_COMBINER")
    summary = pd.concat([summary, combiner_row.to_frame().T])
    summary.to_csv(f"{out_prefix}_summary.csv")
    print(f"\n✓ Saved {out_prefix}_summary.csv")

    result["feature_importances"].to_csv(
        f"{out_prefix}_weights.csv", header=["importance"])
    print(f"✓ Saved {out_prefix}_weights.csv")

    result["fold_metrics"].to_csv(
        f"{out_prefix}_folds.csv", index=False)
    print(f"✓ Saved {out_prefix}_folds.csv")

    print("\n" + "=" * 60)
    print(f"  Signal Combiner Summary  [{universe_name}]")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60)
