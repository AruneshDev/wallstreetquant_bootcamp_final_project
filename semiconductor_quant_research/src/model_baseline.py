import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'mom_1d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_60d',
    'vol_5d', 'vol_20d', 'reversal_1d', 'vol_ratio',
    'dist_52w_high', 'cs_rank_mom10'
]
TARGET_COL = 'fwd_ret_5d'


# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ══════════════════════════════════════════════════════════════════

def walk_forward_splits(dates: pd.Index,
                         train_days: int = 252,
                         val_days:   int = 63,
                         test_days:  int = 63,
                         step_days:  int = 63):
    """
    Generator of (train_dates, val_dates, test_dates).
    With 2 years of data use shorter windows:
      train=252d (~1yr), val=63d, test=63d, step=63d
    Yields ~3 folds from 535 days.
    """
    total = len(dates)
    start = 0
    while start + train_days + val_days + test_days <= total:
        train = dates[start : start + train_days]
        val   = dates[start + train_days : start + train_days + val_days]
        test  = dates[start + train_days + val_days :
                      start + train_days + val_days + test_days]
        yield train, val, test
        start += step_days


# ══════════════════════════════════════════════════════════════════
# IC METRICS
# ══════════════════════════════════════════════════════════════════

def daily_ic(pred_df: pd.DataFrame,
             target_df: pd.DataFrame) -> pd.Series:
    """
    pred_df / target_df: (date × ticker) DataFrames.
    Returns daily Pearson IC series.
    """
    ics = {}
    for date in pred_df.index:
        if date not in target_df.index:
            continue
        p = pred_df.loc[date].dropna()
        t = target_df.loc[date].reindex(p.index).dropna()
        both = pd.concat([p, t], axis=1).dropna()
        if len(both) < 4:
            continue
        r, _ = pearsonr(both.iloc[:, 0], both.iloc[:, 1])
        ics[date] = r
    return pd.Series(ics).sort_index()


def daily_rank_ic(pred_df: pd.DataFrame,
                   target_df: pd.DataFrame) -> pd.Series:
    rank_ics = {}
    for date in pred_df.index:
        if date not in target_df.index:
            continue
        p = pred_df.loc[date].dropna()
        t = target_df.loc[date].reindex(p.index).dropna()
        both = pd.concat([p, t], axis=1).dropna()
        if len(both) < 4:
            continue
        r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
        rank_ics[date] = r
    return pd.Series(rank_ics).sort_index()


def ic_report(ic: pd.Series, rank_ic: pd.Series, label: str):
    ic    = ic.dropna()
    ric   = rank_ic.dropna()
    icir  = ic.mean()  / ic.std()  if ic.std()  > 0 else np.nan
    ricir = ric.mean() / ric.std() if ric.std() > 0 else np.nan
    print(f"\n{'='*55}")
    print(f"  IC Report — {label}")
    print(f"{'='*55}")
    print(f"  IC mean     : {ic.mean():.5f}")
    print(f"  IC std      : {ic.std():.5f}")
    print(f"  ICIR        : {icir:.4f}")
    print(f"  RankIC mean : {ric.mean():.5f}")
    print(f"  RankICIR    : {ricir:.4f}")
    print(f"  IC pos%     : {(ic > 0).mean()*100:.1f}%")
    print(f"  N days      : {len(ic)}")
    print(f"{'='*55}")
    return {'IC': ic.mean(), 'ICIR': icir,
            'RankIC': ric.mean(), 'RankICIR': ricir,
            'IC_pos_pct': (ic > 0).mean(), 'N': len(ic)}


# ══════════════════════════════════════════════════════════════════
# RANDOM FOREST BASELINE
# ══════════════════════════════════════════════════════════════════

def run_random_forest(panel: pd.DataFrame,
                       n_estimators: int = 200) -> dict:
    """
    Walk-forward Random Forest.
    Predicts next-day return for each SEMI stock.
    Returns: IC series, predictions DataFrame, feature importances.
    """
    print(f"\nRunning Random Forest (n_est={n_estimators})...")

    dates   = panel.index.get_level_values('date').unique().sort_values()
    all_pred, all_true = {}, {}

    splits = list(walk_forward_splits(dates))
    print(f"  Walk-forward folds: {len(splits)}")

    feat_imp_all = pd.Series(0.0, index=FEATURE_COLS)

    for fold_i, (train_d, val_d, test_d) in enumerate(splits):
        # ── Build matrices ──
        train_data = panel.loc[
            panel.index.get_level_values('date').isin(train_d)
        ][FEATURE_COLS + [TARGET_COL]].dropna()

        test_data = panel.loc[
            panel.index.get_level_values('date').isin(test_d)
        ][FEATURE_COLS + [TARGET_COL]].dropna()

        if len(train_data) < 50 or len(test_data) < 5:
            continue

        X_train = train_data[FEATURE_COLS].values
        y_train = train_data[TARGET_COL].values
        X_test  = test_data[FEATURE_COLS].values
        y_test  = test_data[TARGET_COL].values

        # ── Scale features ──
        scaler  = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # ── Train RF ──
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # ── Store results ──
        test_idx = test_data.index
        for (date, ticker), pred, true in zip(test_idx, preds, y_test):
            if date not in all_pred:
                all_pred[date] = {}
                all_true[date] = {}
            all_pred[date][ticker] = pred
            all_true[date][ticker] = true

        feat_imp_all += pd.Series(
            model.feature_importances_, index=FEATURE_COLS)

        r2 = r2_score(y_test, preds)
        print(f"  Fold {fold_i+1}: test_days={len(test_d)} "
              f"train_rows={len(train_data)} R²={r2:.4f}")

    pred_df   = pd.DataFrame(all_pred).T.sort_index()
    target_df = pd.DataFrame(all_true).T.sort_index()
    feat_imp  = (feat_imp_all / len(splits)).sort_values(ascending=False)

    ic      = daily_ic(pred_df, target_df)
    rank_ic = daily_rank_ic(pred_df, target_df)
    metrics = ic_report(ic, rank_ic, "Random Forest")

    print(f"\n  Feature importances:")
    for feat, imp in feat_imp.items():
        print(f"    {feat:<20}: {imp:.4f}")

    return {
        'model': 'RandomForest',
        'ic_series': ic,
        'rank_ic_series': rank_ic,
        'metrics': metrics,
        'pred_df': pred_df,
        'target_df': target_df,
        'feature_importances': feat_imp
    }


# ══════════════════════════════════════════════════════════════════
# GRADIENT BOOSTING (challenger)
# ══════════════════════════════════════════════════════════════════

def run_gradient_boosting(panel: pd.DataFrame) -> dict:
    """
    GBM challenger — compare IC vs Random Forest.
    """
    print(f"\nRunning Gradient Boosting...")

    dates   = panel.index.get_level_values('date').unique().sort_values()
    all_pred, all_true = {}, {}

    splits = list(walk_forward_splits(dates))

    for fold_i, (train_d, val_d, test_d) in enumerate(splits):
        train_data = panel.loc[
            panel.index.get_level_values('date').isin(train_d)
        ][FEATURE_COLS + [TARGET_COL]].dropna()

        test_data = panel.loc[
            panel.index.get_level_values('date').isin(test_d)
        ][FEATURE_COLS + [TARGET_COL]].dropna()

        if len(train_data) < 50 or len(test_data) < 5:
            continue

        scaler  = RobustScaler()
        X_train = scaler.fit_transform(train_data[FEATURE_COLS].values)
        X_test  = scaler.transform(test_data[FEATURE_COLS].values)
        y_train = train_data[TARGET_COL].values
        y_test  = test_data[TARGET_COL].values

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        test_idx = test_data.index
        for (date, ticker), pred, true in zip(test_idx, preds, y_test):
            if date not in all_pred:
                all_pred[date] = {}
                all_true[date] = {}
            all_pred[date][ticker] = pred
            all_true[date][ticker] = true

        print(f"  Fold {fold_i+1}: R²={r2_score(y_test, preds):.4f}")

    pred_df   = pd.DataFrame(all_pred).T.sort_index()
    target_df = pd.DataFrame(all_true).T.sort_index()
    ic        = daily_ic(pred_df, target_df)
    rank_ic   = daily_rank_ic(pred_df, target_df)
    metrics   = ic_report(ic, rank_ic, "Gradient Boosting")

    return {
        'model': 'GradientBoosting',
        'ic_series': ic,
        'rank_ic_series': rank_ic,
        'metrics': metrics,
        'pred_df': pred_df,
        'target_df': target_df,
    }


# ══════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════

def save_results(rf_results: dict, gbm_results: dict):
    Path("results").mkdir(exist_ok=True)

    rf_results['ic_series'].to_csv("results/rf_ic.csv",  header=['ic'])
    rf_results['rank_ic_series'].to_csv("results/rf_rank_ic.csv",
                                         header=['rank_ic'])
    rf_results['feature_importances'].to_csv("results/rf_feature_importance.csv",
                                              header=['importance'])

    gbm_results['ic_series'].to_csv("results/gbm_ic.csv", header=['ic'])
    gbm_results['rank_ic_series'].to_csv("results/gbm_rank_ic.csv",
                                          header=['rank_ic'])

    # ── Comparison table ──
    comp = pd.DataFrame([
        {'Model': 'RandomForest',      **rf_results['metrics']},
        {'Model': 'GradientBoosting',  **gbm_results['metrics']},
    ]).set_index('Model')
    comp.to_csv("results/ml_baseline_comparison.csv")

    print("\n✓ Results saved to results/")
    print("\nBaseline comparison:")
    print(comp.round(4).to_string())
    return comp


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.features import load_features

    panel = load_features()
    print(f"Panel: {panel.shape} | "
          f"Dates: {panel.index.get_level_values('date').nunique()} | "
          f"Tickers: {panel.index.get_level_values('ticker').nunique()}")

    rf_results  = run_random_forest(panel)
    gbm_results = run_gradient_boosting(panel)
    comp        = save_results(rf_results, gbm_results)
