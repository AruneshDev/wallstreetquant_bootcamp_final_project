import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_daily_ic(signal_series: pd.Series,
                     target_series: pd.Series) -> pd.Series:
    """
    Compute daily IC between signal and forward return.
    Both series must share a (date, ticker) MultiIndex.
    Returns a Series indexed by date.
    """
    results = {}
    dates = signal_series.index.get_level_values('date').unique()
    for d in dates:
        try:
            sig = signal_series.xs(d, level='date').dropna()
            tgt = target_series.xs(d, level='date').reindex(sig.index).dropna()
            both = pd.concat([sig, tgt], axis=1).dropna()
            if len(both) < 5:
                continue
            r, _ = pearsonr(both.iloc[:, 0], both.iloc[:, 1])
            results[d] = r
        except Exception:
            continue
    return pd.Series(results).sort_index()


def compute_daily_rank_ic(signal_series: pd.Series,
                          target_series: pd.Series) -> pd.Series:
    """Spearman RankIC — more robust to outliers than Pearson IC."""
    results = {}
    dates = signal_series.index.get_level_values('date').unique()
    for d in dates:
        try:
            sig = signal_series.xs(d, level='date').dropna()
            tgt = target_series.xs(d, level='date').reindex(sig.index).dropna()
            both = pd.concat([sig, tgt], axis=1).dropna()
            if len(both) < 5:
                continue
            r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
            results[d] = r
        except Exception:
            continue
    return pd.Series(results).sort_index()


def signal_report(ic_series: pd.Series,
                  rank_ic_series: pd.Series,
                  label: str = "Signal") -> dict:
    """
    Print and return full signal quality metrics.

    IC interpretation:
      IC mean > 0.03 : weak but potentially useful
      IC mean > 0.05 : tradable
      IC mean > 0.10 : strong
      ICIR > 0.5     : consistent
      IC pos% > 55%  : fires in right direction most days
    """
    ic   = ic_series.dropna()
    ric  = rank_ic_series.dropna()

    metrics = {
        'IC_mean':       round(ic.mean(), 5),
        'IC_std':        round(ic.std(),  5),
        'ICIR':          round(ic.mean() / ic.std(), 4) if ic.std() > 0 else np.nan,
        'RankIC_mean':   round(ric.mean(), 5),
        'RankIC_std':    round(ric.std(),  5),
        'RankICIR':      round(ric.mean() / ric.std(), 4) if ric.std() > 0 else np.nan,
        'IC_pos_pct':    round((ic > 0).mean(), 4),
        'N_days':        len(ic),
    }

    print(f"\n{'='*50}")
    print(f"  Signal Report: {label}")
    print(f"{'='*50}")
    print(f"  IC mean      : {metrics['IC_mean']:.5f}")
    print(f"  IC std       : {metrics['IC_std']:.5f}")
    print(f"  ICIR         : {metrics['ICIR']:.4f}")
    print(f"  RankIC mean  : {metrics['RankIC_mean']:.5f}")
    print(f"  RankICIR     : {metrics['RankICIR']:.4f}")
    print(f"  IC pos%      : {metrics['IC_pos_pct']*100:.1f}%")
    print(f"  N days       : {metrics['N_days']}")
    print(f"{'='*50}\n")
    return metrics


def backtest_report(port_ret: pd.Series, label: str = "Strategy") -> dict:
    """
    Print and return full backtest performance metrics.
    """
    r   = port_ret.dropna()
    ar  = r.mean() * 252
    av  = r.std()  * np.sqrt(252)
    sr  = ar / av if av > 0 else np.nan
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    neg = r[r < 0]
    dv  = neg.std() * np.sqrt(252) if len(neg) > 0 else np.nan
    sortino = ar / dv if dv and dv > 0 else np.nan
    total_r = cum.iloc[-1] - 1
    win_r   = (r > 0).mean()

    metrics = {
        'annual_return': round(ar, 5),
        'annual_vol':    round(av, 5),
        'sharpe':        round(sr, 4),
        'sortino':       round(sortino, 4),
        'max_drawdown':  round(mdd, 5),
        'total_return':  round(total_r, 5),
        'win_rate':      round(win_r, 4),
        'n_days':        len(r),
    }

    print(f"\n{'='*50}")
    print(f"  Backtest Report: {label}")
    print(f"{'='*50}")
    print(f"  Annual Return  : {ar*100:.2f}%")
    print(f"  Annual Vol     : {av*100:.2f}%")
    print(f"  Sharpe         : {sr:.3f}")
    print(f"  Sortino        : {sortino:.3f}")
    print(f"  Max Drawdown   : {mdd*100:.2f}%")
    print(f"  Total Return   : {total_r*100:.2f}%")
    print(f"  Win Rate       : {win_r*100:.1f}%")
    print(f"  N days         : {len(r)}")
    print(f"{'='*50}\n")
    return metrics


if __name__ == "__main__":
    print("evaluate.py — import and use compute_daily_ic, signal_report, backtest_report")
