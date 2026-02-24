import pandas as pd
import numpy as np
from pathlib import Path


def build_features(close: pd.DataFrame,
                   volume: pd.DataFrame,
                   ret: pd.DataFrame) -> pd.DataFrame:
    """
    Build a flat feature matrix: (date × ticker, features).
    All features are lagged by 1 day — zero lookahead.

    Features per stock per day:
      Momentum  : 1d, 5d, 10d, 20d, 60d log return
      Volatility: 5d, 20d realized vol
      Reversal  : 1d return (short-term mean reversion signal)
      Volume    : 5d / 20d volume ratio (turnover proxy)
      CS Rank   : cross-sectional rank of 10d momentum within SEMI
    Target:
      fwd_ret_1d: next-day log return (what we predict)
    """
    SEMI = ['NVDA','AMD','AVGO','TSM','QCOM','AMAT',
            'LRCX','MU','KLAC','TXN','ASML','MRVL']
    semi_tickers = [t for t in SEMI if t in ret.columns]

    records = []

    for ticker in semi_tickers:
        r  = ret[ticker]
        c  = close[ticker]
        v  = volume[ticker]

        df = pd.DataFrame(index=r.index)

        # ── Momentum features (lagged 1 to avoid lookahead) ──
        df['mom_1d']  = r.shift(1)
        df['mom_5d']  = r.rolling(5).sum().shift(1)
        df['mom_10d'] = r.rolling(10).sum().shift(1)
        df['mom_20d'] = r.rolling(20).sum().shift(1)
        df['mom_60d'] = r.rolling(60).sum().shift(1)

        # ── Volatility features ──
        df['vol_5d']  = r.rolling(5).std().shift(1)
        df['vol_20d'] = r.rolling(20).std().shift(1)

        # ── Reversal ──
        df['reversal_1d'] = -r.shift(1)   # negative of yesterday's return

        # ── Volume ratio (activity indicator) ──
        vol_5  = v.rolling(5).mean().shift(1)
        vol_20 = v.rolling(20).mean().shift(1)
        df['vol_ratio'] = (vol_5 / vol_20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

        # ── Price distance from 52-week high ──
        high_252 = c.rolling(252).max().shift(1)
        df['dist_52w_high'] = (c.shift(1) / high_252.replace(0, np.nan) - 1)

        # ── Target: next-day return ──
        df['fwd_ret_1d'] = r   # today's return = yesterday's prediction target

        df['ticker'] = ticker
        records.append(df)

    panel = pd.concat(records)
    panel = panel.reset_index().rename(columns={'index': 'date', 'Date': 'date'})

    # ── Cross-sectional rank of 10d momentum (within SEMI, per day) ──
    pivot_mom10 = panel.pivot(index='date', columns='ticker', values='mom_10d')
    cs_rank     = pivot_mom10.rank(axis=1, pct=True)   # 0–1 percentile rank
    cs_rank_long = cs_rank.stack().reset_index()
    cs_rank_long.columns = ['date', 'ticker', 'cs_rank_mom10']
    panel = panel.merge(cs_rank_long, on=['date','ticker'], how='left')

    panel = panel.set_index(['date','ticker']).sort_index()
    panel = panel.dropna(subset=['fwd_ret_1d'])

    print(f"Feature matrix: {panel.shape}")
    print(f"Columns: {panel.columns.tolist()}")
    return panel


FEATURE_COLS = [
    'mom_1d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_60d',
    'vol_5d', 'vol_20d', 'reversal_1d', 'vol_ratio',
    'dist_52w_high', 'cs_rank_mom10'
]
TARGET_COL = 'fwd_ret_1d'


def save_features(panel: pd.DataFrame, path: str = "data/features.parquet"):
    panel.to_parquet(path)
    print(f"✓ Saved features to {path}")


def load_features(path: str = "data/features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


# Replace the if __name__ == "__main__": block at the bottom of features.py
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load
    close, volume, ret = load()
    panel = build_features(close, volume, ret)
    save_features(panel)

