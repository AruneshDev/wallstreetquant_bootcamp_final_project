import pandas as pd
import numpy as np
from pathlib import Path


SEMI = ['NVDA','AMD','AVGO','TSM','QCOM','AMAT',
        'LRCX','MU','KLAC','TXN','ASML','MRVL']

FEATURE_COLS = [
    'mom_1d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_60d',
    'vol_5d', 'vol_20d',
    'reversal_1d',
    'vol_ratio',
    'dist_52w_high', 'dist_52w_low',
    'rsi_norm',
    'macd_hist',
    'cs_rank_mom10',
]
TARGET_COL = 'fwd_ret_5d'    # ← was fwd_ret_1d


def build_features(close: pd.DataFrame,
                   volume: pd.DataFrame,
                   ret:    pd.DataFrame) -> pd.DataFrame:
    """
    Flat feature matrix: (date × ticker, features).
    All features lagged by 1 day — zero lookahead.

    Features:
      Momentum    : 1d, 5d, 10d, 20d, 60d log return
      Volatility  : 5d, 20d realised vol
      Reversal    : -1d return (mean-reversion signal)
      Volume ratio: 5d / 20d turnover proxy
      52w position: distance from 52w high and 52w low
      RSI (14d)   : centred at 0  (rsi_norm = (RSI-50)/50)
      MACD hist   : (EMA12 - EMA26 - signal9) / price
      CS Rank     : cross-sectional percentile rank of 10d mom
    Target:
      fwd_ret_5d  : next 5-day cumulative log return
                    (~2x less noisy than 1d, improves IC)
    """
    semi_tickers = [t for t in SEMI if t in ret.columns]
    records = []

    for ticker in semi_tickers:
        r = ret[ticker]
        c = close[ticker]
        v = volume[ticker]

        df = pd.DataFrame(index=r.index)

        # ── Momentum (lagged 1d) ──────────────────────────────
        for w in [1, 5, 10, 20, 60]:
            df[f'mom_{w}d'] = r.rolling(w).sum().shift(1)

        # ── Volatility ────────────────────────────────────────
        df['vol_5d']  = r.rolling(5).std().shift(1)
        df['vol_20d'] = r.rolling(20).std().shift(1)

        # ── Short-term reversal ───────────────────────────────
        df['reversal_1d'] = -r.shift(1)

        # ── Volume ratio ──────────────────────────────────────
        vol_5  = v.rolling(5).mean().shift(1)
        vol_20 = v.rolling(20).mean().shift(1)
        df['vol_ratio'] = (
            vol_5 / vol_20.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

        # ── 52-week high / low distance ───────────────────────
        high_252 = c.rolling(252).max().shift(1)
        low_252  = c.rolling(252).min().shift(1)
        c_lag    = c.shift(1)
        df['dist_52w_high'] = (c_lag / high_252.replace(0, np.nan) - 1)
        df['dist_52w_low']  = (c_lag / low_252.replace(0, np.nan)  - 1)

        # ── RSI (14d), centred ────────────────────────────────
        delta = r.shift(1)
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)
        df['rsi_norm'] = (rsi - 50) / 50

        # ── MACD histogram (price-normalised) ────────────────
        ema12     = c.ewm(span=12, adjust=False).mean().shift(1)
        ema26     = c.ewm(span=26, adjust=False).mean().shift(1)
        macd      = ema12 - ema26
        macd_sig  = macd.ewm(span=9, adjust=False).mean()
        price_lag = c.shift(1).replace(0, np.nan)
        df['macd_hist'] = (macd - macd_sig) / price_lag

        # ── Target: 5-day forward return (NO shift — lookahead
        #    is intentional here, this is the label not a feature)
        df['fwd_ret_5d'] = r.rolling(5).sum().shift(-5)  # ← key change

        df['ticker'] = ticker
        records.append(df)

    # ── Stack all tickers ─────────────────────────────────────
    panel = pd.concat(records)
    panel = (panel
             .reset_index()
             .rename(columns={'index': 'date', 'Date': 'date'}))

    # ── Cross-sectional rank of 10d momentum ─────────────────
    pivot_mom10  = panel.pivot(index='date', columns='ticker',
                               values='mom_10d')
    cs_rank      = pivot_mom10.rank(axis=1, pct=True)
    cs_rank_long = (cs_rank
                    .stack()
                    .reset_index()
                    .rename(columns={0: 'cs_rank_mom10'}))
    panel = panel.merge(cs_rank_long, on=['date', 'ticker'], how='left')

    panel = (panel
             .set_index(['date', 'ticker'])
             .sort_index()
             .dropna(subset=['fwd_ret_5d']))   # ← was fwd_ret_1d

    print(f"Feature matrix: {panel.shape}")
    print(f"Columns: {[c for c in panel.columns if c not in ['fwd_ret_5d']]}")
    return panel


def save_features(panel: pd.DataFrame,
                  path:  str = "data/features.parquet"):
    panel.to_parquet(path)
    print(f"✓ Saved features to {path}")


def load_features(path: str = "data/features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load
    close, volume, ret = load()
    panel = build_features(close, volume, ret)
    save_features(panel)
    print(f"\nPeriod : {panel.index.get_level_values('date').min()} "
          f"→ {panel.index.get_level_values('date').max()}")
    print(f"Days   : {panel.index.get_level_values('date').nunique()} "
          f"| Tickers: {panel.index.get_level_values('ticker').nunique()}")
    print(f"\nFeature stats:\n{panel[FEATURE_COLS].describe().round(4)}")
