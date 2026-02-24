import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# ── Sector ETFs (GICS sectors) ─────────────────────────────────
SECTOR_ETFS = {
    'Semiconductors': 'SOXX',
    'Technology':     'XLK',
    'Financials':     'XLF',
    'Healthcare':     'XLV',
    'Energy':         'XLE',
    'Industrials':    'XLI',
    'Consumer Disc':  'XLY',
    'Consumer Stapl': 'XLP',
    'Utilities':      'XLU',
    'Real Estate':    'XLRE',
    'Materials':      'XLB',
    'Communication':  'XLC',
    'Nasdaq-100':     'QQQ',
    'S&P 500':        'SPY',
    'Russell 2000':   'IWM',
}

def load_sector_returns(start='2020-01-01',
                        end='2026-02-20') -> pd.DataFrame:
    tickers = list(SECTOR_ETFS.values())
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)['Close']
    ret = raw.pct_change().dropna()
    ret.index = pd.to_datetime(ret.index).tz_localize(None)
# CORRECT — renames each column by matching its ticker symbol to the dict
    ticker_to_name = {v: k for k, v in SECTOR_ETFS.items()}
    ret = ret.rename(columns=ticker_to_name)

    return ret


def sector_correlation_matrix(sector_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Full-period Pearson correlation matrix: semis vs every sector.
    This shows how much of each sector's daily return is explained
    by semiconductor moves.
    """
    corr = sector_ret.corr().round(3)
    print("\n  Sector Correlation Matrix (Full Period 2020–2026)")
    print(f"{'─'*70}")
    print(corr.to_string())
    print(f"{'─'*70}")
    return corr


def semi_beta_to_sectors(sector_ret: pd.DataFrame,
                          semi_col: str = 'Semiconductors') -> pd.DataFrame:
    """
    Regress each sector on SOXX:
        R_sector = α + β_semi × R_SOXX + ε

    Reports:
      - β_semi (how many $ of sector move per $1 of SOXX move)
      - R² (what % of sector variance is explained by semis)
      - α (sector's own daily alpha vs semis)
    """
    soxx = sector_ret[semi_col]
    X = sm.add_constant(soxx)

    rows = []
    for sector in sector_ret.columns:
        if sector == semi_col:
            continue
        y = sector_ret[sector].dropna()
        X_ = X.reindex(y.index).dropna()
        y = y.reindex(X_.index)
        ols = sm.OLS(y, X_).fit()

        rows.append({
            'sector':     sector,
            'beta_semi':  round(ols.params[semi_col], 4),
            'alpha_ann':  round(ols.params['const'] * 252 * 100, 2),
            'r2':         round(ols.rsquared, 4),
            'corr':       round(y.corr(soxx), 4),
            'n_days':     len(y),
        })

    df = pd.DataFrame(rows).set_index('sector').sort_values(
        'r2', ascending=False)

    print(f"\n{'='*65}")
    print(f"  Semiconductor β to Each Sector (SOXX as driver)")
    print(f"{'='*65}")
    print(f"  {'Sector':<18} {'β_semi':>8} {'α%/yr':>8} {'R²':>7} {'Corr':>8}")
    print(f"  {'─'*55}")
    for s, row in df.iterrows():
        print(f"  {s:<18} {row['beta_semi']:>8.4f} "
              f"{row['alpha_ann']:>8.2f}% {row['r2']:>7.4f} "
              f"{row['corr']:>8.4f}")
    print(f"{'='*65}")
    return df
def semi_market_cap_impact(sector_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Quantify: on days when semis move ±2σ, what happens to SPY and QQQ?
    This captures the 'trillions of dollars moving' dynamic.
    """
    soxx = sector_ret['Semiconductors']
    spy  = sector_ret['S&P 500']
    qqq  = sector_ret['Nasdaq-100']

    sigma     = soxx.std()
    up_days   = soxx[soxx >  2 * sigma].index
    down_days = soxx[soxx < -2 * sigma].index

    def stats(series, days, label):
        sub = series.reindex(days).dropna()
        return {
            'event':       label,
            'n_days':      len(sub),
            'SPY_mean%':   round(spy.reindex(days).dropna().mean() * 100, 3),
            'QQQ_mean%':   round(qqq.reindex(days).dropna().mean() * 100, 3),
            'SOXX_mean%':  round(soxx.reindex(days).dropna().mean() * 100, 3),
        }

    rows = [
        stats(spy, up_days,   'SOXX up >2σ'),
        stats(spy, down_days, 'SOXX down >2σ'),
    ]

    df = pd.DataFrame(rows).set_index('event')
    print(f"\n{'='*65}")
    print("  Semi Shock Impact on Broad Market (±2σ SOXX days)")
    print(f"{'='*65}")
    print(df.to_string())
    print(f"{'='*65}")
    return df
def rolling_semi_dominance(sector_ret: pd.DataFrame,
                            window: int = 126) -> pd.DataFrame:
    """
    Rolling 6-month R² of SOXX vs SPY and QQQ.
    Shows how semiconductor dominance has grown over 2020–2026:
    - Pre-AI (2020–2022): lower R²
    - Post-ChatGPT (2023–2026): higher R²
    """
    soxx = sector_ret['Semiconductors']
    results = {}

    for i in range(window, len(sector_ret)):
        win  = sector_ret.iloc[i - window: i]
        date = sector_ret.index[i]
        row  = {}
        for col in ['S&P 500', 'Nasdaq-100', 'Technology',
                    'Financials', 'Industrials']:
            if col not in win.columns:
                continue
            X   = sm.add_constant(win['Semiconductors'])
            ols = sm.OLS(win[col], X).fit()
            row[f'r2_{col}'] = ols.rsquared
        results[date] = row

    df = pd.DataFrame(results).T.round(4)
    print(f"\n✓ Rolling semi dominance computed ({window}d window)")
    return df
def nvda_market_weight_impact(semi_ret: pd.DataFrame,
                              spy_ret: pd.Series) -> pd.DataFrame:
    SPY_WEIGHTS = {
        'NVDA': 0.074, 'AVGO': 0.022, 'TSM': 0.015,
        'AMD':  0.007, 'QCOM': 0.007, 'AMAT': 0.004,
        'LRCX': 0.003, 'MU':   0.003, 'KLAC': 0.003,
        'TXN':  0.005, 'ASML': 0.003, 'MRVL': 0.002,
    }

    rows = []
    X = sm.add_constant(spy_ret.rename('SPY'))
    for tkr, wt in SPY_WEIGHTS.items():
        if tkr not in semi_ret.columns:
            continue
        y   = semi_ret[tkr].dropna()
        X_  = X.reindex(y.index).dropna()
        y   = y.reindex(X_.index)
        ols = sm.OLS(y, X_).fit()

        ann_vol = y.std() * np.sqrt(252)

        rows.append({
            'ticker':             tkr,
            'spy_weight%':        round(wt * 100, 2),
            'beta_to_spy':        round(ols.params['SPY'], 4),
            'r2':                 round(ols.rsquared, 4),
            'ann_vol%':           round(ann_vol * 100, 2),
            # Euler marginal risk contribution = w_i × β_i
            'risk_contribution%': round(wt * ols.params['SPY'] * 100, 3),
        })

    df = pd.DataFrame(rows).set_index('ticker').sort_values(
        'risk_contribution%', ascending=False)

    print(f"\n{'='*70}")
    print("  Semi Marginal Risk Contribution to SPY (Euler Decomposition)")
    print(f"{'='*70}")
    print(f"  {'Ticker':<8} {'Wt%':>6} {'β→SPY':>8} {'R²':>7} "
          f"{'Vol%':>7} {'RiskContrib%':>14}")
    print(f"  {'─'*60}")
    for t, row in df.iterrows():
        print(f"  {t:<8} {row['spy_weight%']:>6.2f} "
              f"{row['beta_to_spy']:>8.4f} {row['r2']:>7.4f} "
              f"{row['ann_vol%']:>7.2f} "
              f"{row['risk_contribution%']:>14.3f}%")

    total = df['risk_contribution%'].sum()
    print(f"  {'─'*60}")
    print(f"  {'Total semi risk contribution to SPY':>54}: "
          f"{total:.3f}%")
    print(f"{'='*70}")
    return df

if __name__ == "__main__":
    from src.data_loader import load

    close, volume, ret = load()

    # Sector returns (15 ETFs)
    sector_ret = load_sector_returns()

    # 1) Correlation matrix
    corr_matrix = sector_correlation_matrix(sector_ret)

    # 2) Beta of each sector to SOXX
    beta_df = semi_beta_to_sectors(sector_ret)

    # 3) Market shock impact on SPY/QQQ
    shock_df = semi_market_cap_impact(sector_ret)

    # 4) Rolling SOXX dominance (R² over time)
    rolling_df = rolling_semi_dominance(sector_ret, window=126)

    # 5) Individual semi contribution to SPY vol
    spy_ret = sector_ret['S&P 500']
    contrib_df = nvda_market_weight_impact(ret, spy_ret)

    # ── Save ──────────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    corr_matrix.to_csv("results/sector_correlation.csv")
    beta_df.to_csv("results/semi_beta_to_sectors.csv")
    shock_df.to_csv("results/semi_shock_impact.csv")
    rolling_df.to_csv("results/rolling_semi_dominance.csv")
    contrib_df.to_csv("results/semi_spy_vol_contribution.csv")

    print("\n✓ Saved results/sector_correlation.csv")
    print("✓ Saved results/semi_beta_to_sectors.csv")
    print("✓ Saved results/semi_shock_impact.csv")
    print("✓ Saved results/rolling_semi_dominance.csv")
    print("✓ Saved results/semi_spy_vol_contribution.csv")
