import pandas as pd
import numpy as np
from pathlib import Path
from src.evaluate import backtest_report


SEMI = ['NVDA','AMD','AVGO','TSM','QCOM','AMAT',
        'LRCX','MU','KLAC','TXN','ASML','MRVL']


# ══════════════════════════════════════════════════════════════════
# STRATEGY 1 — Cross-Sectional Momentum
# ══════════════════════════════════════════════════════════════════

def run_cs_momentum(
    ret:      pd.DataFrame,
    semi:     list  = SEMI,
    mom_win:  int   = 10,
    n_long:   int   = 3,
    n_short:  int   = 3,
    cost:     float = 0.0007,
    vol_tgt:  float = 0.15,
    label:    str   = "CS Momentum"
) -> pd.Series:
    """
    Cross-sectional momentum on SEMI basket.
    Rank by rolling mom_win return (lagged 1d, no lookahead).
    Long top n_long, short bottom n_short, equal-weight.
    Vol targeting activated only after 63 days of real PnL history.
    """
    semi_ret = ret[[t for t in semi if t in ret.columns]]
    mom_sig  = semi_ret.rolling(mom_win).sum().shift(1)

    port     = pd.Series(0.0, index=semi_ret.index)
    prev_pos = pd.Series(0.0, index=semi_ret.columns)
    VOL_LOOKBACK = 63   # minimum PnL history before vol-targeting kicks in

    for i in range(mom_win + 1, len(semi_ret)):
        sig = mom_sig.iloc[i]
        if sig.isna().any():
            continue

        ranked = sig.rank(ascending=False)
        n      = len(semi_ret.columns)
        longs  = ranked[ranked <= n_long].index
        shorts = ranked[ranked > n - n_short].index

        new_pos = pd.Series(0.0, index=semi_ret.columns)
        new_pos[longs]  = +1.0 / n_long
        new_pos[shorts] = -1.0 / n_short

        # Vol targeting — only after enough real PnL history
        if vol_tgt is not None and i > mom_win + VOL_LOOKBACK:
            hist     = port.iloc[i - VOL_LOOKBACK : i]
            active_h = hist[hist != 0]
            if len(active_h) > 10:
                pvol = active_h.std() * np.sqrt(252)
                if pvol > 0:
                    new_pos = new_pos * min(vol_tgt / pvol, 2.0)

        delta        = (new_pos - prev_pos).abs().sum()
        port.iloc[i] = (new_pos * semi_ret.iloc[i]).sum() - delta * cost
        prev_pos     = new_pos

    result = port.iloc[mom_win + VOL_LOOKBACK + 2:]
    backtest_report(result, label=label)
    return result


# ══════════════════════════════════════════════════════════════════
# STRATEGY 2 — Pairs Trade (log-spread, cointegration-consistent)
# ══════════════════════════════════════════════════════════════════

def run_pairs_trade(
    ret:       pd.DataFrame,
    close:     pd.DataFrame,
    ticker_a:  str   = 'NVDA',
    ticker_b:  str   = 'TXN',
    win:       int   = 120,
    entry_z:   float = 1.5,
    exit_z:    float = 0.3,
    cost:      float = 0.0007,
    label:     str   = "Pairs Trade"
) -> pd.Series:
    """
    Log-price spread mean reversion.
    Uses log(A) - log(B) instead of A/B ratio — removes the
    trend drift that destroyed the price-ratio version.
    Entry at ±entry_z standard deviations, exit at ±exit_z.
    """
    log_spread = np.log(close[ticker_a]) - np.log(close[ticker_b])
    mu         = log_spread.rolling(win).mean()
    sigma      = log_spread.rolling(win).std()
    z_score    = ((log_spread - mu) / sigma).dropna()

    signal   = pd.Series(0.0, index=z_score.index)
    position = 0

    for i in range(1, len(z_score)):
        z = z_score.iloc[i - 1]   # yesterday's signal → today's execution
        if position == 0:
            if   z >  entry_z:  position = -1   # spread high → short A, long B
            elif z < -entry_z:  position = +1   # spread low  → long A, short B
        elif position == +1 and z > -exit_z:
            position = 0
        elif position == -1 and z <  exit_z:
            position = 0
        signal.iloc[i] = position

    ret_a      = ret[ticker_a].reindex(signal.index)
    ret_b      = ret[ticker_b].reindex(signal.index)
    spread_ret = signal * (ret_a - ret_b) * 0.5
    tc         = signal.diff().abs() * cost
    port       = (spread_ret - tc).dropna()
    active     = port[signal.reindex(port.index) != 0]

    backtest_report(active, label=label)
    print(f"  Trades: {int((signal.diff().abs() > 0).sum())} | "
          f"Active days: {len(active)}\n")
    return active


# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Robustness table across momentum windows
# ══════════════════════════════════════════════════════════════════

def run_momentum_robustness(
    ret:  pd.DataFrame,
    semi: list  = SEMI,
    cost: float = 0.0007
) -> pd.DataFrame:
    """
    Sweep CS momentum across 8 lookback windows.
    Reveals regime: reversal zone (short) vs momentum zone (long).
    """
    print("\n" + "="*60)
    print("  CS Momentum Robustness — Window Sensitivity")
    print("="*60)
    print(f"  {'Window':>8} | {'AR%':>7} | {'Vol%':>7} | "
          f"{'Sharpe':>7} | {'MDD%':>8} | {'WinRate':>8}")
    print("-"*60)

    rows = []
    for win in [3, 5, 10, 15, 20, 30, 45, 60]:
        port = run_cs_momentum(ret, semi, mom_win=win,
                               cost=cost, vol_tgt=0.15,
                               label=f"__silent_{win}")
        ar  = port.mean() * 252
        av  = port.std()  * np.sqrt(252)
        sr  = ar / av if av > 0 else np.nan
        cum = (1 + port).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        wr  = (port > 0).mean()
        print(f"  {win:>6}d | {ar*100:>6.1f}% | {av*100:>6.1f}% | "
              f"{sr:>7.3f} | {mdd*100:>7.1f}% | {wr*100:>7.1f}%")
        rows.append({'window': win, 'AR': ar, 'Vol': av,
                     'Sharpe': sr, 'MDD': mdd, 'WinRate': wr})

    print("="*60)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Annual breakdown
# ══════════════════════════════════════════════════════════════════

def annual_analysis(
    port_ret: pd.Series,
    label:    str = "Strategy"
) -> pd.DataFrame:
    """
    Per-year performance metrics.
    If all years are positive → signal is consistent across regimes.
    If only one year is positive → regime-dependent, needs more testing.
    """
    r    = port_ret.dropna()
    rows = []

    for year, grp in r.groupby(r.index.year):
        ar  = grp.mean() * 252
        av  = grp.std()  * np.sqrt(252)
        sr  = ar / av if av > 0 else np.nan
        cum = (1 + grp).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        tr  = cum.iloc[-1] - 1
        neg = grp[grp < 0]
        dv  = neg.std() * np.sqrt(252) if len(neg) > 5 else np.nan
        sortino = ar / dv if (dv and dv > 0) else np.nan

        rows.append({
            'year':        year,
            'total_ret%':  round(tr  * 100, 2),
            'annual_ret%': round(ar  * 100, 2),
            'vol%':        round(av  * 100, 2),
            'sharpe':      round(sr,  3),
            'sortino':     round(sortino, 3),
            'mdd%':        round(mdd * 100, 2),
            'win_rate%':   round((grp > 0).mean() * 100, 1),
            'n_days':      len(grp)
        })

    df = pd.DataFrame(rows).set_index('year')
    print(f"\n{'='*75}")
    print(f"  Annual Analysis — {label}")
    print(f"{'='*75}")
    print(df.to_string())
    print(f"{'='*75}\n")
    return df


# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Monthly returns heatmap matrix
# ══════════════════════════════════════════════════════════════════

def monthly_returns_heatmap(
    port_ret: pd.Series,
    label:    str = "Strategy"
) -> pd.DataFrame:
    """
    Compound monthly returns → year × month matrix.
    Used as Plotly heatmap in Streamlit.
    Reveals seasonality: e.g., Jan effect, summer lull, Q4 rally.
    """
    r       = port_ret.dropna().copy()
    r.index = pd.to_datetime(r.index)
    monthly = r.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    pivot = pd.DataFrame({
        'year':  monthly.index.year,
        'month': monthly.index.month,
        'ret':   monthly.values
    }).pivot(index='year', columns='month', values='ret')

    pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                     'Jul','Aug','Sep','Oct','Nov','Dec']

    print(f"\nMonthly Returns (%) — {label}")
    print((pivot * 100).round(1).to_string())
    return pivot


# ══════════════════════════════════════════════════════════════════
# MAIN — Run full pipeline
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load

    Path("results").mkdir(exist_ok=True)

    close, volume, ret = load()

    print(f"Period : {ret.index[0].date()} → {ret.index[-1].date()}")
    print(f"Days   : {len(ret)} | Tickers: {len(ret.columns)}\n")

    # ── 1. Silence individual reports inside robustness sweep ──
    import io, contextlib

    def silent_run(win):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            p = run_cs_momentum(ret, SEMI, mom_win=win,
                                cost=0.0007, vol_tgt=0.15,
                                label=f"__silent_{win}")
        return p

    # ── 2. Robustness sweep ──
    print("\n" + "="*60)
    print("  CS Momentum Robustness — Window Sensitivity")
    print("="*60)
    print(f"  {'Window':>8} | {'AR%':>7} | {'Vol%':>7} | "
          f"{'Sharpe':>7} | {'MDD%':>8} | {'WinRate':>8}")
    print("-"*60)

    rob_rows = []
    for win in [3, 5, 10, 15, 20, 30, 45, 60]:
        port = silent_run(win)
        ar   = port.mean() * 252
        av   = port.std()  * np.sqrt(252)
        sr   = ar / av if av > 0 else np.nan
        cum  = (1 + port).cumprod()
        mdd  = (cum / cum.cummax() - 1).min()
        wr   = (port > 0).mean()
        print(f"  {win:>6}d | {ar*100:>6.1f}% | {av*100:>6.1f}% | "
              f"{sr:>7.3f} | {mdd*100:>7.1f}% | {wr*100:>7.1f}%")
        rob_rows.append({'window': win, 'AR': ar, 'Vol': av,
                         'Sharpe': sr, 'MDD': mdd, 'WinRate': wr})

    print("="*60)
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv("results/cs_momentum_robustness.csv", index=False)

    # ── 3. Full run with best window ──
    best_win = int(rob_df.loc[rob_df['Sharpe'].idxmax(), 'window'])
    print(f"\nBest window by Sharpe: {best_win}d\n")

    cs_port = run_cs_momentum(
        ret, SEMI,
        mom_win=best_win,
        label=f"CS Momentum ({best_win}d)"
    )

    ann_df = annual_analysis(cs_port, f"CS Momentum ({best_win}d)")
    ann_df.to_csv("results/cs_momentum_annual.csv")

    mon_df = monthly_returns_heatmap(cs_port, f"CS Momentum ({best_win}d)")
    mon_df.to_csv("results/cs_momentum_monthly.csv")

    # ── 4. Pairs trade ──
    print("─"*60)
    print("Running NVDA/TXN Pairs Trade (log-spread, win=120d)...\n")
    pairs_port = run_pairs_trade(ret, close, 'NVDA', 'TXN', win=120)

    if len(pairs_port) > 30:
        pairs_ann = annual_analysis(pairs_port, "NVDA/TXN Pairs")
        pairs_ann.to_csv("results/pairs_annual.csv")
        monthly_returns_heatmap(pairs_port, "NVDA/TXN Pairs") \
            .to_csv("results/pairs_monthly.csv")
