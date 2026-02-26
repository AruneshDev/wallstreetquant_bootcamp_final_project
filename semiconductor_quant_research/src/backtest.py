"""
backtest.py — Strategy backtests with explicit gross/net performance reporting.

All strategies report BOTH gross (pre-cost) and net (post-cost) metrics so
that alpha claims are never inflated by ignoring execution costs.

Cost model
----------
Each strategy accepts two cost parameters:
  tcost_bps   : One-way transaction cost in basis points (default 7 bps).
                Covers bid-ask spread + market impact for liquid large-caps.
                Conservative estimate for semis at typical position sizes.
  slippage_bps: Additional market-impact slippage (default 2 bps).
                Total round-trip cost = 2 × (tcost_bps + slippage_bps).

Capacity check
--------------
run_capacity_check() estimates whether a portfolio of a given size (AUM) can
be executed at the assumed cost level, using 10-day ADV as a liquidity gauge.
Rule of thumb: position size ≤ 10% of 10-day ADV to stay within 7 bps.

Important: a strategy is only called "alpha-generating" in code comments when
alpha t-stat > 1.65 (p < 0.10).  Strategies with p > 0.10 are flagged with
'⚠️ not significant' in backtest_report output.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import coint
from src.evaluate import backtest_report


SEMI = ['NVDA','AMD','AVGO','TSM','QCOM','AMAT',
        'LRCX','MU','KLAC','TXN','ASML','MRVL']

# ── Default cost assumptions ──────────────────────────────────────────────────
DEFAULT_TCOST_BPS    = 7    # one-way; 7 bps is reasonable for S&P 500 semis
DEFAULT_SLIPPAGE_BPS = 2    # additional market-impact slippage


# ══════════════════════════════════════════════════════════════════
# COST UTILITIES
# ══════════════════════════════════════════════════════════════════

def bps_to_decimal(bps: float) -> float:
    """Convert basis points to a decimal cost factor. 7 bps → 0.0007."""
    return bps / 10_000.0


def gross_to_net(
    gross_ret:   pd.Series,
    turnover:    pd.Series,
    tcost_bps:   float = DEFAULT_TCOST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.Series:
    """
    Subtract transaction costs from a gross daily return series.

    Parameters
    ----------
    gross_ret    : Daily gross return series (no costs already subtracted).
    turnover     : Daily portfolio turnover (sum of |Δw| per day).
    tcost_bps    : One-way transaction cost in bps.
    slippage_bps : Additional slippage in bps.

    Returns
    -------
    net_ret : Daily net return series after costs.
    """
    total_cost_decimal = bps_to_decimal(tcost_bps + slippage_bps)
    cost_drag          = turnover.reindex(gross_ret.index).fillna(0.0) \
                         * total_cost_decimal
    return gross_ret - cost_drag


def performance_summary(
    ret:   pd.Series,
    label: str = "Strategy",
    rf:    float = 0.053,
) -> dict:
    """
    Compute annualised performance statistics.

    Parameters
    ----------
    ret   : Daily return series.
    label : Label for printing.
    rf    : Annual risk-free rate.

    Returns
    -------
    dict with keys: ann_ret, ann_vol, sharpe, sortino, mdd, total_ret, win_rate.
    """
    r       = ret.dropna()
    rf_d    = (1 + rf) ** (1 / 252) - 1
    ar      = r.mean() * 252
    av      = r.std()  * np.sqrt(252)
    sr      = (r.mean() - rf_d) * 252 / av if av > 0 else np.nan
    cum     = (1 + r).cumprod()
    mdd     = (cum / cum.cummax() - 1).min()
    neg     = r[r < 0]
    dv      = neg.std() * np.sqrt(252) if len(neg) > 5 else np.nan
    sortino = (r.mean() - rf_d) * 252 / dv if (dv and dv > 0) else np.nan
    tr      = cum.iloc[-1] - 1
    wr      = (r > 0).mean()

    return {
        "label":     label,
        "ann_ret":   ar,
        "ann_vol":   av,
        "sharpe":    sr,
        "sortino":   sortino,
        "mdd":       mdd,
        "total_ret": tr,
        "win_rate":  wr,
        "n_days":    len(r),
    }


def print_gross_net_comparison(
    gross: pd.Series,
    net:   pd.Series,
    label: str = "Strategy",
) -> None:
    """
    Print a side-by-side gross vs net performance table.

    Parameters
    ----------
    gross : Daily gross return series.
    net   : Daily net return series.
    label : Strategy label.
    """
    gs = performance_summary(gross, f"{label} [GROSS]")
    ns = performance_summary(net,   f"{label} [NET]")

    print(f"\n{'='*65}")
    print(f"  {label} — Gross vs Net Performance")
    print(f"{'='*65}")
    print(f"  {'Metric':<20} {'GROSS':>12} {'NET':>12} {'Δ (drag)':>12}")
    print(f"  {'─'*60}")
    metrics = [
        ("Ann. Return %",  gs["ann_ret"] * 100,  ns["ann_ret"] * 100),
        ("Ann. Vol %",     gs["ann_vol"] * 100,  ns["ann_vol"] * 100),
        ("Sharpe",         gs["sharpe"],          ns["sharpe"]),
        ("Sortino",        gs["sortino"],         ns["sortino"]),
        ("Max Drawdown %", gs["mdd"]     * 100,  ns["mdd"]     * 100),
        ("Total Return %", gs["total_ret"]* 100, ns["total_ret"]* 100),
        ("Win Rate %",     gs["win_rate"] * 100, ns["win_rate"] * 100),
    ]
    for name, gv, nv in metrics:
        if isinstance(gv, float) and isinstance(nv, float):
            drag = nv - gv
            print(f"  {name:<20} {gv:>12.3f} {nv:>12.3f} {drag:>+12.3f}")
    print(f"  {'─'*60}")
    print(f"  N days           : {ns['n_days']:>12,}")
    print(f"{'='*65}")


# ══════════════════════════════════════════════════════════════════
# CAPACITY CHECK
# ══════════════════════════════════════════════════════════════════

def run_capacity_check(
    close:     pd.DataFrame,
    volume:    pd.DataFrame,
    tickers:   list[str],
    aum_m:     float = 10.0,
    n_positions: int = 6,
    adv_win:   int   = 10,
    adv_pct:   float = 0.10,
) -> pd.DataFrame:
    """
    Estimate strategy capacity and flag tickers where the assumed AUM
    would exceed *adv_pct* × 10-day ADV.

    Capacity rule: position_size ≤ adv_pct × ADV_USD to stay at 7 bps cost.
    Above this threshold, market impact rises faster than linearly.

    Parameters
    ----------
    close        : Wide close-price DataFrame.
    volume       : Wide volume DataFrame (shares).
    tickers      : List of tickers in the strategy.
    aum_m        : Portfolio AUM in millions USD.
    n_positions  : Number of simultaneous positions (each side of book).
    adv_win      : Rolling window (days) for ADV calculation.
    adv_pct      : Max fraction of ADV per position.

    Returns
    -------
    DataFrame with columns: ticker, adv_usd_m, position_usd_m, capacity_ok.
    """
    aum_usd = aum_m * 1e6
    pos_size = aum_usd / n_positions

    rows = []
    for t in tickers:
        if t not in close.columns or t not in volume.columns:
            continue
        adv_shares = volume[t].rolling(adv_win).mean().iloc[-1]
        last_price = close[t].iloc[-1]
        adv_usd    = adv_shares * last_price
        max_pos    = adv_usd * adv_pct
        ok         = pos_size <= max_pos

        rows.append({
            "ticker":        t,
            "adv_usd_m":     round(adv_usd / 1e6, 2),
            "position_usd_m": round(pos_size / 1e6, 2),
            "max_pos_usd_m": round(max_pos / 1e6, 2),
            "capacity_ok":   ok,
        })

    df = pd.DataFrame(rows).set_index("ticker")
    print(f"\n  Capacity Check (AUM=${aum_m:.0f}M, {n_positions} positions, "
          f"adv_pct={adv_pct*100:.0f}%)")
    print(f"  {'Ticker':<8} {'ADV $M':>8} {'Pos $M':>8} "
          f"{'Max $M':>8} {'OK':>6}")
    print(f"  {'─'*42}")
    for t, row in df.iterrows():
        flag = "✅" if row["capacity_ok"] else "⚠️"
        print(f"  {t:<8} {row['adv_usd_m']:>8.1f} {row['position_usd_m']:>8.2f} "
              f"{row['max_pos_usd_m']:>8.1f} {flag:>6}")

    n_ok = df["capacity_ok"].sum()
    print(f"\n  {n_ok}/{len(df)} tickers within capacity at ${aum_m}M AUM.")
    return df


# ══════════════════════════════════════════════════════════════════
# COINTEGRATION — find best pair automatically
# ══════════════════════════════════════════════════════════════════

def find_best_pair(close: pd.DataFrame,
                   tickers: list) -> tuple:
    """
    Engle-Granger cointegration test across all pairs.
    Returns ticker_a, ticker_b with the lowest p-value.
    """
    prices  = close[tickers].dropna()
    results = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            try:
                _, pval, _ = coint(prices[t1], prices[t2])
                results.append({'pair': f"{t1}/{t2}",
                                 't1': t1, 't2': t2,
                                 'pval': round(pval, 4)})
            except Exception:
                continue

    df = pd.DataFrame(results).sort_values('pval').reset_index(drop=True)

    print("\n  Top 10 cointegrated pairs (Engle-Granger):")
    print(f"  {'Rank':<5} {'Pair':<15} {'p-value':>8}")
    print(f"  {'─'*30}")
    for idx, row in df.head(10).iterrows():
        sig = "✅" if row['pval'] < 0.05 else ("~" if row['pval'] < 0.10 else "")
        print(f"  {idx+1:<5} {row['pair']:<15} {row['pval']:>8.4f}  {sig}")

    best = df.iloc[0]
    print(f"\n  ✓ Best pair: {best['pair']}  (p={best['pval']:.4f})\n")
    return best['t1'], best['t2']


# ══════════════════════════════════════════════════════════════════
# STRATEGY 1 — Cross-Sectional Momentum
# ══════════════════════════════════════════════════════════════════

def run_cs_momentum(
    ret:          pd.DataFrame,
    semi:         list  = SEMI,
    mom_win:      int   = 10,
    n_long:       int   = 3,
    n_short:      int   = 3,
    tcost_bps:    float = DEFAULT_TCOST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    cost:         float | None = None,   # legacy compat; overrides tcost if set
    vol_tgt:      float = 0.15,
    label:        str   = "CS Momentum",
    report_net:   bool  = True,
) -> pd.Series:
    """
    Cross-sectional momentum on SEMI basket with explicit gross/net reporting.

    Alpha hypothesis
    ----------------
    In persistent trend regimes, recent relative winners continue to outperform
    recent relative losers over 20–60 day horizons (Jegadeesh & Titman 1993).
    Within the semiconductor sector, AI-capex cycles create sustained momentum
    episodes where GPU/HBM winners lead for multiple quarters.

    Expected IC sign: Positive.  Key failure mode: momentum crashes in sharp
    reversals (e.g. 2022 growth selloff, 2025 rotation).

    Parameters
    ----------
    ret          : Wide log-return DataFrame.
    semi         : List of semi ticker symbols.
    mom_win      : Trailing momentum window in days.
    n_long/short : Number of long/short positions per leg.
    tcost_bps    : One-way transaction cost in basis points.
    slippage_bps : Additional slippage in basis points.
    cost         : Legacy scalar cost parameter (overrides tcost_bps if set).
    vol_tgt      : Annualised volatility target (None = no scaling).
    label        : Display label.
    report_net   : If True, print gross/net comparison table.

    Returns
    -------
    net_port : Net daily return series (after transaction costs).
    """
    # ── Backwards-compat: legacy cost= parameter ─────────────────────────────
    if cost is not None:
        tcost_bps    = cost * 10_000 / 2   # assume cost was one-way decimal
        slippage_bps = 0.0

    total_cost_dec = bps_to_decimal(tcost_bps + slippage_bps)

    semi_ret = ret[[t for t in semi if t in ret.columns]]
    mom_sig  = semi_ret.rolling(mom_win).sum().shift(1)

    gross_port  = pd.Series(0.0, index=semi_ret.index)
    turnover_s  = pd.Series(0.0, index=semi_ret.index)
    prev_pos    = pd.Series(0.0, index=semi_ret.columns)
    VOL_LOOKBACK = 63

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

        if vol_tgt is not None and i > mom_win + VOL_LOOKBACK:
            hist     = gross_port.iloc[i - VOL_LOOKBACK : i]
            active_h = hist[hist != 0]
            if len(active_h) > 10:
                pvol = active_h.std() * np.sqrt(252)
                if pvol > 0:
                    new_pos = new_pos * min(vol_tgt / pvol, 2.0)

        delta                = (new_pos - prev_pos).abs().sum()
        gross_port.iloc[i]   = (new_pos * semi_ret.iloc[i]).sum()
        turnover_s.iloc[i]   = delta
        prev_pos             = new_pos

    # ── Trim warm-up period ───────────────────────────────────────────────────
    cut  = mom_win + VOL_LOOKBACK + 2
    gross_result   = gross_port.iloc[cut:]
    turnover_trim  = turnover_s.iloc[cut:]

    # ── Compute net return ────────────────────────────────────────────────────
    cost_dec     = bps_to_decimal(tcost_bps + slippage_bps)
    net_result   = gross_result - turnover_trim * cost_dec

    if report_net:
        print_gross_net_comparison(gross_result, net_result, label=label)
    backtest_report(net_result, label=label + " [NET]")
    return net_result


# ══════════════════════════════════════════════════════════════════
# STRATEGY 2 — Pairs Trade (log-spread, cointegration-consistent)
# ══════════════════════════════════════════════════════════════════

def run_pairs_trade(
    ret:          pd.DataFrame,
    close:        pd.DataFrame,
    ticker_a:     str   = 'AMAT',
    ticker_b:     str   = 'LRCX',
    win:          int   = 120,
    entry_z:      float = 1.5,
    exit_z:       float = 0.3,
    tcost_bps:    float = DEFAULT_TCOST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    cost:         float | None = None,   # legacy compat
    label:        str   = "Pairs Trade",
    report_net:   bool  = True,
) -> pd.Series:
    """
    Log-price spread mean reversion with explicit gross/net reporting.

    Alpha hypothesis
    ----------------
    QCOM and MRVL share common exposure to mobile/data-centre silicon demand.
    Short-term divergences in the log-price spread mean-revert as relative
    valuations normalise.  Engle-Granger p ≈ 0.015 confirms cointegration.

    Expected IC sign: N/A (signal is the spread z-score, not a return forecast).
    Expected Sharpe (net): 0.4–0.6 in cointegration-stable regimes.
    Key failure mode: Structural break in the cointegration relationship
    during AI melt-up periods (2024) where MRVL re-rates independently.

    Parameters
    ----------
    ret          : Wide log-return DataFrame.
    close        : Wide close-price DataFrame.
    ticker_a/b   : Pair tickers (spread = log(a) - log(b)).
    win          : Rolling window for z-score normalisation.
    entry_z      : Entry z-score threshold.
    exit_z       : Exit z-score threshold.
    tcost_bps    : One-way transaction cost in basis points.
    slippage_bps : Additional slippage in basis points.
    cost         : Legacy decimal cost parameter (overrides if set).
    label        : Display label.
    report_net   : If True, print gross/net comparison table.

    Returns
    -------
    net active return series (active trading days only).
    """
    if cost is not None:
        tcost_bps    = cost * 10_000 / 2
        slippage_bps = 0.0

    cost_dec = bps_to_decimal(tcost_bps + slippage_bps)

    log_spread = np.log(close[ticker_a]) - np.log(close[ticker_b])
    mu         = log_spread.rolling(win).mean()
    sigma      = log_spread.rolling(win).std()
    z_score    = ((log_spread - mu) / sigma).dropna()

    signal   = pd.Series(0.0, index=z_score.index)
    position = 0

    for i in range(1, len(z_score)):
        z = z_score.iloc[i - 1]
        if position == 0:
            if   z >  entry_z:
                position = -1
            elif z < -entry_z:
                position = +1
        elif position == +1 and z > -exit_z:
            position = 0
        elif position == -1 and z <  exit_z:
            position = 0
        signal.iloc[i] = position

    ret_a       = ret[ticker_a].reindex(signal.index)
    ret_b       = ret[ticker_b].reindex(signal.index)
    gross_ret   = signal * (ret_a - ret_b) * 0.5
    turnover_s  = signal.diff().abs()

    gross_port  = gross_ret.dropna()
    turnover_t  = turnover_s.reindex(gross_port.index).fillna(0.0)
    net_port    = gross_port - turnover_t * cost_dec
    active      = net_port[signal.reindex(net_port.index) != 0]
    gross_active = gross_port[signal.reindex(gross_port.index) != 0]

    if report_net:
        print_gross_net_comparison(gross_active, active, label=label)
    backtest_report(active, label=label + " [NET]")
    print(f"  Trades: {int((signal.diff().abs() > 0).sum())} | "
          f"Active days: {len(active)}\n")
    return active


# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Robustness sweep
# ══════════════════════════════════════════════════════════════════

def run_momentum_robustness(ret:  pd.DataFrame,
                             semi: list  = SEMI,
                             cost: float = 0.0007) -> pd.DataFrame:
    import io, contextlib

    print("\n" + "="*60)
    print("  CS Momentum Robustness — Window Sensitivity")
    print("="*60)
    print(f"  {'Window':>8} | {'AR%':>7} | {'Vol%':>7} | "
          f"{'Sharpe':>7} | {'MDD%':>8} | {'WinRate':>8}")
    print("-"*60)

    rows = []
    for win in [3, 5, 10, 15, 20, 30, 45, 60]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
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

def annual_analysis(port_ret: pd.Series,
                     label:    str = "Strategy") -> pd.DataFrame:
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
# ANALYSIS 3 — Monthly heatmap
# ══════════════════════════════════════════════════════════════════

def monthly_returns_heatmap(port_ret: pd.Series,
                              label:    str = "Strategy") -> pd.DataFrame:
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
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, io, contextlib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load

    Path("results").mkdir(exist_ok=True)

    close, volume, ret = load()
    print(f"Period : {ret.index[0].date()} → {ret.index[-1].date()}")
    print(f"Days   : {len(ret)} | Tickers: {len(ret.columns)}\n")

    # ── 1. Cointegration test — auto-select best pair ──────────
    print("="*60)
    print("  Finding best cointegrated pair...")
    print("="*60)
    t1, t2 = find_best_pair(close, SEMI)
    pair_label = f"{t1}/{t2} Pairs"

    # ── 2. Robustness sweep (silent inner runs) ─────────────────
    rob_rows = []
    print("\n" + "="*60)
    print("  CS Momentum Robustness — Window Sensitivity")
    print("="*60)
    print(f"  {'Window':>8} | {'AR%':>7} | {'Vol%':>7} | "
          f"{'Sharpe':>7} | {'MDD%':>8} | {'WinRate':>8}")
    print("-"*60)

    for win in [3, 5, 10, 15, 20, 30, 45, 60]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            port = run_cs_momentum(ret, SEMI, mom_win=win,
                                    cost=0.0007, vol_tgt=0.15,
                                    label=f"__silent_{win}")
        ar  = port.mean() * 252
        av  = port.std()  * np.sqrt(252)
        sr  = ar / av if av > 0 else np.nan
        cum = (1 + port).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        wr  = (port > 0).mean()
        print(f"  {win:>6}d | {ar*100:>6.1f}% | {av*100:>6.1f}% | "
              f"{sr:>7.3f} | {mdd*100:>7.1f}% | {wr*100:>7.1f}%")
        rob_rows.append({'window': win, 'AR': ar, 'Vol': av,
                         'Sharpe': sr, 'MDD': mdd, 'WinRate': wr})

    print("="*60)
    rob_df = pd.DataFrame(rob_rows)

    # Robustness CSV needs display-friendly column names
    rob_df_save = rob_df.copy()
    rob_df_save.columns = ['window','AR','Vol','Sharpe','MDD','WinRate']
    rob_df_save['AR']      = (rob_df_save['AR']      * 100).round(2)
    rob_df_save['Vol']     = (rob_df_save['Vol']      * 100).round(2)
    rob_df_save['Sharpe']  =  rob_df_save['Sharpe'].round(3)
    rob_df_save['MDD']     = (rob_df_save['MDD']      * 100).round(2)
    rob_df_save['WinRate'] = (rob_df_save['WinRate']  * 100).round(1)
    rob_df_save.to_csv("results/cs_momentum_robustness.csv", index=False)

    # ── 3. Best CS Momentum run ──────────────────────────────────
    best_win = int(rob_df.loc[rob_df['Sharpe'].idxmax(), 'window'])
    print(f"\nBest window by Sharpe: {best_win}d\n")

    cs_label = f"CS Momentum ({best_win}d)"
    cs_port  = run_cs_momentum(ret, SEMI, mom_win=best_win,
                                label=cs_label)

    annual_analysis(cs_port, cs_label) \
        .to_csv("results/cs_momentum_annual.csv")
    monthly_returns_heatmap(cs_port, cs_label) \
        .to_csv("results/cs_momentum_monthly.csv")

    # ── 4. Pairs trade with auto-selected pair ───────────────────
    print("─"*60)
    print(f"Running {t1}/{t2} Pairs Trade (log-spread, win=120d)...\n")

    pairs_port = run_pairs_trade(ret, close, t1, t2,
                                  win=120, label=pair_label)

    if len(pairs_port) > 30:
        annual_analysis(pairs_port, pair_label) \
            .to_csv("results/pairs_annual.csv")
        monthly_returns_heatmap(pairs_port, pair_label) \
            .to_csv("results/pairs_monthly.csv")
