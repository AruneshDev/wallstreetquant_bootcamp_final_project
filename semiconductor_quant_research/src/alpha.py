import pandas as pd
import numpy as np
import sys
from pathlib import Path
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════════════════════════
# BENCHMARKS
# ══════════════════════════════════════════════════════════════════

def load_benchmarks():
    import yfinance as yf
    tickers = ['SOXX', 'SPY']
    raw     = yf.download(tickers, start='2020-01-01',   # ← was 2024
                           end='2026-02-22', auto_adjust=True,
                           progress=False)['Close']
    ret     = raw.pct_change().dropna()
    ret.index = pd.to_datetime(ret.index).tz_localize(None)
    return ret


# ══════════════════════════════════════════════════════════════════
# ALPHA DECOMPOSITION
# ══════════════════════════════════════════════════════════════════

def alpha_decomposition(strat_ret: pd.Series,
                         bench_ret: pd.Series,
                         rf_annual: float = 0.053,
                         label:     str   = "Strategy") -> dict:
    """
    R_p - R_f  =  α  +  β × (R_m - R_f)  +  ε
    """
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    both = pd.concat([strat_ret, bench_ret], axis=1).dropna()
    both.columns = ['strat', 'bench']
    both['strat_excess'] = both['strat'] - rf_daily
    both['bench_excess'] = both['bench'] - rf_daily

    X   = sm.add_constant(both['bench_excess'])
    y   = both['strat_excess']
    ols = sm.OLS(y, X).fit()

    alpha_daily = ols.params['const']
    beta        = ols.params['bench_excess']
    alpha_ann   = alpha_daily * 252
    alpha_tstat = ols.tvalues['const']
    alpha_pval  = ols.pvalues['const']
    r2          = ols.rsquared

    resid          = ols.resid
    tracking_error = resid.std() * np.sqrt(252)
    info_ratio     = alpha_ann / tracking_error if tracking_error > 0 else np.nan
    market_corr    = both['strat'].corr(both['bench'])

    ann_ret = both['strat'].mean() * 252
    ann_vol = both['strat'].std()  * np.sqrt(252)
    sharpe  = (both['strat_excess'].mean() * 252) / ann_vol

    print(f"\n{'='*60}")
    print(f"  Alpha Decomposition — {label}")
    print(f"{'='*60}")
    print(f"  Benchmark        : {bench_ret.name}")
    print(f"  N days           : {len(both)}")
    print(f"  Risk-free rate   : {rf_annual*100:.1f}% annual")
    print(f"{'─'*60}")
    print(f"  Annual Return    : {ann_ret*100:.2f}%")
    print(f"  Sharpe Ratio     : {sharpe:.3f}")
    print(f"{'─'*60}")
    print(f"  Jensen's Alpha   : {alpha_ann*100:.2f}% / yr")
    print(f"  Alpha t-stat     : {alpha_tstat:.3f}")
    print(f"  Alpha p-value    : {alpha_pval:.4f}  "
          f"{'✅ significant' if alpha_pval < 0.1 else '⚠️  not significant'}")
    print(f"  Beta             : {beta:.4f}")
    print(f"  R²               : {r2:.4f}  "
          f"({'high beta exposure' if r2 > 0.3 else 'low beta exposure ✅'})")
    print(f"  Market Corr      : {market_corr:.4f}")
    print(f"{'─'*60}")
    print(f"  Tracking Error   : {tracking_error*100:.2f}%")
    print(f"  Information Ratio: {info_ratio:.3f}")
    print(f"{'='*60}")

    return {
        'label':          label,
        'benchmark':      bench_ret.name,
        'n_days':         len(both),
        'ann_return_pct': round(ann_ret   * 100, 2),
        'sharpe':         round(sharpe,     3),
        'alpha_ann_pct':  round(alpha_ann * 100, 2),
        'alpha_tstat':    round(alpha_tstat, 3),
        'alpha_pval':     round(alpha_pval,  4),
        'beta':           round(beta,         4),
        'r2':             round(r2,           4),
        'market_corr':    round(market_corr,  4),
        'tracking_error': round(tracking_error * 100, 2),
        'info_ratio':     round(info_ratio,   3),
    }


# ══════════════════════════════════════════════════════════════════
# ROLLING ALPHA
# ══════════════════════════════════════════════════════════════════

def rolling_alpha(strat_ret: pd.Series,
                   bench_ret: pd.Series,
                   window:    int = 126) -> pd.Series:
    """Rolling annualised Jensen's alpha (126d ≈ 6 months)."""
    both = pd.concat([strat_ret, bench_ret], axis=1).dropna()
    both.columns = ['strat', 'bench']

    alphas = {}
    for i in range(window, len(both)):
        win = both.iloc[i - window : i]
        X   = sm.add_constant(win['bench'])
        ols = sm.OLS(win['strat'], X).fit()
        alphas[both.index[i]] = ols.params['const'] * 252

    return pd.Series(alphas)


# ══════════════════════════════════════════════════════════════════
# STRATEGY CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════

def strategy_correlation(cs_port:    pd.Series,
                          pairs_port: pd.Series,
                          soxx_ret:   pd.Series,
                          spy_ret:    pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({
        'CS Momentum': cs_port,
        'Pairs Trade': pairs_port,
        'SOXX':        soxx_ret,
        'SPY':         spy_ret
    }).dropna()

    corr = df.corr().round(3)
    print("\n  Strategy & Benchmark Correlation Matrix")
    print(f"{'─'*50}")
    print(corr.to_string())
    print(f"{'─'*50}")
    return corr

def yearly_alpha(strat_ret: pd.Series,
                 bench_ret: pd.Series,
                 rf_annual: float = 0.053) -> pd.DataFrame:
    """
    Per-year Jensen's alpha and beta vs a benchmark.

    Returns rows:
      year, ann_ret%, sharpe, alpha%, beta, r2, n_days
    """
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    both = pd.concat([strat_ret, bench_ret], axis=1).dropna()
    both.columns = ['strat', 'bench']

    rows = []

    for year, grp in both.groupby(both.index.year):
        if len(grp) < 50:
            continue

        strat = grp['strat']
        bench = grp['bench']

        strat_ex = strat - rf_daily
        bench_ex = bench - rf_daily

        X   = sm.add_constant(bench_ex)
        y   = strat_ex
        ols = sm.OLS(y, X).fit()

        alpha_daily = ols.params['const']
        beta        = ols.params['bench_excess']
        alpha_ann   = alpha_daily * 252
        r2          = ols.rsquared

        ann_ret = strat.mean() * 252
        ann_vol = strat.std()  * np.sqrt(252)
        sharpe  = (strat_ex.mean() * 252) / ann_vol if ann_vol > 0 else np.nan

        rows.append({
            'year':        year,
            'ann_ret_pct': round(ann_ret   * 100, 2),
            'sharpe':      round(sharpe,     3),
            'alpha_pct':   round(alpha_ann * 100, 2),
            'beta':        round(beta,        4),
            'r2':          round(r2,          4),
            'n_days':      int(len(grp)),
        })

    return pd.DataFrame(rows).set_index('year')



# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.data_loader import load
    from src.backtest import run_cs_momentum, run_pairs_trade
    import contextlib, io

    SEMI = ['NVDA','AMD','AVGO','TSM','QCOM','AMAT',
            'LRCX','MU','KLAC','TXN','ASML','MRVL']

    close, volume, ret = load()

    def silent(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return fn(*args, **kwargs)

    cs_port    = silent(run_cs_momentum, ret, SEMI, mom_win=45)
    pairs_port = silent(run_pairs_trade, ret, close, 'QCOM', 'MRVL', win=120)

    bench = load_benchmarks()
    soxx  = bench['SOXX'].rename('SOXX')
    spy   = bench['SPY'].rename('SPY')

    # ── Full-period alpha decomposition ──────────────────────
    results = []
    results.append(alpha_decomposition(cs_port,    soxx, label="CS Momentum (45d) vs SOXX"))
    results.append(alpha_decomposition(cs_port,    spy,  label="CS Momentum (45d) vs SPY"))
    results.append(alpha_decomposition(pairs_port, soxx, label="QCOM/MRVL Pairs vs SOXX"))
    results.append(alpha_decomposition(pairs_port, spy,  label="QCOM/MRVL Pairs vs SPY"))

    corr_df       = strategy_correlation(cs_port, pairs_port, soxx, spy)
    cs_roll_alpha = rolling_alpha(cs_port,    soxx, window=126)
    p_roll_alpha  = rolling_alpha(pairs_port, soxx, window=126)

    # ── Year-on-year alpha vs SOXX ────────────────────────────
    def yearly_alpha(strat_ret:  pd.Series,
                     bench_ret:  pd.Series,
                     rf_annual:  float = 0.053,
                     label:      str   = "Strategy") -> pd.DataFrame:
        """Per-year Jensen's alpha, beta, Sharpe vs a benchmark."""
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1

        both = pd.concat([strat_ret, bench_ret], axis=1).dropna()
        both.columns = ['strat', 'bench']

        rows = []
        for year, grp in both.groupby(both.index.year):
            if len(grp) < 30:           # skip stub years
                continue

            strat_ex = grp['strat'] - rf_daily
            bench_ex = grp['bench'] - rf_daily

            X   = sm.add_constant(bench_ex)
            ols = sm.OLS(strat_ex, X).fit()

            alpha_ann = ols.params['const'] * 252
            beta      = ols.params[bench_ex.name]
            r2        = ols.rsquared

            ann_ret = grp['strat'].mean() * 252
            ann_vol = grp['strat'].std()  * np.sqrt(252)
            sharpe  = (strat_ex.mean() * 252) / ann_vol if ann_vol > 0 else np.nan

            rows.append({
                'year':        year,
                'n_days':      len(grp),
                'ann_ret%':    round(ann_ret   * 100, 2),
                'sharpe':      round(sharpe,    3),
                'alpha%':      round(alpha_ann * 100, 2),
                'beta':        round(beta,       4),
                'r2':          round(r2,         4),
            })

        df = pd.DataFrame(rows).set_index('year')

        print(f"\n{'='*70}")
        print(f"  Year-on-Year Alpha — {label}")
        print(f"{'='*70}")
        print(f"  {'Year':<6} {'Days':>5} {'AnnRet%':>9} {'Sharpe':>8} "
              f"{'Alpha%':>8} {'Beta':>8} {'R²':>7}")
        print(f"  {'─'*60}")
        for yr, row in df.iterrows():
            stub = "  ⚠️ partial" if row['n_days'] < 60 else ""
            print(f"  {yr:<6} {row['n_days']:>5} {row['ann_ret%']:>9.2f} "
                  f"{row['sharpe']:>8.3f} {row['alpha%']:>8.2f} "
                  f"{row['beta']:>8.4f} {row['r2']:>7.4f}{stub}")
        print(f"{'='*70}")
        return df

    # CS Momentum
    cs_yearly_soxx  = yearly_alpha(cs_port,    soxx, label="CS Momentum (45d) vs SOXX")
    cs_yearly_spy   = yearly_alpha(cs_port,    spy,  label="CS Momentum (45d) vs SPY")

    # Pairs Trade
    p_yearly_soxx   = yearly_alpha(pairs_port, soxx, label="QCOM/MRVL Pairs vs SOXX")
    p_yearly_spy    = yearly_alpha(pairs_port, spy,  label="QCOM/MRVL Pairs vs SPY")

    # ── Combined portfolio (50/50) ────────────────────────────
    combined = pd.concat([cs_port, pairs_port], axis=1).dropna()
    combined.columns = ['cs', 'pairs']
    combined_port = combined.mean(axis=1)
    combined_port.name = "Combined (50/50)"

    print("\n" + "="*50)
    print("  Combined Portfolio (50/50 CS + Pairs)")
    print("="*50)
    ann_r = combined_port.mean() * 252
    ann_v = combined_port.std()  * np.sqrt(252)
    sr    = ann_r / ann_v
    cum   = (1 + combined_port).cumprod()
    mdd   = (cum / cum.cummax() - 1).min()
    print(f"  Annual Return : {ann_r*100:.2f}%")
    print(f"  Annual Vol    : {ann_v*100:.2f}%")
    print(f"  Sharpe        : {sr:.3f}")
    print(f"  Max Drawdown  : {mdd*100:.2f}%")
    print(f"  Corr(CS,Pairs): {combined['cs'].corr(combined['pairs']):.3f}")
    print("="*50)

    combined_yearly = yearly_alpha(combined_port, soxx,
                                    label="Combined (50/50) vs SOXX")

    # ── Save everything ───────────────────────────────────────
    Path("results").mkdir(exist_ok=True)

    pd.DataFrame(results).set_index('label').to_csv(
        "results/alpha_decomposition.csv")
    corr_df.to_csv("results/strategy_correlation.csv")
    cs_roll_alpha.to_csv("results/cs_rolling_alpha.csv",   header=['alpha_ann'])
    p_roll_alpha.to_csv("results/pairs_rolling_alpha.csv", header=['alpha_ann'])
    cs_yearly_soxx.to_csv("results/cs_yearly_alpha_soxx.csv")
    cs_yearly_spy.to_csv("results/cs_yearly_alpha_spy.csv")
    p_yearly_soxx.to_csv("results/pairs_yearly_alpha_soxx.csv")
    p_yearly_spy.to_csv("results/pairs_yearly_alpha_spy.csv")
    combined_yearly.to_csv("results/combined_yearly_alpha.csv")

    print("\n✓ Saved results/alpha_decomposition.csv")
    print("✓ Saved results/strategy_correlation.csv")
    print("✓ Saved results/cs_rolling_alpha.csv")
    print("✓ Saved results/pairs_rolling_alpha.csv")
    print("✓ Saved results/cs_yearly_alpha_soxx.csv")
    print("✓ Saved results/cs_yearly_alpha_spy.csv")
    print("✓ Saved results/pairs_yearly_alpha_soxx.csv")
    print("✓ Saved results/pairs_yearly_alpha_spy.csv")
    print("✓ Saved results/combined_yearly_alpha.csv")
