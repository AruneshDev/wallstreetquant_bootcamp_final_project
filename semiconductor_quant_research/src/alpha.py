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
    pairs_port = silent(run_pairs_trade, ret, close,
                        'AMAT', 'LRCX', win=120)          # ← was NVDA/TXN

    # ── Benchmarks ──
    bench = load_benchmarks()
    soxx  = bench['SOXX'].rename('SOXX')
    spy   = bench['SPY'].rename('SPY')

    results = []

    r = alpha_decomposition(cs_port, soxx,
                             label="CS Momentum (45d) vs SOXX")
    results.append(r)

    r = alpha_decomposition(cs_port, spy,
                             label="CS Momentum (45d) vs SPY")
    results.append(r)

    r = alpha_decomposition(pairs_port, soxx,
                             label="AMAT/LRCX Pairs vs SOXX") # ← was NVDA/TXN
    results.append(r)

    r = alpha_decomposition(pairs_port, spy,
                             label="AMAT/LRCX Pairs vs SPY")  # ← was NVDA/TXN
    results.append(r)

    corr_df       = strategy_correlation(cs_port, pairs_port, soxx, spy)
    cs_roll_alpha = rolling_alpha(cs_port,    soxx, window=126)
    p_roll_alpha  = rolling_alpha(pairs_port, soxx, window=126)

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(results).set_index('label').to_csv(
        "results/alpha_decomposition.csv")
    corr_df.to_csv("results/strategy_correlation.csv")
    cs_roll_alpha.to_csv("results/cs_rolling_alpha.csv",
                          header=['alpha_ann'])
    p_roll_alpha.to_csv("results/pairs_rolling_alpha.csv",
                         header=['alpha_ann'])

    print("\n✓ Saved results/alpha_decomposition.csv")
    print("✓ Saved results/strategy_correlation.csv")
    print("✓ Saved results/cs_rolling_alpha.csv")
    print("✓ Saved results/pairs_rolling_alpha.csv")
