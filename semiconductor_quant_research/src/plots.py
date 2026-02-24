import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
from pathlib import Path

pio.templates.default = "plotly_dark"

CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════

def save(fig: go.Figure, name: str, caption: str, desc: str = ""):
    png  = CHART_DIR / f"{name}.png"
    meta = CHART_DIR / f"{name}.png.meta.json"
    fig.write_image(str(png))
    with meta.open("w") as f:
        json.dump({"caption": caption, "description": desc}, f)
    print(f"  ✓ charts/{name}.png")
    return png


# ══════════════════════════════════════════════════════════════════
# EDA CHARTS
# ══════════════════════════════════════════════════════════════════

def plot_normalized_prices(close: pd.DataFrame,
                            tickers: list,
                            title: str = "Normalized prices (base=100)") -> go.Figure:
    base = close[tickers].div(close[tickers].iloc[0]) * 100
    fig  = go.Figure()
    for t in tickers:
        fig.add_trace(go.Scatter(x=base.index, y=base[t],
                                 name=t, mode='lines'))
    fig.update_layout(
        title=title,
        legend=dict(orientation='h', yanchor='bottom',
                    y=1.05, xanchor='center', x=0.5)
    )
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="indexed price")
    return fig


def plot_corr_heatmap(ret: pd.DataFrame,
                       tickers: list,
                       title: str = "Correlation matrix") -> go.Figure:
    corr = ret[tickers].corr()
    fig  = px.imshow(corr,
                     text_auto=".2f",
                     color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1,
                     title=title)
    fig.update_xaxes(title_text="ticker")
    fig.update_yaxes(title_text="ticker")
    return fig


def plot_corr_bar(ret: pd.DataFrame,
                   ref: str = 'NVDA',
                   tickers: list = None) -> go.Figure:
    universe = tickers or ret.columns.tolist()
    corr = ret[universe].corrwith(ret[ref]).drop(ref, errors='ignore')
    corr = corr.sort_values(ascending=False)
    df   = corr.reset_index()
    df.columns = ['ticker', 'corr']
    fig  = px.bar(df, x='ticker', y='corr',
                  title=f"Same-day correlation with {ref}")
    fig.update_xaxes(title_text="ticker")
    fig.update_yaxes(title_text="pearson corr")
    fig.update_layout(showlegend=False)
    return fig


def plot_rolling_vol(ret: pd.DataFrame,
                      tickers: list,
                      window: int = 21) -> go.Figure:
    fig = go.Figure()
    for t in tickers:
        rvol = ret[t].rolling(window).std() * np.sqrt(252) * 100
        fig.add_trace(go.Scatter(x=rvol.index, y=rvol,
                                 name=t, mode='lines'))
    fig.update_layout(
        title=f"Rolling {window}d annualised vol (%)",
        legend=dict(orientation='h', yanchor='bottom',
                    y=1.05, xanchor='center', x=0.5)
    )
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="ann. vol %")
    return fig


# ══════════════════════════════════════════════════════════════════
# STRATEGY CHARTS
# ══════════════════════════════════════════════════════════════════

def plot_equity_curve(port_ret: pd.Series,
                       title: str = "Equity curve") -> go.Figure:
    r   = port_ret.dropna()
    cum = (1 + r).cumprod()
    dd  = (cum / cum.cummax() - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values,
        name='Equity', mode='lines', fill='tozeroy',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name='Drawdown %',
        line=dict(color='red', dash='dot', width=1),
        yaxis='y2'
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(title="equity (×)"),
        yaxis2=dict(title="drawdown %", overlaying='y',
                    side='right', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom',
                    y=1.05, xanchor='center', x=0.5)
    )
    fig.update_xaxes(title_text="date")
    return fig


def plot_annual_bar(ann_df: pd.DataFrame,
                     label: str = "Strategy") -> go.Figure:
    """
    Bar chart of per-year total return with colour coding:
    green = positive, red = negative.
    """
    df     = ann_df.reset_index()
    colors = ['#22c55e' if v >= 0 else '#ef4444'
              for v in df['total_ret%']]
    fig = go.Figure(go.Bar(
        x=df['year'].astype(str),
        y=df['total_ret%'],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df['total_ret%']],
        textposition='outside'
    ))
    fig.update_layout(
        title=f"Annual returns — {label}",
        showlegend=False
    )
    fig.update_xaxes(title_text="year")
    fig.update_yaxes(title_text="total return %")
    return fig


def plot_monthly_heatmap(monthly_df: pd.DataFrame,
                          label: str = "Strategy") -> go.Figure:
    """
    Month × year return heatmap.
    Red = negative months, green = positive.
    """
    fig = px.imshow(
        (monthly_df * 100).round(1),
        text_auto=".1f",
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title=f"Monthly returns (%) — {label}"
    )
    fig.update_xaxes(title_text="month")
    fig.update_yaxes(title_text="year")
    return fig


def plot_robustness_bar(rob_df: pd.DataFrame) -> go.Figure:
    """
    Sharpe vs momentum window — shows reversal/momentum crossover.
    """
    colors = ['#22c55e' if v >= 0 else '#ef4444'
              for v in rob_df['Sharpe']]
    fig = go.Figure(go.Bar(
        x=rob_df['window'].astype(str) + 'd',
        y=rob_df['Sharpe'].round(3),
        marker_color=colors,
        text=[f"{v:.3f}" for v in rob_df['Sharpe']],
        textposition='outside'
    ))
    fig.add_hline(y=0, line_dash='dash',
                  line_color='white', opacity=0.5)
    fig.update_layout(
        title="CS Momentum — Sharpe vs lookback window",
        showlegend=False
    )
    fig.update_xaxes(title_text="momentum window")
    fig.update_yaxes(title_text="sharpe ratio")
    return fig


def plot_pairs_zscore(close: pd.DataFrame,
                       ticker_a: str = 'NVDA',
                       ticker_b: str = 'TXN',
                       win: int = 120) -> go.Figure:
    """
    Log-spread z-score over time with entry/exit thresholds.
    """
    log_spread = np.log(close[ticker_a]) - np.log(close[ticker_b])
    mu     = log_spread.rolling(win).mean()
    sigma  = log_spread.rolling(win).std()
    z      = (log_spread - mu) / sigma

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z.index, y=z.values,
                             name='z-score', mode='lines',
                             line=dict(width=1.5)))
    for level, color, name in [
        ( 1.5, 'orange', 'entry +1.5'),
        (-1.5, 'orange', 'entry -1.5'),
        ( 0.3, 'grey',   'exit  +0.3'),
        (-0.3, 'grey',   'exit  -0.3'),
    ]:
        fig.add_hline(y=level, line_dash='dash',
                      line_color=color, opacity=0.6,
                      annotation_text=name,
                      annotation_position='right')
    fig.add_hline(y=0, line_color='white', opacity=0.3)
    fig.update_layout(
        title=f"{ticker_a}/{ticker_b} log-spread z-score ({win}d rolling)",
        showlegend=False
    )
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="z-score")
    return fig


def plot_rolling_sharpe(port_ret: pd.Series,
                         window: int = 126,
                         title: str = "Rolling Sharpe (6m)") -> go.Figure:
    """
    Rolling annualised Sharpe — reveals regime changes.
    """
    roll_sr = (port_ret.rolling(window).mean() /
               port_ret.rolling(window).std()) * np.sqrt(252)
    fig = px.line(x=roll_sr.index, y=roll_sr.values, title=title)
    fig.add_hline(y=0,   line_dash='dash', line_color='white',  opacity=0.4)
    fig.add_hline(y=0.5, line_dash='dot',  line_color='#22c55e', opacity=0.5)
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="rolling sharpe")
    return fig


# ══════════════════════════════════════════════════════════════════
# GENERATE ALL CHARTS
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load
    from src.backtest import (run_cs_momentum, run_pairs_trade,
                               annual_analysis, monthly_returns_heatmap)
    import io, contextlib

    SEMI = ['NVDA','AMD','AVGO','TSM','QCOM','AMAT',
            'LRCX','MU','KLAC','TXN','ASML','MRVL']

    close, volume, ret = load()

    # ── Load saved results ──
    rob_df  = pd.read_csv("results/cs_momentum_robustness.csv")
    ann_cs  = pd.read_csv("results/cs_momentum_annual.csv",
                          index_col=0)
    mon_cs  = pd.read_csv("results/cs_momentum_monthly.csv",
                          index_col=0)
    ann_p   = pd.read_csv("results/pairs_annual.csv",
                          index_col=0)
    mon_p   = pd.read_csv("results/pairs_monthly.csv",
                          index_col=0)

    # ── Re-run strategies to get PnL series ──
    def silent(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return fn(*args, **kwargs)

    cs_port    = silent(run_cs_momentum, ret, SEMI, mom_win=45,
                        label="CS Momentum (45d)")
    pairs_port = silent(run_pairs_trade, ret, close,
                        'NVDA', 'TXN', win=120)

    print("Generating charts...\n")

    # 1. Normalized prices
    fig = plot_normalized_prices(
        close, SEMI,
        title="Semiconductor prices — normalized to 100 (2024–2026)")
    save(fig, "01_semi_prices",
         "Semiconductor normalized prices 2024–2026")

    # 2. Correlation heatmap
    fig = plot_corr_heatmap(ret, SEMI,
                            title="Semiconductor correlation matrix (2024–2026)")
    save(fig, "02_corr_heatmap",
         "Semiconductor cross-correlation matrix")

    # 3. NVDA correlation bar
    fig = plot_corr_bar(ret, ref='NVDA', tickers=SEMI)
    save(fig, "03_nvda_corr_bar",
         "Same-day correlation of all semis vs NVDA")

    # 4. Rolling volatility
    fig = plot_rolling_vol(ret, SEMI[:6], window=21)
    save(fig, "04_rolling_vol",
         "Rolling 21d annualised volatility — top 6 semis")

    # 5. CS momentum equity curve
    fig = plot_equity_curve(cs_port,
                            title="CS Momentum (45d) — equity curve 2024–2026")
    save(fig, "05_cs_mom_equity",
         "Cross-sectional momentum equity curve and drawdown")

    # 6. CS momentum robustness
    fig = plot_robustness_bar(rob_df)
    save(fig, "06_cs_mom_robustness",
         "CS Momentum Sharpe vs lookback window")

    # 7. CS momentum annual bar
    fig = plot_annual_bar(ann_cs, label="CS Momentum (45d)")
    save(fig, "07_cs_mom_annual",
         "CS Momentum annual returns by year")

    # 8. CS momentum monthly heatmap
    fig = plot_monthly_heatmap(mon_cs, label="CS Momentum (45d)")
    save(fig, "08_cs_mom_monthly",
         "CS Momentum monthly returns heatmap")

    # 9. Pairs trade equity curve
    fig = plot_equity_curve(pairs_port,
                            title="NVDA/TXN Pairs Trade — equity curve 2024–2026")
    save(fig, "09_pairs_equity",
         "NVDA/TXN pairs trade equity curve and drawdown")

    # 10. Pairs z-score
    fig = plot_pairs_zscore(close, 'NVDA', 'TXN', win=120)
    save(fig, "10_pairs_zscore",
         "NVDA/TXN log-spread z-score with entry/exit thresholds")

    # 11. Pairs annual bar
    fig = plot_annual_bar(ann_p, label="NVDA/TXN Pairs")
    save(fig, "11_pairs_annual",
         "NVDA/TXN pairs trade annual returns by year")

    # 12. Rolling Sharpe — CS momentum
    fig = plot_rolling_sharpe(cs_port, window=63,
                               title="CS Momentum — rolling Sharpe (63d)")
    save(fig, "12_cs_mom_rolling_sharpe",
         "CS Momentum rolling 63-day Sharpe ratio")

    # 13. Rolling Sharpe — pairs
    fig = plot_rolling_sharpe(pairs_port, window=63,
                               title="NVDA/TXN Pairs — rolling Sharpe (63d)")
    save(fig, "13_pairs_rolling_sharpe",
         "NVDA/TXN pairs rolling 63-day Sharpe ratio")

    print(f"\nAll charts saved to charts/")
