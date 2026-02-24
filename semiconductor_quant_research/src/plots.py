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
                                 name=t, mode="lines"))
    fig.update_layout(
        title=title,
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.05, xanchor="center", x=0.5)
    )
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="indexed price")
    return fig

def plot_corr_heatmap(ret: pd.DataFrame,
                      tickers: list,
                      title: str = "Correlation matrix") -> go.Figure:
    corr = ret[tickers].corr()
    fig  = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=title
    )
    fig.update_xaxes(title_text="ticker")
    fig.update_yaxes(title_text="ticker")
    return fig

def plot_corr_bar(ret: pd.DataFrame,
                  ref: str = "NVDA",
                  tickers: list = None,
                  title: str = None) -> go.Figure:
    universe = tickers or ret.columns.tolist()
    corr = ret[universe].corrwith(ret[ref]).drop(ref, errors="ignore")
    corr = corr.sort_values(ascending=False)
    df   = corr.reset_index()
    df.columns = ["ticker", "corr"]
    fig  = px.bar(df, x="ticker", y="corr",
                  title=title or f"Same-day correlation with {ref}")
    fig.update_xaxes(title_text="ticker")
    fig.update_yaxes(title_text="pearson corr")
    fig.update_layout(showlegend=False)
    return fig

def plot_rolling_vol(ret: pd.DataFrame,
                     tickers: list,
                     window: int = 21,
                     title: str = None) -> go.Figure:
    fig = go.Figure()
    for t in tickers:
        rvol = ret[t].rolling(window).std() * np.sqrt(252) * 100
        fig.add_trace(go.Scatter(x=rvol.index, y=rvol,
                                 name=t, mode="lines"))
    fig.update_layout(
        title=title or f"Rolling {window}d annualised vol (%)",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.05, xanchor="center", x=0.5)
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
        name="Equity", mode="lines",
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown %",
        line=dict(color="red", dash="dot", width=1),
        yaxis="y2"
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(title="equity (×)"),
        yaxis2=dict(title="drawdown %", overlaying="y",
                    side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.05, xanchor="center", x=0.5)
    )
    fig.update_xaxes(title_text="date")
    return fig

def plot_annual_bar(ann_df: pd.DataFrame,
                    label: str = "Strategy") -> go.Figure:
    df     = ann_df.reset_index()
    colors = ["#22c55e" if v >= 0 else "#ef4444"
              for v in df["total_ret%"]]
    fig = go.Figure(go.Bar(
        x=df["year"].astype(str),
        y=df["total_ret%"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df["total_ret%"]],
        textposition="outside"
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
    fig = px.imshow(
        (monthly_df * 100).round(1),
        text_auto=".1f",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title=f"Monthly returns (%) — {label}"
    )
    fig.update_xaxes(title_text="month")
    fig.update_yaxes(title_text="year")
    return fig

def plot_robustness_bar(rob_df: pd.DataFrame) -> go.Figure:
    colors = ["#22c55e" if v >= 0 else "#ef4444"
              for v in rob_df["Sharpe"]]
    fig = go.Figure(go.Bar(
        x=rob_df["window"].astype(str) + "d",
        y=rob_df["Sharpe"].round(3),
        marker_color=colors,
        text=[f"{v:.3f}" for v in rob_df["Sharpe"]],
        textposition="outside"
    ))
    fig.add_hline(y=0, line_dash="dash",
                  line_color="white", opacity=0.5)
    fig.update_layout(
        title="CS Momentum — Sharpe vs lookback window",
        showlegend=False
    )
    fig.update_xaxes(title_text="momentum window")
    fig.update_yaxes(title_text="sharpe ratio")
    return fig

def plot_pairs_zscore(close: pd.DataFrame,
                      ticker_a: str = "QCOM",
                      ticker_b: str = "MRVL",
                      win: int = 120) -> go.Figure:
    log_spread = np.log(close[ticker_a]) - np.log(close[ticker_b])
    mu     = log_spread.rolling(win).mean()
    sigma  = log_spread.rolling(win).std()
    z      = (log_spread - mu) / sigma

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z.index, y=z.values,
        name="z-score", mode="lines",
        line=dict(width=1.5)
    ))
    for level, color, name in [
        ( 1.5, "orange", "entry +1.5σ"),
        (-1.5, "orange", "entry -1.5σ"),
        ( 0.3, "grey",   "exit  +0.3σ"),
        (-0.3, "grey",   "exit  -0.3σ"),
    ]:
        fig.add_hline(y=level, line_dash="dash",
                      line_color=color, opacity=0.6,
                      annotation_text=name,
                      annotation_position="right")
    fig.add_hline(y=0, line_color="white", opacity=0.3)
    fig.update_layout(
        title=f"{ticker_a}/{ticker_b} log-spread z-score ({win}d rolling)",
        showlegend=False
    )
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="z-score")
    return fig

def plot_rolling_sharpe(port_ret: pd.Series,
                        window: int = 126,
                        title: str = "Rolling Sharpe") -> go.Figure:
    roll_sr = (port_ret.rolling(window).mean() /
               port_ret.rolling(window).std()) * np.sqrt(252)
    fig = px.line(x=roll_sr.index, y=roll_sr.values, title=title)
    fig.add_hline(y=0,   line_dash="dash", line_color="white",   opacity=0.4)
    fig.add_hline(y=0.5, line_dash="dot",  line_color="#22c55e", opacity=0.5)
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="rolling sharpe")
    return fig

# ══════════════════════════════════════════════════════════════════
# INDUSTRIAL CORRELATION / MARKET IMPACT CHARTS
# ══════════════════════════════════════════════════════════════════

def plot_sector_corr_heatmap(sector_corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        sector_corr,
        text_auto=".3f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Sector & index correlation matrix (2020–2026)",
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    return fig

def plot_semi_beta_bar(semi_beta: pd.DataFrame) -> go.Figure:
    df = semi_beta.sort_values("beta_semi", ascending=False)
    fig = go.Figure(go.Bar(
        x=df.index,
        y=df["beta_semi"],
        marker_color="#6366f1",
        text=[f"{v:.3f}" for v in df["beta_semi"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash",
                  line_color="white", opacity=0.4)
    fig.update_layout(
        title="β (sector / index return) vs SOXX daily return",
        showlegend=False
    )
    fig.update_xaxes(title_text="sector")
    fig.update_yaxes(title_text="beta to SOXX")
    return fig

def plot_semi_shock_bar(semi_shock: pd.DataFrame) -> go.Figure:
    """
    Bar chart: average SPY / QQQ moves on SOXX ±2σ days.
    """
    fig = go.Figure()
    for col, color in [("SPY_mean%", "#22c55e"),
                       ("QQQ_mean%", "#6366f1"),
                       ("SOXX_mean%", "#f97316")]:
        fig.add_trace(go.Bar(
            x=semi_shock.index,
            y=semi_shock[col],
            name=col.replace("_mean%", ""),
        ))
    fig.update_layout(
        barmode="group",
        title="Average moves on SOXX ±2σ days",
        showlegend=True
    )
    fig.update_xaxes(title_text="event")
    fig.update_yaxes(title_text="average return %")
    return fig

def plot_semi_spy_risk_bar(semi_spy_risk: pd.DataFrame) -> go.Figure:
    """
    Euler marginal risk contribution (w × β) of each semi to SPY.
    """
    df = semi_spy_risk.sort_values("risk_contribution%", ascending=False)
    fig = go.Figure(go.Bar(
        x=df.index,
        y=df["risk_contribution%"],
        marker_color="#22c55e",
        text=[f"{v:.2f}%" for v in df["risk_contribution%"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Semi marginal risk contribution to SPY (w × β)",
        showlegend=False
    )
    fig.update_xaxes(title_text="ticker")
    fig.update_yaxes(title_text="risk contribution to SPY (%)")
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

    # Load saved results
    rob_df  = pd.read_csv("results/cs_momentum_robustness.csv")
    ann_cs  = pd.read_csv("results/cs_momentum_annual.csv",  index_col=0)
    mon_cs  = pd.read_csv("results/cs_momentum_monthly.csv", index_col=0)
    ann_p   = pd.read_csv("results/pairs_annual.csv",        index_col=0)
    mon_p   = pd.read_csv("results/pairs_monthly.csv",       index_col=0)

    # Industrial correlation results
    sector_corr   = pd.read_csv("results/sector_correlation.csv",           index_col=0)
    semi_beta     = pd.read_csv("results/semi_beta_to_sectors.csv",         index_col=0)
    semi_shock    = pd.read_csv("results/semi_shock_impact.csv",            index_col=0)
    semi_spy_risk = pd.read_csv("results/semi_spy_vol_contribution.csv",    index_col=0)

    # Re-run strategies to get PnL series
    def silent(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return fn(*args, **kwargs)

    cs_port    = silent(run_cs_momentum, ret, SEMI, mom_win=45,
                        label="CS Momentum (45d)")
    pairs_port = silent(run_pairs_trade, ret, close,
                        'QCOM', 'MRVL', win=120)

    print("Generating charts...\n")

    # 1. Normalized prices
    fig = plot_normalized_prices(
        close, SEMI,
        title="Semiconductor prices — normalized to 100 (2020–2026)")
    save(fig, "01_semi_prices",
         "Semiconductor normalized prices 2020–2026")

    # 2. Correlation heatmap
    fig = plot_corr_heatmap(ret, SEMI,
                            title="Semiconductor correlation matrix (2020–2026)")
    save(fig, "02_corr_heatmap",
         "Semiconductor cross-correlation matrix")

    # 3. NVDA correlation bar
    fig = plot_corr_bar(ret, ref="NVDA", tickers=SEMI,
                        title="Same-day correlation of all semis vs NVDA")
    save(fig, "03_nvda_corr_bar",
         "Same-day correlation of all semiconductors vs NVDA")

    # 4. Rolling volatility
    fig = plot_rolling_vol(ret, SEMI[:6], window=21,
                           title="Rolling 21d annualised volatility — top 6 semis")
    save(fig, "04_rolling_vol",
         "Rolling 21-day annualised volatility for top 6 semis")

    # 5. CS momentum equity curve
    fig = plot_equity_curve(cs_port,
                            title="CS Momentum (45d) — equity curve 2020–2026")
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
                            title="QCOM/MRVL Pairs Trade — equity curve 2020–2026")
    save(fig, "09_pairs_equity",
         "QCOM/MRVL pairs trade equity curve and drawdown")

    # 10. Pairs z-score
    fig = plot_pairs_zscore(close, "QCOM", "MRVL", win=120)
    save(fig, "10_pairs_zscore",
         "QCOM/MRVL log-spread z-score with entry/exit thresholds")

    # 11. Pairs annual bar
    fig = plot_annual_bar(ann_p, label="QCOM/MRVL Pairs")
    save(fig, "11_pairs_annual",
         "QCOM/MRVL pairs trade annual returns by year")

    # 12. Rolling Sharpe — CS momentum
    fig = plot_rolling_sharpe(cs_port, window=63,
                              title="CS Momentum — rolling Sharpe (63d)")
    save(fig, "12_cs_mom_rolling_sharpe",
         "CS Momentum rolling 63-day Sharpe ratio")

    # 13. Rolling Sharpe — pairs
    fig = plot_rolling_sharpe(pairs_port, window=63,
                              title="QCOM/MRVL Pairs — rolling Sharpe (63d)")
    save(fig, "13_pairs_rolling_sharpe",
         "QCOM/MRVL pairs rolling 63-day Sharpe ratio")

    # 14. Sector correlation heatmap
    fig = plot_sector_corr_heatmap(sector_corr)
    save(fig, "14_sector_corr",
         "Correlation matrix across sectors and indices")

    # 15. SOXX beta into sectors/indices
    fig = plot_semi_beta_bar(semi_beta)
    save(fig, "15_semi_beta",
         "β (sector / index) vs SOXX daily returns")

    # 16. SOXX shock day impact on SPY/QQQ
    fig = plot_semi_shock_bar(semi_shock)
    save(fig, "16_semi_shock",
         "Average SPY/QQQ/SOXX moves on ±2σ SOXX days")

    # 17. Semi marginal risk contribution to SPY
    fig = plot_semi_spy_risk_bar(semi_spy_risk)
    save(fig, "17_semi_spy_risk",
         "Euler marginal risk contribution of each semi to SPY")

    print("\nAll charts saved to charts/")
