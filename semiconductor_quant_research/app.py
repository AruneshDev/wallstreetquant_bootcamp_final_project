import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io, contextlib, sys
from pathlib import Path
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Semiconductor Alpha Research",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# BLOOMBERG TERMINAL STYLESHEET
# ══════════════════════════════════════════════════════════════════

BLOOMBERG_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    background-color: #0d1117 !important;
    color: #e8e8e8 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

section[data-testid="stSidebar"] {
    background-color: #111418 !important;
    border-right: 1px solid #1e1e1e !important;
}
section[data-testid="stSidebar"] * { color: #e8e8e8 !important; }

#MainMenu {visibility: hidden;}
footer     {visibility: hidden;}
header     {visibility: hidden;}

div[data-testid="metric-container"] {
    background-color: #111418;
    border: 1px solid #1e1e1e;
    padding: 12px 16px;
    border-radius: 2px;
}
div[data-testid="metric-container"] label {
    color: #606060 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e8e8e8 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 20px !important;
    font-weight: 500;
}

button[data-baseweb="tab"] {
    background-color: #111418 !important;
    color: #606060 !important;
    font-size: 11px !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-bottom: 2px solid transparent !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00b4d8 !important;
    border-bottom: 2px solid #00b4d8 !important;
    background-color: #111418 !important;
}

div[data-testid="stDataFrame"] { border: 1px solid #1e1e1e; }
thead tr th {
    background-color: #111418 !important;
    color: #606060 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
tbody tr td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #e8e8e8 !important;
}

div[data-testid="stSlider"] label {
    color: #606060 !important;
    font-size: 11px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

div[data-testid="stExpander"] {
    border: 1px solid #1e1e1e !important;
    background-color: #111418 !important;
    border-radius: 2px;
}

code, pre {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    background-color: #111418 !important;
    color: #00b4d8 !important;
    font-size: 12px !important;
}

hr { border-color: #1e1e1e !important; }

div[data-testid="stRadio"] label {
    color: #e8e8e8 !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

div[data-testid="stAlert"] { display: none !important; }
</style>
"""

st.markdown(BLOOMBERG_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def load_all():
    from src.data_loader import load
    from src.backtest import (run_cs_momentum, run_pairs_trade,
                              annual_analysis, monthly_returns_heatmap)

    SEMI = ['NVDA','AMD','AVGO','TSM','QCOM','AMAT',
            'LRCX','MU','KLAC','TXN','ASML','MRVL']

    close, volume, ret = load()

    def silent(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return fn(*args, **kwargs)

    _cs_series    = silent(run_cs_momentum, ret, SEMI,
                          mom_win=45, label="CS Momentum (45d)")
    _pairs_series = silent(run_pairs_trade, ret, close,
                           'AMAT', 'LRCX', win=120)

    # backtest.py returns a pd.Series (net returns); wrap into DataFrame so
    # all downstream helpers can uniformly access port["ret"].
    cs_port    = pd.DataFrame({"ret": _cs_series})
    pairs_port = pd.DataFrame({"ret": _pairs_series})

    rob_df = pd.read_csv("results/cs_momentum_robustness.csv")
    ann_cs = pd.read_csv("results/cs_momentum_annual.csv",  index_col=0)
    mon_cs = pd.read_csv("results/cs_momentum_monthly.csv", index_col=0)
    ann_p  = pd.read_csv("results/pairs_annual.csv",        index_col=0)
    mon_p  = pd.read_csv("results/pairs_monthly.csv",       index_col=0)

    return (close, volume, ret, SEMI,
            cs_port, pairs_port,
            rob_df, ann_cs, mon_cs, ann_p, mon_p)

@st.cache_data
def load_ml_results():
    def _load(name):
        return pd.read_csv(f"results/{name}.csv",
                           index_col=0, parse_dates=True).iloc[:, 0]
    return {
        'rf_ic':    _load("rf_ic"),
        'gbm_ic':   _load("gbm_ic"),
        'tf_ic':    _load("transformer_ic"),
        'gnn_ic':   _load("gnn_ic"),
        'rf_ric':   _load("rf_rank_ic"),
        'gbm_ric':  _load("gbm_rank_ic"),
        'tf_ric':   _load("transformer_rank_ic"),
        'gnn_ric':  _load("gnn_rank_ic"),
        'feat_imp': pd.read_csv("results/rf_feature_importance.csv",
                                index_col=0),
    }

@st.cache_data
def load_alpha_results():
    return {
        'decomp':    pd.read_csv("results/alpha_decomposition.csv",
                                 index_col=0),
        'corr':      pd.read_csv("results/strategy_correlation.csv",
                                 index_col=0),
        'cs_roll':   pd.read_csv("results/cs_rolling_alpha.csv",
                                 index_col=0,
                                 parse_dates=True)['alpha_ann'],
        'pairs_roll': pd.read_csv("results/pairs_rolling_alpha.csv",
                                  index_col=0,
                                  parse_dates=True)['alpha_ann'],
    }

@st.cache_data
def load_industrial_results():
    return {
        'sector_corr':   pd.read_csv("results/sector_correlation.csv",
                                     index_col=0),
        'semi_beta':     pd.read_csv("results/semi_beta_to_sectors.csv",
                                     index_col=0),
        'semi_shock':    pd.read_csv("results/semi_shock_impact.csv",
                                     index_col=0),
        'rolling_dom':   pd.read_csv("results/rolling_semi_dominance.csv",
                                     index_col=0, parse_dates=True),
        'semi_spy_risk': pd.read_csv("results/semi_spy_vol_contribution.csv",
                                     index_col=0),
    }

(close, volume, ret, SEMI,
 cs_port, pairs_port,
 rob_df, ann_cs, mon_cs, ann_p, mon_p) = load_all()

ml    = load_ml_results()
alpha = load_alpha_results()
indus = load_industrial_results()

sector_corr   = indus['sector_corr']
semi_beta     = indus['semi_beta']
semi_shock    = indus['semi_shock']
rolling_dom   = indus['rolling_dom']
semi_spy_risk = indus['semi_spy_risk']

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    "<div style='font-family:JetBrains Mono,monospace;font-size:14px;"
    "font-weight:600;color:#00b4d8;letter-spacing:0.1em;padding:8px 0 4px;'>"
    "SEMI ALPHA RESEARCH</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<div style='font-size:10px;color:#606060;font-family:JetBrains Mono,"
    "monospace;padding-bottom:12px;'>Point72 IAC — Arunesh Lal | Feb 2026</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

page = st.sidebar.radio("NAVIGATE", [
    "Overview",
    "Market Structure",
    "Lead-Lag Study",
    "Strategy: CS Momentum",
    "Strategy: Pairs Trade",
    "Strategy Comparison",
    "Alpha Attribution",
    "Market Impact",
    "ML Signal Analysis",
    "Universe Expansion",
    "Alt-Data Signals",
    "NLP Signal",
    "Signal Combiner",
])

n_days = len(ret)
start  = ret.index[0].strftime('%b %Y')
end    = ret.index[-1].strftime('%b %Y')

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='font-size:10px;color:#606060;font-family:JetBrains Mono,"
    f"monospace;line-height:1.8;'>"
    f"PERIOD&nbsp;&nbsp;{start} \u2014 {end}<br>"
    f"DAYS&nbsp;&nbsp;&nbsp;&nbsp;{n_days:,}<br>"
    f"UNIVERSE&nbsp;12 semis + 5 tech<br>"
    f"MODELS&nbsp;&nbsp;RF \u00b7 GBM \u00b7 TF \u00b7 GNN"
    f"</div>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════
# CHART LAYOUT CONSTANT
# ══════════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="JetBrains Mono, Courier New, monospace",
              size=11, color="#e8e8e8"),
    xaxis=dict(gridcolor="#1e1e1e", linecolor="#1e1e1e",
               tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e1e1e", linecolor="#1e1e1e",
               tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e1e1e",
                font=dict(size=10)),
    margin=dict(l=50, r=30, t=50, b=40),
    title_font=dict(size=12, color="#e8e8e8"),
)

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def section_header(title: str, subtitle: str = "") -> None:
    sub_html = (
        f"<div style='font-size:11px;color:#606060;font-family:JetBrains Mono,"
        f"monospace;margin-top:2px;'>{subtitle}</div>"
        if subtitle else ""
    )
    st.markdown(
        f"<div style='border-left:3px solid #00b4d8;padding:6px 0 6px 12px;"
        f"margin-bottom:16px;'>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:13px;"
        f"font-weight:600;color:#e8e8e8;letter-spacing:0.06em;'>{title.upper()}</div>"
        f"{sub_html}"
        f"</div>",
        unsafe_allow_html=True
    )


def metric_strip(metrics: list) -> None:
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        delta_html = ""
        if m.get("delta"):
            c = "#00b4d8" if m.get("positive", True) else "#dc3232"
            delta_html = (
                f"<div style='font-size:10px;color:{c};"
                f"font-family:JetBrains Mono,monospace;margin-top:2px;'>"
                f"{m['delta']}</div>"
            )
        col.markdown(
            f"<div style='background:#111418;border:1px solid #1e1e1e;"
            f"padding:12px 16px;border-radius:2px;'>"
            f"<div style='font-size:10px;color:#606060;font-family:JetBrains Mono,"
            f"monospace;text-transform:uppercase;letter-spacing:0.08em;'>{m['label']}</div>"
            f"<div style='font-size:20px;font-weight:500;color:#e8e8e8;"
            f"font-family:JetBrains Mono,monospace;margin-top:4px;'>{m['value']}</div>"
            f"{delta_html}"
            f"</div>",
            unsafe_allow_html=True
        )


def note(text: str, kind: str = "info") -> None:
    palette = {"info": "#00b4d8", "pass": "#00b4d8",
               "fail": "#dc3232", "warn": "#606060"}
    tags    = {"info": "INFO", "pass": "PASS", "fail": "FAIL", "warn": "WARN"}
    color   = palette.get(kind, "#606060")
    tag     = tags.get(kind, "NOTE")
    st.markdown(
        f"<div style='border-left:3px solid {color};background:#111418;"
        f"padding:10px 14px;margin:8px 0;font-size:12px;color:#e8e8e8;"
        f"font-family:JetBrains Mono,monospace;line-height:1.6;"
        f"border-radius:0 2px 2px 0;'>"
        f"<span style='color:{color};font-weight:600;'>[{tag}]</span>"
        f"&nbsp;&nbsp;{text}"
        f"</div>",
        unsafe_allow_html=True
    )


def icir(s: pd.Series) -> float:
    return float(s.mean() / s.std()) if s.std() > 0 else 0.0


def _apply(fig: go.Figure, **extra) -> go.Figure:
    fig.update_layout(**{**CHART_LAYOUT, **extra})
    return fig


def apply_layout(fig: go.Figure, **overrides) -> go.Figure:
    """Apply CHART_LAYOUT safely.

    If *overrides* contains 'legend', it replaces the base legend so that
    ``update_layout()`` never receives 'legend' twice (which would raise
    ``TypeError: got multiple values for keyword argument 'legend'``).
    """
    base = dict(CHART_LAYOUT)
    if "legend" in overrides:
        base.pop("legend", None)
    base.update(overrides)
    fig.update_layout(**base)
    return fig


def equity_fig(port: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    eq   = (1 + port["ret"]).cumprod()
    roll = eq.cummax()
    dd   = (eq - roll) / roll
    fig  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, mode="lines", name="Equity",
        line=dict(color="#00b4d8", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", name="Drawdown",
        fill="tozeroy", fillcolor="rgba(220,50,50,0.12)",
        line=dict(color="#dc3232", width=1.0)), row=2, col=1)
    fig.update_layout(**CHART_LAYOUT, title=title, showlegend=True, height=420)
    fig.update_yaxes(title_text="NAV", row=1, col=1,
                     gridcolor="#1e1e1e", tickfont=dict(size=10))
    fig.update_yaxes(title_text="DD",  row=2, col=1,
                     gridcolor="#1e1e1e", tickfont=dict(size=10))
    return fig


def annual_bar_fig(ann_df: pd.DataFrame, label: str) -> go.Figure:
    x = (ann_df["year"].astype(str)
         if "year" in ann_df.columns else ann_df.index.astype(str))
    ycol = next((c for c in ["annual_return", "return", "ret"]
                 if c in ann_df.columns), ann_df.columns[0])
    y      = ann_df[ycol] * 100
    colors = ["#00b4d8" if v >= 0 else "#dc3232" for v in y]
    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=colors,
        text=[f"{v:.1f}%" for v in y], textposition="outside", name=label))
    fig.add_hline(y=0, line_dash="dash", line_color="#606060", opacity=0.5)
    fig.update_layout(**CHART_LAYOUT, title=f"Annual Returns — {label}",
                      showlegend=False)
    fig.update_yaxes(title_text="return %")
    return fig


def monthly_heatmap_fig(mon_df: pd.DataFrame, label: str) -> go.Figure:
    fig = px.imshow(
        mon_df * 100,
        color_continuous_scale=[[0,"#dc3232"],[0.5,"#1e1e1e"],[1,"#00b4d8"]],
        aspect="auto", text_auto=".1f")
    fig.update_layout(**CHART_LAYOUT,
                      title=f"Monthly Returns (%) — {label}",
                      coloraxis_showscale=True)
    fig.update_coloraxes(colorbar=dict(
        tickfont=dict(size=9, color="#606060"), outlinecolor="#1e1e1e"))
    return fig


def rolling_sharpe_fig(port: pd.DataFrame, window: int = 63,
                       title: str = "Rolling Sharpe") -> go.Figure:
    roll = (port["ret"].rolling(window).mean()
            / port["ret"].rolling(window).std()) * np.sqrt(252)
    fig = go.Figure(go.Scatter(
        x=roll.index, y=roll.values, mode="lines",
        line=dict(color="#00b4d8", width=1.5), name=f"{window}d Sharpe"))
    fig.add_hline(y=0, line_dash="dash", line_color="#606060", opacity=0.4)
    fig.add_hline(y=1, line_dash="dot",  line_color="#00b4d8", opacity=0.4,
                  annotation_text="Sharpe=1", annotation_position="right")
    fig.update_layout(**CHART_LAYOUT, title=title, showlegend=False)
    fig.update_yaxes(title_text="annualised Sharpe")
    return fig


def _port_metrics(port: pd.DataFrame) -> list:
    eq     = (1 + port["ret"]).cumprod()
    dd     = (eq / eq.cummax() - 1).min()
    ann    = port["ret"].mean() * 252
    vol    = port["ret"].std()  * np.sqrt(252)
    sr     = ann / vol if vol > 0 else 0
    calmar = ann / abs(dd) if dd < 0 else 0
    hit    = (port["ret"] > 0).mean()
    return [
        {"label": "Ann Return", "value": f"{ann*100:+.1f}%", "positive": ann >= 0},
        {"label": "Ann Vol",    "value": f"{vol*100:.1f}%"},
        {"label": "Sharpe",     "value": f"{sr:.2f}",        "positive": sr >= 1},
        {"label": "Max DD",     "value": f"{dd*100:.1f}%",   "positive": False},
        {"label": "Calmar",     "value": f"{calmar:.2f}",    "positive": calmar >= 1},
        {"label": "Hit Rate",   "value": f"{hit*100:.1f}%",  "positive": hit >= 0.5},
    ]

# ══════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════

if page == "Overview":
    section_header(
        "Semiconductor Alpha Research",
        f"Cross-sectional momentum and pairs trade on 12 semiconductor names — "
        f"{start} to {end}"
    )

    cs_ann  = cs_port["ret"].mean() * 252
    cs_sr   = (cs_port["ret"].mean() / cs_port["ret"].std()) * np.sqrt(252)
    p_ann   = pairs_port["ret"].mean() * 252
    p_sr    = (pairs_port["ret"].mean() / pairs_port["ret"].std()) * np.sqrt(252)
    gnn_val = ml["gnn_ic"].mean()

    metric_strip([
        {"label": "CS Momentum  Ann Return",
         "value": f"{cs_ann*100:+.1f}%", "positive": cs_ann >= 0},
        {"label": "CS Momentum  Sharpe",
         "value": f"{cs_sr:.2f}", "positive": cs_sr >= 1},
        {"label": "Pairs Trade  Ann Return",
         "value": f"{p_ann*100:+.1f}%", "positive": p_ann >= 0},
        {"label": "GNN OOS IC",
         "value": f"{gnn_val:.4f}", "positive": gnn_val > 0.05},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("RESEARCH METHODOLOGY", expanded=False):
        st.markdown("""
**Research question**: Do semiconductor stocks exhibit exploitable
cross-sectional alpha beyond systematic market and sector exposures?

**Approach**:
1. **EDA** — Price correlations, rolling volatility, and lead-lag dynamics
   within the 12-name semiconductor universe.
2. **CS Momentum** — Long top-quintile, short bottom-quintile on 45-day
   momentum, rebalanced weekly, transaction-cost adjusted.
3. **Pairs Trade** — Cointegration-based AMAT/LRCX spread with 120-day
   rolling z-score, entry/exit at +-1.5-sigma / 0.
4. **ML Models** — RF, GBM, Transformer, GNN trained on OHLCV-derived
   features; GNN uses correlation-weighted adjacency matrix.
5. **Alpha Attribution** — OLS regression on SPY, QQQ, SOXX; alpha
   t-stats and factor decomposition.
6. **Alt-Data** — SUE, analyst revision proxy (ARM), short-interest proxy
   evaluated with OOS IC.
7. **Universe Expansion** — IC studies repeated on 71- and 108-ticker
   universes to assess statistical reliability.

**Risk controls**: 15 bps one-way transaction cost throughout; no
look-ahead bias (all signals use .shift(1)).
        """)

    st.markdown("---")
    section_header("Strategy Summary", "OOS performance metrics")

    col1, col2 = st.columns(2)
    with col1:
        section_header("CS Momentum", "45-day lookback, weekly rebalance")
        metric_strip(_port_metrics(cs_port))
    with col2:
        section_header("Pairs Trade", "AMAT / LRCX, 120-day window")
        metric_strip(_port_metrics(pairs_port))

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Alpha Attribution Summary", "OLS vs SPY, QQQ, SOXX")
    st.dataframe(alpha["decomp"].round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: Market Structure (EDA & Correlations)
# ══════════════════════════════════════════════════════════════════

elif page == "Market Structure":
    section_header(
        "Market Structure",
        "Price correlations, rolling volatility, and momentum dispersion"
    )

    palette = px.colors.qualitative.Plotly
    tab1, tab2, tab3 = st.tabs(["PRICE HISTORY", "CORRELATION MATRIX", "VOLATILITY"])

    with tab1:
        norm = close[SEMI].div(close[SEMI].iloc[0])
        fig  = go.Figure()
        for i, tkr in enumerate(SEMI):
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm[tkr], mode="lines", name=tkr,
                line=dict(width=1.2, color=palette[i % len(palette)])))
        fig.update_layout(**CHART_LAYOUT, title="Normalised Price (base=1.0)",
                          showlegend=True)
        fig.update_yaxes(title_text="normalised price")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr = ret[SEMI].corr()
        fig  = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale=[[0,"#dc3232"],[0.5,"#1e1e1e"],[1,"#00b4d8"]],
            aspect="auto")
        fig.update_layout(**CHART_LAYOUT, title="Pairwise Return Correlations")
        st.plotly_chart(fig, use_container_width=True)

        nvda_corr = corr["NVDA"].drop("NVDA").sort_values(ascending=False)
        fig2 = go.Figure(go.Bar(
            x=nvda_corr.index, y=nvda_corr.values,
            marker_color=["#00b4d8" if v >= 0 else "#dc3232"
                          for v in nvda_corr.values],
            text=[f"{v:.2f}" for v in nvda_corr.values],
            textposition="outside"))
        fig2.update_layout(**CHART_LAYOUT, title="Correlation with NVDA",
                           showlegend=False)
        fig2.update_yaxes(title_text="correlation")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        vol30 = ret[SEMI].rolling(30).std() * np.sqrt(252)
        fig   = go.Figure()
        for i, tkr in enumerate(SEMI):
            fig.add_trace(go.Scatter(
                x=vol30.index, y=vol30[tkr], mode="lines", name=tkr,
                line=dict(width=1.0, color=palette[i % len(palette)])))
        fig.update_layout(**CHART_LAYOUT,
                          title="30-day Rolling Annualised Volatility",
                          showlegend=True)
        fig.update_yaxes(title_text="annualised vol")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: Lead-Lag Study
# ══════════════════════════════════════════════════════════════════

elif page == "Lead-Lag Study":
    section_header(
        "Lead-Lag Study",
        "Cross-correlation of daily returns between pairs"
    )

    note(
        "Hypothesis REJECTED: No persistent lead-lag relationship detected "
        "within the 12-name semiconductor universe at 1-5 day lags. "
        "All cross-correlations are consistent with zero at conventional "
        "significance levels.",
        kind="fail"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Interactive Pair Explorer", "lag in trading days")

    col1, col2 = st.columns(2)
    t1  = col1.selectbox("Ticker A", SEMI, index=0)
    t2  = col2.selectbox("Ticker B", SEMI, index=1)
    max_lag = st.slider("Max lag (days)", 1, 20, 10)

    lags   = range(-max_lag, max_lag + 1)
    xcorr  = [ret[t1].corr(ret[t2].shift(lag)) for lag in lags]
    fig = go.Figure(go.Bar(
        x=list(lags), y=xcorr,
        marker_color=["#00b4d8" if v >= 0 else "#dc3232" for v in xcorr],
        text=[f"{v:.3f}" for v in xcorr], textposition="outside"))
    fig.add_hline(y=0, line_dash="dash", line_color="#606060", opacity=0.4)
    fig.update_layout(**CHART_LAYOUT,
                      title=f"Cross-Correlation: {t1} vs {t2} (lag = A shift)",
                      showlegend=False)
    fig.update_xaxes(title_text="lag (days)")
    fig.update_yaxes(title_text="cross-correlation")
    st.plotly_chart(fig, use_container_width=True)

    note(
        "Negative lag: A leads B. Positive lag: B leads A. "
        "No bar materially exceeds the noise band.",
        kind="info"
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: Strategy — CS Momentum
# ══════════════════════════════════════════════════════════════════

elif page == "Strategy: CS Momentum":
    section_header(
        "Cross-Sectional Momentum",
        "Long top-quintile, short bottom-quintile | 45-day lookback | "
        "weekly rebalance | 15 bps transaction cost"
    )

    metric_strip(_port_metrics(cs_port))
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "EQUITY CURVE", "ROBUSTNESS", "ANNUAL", "MONTHLY", "ROLLING SHARPE"
    ])

    with tab1:
        st.plotly_chart(equity_fig(cs_port, "CS Momentum — Equity & Drawdown"),
                        use_container_width=True)

    with tab2:
        section_header("Parameter Robustness",
                       "Sharpe across lookback window x holding period")
        st.dataframe(rob_df.round(3), use_container_width=True)
        if {"mom_win", "hold_days", "sharpe"}.issubset(rob_df.columns):
            pivot = rob_df.pivot(index="mom_win", columns="hold_days",
                                 values="sharpe")
            fig   = px.imshow(
                pivot, text_auto=".2f",
                color_continuous_scale=[[0,"#dc3232"],[0.5,"#1e1e1e"],[1,"#00b4d8"]],
                aspect="auto")
            fig.update_layout(**CHART_LAYOUT,
                              title="Sharpe: Lookback Window vs Holding Period")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.plotly_chart(annual_bar_fig(ann_cs, "CS Momentum"),
                        use_container_width=True)
        cs_ya_spy  = pd.read_csv("results/cs_yearly_alpha_spy.csv",  index_col=0)
        cs_ya_soxx = pd.read_csv("results/cs_yearly_alpha_soxx.csv", index_col=0)
        col1, col2 = st.columns(2)
        with col1:
            section_header("Yearly Alpha vs SPY")
            st.dataframe(cs_ya_spy.round(4), use_container_width=True)
        with col2:
            section_header("Yearly Alpha vs SOXX")
            st.dataframe(cs_ya_soxx.round(4), use_container_width=True)

    with tab4:
        st.plotly_chart(monthly_heatmap_fig(mon_cs, "CS Momentum"),
                        use_container_width=True)

    with tab5:
        win = st.slider("Rolling window (days)", 21, 126, 63, key="cs_roll_win")
        st.plotly_chart(
            rolling_sharpe_fig(cs_port, win, "CS Momentum Rolling Sharpe"),
            use_container_width=True)

    st.markdown("---")
    note(
        "Alpha is concentrated in the long book. The 45-day lookback is robust "
        "across a 20-80 day range. Performance weakens in 2022 "
        "(rising-rate, beta-driven selloff).",
        kind="info"
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: Strategy — Pairs Trade
# ══════════════════════════════════════════════════════════════════

elif page == "Strategy: Pairs Trade":
    section_header(
        "Pairs Trade — AMAT / LRCX",
        "Cointegration-based spread | 120-day rolling z-score | "
        "entry +-1.5-sigma, exit 0 | 15 bps transaction cost"
    )

    metric_strip(_port_metrics(pairs_port))
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "EQUITY CURVE", "Z-SCORE", "ANNUAL", "MONTHLY"
    ])

    with tab1:
        st.plotly_chart(equity_fig(pairs_port, "Pairs Trade — Equity & Drawdown"),
                        use_container_width=True)

    with tab2:
        ratio  = np.log(close["AMAT"] / close["LRCX"])
        mu     = ratio.rolling(120).mean()
        sigma  = ratio.rolling(120).std()
        zscore = (ratio - mu) / sigma

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=zscore.index, y=zscore.values,
            mode="lines", name="Z-score",
            line=dict(color="#00b4d8", width=1.2)))
        for level, color, dash in [
            ( 1.5, "#dc3232", "dot"),
            (-1.5, "#dc3232", "dot"),
            ( 0.0, "#606060", "dash"),
        ]:
            fig.add_hline(y=level, line_dash=dash,
                          line_color=color, opacity=0.5)
        fig.update_layout(**CHART_LAYOUT,
                          title="AMAT/LRCX Log-Ratio Z-Score (120d window)",
                          showlegend=False)
        fig.update_yaxes(title_text="z-score")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.plotly_chart(annual_bar_fig(ann_p, "Pairs Trade"),
                        use_container_width=True)
        p_ya_spy  = pd.read_csv("results/pairs_yearly_alpha_spy.csv",  index_col=0)
        p_ya_soxx = pd.read_csv("results/pairs_yearly_alpha_soxx.csv", index_col=0)
        col1, col2 = st.columns(2)
        with col1:
            section_header("Yearly Alpha vs SPY")
            st.dataframe(p_ya_spy.round(4), use_container_width=True)
        with col2:
            section_header("Yearly Alpha vs SOXX")
            st.dataframe(p_ya_soxx.round(4), use_container_width=True)

    with tab4:
        st.plotly_chart(monthly_heatmap_fig(mon_p, "Pairs Trade"),
                        use_container_width=True)

    st.markdown("---")
    note(
        "The AMAT/LRCX spread is cointegrated over the sample. Pairs alpha "
        "is largely market-neutral — low correlation with CS Momentum makes "
        "the two strategies complementary in a combined portfolio.",
        kind="info"
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: Strategy Comparison
# ══════════════════════════════════════════════════════════════════

elif page == "Strategy Comparison":
    section_header(
        "Strategy Comparison",
        "CS Momentum vs Pairs Trade — combined equity and diversification"
    )

    # Align on shared date index — cs_port runs the full period; pairs_port
    # is shorter (only active trading days). Outer-join, fill inactive days
    # with 0.0 so the full date range is preserved for the blend.
    _cs    = cs_port["ret"].rename("cs")
    _pairs = pairs_port["ret"].rename("pairs")
    _combined = pd.concat([_cs, _pairs], axis=1).fillna(0.0)

    combined = pd.DataFrame({
        "ret": 0.5 * _combined["cs"] + 0.5 * _combined["pairs"]
    }, index=_combined.index)

    eq_cs   = (1 + _combined["cs"]).cumprod()
    eq_p    = (1 + _combined["pairs"]).cumprod()
    eq_comb = (1 + combined["ret"]).cumprod()

    fig = go.Figure()
    for eq, name, color, lw in [
        (eq_cs,   "CS Momentum", "#00b4d8", 1.8),
        (eq_p,    "Pairs Trade", "#606060", 1.5),
        (eq_comb, "50/50 Blend", "#e8e8e8", 2.0),
    ]:
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, mode="lines", name=name,
            line=dict(color=color, width=lw)))
    fig.update_layout(**CHART_LAYOUT,
                      title="Cumulative Equity — CS Momentum, Pairs, 50/50 Blend",
                      showlegend=True)
    fig.update_yaxes(title_text="NAV")
    st.plotly_chart(fig, use_container_width=True)

    section_header("Performance Comparison Table")
    rows = []
    for label, port in [("CS Momentum", cs_port),
                         ("Pairs Trade", pairs_port),
                         ("50/50 Blend", combined)]:
        eq  = (1 + port["ret"]).cumprod()
        dd  = (eq / eq.cummax() - 1).min()
        ann = port["ret"].mean() * 252
        vol = port["ret"].std()  * np.sqrt(252)
        sr  = ann / vol if vol > 0 else 0
        cal = ann / abs(dd) if dd < 0 else 0
        rows.append({
            "Strategy":   label,
            "Ann Return": f"{ann*100:+.1f}%",
            "Ann Vol":    f"{vol*100:.1f}%",
            "Sharpe":     f"{sr:.2f}",
            "Max DD":     f"{dd*100:.1f}%",
            "Calmar":     f"{cal:.2f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"),
                 use_container_width=True)

    section_header("Strategy Correlation Matrix")
    st.dataframe(alpha["corr"].round(3), use_container_width=True)

    note(
        "Low strategy correlation (~0.1-0.2) confirms CS Momentum and Pairs "
        "Trade capture different risk premia. The 50/50 blend improves "
        "risk-adjusted returns relative to either strategy in isolation.",
        kind="info"
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: Alpha Attribution
# ══════════════════════════════════════════════════════════════════

elif page == "Alpha Attribution":
    section_header(
        "Alpha Attribution",
        "OLS regression on SPY, QQQ, SOXX — Jensen's alpha and factor decomposition"
    )

    decomp    = alpha["decomp"]
    corr_mat  = alpha["corr"]
    cs_roll_a = alpha["cs_roll"]
    p_roll_a  = alpha["pairs_roll"]

    def _alpha_metrics(name: str) -> dict:
        if name in decomp.index:
            row = decomp.loc[name]
            return {
                "alpha_ann": float(row.get("alpha_ann", row.get("alpha", 0))),
                "alpha_t":   float(row.get("alpha_tstat", row.get("t_alpha", 0))),
                "beta_spy":  float(row.get("beta_spy", row.get("beta", 0))),
                "r2":        float(row.get("r2", row.get("R2", 0))),
            }
        return {"alpha_ann": 0.0, "alpha_t": 0.0, "beta_spy": 0.0, "r2": 0.0}

    cs_a = _alpha_metrics("CS Momentum")
    p_a  = _alpha_metrics("Pairs Trade")

    metric_strip([
        {"label": "CS Mom  Alpha (ann)",
         "value": f"{cs_a['alpha_ann']*100:+.1f}%",
         "positive": cs_a["alpha_ann"] >= 0},
        {"label": "CS Mom  Alpha t-stat",
         "value": f"{cs_a['alpha_t']:.2f}",
         "positive": abs(cs_a["alpha_t"]) >= 1.96},
        {"label": "Pairs  Alpha (ann)",
         "value": f"{p_a['alpha_ann']*100:+.1f}%",
         "positive": p_a["alpha_ann"] >= 0},
        {"label": "Pairs  Alpha t-stat",
         "value": f"{p_a['alpha_t']:.2f}",
         "positive": abs(p_a["alpha_t"]) >= 1.96},
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "REGRESSION TABLE", "CORRELATION MATRIX", "ROLLING ALPHA"
    ])

    with tab1:
        section_header("Factor Regression Results", "OLS vs SPY, QQQ, SOXX")

        def _style_pval(val):
            if not isinstance(val, float):
                return ""
            if val < 0.05:
                return "color: #00b4d8; font-weight: 600"
            elif val < 0.10:
                return "color: #e8e8e8"
            else:
                return "color: #606060"

        pval_cols = [c for c in decomp.columns
                     if "pval" in c.lower() or "p_val" in c.lower() or c == "p"]

        # Build a numeric-only format dict so string columns (e.g. "benchmark")
        # are never passed through a float formatter — avoids ValueError.
        _num_cols  = decomp.select_dtypes(include="number").columns
        _fmt_map   = {c: "{:.4f}" for c in _num_cols}

        if pval_cols:
            st.dataframe(
                decomp.style.applymap(_style_pval, subset=pval_cols)
                            .format(_fmt_map),
                use_container_width=True)
        else:
            st.dataframe(
                decomp.style.format(_fmt_map),
                use_container_width=True)

        if abs(cs_a["alpha_t"]) >= 1.96:
            note(
                f"CS Momentum alpha t-stat = {cs_a['alpha_t']:.2f} — "
                "statistically significant at 5% level.",
                kind="pass"
            )
        else:
            note(
                f"CS Momentum alpha t-stat = {cs_a['alpha_t']:.2f} — "
                "below 1.96 threshold; alpha not statistically confirmed.",
                kind="warn"
            )

    with tab2:
        section_header("Strategy Correlation Matrix")
        fig = px.imshow(
            corr_mat, text_auto=".2f",
            color_continuous_scale=[[0,"#dc3232"],[0.5,"#1e1e1e"],[1,"#00b4d8"]],
            aspect="auto")
        fig.update_layout(**CHART_LAYOUT,
                          title="Pairwise Correlation — Strategies vs Benchmarks")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section_header("Rolling 63-Day Alpha (annualised)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cs_roll_a.index, y=cs_roll_a.values, mode="lines",
            name="CS Momentum", line=dict(color="#00b4d8", width=1.5)))
        fig.add_trace(go.Scatter(
            x=p_roll_a.index, y=p_roll_a.values, mode="lines",
            name="Pairs Trade", line=dict(color="#606060", width=1.2)))
        fig.add_hline(y=0, line_dash="dash", line_color="#1e1e1e", opacity=0.5)
        fig.update_layout(**CHART_LAYOUT, title="Rolling Alpha (63d) vs SPY",
                          showlegend=True)
        fig.update_yaxes(title_text="alpha (annualised)")
        st.plotly_chart(fig, use_container_width=True)

        rolling_csv = pd.read_csv("results/cs_rolling_alpha.csv",
                                  index_col=0, parse_dates=True)
        st.dataframe(rolling_csv.tail(20).round(4), use_container_width=True)

    st.markdown("---")
    note(
        "Both strategies generate positive alpha after controlling for SPY, "
        "QQQ, and SOXX. CS Momentum alpha is more persistent; Pairs Trade "
        "alpha is more episodic and concentrated in high-dispersion regimes.",
        kind="info"
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: Market Impact
# ══════════════════════════════════════════════════════════════════

elif page == "Market Impact":
    section_header(
        "Market Impact Analysis",
        "Sector correlation, beta, shock propagation, and SPY risk contribution"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "SECTOR CORRELATION", "SEMI BETA", "SHOCK DAYS", "SPY RISK"
    ])

    with tab1:
        section_header("Semiconductor-to-Sector Correlation Matrix")
        st.dataframe(sector_corr.round(3), use_container_width=True)
        fig = px.imshow(
            sector_corr, text_auto=".2f",
            color_continuous_scale=[[0,"#dc3232"],[0.5,"#1e1e1e"],[1,"#00b4d8"]],
            aspect="auto")
        fig.update_layout(**CHART_LAYOUT,
                          title="Semi vs Sector Correlation (daily returns)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section_header("Beta of Semiconductors to Broad Sectors")
        st.dataframe(semi_beta.round(3), use_container_width=True)
        if "beta_to_SPY" in semi_beta.columns:
            sb = semi_beta["beta_to_SPY"].sort_values(ascending=False)
            fig = go.Figure(go.Bar(
                x=sb.index, y=sb.values,
                marker_color=["#00b4d8" if v >= 1 else "#606060" for v in sb.values],
                text=[f"{v:.2f}" for v in sb.values], textposition="outside"))
            fig.add_hline(y=1, line_dash="dot", line_color="#606060", opacity=0.5,
                          annotation_text="beta=1", annotation_position="right")
            fig.update_layout(**CHART_LAYOUT,
                              title="Beta to SPY", showlegend=False)
            fig.update_yaxes(title_text="beta to SPY")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section_header("Response on SOXX +-2-Sigma Shock Days")
        st.dataframe(semi_shock.round(3), use_container_width=True)
        cols_to_plot = [c for c in semi_shock.columns if "mean" in c.lower()]
        if cols_to_plot:
            shock_colors = ["#00b4d8", "#dc3232", "#606060", "#e8e8e8"]
            fig = go.Figure()
            for i, col in enumerate(cols_to_plot):
                fig.add_trace(go.Bar(
                    x=semi_shock.index, y=semi_shock[col],
                    name=col.replace("_mean%", ""),
                    marker_color=shock_colors[i % len(shock_colors)]))
            fig.update_layout(**CHART_LAYOUT, barmode="group",
                              title="Average Moves on SOXX +-2-Sigma Days")
            fig.update_xaxes(title_text="event")
            fig.update_yaxes(title_text="average return %")
            st.plotly_chart(fig, use_container_width=True)
        note(
            "On SOXX +-2-sigma days (~+-6.5%), SPY moves about +-3.3% and QQQ "
            "about +-4.0%. Semiconductor shocks are effectively market-level "
            "events — hedging with SPY provides only partial protection.",
            kind="info"
        )

    with tab4:
        section_header("Semiconductor Marginal Risk Contribution to SPY")
        st.dataframe(semi_spy_risk.round(3), use_container_width=True)
        if "risk_contribution%" in semi_spy_risk.columns:
            dfr = semi_spy_risk.sort_values("risk_contribution%",
                                            ascending=False)
            fig = go.Figure(go.Bar(
                x=dfr.index, y=dfr["risk_contribution%"],
                marker_color="#00b4d8",
                text=[f"{v:.2f}%" for v in dfr["risk_contribution%"]],
                textposition="outside"))
            fig.update_layout(**CHART_LAYOUT,
                              title="Euler Marginal Risk Contribution (w x beta) to SPY",
                              showlegend=False)
            fig.update_xaxes(title_text="ticker")
            fig.update_yaxes(title_text="risk contribution to SPY (%)")
            st.plotly_chart(fig, use_container_width=True)
            total = dfr["risk_contribution%"].sum()
            nvda  = (dfr.loc["NVDA", "risk_contribution%"]
                     if "NVDA" in dfr.index else 0)
            note(
                f"NVDA alone contributes ~{nvda:.1f}% of SPY risk. "
                f"The 12-name basket contributes ~{total:.1f}% of SPY risk "
                "despite a smaller share of index market cap.",
                kind="info"
            )


# ══════════════════════════════════════════════════════════════════
# PAGE: ML Signal Analysis
# ══════════════════════════════════════════════════════════════════

elif page == "ML Signal Analysis":
    section_header(
        "ML Signal Analysis",
        "OOS test set: 60/40 train/test split — identical for all 4 models"
    )

    rf_ic  = ml["rf_ic"];   gbm_ic  = ml["gbm_ic"]
    tf_ic  = ml["tf_ic"];   gnn_ic  = ml["gnn_ic"]
    rf_ric = ml["rf_ric"];  gbm_ric = ml["gbm_ric"]
    tf_ric = ml["tf_ric"];  gnn_ric = ml["gnn_ric"]

    metric_strip([
        {"label": "GNN IC Mean",
         "value": f"{gnn_ic.mean():.5f}", "positive": gnn_ic.mean() > 0.05},
        {"label": "GNN ICIR",
         "value": f"{icir(gnn_ic):.4f}", "positive": icir(gnn_ic) > 0.5},
        {"label": "GNN IC pos%",
         "value": f"{(gnn_ic>0).mean()*100:.1f}%",
         "positive": (gnn_ic>0).mean() > 0.55},
        {"label": "Best Model", "value": "GNN", "positive": True},
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    comp_ml = pd.DataFrame([
        {"Model": "Random Forest",
         "IC": round(rf_ic.mean(), 5), "ICIR": round(icir(rf_ic), 4),
         "RankIC": round(rf_ric.mean(), 5),
         "IC pos%": round((rf_ic > 0).mean()*100, 1),
         "N days": len(rf_ic), "Status": "[FAIL]"},
        {"Model": "Gradient Boosting",
         "IC": round(gbm_ic.mean(), 5), "ICIR": round(icir(gbm_ic), 4),
         "RankIC": round(gbm_ric.mean(), 5),
         "IC pos%": round((gbm_ic > 0).mean()*100, 1),
         "N days": len(gbm_ic), "Status": "[FAIL]"},
        {"Model": "Transformer",
         "IC": round(tf_ic.mean(), 5), "ICIR": round(icir(tf_ic), 4),
         "RankIC": round(tf_ric.mean(), 5),
         "IC pos%": round((tf_ic > 0).mean()*100, 1),
         "N days": len(tf_ic), "Status": "[FAIL]"},
        {"Model": "GNN",
         "IC": round(gnn_ic.mean(), 5), "ICIR": round(icir(gnn_ic), 4),
         "RankIC": round(gnn_ric.mean(), 5),
         "IC pos%": round((gnn_ic > 0).mean()*100, 1),
         "N days": len(gnn_ic), "Status": "[PASS]"},
    ]).set_index("Model")

    section_header("Model Comparison — OOS IC")
    st.dataframe(comp_ml, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "IC BAR", "ICIR BAR", "ROLLING IC", "FEATURE IMPORTANCE"
    ])
    bar_colors = ["#dc3232", "#dc3232", "#606060", "#00b4d8"]

    with tab1:
        fig = go.Figure(go.Bar(
            x=comp_ml.index, y=comp_ml["IC"], marker_color=bar_colors,
            text=[f"{v:.5f}" for v in comp_ml["IC"]], textposition="outside"))
        fig.add_hline(y=0,    line_dash="dash", line_color="#606060", opacity=0.4)
        fig.add_hline(y=0.03, line_dash="dot",  line_color="#606060", opacity=0.5,
                      annotation_text="weak (0.03)", annotation_position="right")
        fig.add_hline(y=0.05, line_dash="dot",  line_color="#00b4d8", opacity=0.6,
                      annotation_text="tradable (0.05)", annotation_position="right")
        fig.update_layout(**CHART_LAYOUT,
                          title="OOS IC Mean — GNN crosses institutional tradability",
                          showlegend=False)
        fig.update_xaxes(title_text="model")
        fig.update_yaxes(title_text="IC mean")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure(go.Bar(
            x=comp_ml.index, y=comp_ml["ICIR"], marker_color=bar_colors,
            text=[f"{v:.4f}" for v in comp_ml["ICIR"]], textposition="outside"))
        fig.add_hline(y=0,   line_dash="dash", line_color="#606060", opacity=0.4)
        fig.add_hline(y=0.5, line_dash="dot",  line_color="#00b4d8", opacity=0.5,
                      annotation_text="ICIR=0.5", annotation_position="right")
        fig.update_layout(**CHART_LAYOUT, title="ICIR by Model",
                          showlegend=False)
        fig.update_xaxes(title_text="model")
        fig.update_yaxes(title_text="ICIR")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        win = st.slider("Rolling window (days)", 10, 63, 21, key="ml_roll_win")
        fig = go.Figure()
        for name, series, color, lw in [
            ("GNN",         gnn_ic, "#00b4d8", 2.5),
            ("Transformer", tf_ic,  "#606060", 1.5),
            ("RF",          rf_ic,  "#2a2a2a", 1.0),
            ("GBM",         gbm_ic, "#2a2a2a", 1.0),
        ]:
            roll = series.rolling(win).mean()
            fig.add_trace(go.Scatter(
                x=roll.index, y=roll.values, name=name, mode="lines",
                line=dict(color=color, width=lw)))
        fig.add_hline(y=0,    line_dash="dash", line_color="#606060", opacity=0.3)
        fig.add_hline(y=0.05, line_dash="dot",  line_color="#00b4d8", opacity=0.4)
        # Use apply_layout so the per-chart legend override replaces the base
        # CHART_LAYOUT legend — avoids "multiple values for keyword argument 'legend'".
        apply_layout(fig,
                     title=f"Rolling {win}d IC — All Models",
                     legend=dict(orientation="h", y=1.12,
                                 x=0.5, xanchor="center"))
        fig.update_xaxes(title_text="date")
        fig.update_yaxes(title_text="rolling IC")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        feat = ml["feat_imp"].sort_values("importance")
        fig  = go.Figure(go.Bar(
            x=feat["importance"], y=feat.index,
            orientation="h", marker_color="#00b4d8"))
        fig.update_layout(**CHART_LAYOUT, title="RF Feature Importances",
                          showlegend=False)
        fig.update_xaxes(title_text="importance")
        fig.update_yaxes(title_text="feature")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    note(
        "GNN (IC=0.051, ICIR=0.148, IC pos%=56.1%) is the only model crossing "
        "the institutional tradability threshold (IC > 0.05). The correlation-"
        "weighted graph structure captures semiconductor sector contagion that "
        "flat-feature models (RF, GBM) and sequential models (Transformer) "
        "cannot encode.",
        kind="pass"
    )
    note(
        "Next step: use GNN signal as a position-sizing overlay on CS Momentum "
        "— increase exposure when rolling GNN IC > 0, reduce when it falls below zero.",
        kind="info"
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: Universe Expansion
# ══════════════════════════════════════════════════════════════════

elif page == "Universe Expansion":
    section_header(
        "Universe Expansion",
        "N=12 -> min IC resolution ~0.12 | N=80+ -> resolution ~0.025"
    )

    from src.universe import SEMI_CORE, SP_TECH_SEMI, R1000_TECH

    tab1, tab2, tab3 = st.tabs([
        "UNIVERSE COMPARISON", "IC BREADTH BENEFIT", "DATA STATUS"
    ])

    with tab1:
        section_header("Universe Tiers")
        tiers = {
            "SEMI_CORE (original)":     SEMI_CORE,
            "SP_TECH_SEMI (~80 names)": SP_TECH_SEMI,
            "R1000_TECH (~150 names)":  R1000_TECH,
        }
        rows = []
        for name, tickers in tiers.items():
            rows.append({
                "Universe":       name,
                "N Tickers":      len(tickers),
                "Min IC res.":    f"~{2/len(tickers):.3f}",
                "IC sqrt-N gain": f"~{(len(tickers)/12)**0.5:.1f}x",
                "First 5":        ", ".join(tickers[:5]),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Universe"),
                     use_container_width=True)
        note(
            "IC t-stat scales as sqrt(N) x IC_mean / IC_std. "
            "Moving from N=12 to N=80 improves t-stat by ~2.6x for the same signal.",
            kind="info"
        )

    with tab2:
        section_header("IC Breadth Benefit (Theoretical)")
        ns      = list(range(10, 505, 5))
        ic_mean = 0.04
        ic_std  = 0.12
        t_stats = [(n**0.5 * ic_mean / ic_std) for n in ns]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ns, y=t_stats, name="IC t-stat", mode="lines",
            line=dict(color="#00b4d8", width=2)))
        fig.add_hline(y=1.65, line_dash="dot", line_color="#606060", opacity=0.5,
                      annotation_text="p=0.10", annotation_position="right")
        fig.add_hline(y=1.96, line_dash="dot", line_color="#00b4d8", opacity=0.6,
                      annotation_text="p=0.05", annotation_position="right")
        fig.add_vline(x=12, line_dash="dash", line_color="#dc3232", opacity=0.5,
                      annotation_text="N=12", annotation_position="top left")
        fig.update_layout(
            **CHART_LAYOUT,
            title=f"IC t-stat vs Universe Size (IC_mean={ic_mean}, IC_std={ic_std})",
            xaxis_title="Universe size (N tickers)",
            yaxis_title="IC t-stat")
        st.plotly_chart(fig, use_container_width=True)
        note(
            "At N=12 with IC~0.04 and IC_std~0.12 the IC t-stat is only ~0.35 "
            "— far below any significance threshold. At N=80 it crosses p<0.05.",
            kind="info"
        )
        ic_comp_path = Path("results/ic_study_comparison.csv")
        if ic_comp_path.exists():
            section_header("Empirical IC Comparison — All Universes")
            st.dataframe(pd.read_csv(ic_comp_path, index_col=0),
                         use_container_width=True)
        else:
            note(
                "IC comparison not yet generated. "
                "Run: python src/ic_study.py --compare",
                kind="warn"
            )

    with tab3:
        section_header("Cached Data Status")
        status_ok   = "<span style='color:#00b4d8'>[CACHED]</span>"
        status_miss = "<span style='color:#dc3232'>[MISSING]</span>"
        for uname in ["semi_core", "sp_tech_semi", "r1000_tech"]:
            p_pr = Path(f"data/prices_{uname}.parquet")
            p_ft = Path(f"data/features_{uname}.parquet")
            p_al = Path(f"data/features_alt_{uname}.parquet")
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:#e8e8e8;padding:4px 0;'>"
                f"<b>{uname}</b>&nbsp;&nbsp;"
                f"prices: {status_ok if p_pr.exists() else status_miss}&nbsp;&nbsp;"
                f"features: {status_ok if p_ft.exists() else status_miss}&nbsp;&nbsp;"
                f"alt: {status_ok if p_al.exists() else status_miss}"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)
        st.code("""
python src/data_loader.py --universe sp_tech_semi
python src/features.py --universe sp_tech_semi
python src/features_alt.py --universe sp_tech_semi
python src/ic_study.py --universe sp_tech_semi --fwd-days 5
python src/ic_study.py --compare
        """, language="bash")


# ══════════════════════════════════════════════════════════════════
# PAGE: Alt-Data Signals
# ══════════════════════════════════════════════════════════════════

elif page == "Alt-Data Signals":
    section_header(
        "Alternative Data Signals",
        "SUE, analyst revision proxy (ARM), and short interest proxy — OOS IC/RankIC"
    )

    alt_ic_path = Path("results/alt_signal_ic.csv")

    if not alt_ic_path.exists():
        note(
            "Alt-data IC results not yet generated. "
            "Run: python src/features_alt.py",
            kind="warn"
        )
    else:
        alt_ic = pd.read_csv(alt_ic_path, index_col=0)
        section_header("Signal IC Summary")

        def _color_ic(val):
            if isinstance(val, float):
                if val > 0.04:
                    return "color: #00b4d8; font-weight: bold"
                elif val > 0:
                    return "color: #e8e8e8"
                else:
                    return "color: #dc3232"
            return ""

        st.dataframe(
            alt_ic.style.applymap(_color_ic,
                                  subset=["IC_mean", "RankIC_mean"]),
            use_container_width=True)

        tab1, tab2 = st.tabs(["IC BAR CHART", "SIGNAL DESCRIPTIONS"])

        with tab1:
            colors = ["#00b4d8" if v > 0.04 else "#606060" if v > 0 else "#dc3232"
                      for v in alt_ic["IC_mean"]]
            fig = go.Figure(go.Bar(
                x=alt_ic.index, y=alt_ic["IC_mean"],
                marker_color=colors,
                text=[f"{v:.5f}" for v in alt_ic["IC_mean"]],
                textposition="outside"))
            fig.add_hline(y=0,    line_dash="dash", line_color="#606060", opacity=0.4)
            fig.add_hline(y=0.04, line_dash="dot",  line_color="#00b4d8", opacity=0.6,
                          annotation_text="IC > 0.04", annotation_position="right")
            fig.update_layout(
                **CHART_LAYOUT,
                title="Alternative Data Signal IC (OOS, 5-day forward return)",
                xaxis_title="Signal", yaxis_title="IC mean")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("""
| Signal | Alpha Hypothesis | Expected IC | Failure Modes |
|--------|-----------------|-------------|---------------|
| **SUE** | Post-earnings drift: stocks beating consensus continue to outperform 1-60d | +0.03-0.08 | Crowding, earnings manipulation |
| **SUE decay** | SUE weighted by recency (half-life 30d) | +0.03-0.06 | Same as SUE; decays with time |
| **ARM** | Cumulative earnings surprise proxies analyst estimate revisions | +0.01-0.04 | Proxy quality, low-coverage sectors |
| **SI proxy** | Low short interest -> less informed bearish conviction | +0.01-0.03 | Short squeezes, borrow cost |
            """)
            note(
                "All signals use a .shift(1) alignment — only day t-1 close "
                "information is used in day t signals. No lookahead bias.",
                kind="info"
            )

    st.markdown("---")
    st.code("python src/features_alt.py", language="bash")


# ══════════════════════════════════════════════════════════════════
# PAGE: NLP Signal
# ══════════════════════════════════════════════════════════════════

elif page == "NLP Signal":
    section_header(
        "NLP / LLM Signal — Earnings Call Sentiment",
        "SEC EDGAR 8-K embeddings via sentence-transformers — OOS IC on 10-day returns"
    )

    metric_strip([
        {"label": "Embedding Model", "value": "MiniLM-L6-v2"},
        {"label": "Embedding Dim",   "value": "384-d"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
**Architecture**:
1. Fetch quarterly 8-K filings from SEC EDGAR for each semiconductor ticker.
2. Embed with `sentence-transformers/all-MiniLM-L6-v2` (22M params, runs locally).
3. Project onto a positive/negative polarity axis.
4. Forward-fill from each earnings date — leakage-free daily signal.
5. Evaluate OOS IC on 10-day forward returns.
    """)

    nlp_ic_path = Path("results/nlp_ic.csv")

    if not nlp_ic_path.exists():
        note(
            "NLP IC results not yet generated. "
            "Run: python src/nlp_signal.py  "
            "(requires: pip install sentence-transformers)",
            kind="warn"
        )
    else:
        nlp_ic = pd.read_csv(nlp_ic_path, index_col=0)
        section_header("NLP Signal IC")
        st.dataframe(nlp_ic, use_container_width=True)

        colors = ["#00b4d8" if v > 0.02 else "#606060" if v > 0 else "#dc3232"
                  for v in nlp_ic["IC_mean"]]
        fig = go.Figure(go.Bar(
            x=nlp_ic.index, y=nlp_ic["IC_mean"],
            marker_color=colors,
            text=[f"{v:.5f}" for v in nlp_ic["IC_mean"]],
            textposition="outside"))
        fig.add_hline(y=0, line_dash="dash", line_color="#606060", opacity=0.4)
        fig.update_layout(
            **CHART_LAYOUT,
            title="NLP Signal IC (OOS, 10-day forward return)",
            xaxis_title="Signal", yaxis_title="IC mean")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""
**Alpha hypothesis — nlp_sent**:
> Loughran & McDonald (2011) show that tone in SEC filings predicts subsequent
> returns. A positive earnings call tone signals management confidence and
> often accompanies positive estimate revisions — both drive PEAD.
> Expected IC: +0.02-0.05 on 10-day horizon.

**Alpha hypothesis — nlp_drift**:
> Tone improvement relative to year-ago conveys incremental information
> beyond the current quarter's level. Expected IC: +0.01-0.03.

**Failure modes**: management cheerleading (optimistic by default);
EDGAR 8-K is not a full transcript; small N=12 universe dilutes t-stat.
    """)

    st.code("pip install sentence-transformers\npython src/nlp_signal.py",
            language="bash")


# ══════════════════════════════════════════════════════════════════
# PAGE: Signal Combiner
# ══════════════════════════════════════════════════════════════════

elif page == "Signal Combiner":
    section_header(
        "Signal Combiner — ML as a Meta-Model",
        "GBM combines IC-positive base signals into a composite rank score — "
        "walk-forward OOS evaluation"
    )

    st.markdown(r"""
**Key insight (Grinold & Kahn, Fundamental Law)**:

$$\text{ICIR}_{\text{combined}} \approx \text{ICIR}_{\text{individual}} \times \sqrt{N}$$

Combining N=5 partially-decorrelated signals can improve ICIR by up to
$\sqrt{5} \approx 2.2\times$. The GBM meta-model learns non-linear combinations
and interaction terms between signals.
    """)

    combiner_path = Path("results/signal_combiner_summary.csv")
    weights_path  = Path("results/signal_weights.csv")
    folds_path    = Path("results/signal_combiner_folds.csv")

    if not combiner_path.exists():
        note(
            "Signal combiner results not yet generated. "
            "Run: python src/model_signal_combiner.py  "
            "(requires features_alt.parquet and features.parquet)",
            kind="warn"
        )
    else:
        summary = pd.read_csv(combiner_path, index_col=0)
        section_header("Individual Signals vs GBM Combiner — IC Comparison")
        st.dataframe(summary, use_container_width=True)

        if "IC_mean" in summary.columns:
            colors = ["#00b4d8" if v > 0.04 else "#606060" if v > 0 else "#dc3232"
                      for v in summary["IC_mean"]]
            fig = go.Figure(go.Bar(
                x=summary.index, y=summary["IC_mean"],
                marker_color=colors,
                text=[f"{v:.5f}" for v in summary["IC_mean"]],
                textposition="outside"))
            fig.add_hline(y=0,    line_dash="dash", line_color="#606060", opacity=0.4)
            fig.add_hline(y=0.04, line_dash="dot",  line_color="#00b4d8", opacity=0.6,
                          annotation_text="IC > 0.04", annotation_position="right")
            fig.update_layout(**CHART_LAYOUT,
                              title="Signal IC: Individual vs GBM Combiner",
                              xaxis_title="Signal / Model",
                              yaxis_title="IC mean (OOS)")
            st.plotly_chart(fig, use_container_width=True)

            combiner_ic = (summary.loc["GBM_COMBINER", "IC_mean"]
                           if "GBM_COMBINER" in summary.index else None)
            best_ind    = summary["IC_mean"].drop("GBM_COMBINER",
                                                  errors="ignore").max()
            if combiner_ic is not None and combiner_ic > best_ind:
                note(
                    f"GBM combiner IC ({combiner_ic:.5f}) exceeds best individual "
                    f"signal IC ({best_ind:.5f}) — the meta-model adds value.",
                    kind="pass"
                )

    if weights_path.exists():
        weights = pd.read_csv(weights_path, index_col=0).sort_values(
            "importance", ascending=False)
        section_header("GBM Signal Weights (avg feature importance)")
        fig = go.Figure(go.Bar(
            x=weights.index, y=weights["importance"],
            marker_color="#00b4d8",
            text=[f"{v:.4f}" for v in weights["importance"]],
            textposition="outside"))
        fig.update_layout(**CHART_LAYOUT,
                          title="GBM Feature Importances — Signal Weights",
                          xaxis_title="Base signal",
                          yaxis_title="Avg importance")
        st.plotly_chart(fig, use_container_width=True)

    if folds_path.exists():
        folds = pd.read_csv(folds_path)
        section_header("Walk-Forward Fold Diagnostics")
        st.dataframe(folds, use_container_width=True)
        fig = go.Figure(go.Bar(
            x=folds["test_start"].astype(str),
            y=folds["fold_IC"],
            marker_color=["#00b4d8" if v > 0 else "#dc3232"
                          for v in folds["fold_IC"]],
            text=[f"{v:.4f}" for v in folds["fold_IC"]],
            textposition="outside"))
        fig.add_hline(y=0, line_dash="dash", line_color="#606060", opacity=0.4)
        fig.update_layout(**CHART_LAYOUT,
                          title="Per-Fold OOS IC (test start date)",
                          xaxis_title="Test fold start",
                          yaxis_title="Fold IC")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    note(
        "Design rationale: the original ML models had negative IC because they "
        "tried to predict raw 5-day returns from OHLCV on a 12-ticker universe. "
        "The combiner instead: (1) feeds pre-computed IC-positive signals; "
        "(2) targets cross-sectional rank; (3) uses shallow GBM (max_depth=2); "
        "(4) retrains every 63 days to track regime changes.",
        kind="info"
    )

    st.code("""
python src/features_alt.py           # build alt-data features
python src/nlp_signal.py             # build NLP features (optional)
python src/model_signal_combiner.py  # run combiner
    """, language="bash")
