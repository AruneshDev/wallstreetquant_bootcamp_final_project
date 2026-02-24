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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


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

    cs_port    = silent(run_cs_momentum, ret, SEMI,
                        mom_win=45, label="CS Momentum (45d)")
    pairs_port = silent(run_pairs_trade, ret, close,
                    'AMAT', 'LRCX', win=120)   

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


(close, volume, ret, SEMI,
 cs_port, pairs_port,
 rob_df, ann_cs, mon_cs, ann_p, mon_p) = load_all()

ml     = load_ml_results()
alpha  = load_alpha_results()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

st.sidebar.title("📈 Semi Alpha Research")
st.sidebar.caption("Point72 IAC — Arunesh Lal | Feb 2026")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 EDA & Correlations",
    "🔍 Lead-Lag Study",
    "📈 CS Momentum",
    "🔗 Pairs Trade",
    "🏆 Strategy Comparison",
    "📐 Alpha Attribution",
    "🤖 ML Signal Analysis",
])

n_days = len(ret)
start  = ret.index[0].strftime('%b %Y')
end    = ret.index[-1].strftime('%b %Y')

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Data**: {start} → {end}  
**Days**: {n_days:,}  
**Universe**: 12 semis + 5 big tech  
**Models**: RF · GBM · Transformer · GNN
""")


# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════
def metric_row(port: pd.Series):
    r       = port.dropna()
    ar      = r.mean() * 252
    av      = r.std()  * np.sqrt(252)
    sr      = ar / av  if av > 0 else 0.0
    cum     = (1 + r).cumprod()
    mdd     = (cum / cum.cummax() - 1).min()
    neg     = r[r < 0]
    dv      = neg.std() * np.sqrt(252) if len(neg) > 5 else np.nan
    sortino = ar / dv   if (dv and dv > 0) else 0.0
    tr      = cum.iloc[-1] - 1

    c1, c2, c3 = st.columns(3)
    c1.metric("Annual Return", f"{ar*100:.2f}%")
    c2.metric("Annual Vol",    f"{av*100:.2f}%")
    c3.metric("Sharpe",        f"{sr:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Sortino",      f"{sortino:.3f}")
    c5.metric("Max Drawdown", f"{mdd*100:.2f}%")
    c6.metric("Total Return", f"{tr*100:.2f}%")
def equity_fig(port: pd.Series, title: str) -> go.Figure:
    r   = port.dropna()
    cum = (1 + r).cumprod()
    dd  = (cum / cum.cummax() - 1) * 100   # always ≤ 0

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Equity line — NO fill, clean line only
    fig.add_trace(
        go.Scatter(
            x=cum.index, y=cum.values,
            name="Equity", mode="lines",
            line=dict(width=2, color="#6366f1")),
        secondary_y=False)

    # Drawdown — filled area below zero
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values,
            name="Drawdown %", mode="lines",
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
            line=dict(color="rgba(239, 68, 68, 0.8)", width=1)),
        secondary_y=True)

    # Pin drawdown axis so it doesn't squash equity
    dd_floor = min(dd.min() * 1.3, -10)
    fig.update_yaxes(title_text="equity (×)",
                     secondary_y=False,
                     showgrid=True)
    fig.update_yaxes(title_text="drawdown %",
                     secondary_y=True,
                     range=[dd_floor, 5],
                     showgrid=False,
                     zeroline=True,
                     zerolinecolor="rgba(255,255,255,0.2)")

    fig.update_layout(
        title=title,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.08,
                    x=0.5, xanchor="center"))
    fig.update_xaxes(title_text="date")
    return fig


def annual_bar_fig(ann_df: pd.DataFrame, label: str) -> go.Figure:
    df     = ann_df.reset_index()
    colors = ["#22c55e" if v >= 0 else "#ef4444"
              for v in df["total_ret%"]]
    fig = go.Figure(go.Bar(
        x=df["year"].astype(str), y=df["total_ret%"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df["total_ret%"]],
        textposition="outside"))
    fig.update_layout(title=f"Annual returns — {label}",
                      template="plotly_dark", showlegend=False)
    fig.update_xaxes(title_text="year")
    fig.update_yaxes(title_text="total return %")
    return fig


def monthly_heatmap_fig(mon_df: pd.DataFrame, label: str) -> go.Figure:
    data = mon_df.copy()
    data.columns = [str(c) for c in data.columns]
    fig = px.imshow(
        (data * 100).round(1),
        text_auto=".1f",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title=f"Monthly returns (%) — {label}",
        template="plotly_dark")
    fig.update_xaxes(title_text="month")
    fig.update_yaxes(title_text="year")
    return fig


def rolling_sharpe_fig(port: pd.Series,
                        window: int, title: str) -> go.Figure:
    rs = (port.rolling(window).mean() /
          port.rolling(window).std()) * np.sqrt(252)
    fig = px.line(x=rs.index, y=rs.values,
                  title=title, template="plotly_dark")
    fig.add_hline(y=0,   line_dash="dash",
                  line_color="white",   opacity=0.4)
    fig.add_hline(y=0.5, line_dash="dot",
                  line_color="#22c55e", opacity=0.5)
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="rolling sharpe")
    return fig


def icir(s: pd.Series) -> float:
    return s.mean() / s.std() if s.std() > 0 else np.nan


# ══════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("Semiconductor Alpha Research")
    st.markdown("""
    > **Research goal**: Identify statistically robust, market-neutral
    > alpha signals in semiconductor equities using daily price data
    > from 2020 to Feb 2026.
    ---
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe",      f"{len(SEMI)} semis + 5 big tech")
    c2.metric("Trading days",  f"{n_days:,}")
    c3.metric("Period",        f"{start} → {end}")
    c4.metric("Best ML model", "GNN  IC=0.051")

    st.markdown("---")
    st.subheader("Research journey")

    j1, j2, j3, j4 = st.columns(4)
    with j1:
        st.markdown("""
        **① Hypothesis**  
        NVDA leads other semis  
        by 1–5 days at daily frequency.
        """)
    with j2:
        st.markdown("""
        **② Falsification**  
        Zero positive-lift pairs  
        across all 272 directed pairs.  
        Lead-lag alpha **rejected**.
        """)
    with j3:
        st.markdown("""
        **③ Pivot**  
        Robustness sweep found  
        momentum zone at 20–60d.  
        Two profitable strategies built.
        """)
    with j4:
        st.markdown("""
        **④ ML + Alpha**  
        GNN IC=0.051 crosses  
        tradability threshold.  
        Beta ≈ 0 vs SOXX confirmed.
        """)

    st.markdown("---")
    st.subheader("Strategy snapshot")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### CS Momentum (45d)")
        metric_row(cs_port)
        st.plotly_chart(
            equity_fig(cs_port, "CS Momentum equity curve"),
            use_container_width=True)
    with col2:
        st.markdown("#### AMAT / LRCX Pairs Trade")
        metric_row(pairs_port)
        st.plotly_chart(
            equity_fig(pairs_port, "Pairs trade equity curve"),
            use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════

elif page == "📊 EDA & Correlations":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Prices", "Correlations", "Volatility"])

    with tab1:
        sel = st.multiselect("Select tickers", SEMI, default=SEMI[:6])
        if sel:
            base = close[sel].div(close[sel].iloc[0]) * 100
            fig  = go.Figure()
            for t in sel:
                fig.add_trace(go.Scatter(x=base.index, y=base[t],
                                         name=t, mode="lines"))
            fig.update_layout(
                title="Normalized prices (base=100)",
                template="plotly_dark",
                legend=dict(orientation="h", y=1.1,
                            x=0.5, xanchor="center"))
            fig.update_xaxes(title_text="date")
            fig.update_yaxes(title_text="indexed price")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        max_win = min(len(ret), 1260)
        win = st.slider("Rolling window (days)", 60, max_win, 252)
        corr = ret[SEMI].tail(win).corr()
        fig  = px.imshow(corr, text_auto=".2f",
                         color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1,
                         title=f"Correlation matrix — last {win}d",
                         template="plotly_dark")
        fig.update_xaxes(title_text="ticker")
        fig.update_yaxes(title_text="ticker")
        st.plotly_chart(fig, use_container_width=True)

        ref   = st.selectbox("Correlation bar vs:", SEMI, index=0)
        cb    = ret[SEMI].corrwith(ret[ref]).drop(ref).sort_values(ascending=False)
        df_cb = cb.reset_index()
        df_cb.columns = ["ticker", "corr"]
        fig2  = px.bar(df_cb, x="ticker", y="corr",
                       title=f"Same-day correlation vs {ref}",
                       template="plotly_dark")
        fig2.update_xaxes(title_text="ticker")
        fig2.update_yaxes(title_text="pearson corr")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        win_v = st.slider("Vol window (days)", 5, 63, 21)
        sel_v = st.multiselect("Tickers", SEMI, default=SEMI[:4])
        if sel_v:
            fig = go.Figure()
            for t in sel_v:
                rv = ret[t].rolling(win_v).std() * np.sqrt(252) * 100
                fig.add_trace(go.Scatter(x=rv.index, y=rv,
                                         name=t, mode="lines"))
            fig.update_layout(
                title=f"Rolling {win_v}d annualised vol (%)",
                template="plotly_dark",
                legend=dict(orientation="h", y=1.1,
                            x=0.5, xanchor="center"))
            fig.update_xaxes(title_text="date")
            fig.update_yaxes(title_text="ann. vol %")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: LEAD-LAG
# ══════════════════════════════════════════════════════════════════

elif page == "🔍 Lead-Lag Study":
    from scipy.stats import pearsonr
    st.title("Lead-Lag Study")

    st.error("""
    **Research finding**: Zero positive-lift pairs found across all
    272 directed pairs tested. Max lift = –0.20.
    Daily lead-lag alpha is **rejected**.
    """)

    st.subheader("Interactive pair explorer")
    col1, col2 = st.columns(2)
    leader   = col1.selectbox("Leader",   SEMI, index=0)
    follower = col2.selectbox("Follower",
                              [t for t in SEMI if t != leader], index=1)
    max_lag  = st.slider("Max lag (days)", 1, 10, 5)

    def ll_corr(a, b, ml):
        s, f, n = a.values, b.values, len(a)
        rows = []
        for k in range(-ml, ml + 1):
            if   k > 0:  r, _ = pearsonr(s[:n-k], f[k:])
            elif k < 0:  r, _ = pearsonr(s[abs(k):], f[:n-abs(k)])
            else:        r, _ = pearsonr(s, f)
            rows.append({"lag": k, "corr": round(r, 4)})
        return pd.DataFrame(rows)

    ll_df = ll_corr(ret[leader], ret[follower], max_lag)
    ll_df["color"] = ll_df["corr"].apply(
        lambda x: "#22c55e" if x > 0 else "#ef4444")

    fig = go.Figure(go.Bar(
        x=ll_df["lag"], y=ll_df["corr"],
        marker_color=ll_df["color"],
        text=[f"{v:.4f}" for v in ll_df["corr"]],
        textposition="outside"))
    fig.add_vline(x=0, line_dash="dash",
                  line_color="yellow", opacity=0.5)
    fig.add_hline(y=0, line_dash="dash",
                  line_color="white",  opacity=0.3)
    fig.update_layout(
        title=f"Lead-lag correlation: {leader} → {follower}",
        template="plotly_dark", showlegend=False)
    fig.update_xaxes(title_text="lag (days, +ve = leader leads)")
    fig.update_yaxes(title_text="pearson corr")
    st.plotly_chart(fig, use_container_width=True)

    c0   = ll_df[ll_df["lag"] == 0]["corr"].values[0]
    best = (ll_df[ll_df["lag"] > 0]
            .sort_values("corr", ascending=False).iloc[0])
    lift = best["corr"] - c0

    m1, m2, m3 = st.columns(3)
    m1.metric("Same-day corr",
              f"{c0:.4f}")
    m2.metric(f"Best lag corr (lag={int(best['lag'])}d)",
              f"{best['corr']:.4f}")
    m3.metric("Lift", f"{lift:.4f}",
              delta=f"{lift:.4f}",
              delta_color="normal" if lift > 0 else "inverse")


# ══════════════════════════════════════════════════════════════════
# PAGE: CS MOMENTUM
# ══════════════════════════════════════════════════════════════════

elif page == "📈 CS Momentum":
    st.title("Cross-Sectional Momentum")
    st.caption("Long top 3 / short bottom 3 semis ranked by N-day return")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Equity Curve", "Robustness", "Annual", "Monthly"])

    with tab1:
        metric_row(cs_port)
        st.plotly_chart(
            equity_fig(cs_port, "CS Momentum (45d) — equity & drawdown"),
            use_container_width=True)
        st.plotly_chart(
            rolling_sharpe_fig(cs_port, 63, "Rolling Sharpe (63d)"),
            use_container_width=True)

    with tab2:
        colors = ["#22c55e" if v >= 0 else "#ef4444"
                  for v in rob_df["Sharpe"]]
        fig = go.Figure(go.Bar(
            x=rob_df["window"].astype(str) + "d",
            y=rob_df["Sharpe"].round(3),
            marker_color=colors,
            text=[f"{v:.3f}" for v in rob_df["Sharpe"]],
            textposition="outside"))
        fig.add_hline(y=0, line_dash="dash",
                      line_color="white", opacity=0.4)
        fig.update_layout(
            title="Sharpe vs momentum window — reversal→momentum crossover",
            template="plotly_dark", showlegend=False)
        fig.update_xaxes(title_text="momentum window")
        fig.update_yaxes(title_text="sharpe ratio")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rob_df.round(3), use_container_width=True)

    with tab3:
        st.plotly_chart(annual_bar_fig(ann_cs, "CS Momentum (45d)"),
                        use_container_width=True)
        st.dataframe(ann_cs.round(3), use_container_width=True)

    with tab4:
        st.plotly_chart(monthly_heatmap_fig(mon_cs, "CS Momentum (45d)"),
                        use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: PAIRS TRADE
# ══════════════════════════════════════════════════════════════════

elif page == "🔗 Pairs Trade":
    st.title("AMAT / LRCX  Pairs Trade")
    st.caption(
        "Log-spread mean reversion | Entry ±1.5σ | Exit ±0.3σ | 120d rolling")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Equity Curve", "Z-Score", "Annual", "Monthly"])

    with tab1:
        metric_row(pairs_port)
        st.plotly_chart(
            equity_fig(pairs_port, "AMAT/LRCX Pairs — equity & drawdown"),
            use_container_width=True)
        st.plotly_chart(
            rolling_sharpe_fig(pairs_port, 63, "Rolling Sharpe (63d)"),
            use_container_width=True)

    with tab2:
        log_spread = np.log(close["AMAT"]) - np.log(close["LRCX"])
        mu    = log_spread.rolling(120).mean()
        sigma = log_spread.rolling(120).std()
        z     = (log_spread - mu) / sigma

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z.index, y=z.values,
                                  name="z-score", mode="lines",
                                  line=dict(width=1.5)))
        for level, color, name in [
            ( 1.5, "orange", "entry +1.5σ"),
            (-1.5, "orange", "entry -1.5σ"),
            ( 0.3, "grey",   "exit  +0.3σ"),
            (-0.3, "grey",   "exit  -0.3σ"),
        ]:
            fig.add_hline(y=level, line_dash="dash",
                          line_color=color, opacity=0.7,
                          annotation_text=name,
                          annotation_position="right")
        fig.add_hline(y=0, line_color="white", opacity=0.3)
        fig.update_layout(
            title="AMAT/LRCX log-spread z-score (120d rolling)",
            template="plotly_dark", showlegend=False)
        fig.update_xaxes(title_text="date")
        fig.update_yaxes(title_text="z-score")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.plotly_chart(annual_bar_fig(ann_p, "AMAT/LRCX Pairs"),
                        use_container_width=True)
        st.dataframe(ann_p.round(3), use_container_width=True)

    with tab4:
        st.plotly_chart(monthly_heatmap_fig(mon_p, "AMAT/LRCX Pairs"),
                        use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: STRATEGY COMPARISON
# ══════════════════════════════════════════════════════════════════

elif page == "🏆 Strategy Comparison":
    st.title("Strategy Comparison")

    both = pd.DataFrame({
        "CS Momentum (45d)": cs_port,
        "AMAT/LRCX Pairs":    pairs_port
    }).dropna(how="all").fillna(0)
    cum_both = (1 + both).cumprod()

    fig = go.Figure()
    for col in cum_both.columns:
        fig.add_trace(go.Scatter(x=cum_both.index, y=cum_both[col],
                                  name=col, mode="lines"))
    fig.update_layout(
        title="Strategy equity curves — side by side",
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1,
                    x=0.5, xanchor="center"))
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="equity (×)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance metrics")

    def mdict(port, label):
        r   = port.dropna()
        ar  = r.mean() * 252
        av  = r.std()  * np.sqrt(252)
        sr  = ar / av  if av > 0 else 0.0
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        neg = r[r < 0]
        dv  = neg.std() * np.sqrt(252) if len(neg) > 5 else np.nan
        sortino = ar / dv if (dv and dv > 0) else 0.0
        return {
            "Strategy":   label,
            "Ann Ret %":  round(ar  * 100, 2),
            "Ann Vol %":  round(av  * 100, 2),
            "Sharpe":     round(sr,  3),
            "Sortino":    round(sortino, 3),
            "Max DD %":   round(mdd * 100, 2),
            "Win Rate %": round((r > 0).mean() * 100, 1),
        }

    comp = pd.DataFrame([
        mdict(cs_port,    "CS Momentum (45d)"),
        mdict(pairs_port, "AMAT/LRCX Pairs"),
    ]).set_index("Strategy")
    st.dataframe(comp, use_container_width=True)

    st.subheader("Combined portfolio (50/50 equal weight)")
    combined = (cs_port.reindex(both.index).fillna(0) * 0.5 +
                pairs_port.reindex(both.index).fillna(0) * 0.5)
    metric_row(combined)
    st.plotly_chart(
        equity_fig(combined, "Combined 50/50 equity curve"),
        use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: ALPHA ATTRIBUTION
# ══════════════════════════════════════════════════════════════════

elif page == "📐 Alpha Attribution":
    st.title("Alpha Attribution")
    st.caption(
        "Jensen's alpha regression vs SOXX and SPY  |  "
        "R_p − R_f  =  α  +  β(R_m − R_f)  +  ε")

    decomp    = alpha["decomp"]
    corr_df   = alpha["corr"].astype(float)
    cs_roll   = alpha["cs_roll"]
    p_roll    = alpha["pairs_roll"]

    # ── Key metric callouts ──
    try:
        cs_soxx = decomp.loc[decomp.index.str.contains("CS Momentum.*SOXX")]
        p_soxx  = decomp.loc[decomp.index.str.contains("Pairs.*SOXX")]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CS Mom Beta (SOXX)",
                  f"{cs_soxx['beta'].values[0]:.3f}",
                  help="Near zero = not leveraged SOXX exposure")
        c2.metric("Pairs Beta (SOXX)",
                  f"{p_soxx['beta'].values[0]:.3f}",
                  help="Negative = structural market hedge")
        c3.metric("CS Mom R² (SOXX)",
                  f"{cs_soxx['r2'].values[0]:.3f}",
                  help="% variance explained by SOXX")
        c4.metric("Strategy Correlation",
                  f"{corr_df.loc['CS Momentum','Pairs Trade']:.3f}",
                  help="Negative = natural diversification")
    except Exception:
        st.info("Run src/alpha.py to generate attribution results.")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["Alpha Table", "Correlation Matrix", "Rolling Alpha"])

    with tab1:
        st.subheader("Jensen's Alpha decomposition")
        st.dataframe(decomp.round(3), use_container_width=True)

        alphas = decomp['alpha_ann_pct'].values
        labels = decomp.index.tolist()
        colors = ["#22c55e" if v > 0 else "#ef4444" for v in alphas]

        fig = go.Figure(go.Bar(
            x=labels, y=alphas,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in alphas],
            textposition="outside"))
        fig.add_hline(y=0, line_dash="dash",
                      line_color="white", opacity=0.4)
        fig.update_layout(
            title="Annualised Jensen's Alpha vs benchmarks",
            template="plotly_dark", showlegend=False)
        fig.update_xaxes(title_text="strategy vs benchmark")
        fig.update_yaxes(title_text="alpha % / yr")
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Alpha is economically substantial (6–9% annualised) but
        t-stats are < 2 with 2 years of data.** With ~23% tracking error,
        reaching statistical significance (t > 2) requires ~5 years at this
        Sharpe level. The key result is **beta ≈ 0** and **R² ≈ 0** — returns
        are genuinely decorrelated from the semiconductor index, not disguised
        SOXX beta. With the extended dataset (2020–2026) the t-stats will rise
        meaningfully.
        """)

    with tab2:
        fig = px.imshow(
            corr_df,
            text_auto=".3f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Strategy & benchmark correlation matrix",
            template="plotly_dark")
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="")
        st.plotly_chart(fig, use_container_width=True)

        st.success("""
        **CS Momentum ↔ Pairs Trade correlation = –0.458.**
        These strategies hedge each other naturally. CS Momentum thrives
        in trending regimes; Pairs Trade is stable across years. Both carry
        near-zero correlation to SPY (0.099 and –0.105 respectively).
        """)

    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cs_roll.index,
            y=(cs_roll * 100).values,
            name="CS Momentum",
            mode="lines",
            line=dict(color="#6366f1", width=2)))
        fig.add_trace(go.Scatter(
            x=p_roll.index,
            y=(p_roll * 100).values,
            name="Pairs Trade",
            mode="lines",
            line=dict(color="#22c55e", width=2)))
        fig.add_hline(y=0, line_dash="dash",
                      line_color="white", opacity=0.4)
        fig.update_layout(
            title="Rolling 126d annualised alpha vs SOXX",
            template="plotly_dark",
            legend=dict(orientation="h", y=1.1,
                        x=0.5, xanchor="center"))
        fig.update_xaxes(title_text="date")
        fig.update_yaxes(title_text="alpha % / yr")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: ML SIGNAL ANALYSIS
# ══════════════════════════════════════════════════════════════════

elif page == "🤖 ML Signal Analysis":
    st.title("ML Signal Analysis")
    st.caption(
        "OOS test set: 60/40 train/test split | identical for all 4 models")

    rf_ic  = ml["rf_ic"];   gbm_ic  = ml["gbm_ic"]
    tf_ic  = ml["tf_ic"];   gnn_ic  = ml["gnn_ic"]
    rf_ric = ml["rf_ric"];  gbm_ric = ml["gbm_ric"]
    tf_ric = ml["tf_ric"];  gnn_ric = ml["gnn_ric"]

    st.subheader("Model comparison — OOS IC")
    comp_ml = pd.DataFrame([
        {"Model": "Random Forest",
         "IC":      round(rf_ic.mean(),   5),
         "ICIR":    round(icir(rf_ic),    4),
         "RankIC":  round(rf_ric.mean(),  5),
         "IC pos%": round((rf_ic  > 0).mean() * 100, 1),
         "N days":  len(rf_ic)},
        {"Model": "Gradient Boosting",
         "IC":      round(gbm_ic.mean(),  5),
         "ICIR":    round(icir(gbm_ic),   4),
         "RankIC":  round(gbm_ric.mean(), 5),
         "IC pos%": round((gbm_ic > 0).mean() * 100, 1),
         "N days":  len(gbm_ic)},
        {"Model": "Transformer",
         "IC":      round(tf_ic.mean(),   5),
         "ICIR":    round(icir(tf_ic),    4),
         "RankIC":  round(tf_ric.mean(),  5),
         "IC pos%": round((tf_ic  > 0).mean() * 100, 1),
         "N days":  len(tf_ic)},
        {"Model": "GNN ✅",
         "IC":      round(gnn_ic.mean(),  5),
         "ICIR":    round(icir(gnn_ic),   4),
         "RankIC":  round(gnn_ric.mean(), 5),
         "IC pos%": round((gnn_ic > 0).mean() * 100, 1),
         "N days":  len(gnn_ic)},
    ]).set_index("Model")
    st.dataframe(comp_ml, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["IC Bar", "ICIR Bar", "Rolling IC", "Feature Importance"])

    colors = ["#ef4444","#ef4444","#f59e0b","#22c55e"]

    with tab1:
        fig = go.Figure(go.Bar(
            x=comp_ml.index, y=comp_ml["IC"],
            marker_color=colors,
            text=[f"{v:.5f}" for v in comp_ml["IC"]],
            textposition="outside"))
        fig.add_hline(y=0,    line_dash="dash",
                      line_color="white",   opacity=0.4)
        fig.add_hline(y=0.03, line_dash="dot",
                      line_color="#f59e0b", opacity=0.6,
                      annotation_text="weak threshold (0.03)",
                      annotation_position="right")
        fig.add_hline(y=0.05, line_dash="dot",
                      line_color="#22c55e", opacity=0.6,
                      annotation_text="tradable threshold (0.05)",
                      annotation_position="right")
        fig.update_layout(
            title="OOS IC mean — GNN crosses institutional tradability",
            template="plotly_dark", showlegend=False)
        fig.update_xaxes(title_text="model")
        fig.update_yaxes(title_text="IC mean")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure(go.Bar(
            x=comp_ml.index, y=comp_ml["ICIR"],
            marker_color=colors,
            text=[f"{v:.4f}" for v in comp_ml["ICIR"]],
            textposition="outside"))
        fig.add_hline(y=0,   line_dash="dash",
                      line_color="white",   opacity=0.4)
        fig.add_hline(y=0.5, line_dash="dot",
                      line_color="#22c55e", opacity=0.5,
                      annotation_text="strong ICIR (0.5)",
                      annotation_position="right")
        fig.update_layout(
            title="ICIR by model (IC mean / IC std)",
            template="plotly_dark", showlegend=False)
        fig.update_xaxes(title_text="model")
        fig.update_yaxes(title_text="ICIR")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        win = st.slider("Rolling window (days)", 10, 63, 21)
        fig = go.Figure()
        for name, series, color, lw in [
            ("GNN",         gnn_ic, "#22c55e", 2.5),
            ("Transformer", tf_ic,  "#f59e0b", 1.5),
            ("RF",          rf_ic,  "#94a3b8", 1.0),
            ("GBM",         gbm_ic, "#64748b", 1.0),
        ]:
            roll = series.rolling(win).mean()
            fig.add_trace(go.Scatter(
                x=roll.index, y=roll.values,
                name=name, mode="lines",
                line=dict(color=color, width=lw)))
        fig.add_hline(y=0,    line_dash="dash",
                      line_color="white",   opacity=0.3)
        fig.add_hline(y=0.05, line_dash="dot",
                      line_color="#22c55e", opacity=0.4)
        fig.update_layout(
            title=f"Rolling {win}d IC — all models",
            template="plotly_dark",
            legend=dict(orientation="h", y=1.1,
                        x=0.5, xanchor="center"))
        fig.update_xaxes(title_text="date")
        fig.update_yaxes(title_text="rolling IC")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        feat = ml["feat_imp"].sort_values("importance")
        fig  = go.Figure(go.Bar(
            x=feat["importance"], y=feat.index,
            orientation="h",
            marker_color="#6366f1"))
        fig.update_layout(
            title="RF feature importances — reversal_1d dominates",
            template="plotly_dark", showlegend=False)
        fig.update_xaxes(title_text="importance")
        fig.update_yaxes(title_text="feature")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.info("""
    **Key finding**: GNN (IC=0.051, ICIR=0.148, IC pos%=56.1%) is the only
    model crossing the institutional tradability threshold (IC > 0.05).
    The correlation-weighted graph structure captures semiconductor sector
    contagion that flat-feature models (RF, GBM) and sequential models
    (Transformer) cannot encode.

    **Next step**: Use GNN signal as a position-sizing overlay on CS Momentum —
    increase exposure when rolling GNN IC > 0, reduce when it falls below zero.
    """)
