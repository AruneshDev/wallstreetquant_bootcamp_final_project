"""
ic_study.py — Cross-universe IC evaluation for all signals.

Loads feature panels for a given universe tier and computes IC / RankIC for
every available signal column.  Designed to be run across all three universe
tiers to find which signals survive at scale.

With N=12 (semi_core) the minimum resolvable IC ≈ 0.12 per date (one rank
swap).  With N=71 (sp_tech_semi) it falls to ≈ 0.025, and with N=108
(r1000_tech) to ≈ 0.019.  IC t-stat improves by:
  √(71/12) ≈ 2.4×  for sp_tech_semi
  √(108/12) ≈ 3.0× for r1000_tech

Running:
  ./.venv/bin/python src/ic_study.py --universe sp_tech_semi
  ./.venv/bin/python src/ic_study.py --universe r1000_tech
  ./.venv/bin/python src/ic_study.py --compare          ← prints cross-universe table

Outputs:
  results/ic_study_{universe_name}.csv   per-signal IC metrics
  results/ic_study_comparison.csv        side-by-side comparison across tiers
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ── Signals to evaluate (column name → display label) ────────────────────────
SIGNAL_META: dict[str, str] = {
    # OHLCV-derived (from features.parquet / features_{univ}.parquet)
    "mom_20d":        "Momentum 20d",
    "mom_60d":        "Momentum 60d",
    "reversal_1d":    "Short-term reversal 1d",
    "vol_ratio":      "Volume ratio (5d/20d)",
    "dist_52w_high":  "Distance from 52w high",
    "cs_rank_mom10":  "CS rank momentum 10d",
    # Alt-data (from features_alt.parquet / features_alt_{univ}.parquet)
    "sue":            "Earnings surprise (SUE)",
    "sue_decay":      "SUE decay-weighted",
    "arm_signal":     "Analyst revision proxy (ARM)",
    "si_proxy":       "Short interest proxy (SI)",
    # NLP (from features_nlp.parquet / features_nlp_{univ}.parquet)
    "nlp_sent":       "NLP sentiment score",
    "nlp_drift":      "NLP tone drift (vs year-ago)",
}

# IC threshold for ✅ / ~ / ❌ flags
IC_PASS_MEAN   = 0.02    # IC_mean > 0.02
IC_PASS_POS    = 0.52    # IC_pos_pct > 52%


# ══════════════════════════════════════════════════════════════════════════════
# CORE IC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ic_series(
    signal_panel: pd.DataFrame,
    signal_col:   str,
    fwd:          pd.DataFrame,      # forward return wide (dates × tickers)
) -> tuple[pd.Series, pd.Series]:
    """
    Compute daily IC and RankIC series for one signal column.

    Parameters
    ----------
    signal_panel : MultiIndex (date, ticker) panel with *signal_col*.
    signal_col   : Column name.
    fwd          : Wide forward return DataFrame (dates × tickers).

    Returns
    -------
    ic_series, rank_ic_series — pd.Series indexed by date.
    """
    ics:  dict[pd.Timestamp, float] = {}
    rks:  dict[pd.Timestamp, float] = {}

    dates = signal_panel.index.get_level_values("date").unique()
    for date in dates:
        try:
            sig_row = signal_panel.xs(date, level="date")[signal_col].dropna()
        except KeyError:
            continue
        if date not in fwd.index or len(sig_row) < 5:
            continue

        fwd_row = fwd.loc[date, [t for t in sig_row.index if t in fwd.columns]]
        both    = pd.concat([sig_row, fwd_row], axis=1).dropna()
        both.columns = ["sig", "fwd"]
        if len(both) < 5:
            continue

        ics[date]  = pearsonr(both["sig"], both["fwd"])[0]
        rks[date]  = spearmanr(both["sig"], both["fwd"])[0]

    return pd.Series(ics).sort_index(), pd.Series(rks).sort_index()


def _summarise_ic(
    ic_s:      pd.Series,
    ric_s:     pd.Series,
    signal:    str,
    fwd_days:  int,
    n_tickers: int,
) -> dict:
    """Compute summary statistics from IC / RankIC series."""
    ic_mean  = ic_s.mean()
    ic_std   = ic_s.std()
    icir     = ic_mean / ic_std if ic_std > 0 else np.nan
    ric_mean = ric_s.mean()
    ric_std  = ric_s.std()
    ricir    = ric_mean / ric_std if ric_std > 0 else np.nan
    ic_pos   = (ic_s > 0).mean()

    # ── IC t-stat (two-tailed, H0: IC=0) ─────────────────────────────────────
    # t = IC_mean / (IC_std / √N_days)
    n_days   = len(ic_s)
    ic_tstat = (ic_mean / (ic_std / np.sqrt(n_days))
                if ic_std > 0 and n_days > 0 else np.nan)

    flag = (
        "✅" if (ic_mean > IC_PASS_MEAN and ic_pos > IC_PASS_POS)
        else ("~" if ic_mean > 0
              else "❌")
    )

    return {
        "signal":       signal,
        "fwd_days":     fwd_days,
        "IC_mean":      round(ic_mean,  5),
        "IC_std":       round(ic_std,   5),
        "ICIR":         round(icir,     4),
        "IC_tstat":     round(ic_tstat, 3),
        "RankIC_mean":  round(ric_mean, 5),
        "RankICIR":     round(ricir,    4),
        "IC_pos_pct":   round(ic_pos,   4),
        "N_days":       n_days,
        "N_tickers":    n_tickers,
        "flag":         flag,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PANEL LOADER — reads cached parquet files for a universe
# ══════════════════════════════════════════════════════════════════════════════

def _load_feature_panels(
    universe_name: str,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame],
           Optional[pd.DataFrame]]:
    """
    Load OHLCV, alt-data, and NLP feature panels for *universe_name*.

    Returns (ohlcv_panel, alt_panel, nlp_panel) — any may be None if the
    file does not exist yet.
    """
    def _read(path: Path) -> Optional[pd.DataFrame]:
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  ✓ Loaded {path}  shape={df.shape}")
            return df
        print(f"  ⚠️  {path} not found — skipping.")
        return None

    if universe_name == "semi_core":
        ohlcv_path = Path("data/features.parquet")
        alt_path   = Path("data/features_alt.parquet")
        nlp_path   = Path("data/features_nlp.parquet")
    else:
        ohlcv_path = Path(f"data/features_{universe_name}.parquet")
        alt_path   = Path(f"data/features_alt_{universe_name}.parquet")
        nlp_path   = Path(f"data/features_nlp_{universe_name}.parquet")

    return _read(ohlcv_path), _read(alt_path), _read(nlp_path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN IC STUDY FUNCTION (Task 4)
# ══════════════════════════════════════════════════════════════════════════════

def run_ic_study(
    universe_name: str,
    fwd_days:      int = 5,
    save_prefix:   str = "",
) -> pd.DataFrame:
    """
    Load all feature panels for *universe_name*, run IC evaluation for every
    available signal, and save results to results/ic_study_{universe_name}.csv.

    Parameters
    ----------
    universe_name : "semi_core" | "sp_tech_semi" | "r1000_tech"
    fwd_days      : Forward return horizon for IC computation.
    save_prefix   : Optional prefix for output file name.

    Returns
    -------
    DataFrame with IC metrics for every signal evaluated, indexed by "signal".
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load

    print(f"\n{'='*70}")
    print(f"  IC Study | universe={universe_name} | fwd={fwd_days}d")
    print(f"{'='*70}")

    # ── Load price/return data ────────────────────────────────────────────────
    close, volume, ret = load(universe_name=universe_name)
    n_tickers          = ret.shape[1]
    print(f"\n  N_tickers = {n_tickers}  |  "
          f"min resolvable IC ≈ {2/n_tickers:.3f}  |  "
          f"IC t-stat boost vs 12: {(n_tickers/12)**0.5:.2f}×")

    # ── Build forward return wide matrix ─────────────────────────────────────
    fwd = ret.rolling(fwd_days).sum().shift(-fwd_days)

    # ── Load feature panels ───────────────────────────────────────────────────
    print("\nLoading feature panels...")
    ohlcv_panel, alt_panel, nlp_panel = _load_feature_panels(universe_name)

    # ── Merge all panels into one tall panel ──────────────────────────────────
    parts: list[pd.DataFrame] = []
    if ohlcv_panel is not None and not ohlcv_panel.empty:
        ohlcv_sigs = [c for c in SIGNAL_META if c in ohlcv_panel.columns]
        if ohlcv_sigs:
            parts.append(ohlcv_panel[ohlcv_sigs])
    if alt_panel is not None and not alt_panel.empty:
        alt_sigs = [c for c in SIGNAL_META if c in alt_panel.columns]
        if alt_sigs:
            parts.append(alt_panel[alt_sigs])
    if nlp_panel is not None and not nlp_panel.empty:
        nlp_sigs = [c for c in SIGNAL_META if c in nlp_panel.columns]
        if nlp_sigs:
            parts.append(nlp_panel[nlp_sigs])

    if not parts:
        print("\n⚠️  No feature panels found.  "
              "Run features.py, features_alt.py, nlp_signal.py first.")
        return pd.DataFrame()

    # Merge on (date, ticker) index — outer join so each signal keeps its dates
    merged = parts[0]
    for extra in parts[1:]:
        new_cols = [c for c in extra.columns if c not in merged.columns]
        if new_cols:
            merged = merged.join(extra[new_cols], how="outer")

    print(f"\n  Merged panel: {merged.shape}  "
          f"({merged.index.get_level_values('ticker').nunique()} tickers, "
          f"{merged.index.get_level_values('date').nunique()} dates)")

    # ── Evaluate each signal ──────────────────────────────────────────────────
    available_signals = [s for s in SIGNAL_META if s in merged.columns]
    print(f"\n  Signals available: {available_signals}\n")

    rows: list[dict] = []
    for sig in available_signals:
        sub = merged[[sig]].dropna()
        if sub.empty:
            continue
        ic_s, ric_s = _compute_ic_series(sub, sig, fwd)
        if ic_s.empty:
            continue
        row = _summarise_ic(ic_s, ric_s, sig, fwd_days, n_tickers)
        rows.append(row)

        label = SIGNAL_META.get(sig, sig)
        print(f"  {row['flag']}  {label:<35} "
              f"IC={row['IC_mean']:+.4f}  ICIR={row['ICIR']:+.3f}  "
              f"t={row['IC_tstat']:+.2f}  pos%={row['IC_pos_pct']*100:.0f}%  "
              f"N={row['N_tickers']}")

    if not rows:
        print("\n⚠️  No IC results produced — check feature panel alignment.")
        return pd.DataFrame()

    result_df = pd.DataFrame(rows).set_index("signal")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    prefix = f"{save_prefix}_" if save_prefix else ""
    out_path = f"results/{prefix}ic_study_{universe_name}.csv"
    result_df.to_csv(out_path)
    print(f"\n✓ Saved {out_path}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    passing = result_df[
        (result_df["IC_mean"]    > IC_PASS_MEAN) &
        (result_df["IC_pos_pct"] > IC_PASS_POS)
    ]
    if passing.empty:
        print(
            "\n📋 Research finding: No IC-positive signals found at scale "
            f"on {universe_name}.\n"
            "   Feature set needs fundamental rethink (earnings quality,\n"
            "   analyst coverage, sector-relative features)."
        )
    else:
        print(f"\n  Signals passing IC threshold on {universe_name}:")
        for sig, row in passing.iterrows():
            print(f"    ✅  {sig:<22}  IC={row['IC_mean']:+.4f}  "
                  f"t={row['IC_tstat']:+.2f}")

    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-UNIVERSE COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def compare_universes(
    universes: list[str] = None,
    fwd_days:  int = 5,
) -> pd.DataFrame:
    """
    Run ic_study for multiple universes and print a side-by-side IC table.

    Outputs results/ic_study_comparison.csv.
    """
    if universes is None:
        universes = ["semi_core", "sp_tech_semi", "r1000_tech"]

    all_results: dict[str, pd.DataFrame] = {}
    for u in universes:
        df = run_ic_study(u, fwd_days=fwd_days)
        if not df.empty:
            all_results[u] = df

    if not all_results:
        print("No IC results available for comparison.")
        return pd.DataFrame()

    # ── Build side-by-side comparison ─────────────────────────────────────────
    all_signals = sorted(
        set(s for df in all_results.values() for s in df.index)
    )
    rows = []
    for sig in all_signals:
        label = SIGNAL_META.get(sig, sig)
        row: dict = {"signal": sig, "label": label}
        best_ic   = -np.inf
        best_univ = "—"
        for u, df in all_results.items():
            if sig in df.index:
                ic   = df.loc[sig, "IC_mean"]
                flag = df.loc[sig, "flag"]
                row[f"{u}_IC"]   = round(ic, 4)
                row[f"{u}_flag"] = flag
                if ic > best_ic:
                    best_ic   = ic
                    best_univ = u
            else:
                row[f"{u}_IC"]   = np.nan
                row[f"{u}_flag"] = "—"
        row["best_universe"] = best_univ
        rows.append(row)

    comp_df = pd.DataFrame(rows).set_index("signal")

    # ── Pretty-print table ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Cross-Universe IC Comparison")
    print(f"{'='*80}")
    hdr_univs = [u for u in universes if u in all_results]
    col_widths = 10
    header = (f"  {'Signal':<22}  {'Label':<30}  "
              + "  ".join(f"{u[:12]:>{col_widths}}" for u in hdr_univs)
              + f"  {'Best':>15}")
    print(header)
    print(f"  {'─'*80}")
    for sig, row in comp_df.iterrows():
        ic_strs = []
        for u in hdr_univs:
            ic  = row.get(f"{u}_IC",   np.nan)
            flg = row.get(f"{u}_flag", "—")
            ic_strs.append(
                f"{ic:>+8.4f} {flg}" if not np.isnan(ic) else f"{'—':>10}"
            )
        print(f"  {sig:<22}  {row['label']:<30}  "
              + "  ".join(ic_strs)
              + f"  {row['best_universe']:>15}")

    print(f"{'='*80}\n")

    out = "results/ic_study_comparison.csv"
    comp_df.to_csv(out)
    print(f"✓ Saved {out}")
    return comp_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    parser = argparse.ArgumentParser(
        description="Run cross-sectional IC study across universe tiers."
    )
    parser.add_argument(
        "--universe", default="sp_tech_semi",
        choices=["semi_core", "sp_tech_semi", "r1000_tech"],
        help="Universe tier to evaluate signals on.",
    )
    parser.add_argument(
        "--fwd-days", type=int, default=5,
        help="Forward return horizon in trading days (default: 5).",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run IC study on all three universes and print comparison table.",
    )
    args = parser.parse_args()

    if args.compare:
        compare_universes(fwd_days=args.fwd_days)
    else:
        run_ic_study(universe_name=args.universe, fwd_days=args.fwd_days)
