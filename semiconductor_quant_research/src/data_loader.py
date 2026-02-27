"""
data_loader.py — Data download, caching, and multi-universe loading.

Supports three universe tiers (see src/universe.py):
  semi_core    : original 12-ticker semiconductor research universe.
  sp_tech_semi : ~80-ticker S&P Tech + Semiconductor expanded universe.
  r1000_tech   : ~150-ticker Russell-1000 tech/semi proxy universe.

Each universe tier is cached to its own parquet files under data/:
  prices_{universe}.parquet
  volume_{universe}.parquet
  returns_{universe}.parquet

This allows IC studies on the broad universe while keeping the original
semi_core strategy backtests untouched.

Minimum data threshold: a ticker needs ≥90% of trading days present to
be included.  This excludes names with IPO dates mid-period (survivorship
at the entry end) but prevents contamination from data gaps.
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# ── Legacy compatibility: keep original names at module level ─────────────────
SEMI  = ['NVDA', 'AMD', 'AVGO', 'TSM', 'QCOM',
         'AMAT', 'LRCX', 'MU', 'KLAC', 'TXN', 'ASML', 'MRVL']
TECH  = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
ALL   = sorted(set(SEMI + TECH))

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

START = '2020-01-01'
END   = '2026-02-26'

# ── Minimum fraction of trading days a ticker must have ──────────────────────
MIN_COVERAGE = 0.90


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _clean(df: pd.DataFrame, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """Drop columns with too many NaNs, then forward-fill remaining gaps."""
    df = df.dropna(axis=1, thresh=int(min_coverage * len(df)))
    df = df.ffill()
    df.index = pd.to_datetime(df.index)
    return df


def _parquet_paths(universe_name: str) -> tuple[Path, Path, Path]:
    """Return (prices, volume, returns) parquet paths for a universe name."""
    tag = universe_name.lower().replace("-", "_")
    return (
        DATA_DIR / f"prices_{tag}.parquet",
        DATA_DIR / f"volume_{tag}.parquet",
        DATA_DIR / f"returns_{tag}.parquet",
    )


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD (any universe)
# ══════════════════════════════════════════════════════════════════════════════

def download(
    tickers:       list[str] | None = None,
    start:         str = START,
    end:           str = END,
    universe_name: str = "semi_core",
    min_coverage:  float = MIN_COVERAGE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download OHLCV data from Yahoo Finance for *tickers* and cache to Parquet.

    Parameters
    ----------
    tickers       : List of ticker symbols.  Defaults to the legacy ALL list
                    (SEMI + TECH) for backwards compatibility.
    start, end    : Date strings for yfinance.
    universe_name : Tag used for parquet filenames.  Use one of
                    "semi_core" | "sp_tech_semi" | "r1000_tech".
    min_coverage  : Minimum fraction of trading-day rows a ticker must have
                    to be retained (drops tickers with major data gaps).

    Returns
    -------
    close, volume, ret : DataFrames with DatetimeIndex, columns = tickers.
    """
    if tickers is None:
        tickers = ALL

    print(f"Downloading {len(tickers)} tickers [{universe_name}]  "
          f"{start} → {end}")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by='column',
        progress=True,
    )

    # ── Handle both MultiIndex (>1 ticker) and flat (1 ticker) outputs ────────
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw['Close'].copy()
        volume = raw['Volume'].copy()
    else:
        close  = raw[['Close']].copy()
        close.columns = tickers[:1]
        volume = raw[['Volume']].copy()
        volume.columns = tickers[:1]

    close  = _clean(close,  min_coverage)
    volume = _clean(volume, min_coverage)
    ret    = np.log(close / close.shift(1)).dropna()

    p_path, v_path, r_path = _parquet_paths(universe_name)
    close.to_parquet(p_path)
    volume.to_parquet(v_path)
    ret.to_parquet(r_path)

    print(f"\n✓ {p_path.name}  : {close.shape}")
    print(f"✓ {v_path.name}  : {volume.shape}")
    print(f"✓ {r_path.name} : {ret.shape}")
    print(f"  Period  : {close.index[0].date()} → {close.index[-1].date()}")
    print(f"  Tickers : {sorted(close.columns.tolist())}")

    return close, volume, ret


# ══════════════════════════════════════════════════════════════════════════════
# LOAD (any universe, with auto-download fallback)
# ══════════════════════════════════════════════════════════════════════════════

def load(
    universe_name: str = "semi_core",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load cached Parquet data for *universe_name*.

    If the parquet files do not yet exist, automatically downloads the
    corresponding universe using the tickers defined in src/universe.py.

    Parameters
    ----------
    universe_name : "semi_core" | "sp_tech_semi" | "r1000_tech"

    Returns
    -------
    close, volume, ret : DataFrames.
    """
    # ── Handle legacy call: load() with no args returns semi_core ─────────────
    p_path, v_path, r_path = _parquet_paths(universe_name)

    # Backwards-compat: original parquet files have no universe suffix
    if universe_name == "semi_core":
        legacy_p = DATA_DIR / "prices.parquet"
        legacy_v = DATA_DIR / "volume.parquet"
        legacy_r = DATA_DIR / "returns.parquet"
        if legacy_p.exists() and not p_path.exists():
            # Rename on first run so both paths work
            import shutil
            shutil.copy(legacy_p, p_path)
            shutil.copy(legacy_v, v_path)
            shutil.copy(legacy_r, r_path)

    if p_path.exists() and v_path.exists() and r_path.exists():
        close  = pd.read_parquet(p_path)
        volume = pd.read_parquet(v_path)
        ret    = pd.read_parquet(r_path)
        return close, volume, ret

    # ── Auto-download if not cached ───────────────────────────────────────────
    print(f"[data_loader] Cache miss for universe '{universe_name}' — "
          f"downloading now...")
    from src.universe import get_universe
    tickers = get_universe(universe_name)
    return download(tickers=tickers, universe_name=universe_name)


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: download all universe tiers at once
# ══════════════════════════════════════════════════════════════════════════════

def download_all_universes(start: str = START, end: str = END) -> None:
    """
    Download and cache data for all three universe tiers.

    Useful as a one-time setup step before running IC studies.
    """
    from src.universe import get_universe

    for uname in ["semi_core", "sp_tech_semi", "r1000_tech"]:
        tickers = get_universe(uname)
        download(tickers=tickers, start=start, end=end,
                 universe_name=uname)
        print()


if __name__ == "__main__":
    # Default: download the original semi_core universe (backwards-compat)
    download(tickers=ALL, universe_name="semi_core")

