from __future__ import annotations

"""
universe.py — Canonical ticker universe definitions for cross-sectional research.

Provides three universe tiers:
  SEMI_CORE    : 12 U.S.-listed semiconductor names (original research universe).
  TECH_CORE    : 5 mega-cap tech names (original).
  SP_TECH_SEMI : ~80-ticker S&P 500 Tech + Semiconductor expanded universe.
                 Adds breadth for cross-sectional IC computation; ~5× more
                 tickers means IC t-stats improve by √5 ≈ 2.2×.
  R1000_TECH   : Approximate Russell-1000 tech/semi proxy (~150 tickers).
                 Full cross-sectional universe for alpha research.

...
"""

# ── Original 12-name semiconductor core ──────────────────────────────────────
SEMI_CORE: list[str] = [
    "NVDA", "AMD", "AVGO", "TSM", "QCOM",
    "AMAT", "LRCX", "MU", "KLAC", "TXN", "ASML", "MRVL",
]

# ── Original 5 mega-cap tech names ───────────────────────────────────────────
TECH_CORE: list[str] = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]

# ── S&P 500 Technology + Semiconductor sector (~80 names) ────────────────────
# Includes all SEMI_CORE + TECH_CORE plus broad S&P Tech constituents.
# Provides ~6–7× cross-sectional breadth vs SEMI_CORE alone.
SP_TECH_SEMI: list[str] = sorted(set(SEMI_CORE + TECH_CORE + [
    # Semiconductors & Semi Equipment
    "ON", "MCHP", "ADI", "NXPI", "SWKS", "QRVO", "INTC", "MPWR",
    "WOLF", "ACLS", "ENTG", "CEVA", "SMTC", "SITM", "DIOD",
    "RMBS", "FORM", "AMBA",
    # Software & Cloud
    "MSFT", "ORCL", "SAP", "NOW", "CRM", "ADSK", "INTU", "ANSS",
    "CDNS", "SNPS", "CTSH", "EPAM", "GLOB",
    # Hyperscalers / Cloud Infrastructure
    "AMZN", "GOOGL", "META", "MSFT",
    "CSCO", "ANET", "HPE", "JNPR", "NTAP",
    # Hardware / Storage
    "AAPL", "DELL", "HPQ", "WDC", "STX", "PSTG",
    # Payments / FinTech (Tech-adjacent in S&P)
    "V", "MA", "PYPL", "FISV", "FIS",
    # Internet / Platform
    "NFLX", "UBER", "LYFT", "SNAP", "PINS", "TWTR",
    # Cybersecurity
    "CRWD", "PANW", "FTNT", "OKTA", "ZS", "S",
    # Data & Analytics
    "SNOW", "DDOG", "MDB", "PLTR", "CFLT",
    # EDA / CAD (adjacent semis)
    "CDNS", "SNPS",
]))

# ── Russell-1000 Tech proxy (~150 names): SP_TECH_SEMI + mid-cap tech ────────
R1000_TECH: list[str] = sorted(set(SP_TECH_SEMI + [
    # Mid-cap semis
    "IPGP", "COHU", "CRUS", "MTSI", "SLAB", "IOSP",
    "ONTO", "ICHR", "ACMR", "KLIC", "AXTI",
    # Mid-cap software
    "PCTY", "MANH", "FRSH", "SPSC", "QTWO",
    "VEEV", "PAYC", "HUBS", "ZM", "DOCU",
    # IT Services
    "ACN", "IBM", "LDOS", "SAIC", "CACI",
    # Telecom / Networks
    "T", "VZ", "TMUS", "LUMN",
    # Consumer Tech
    "SONO", "ROKU", "SPOT",
    # Gaming
    "TTWO", "EA", "RBLX", "U",
    # Cloud enablers
    "TWLO", "BAND", "ESTC", "NEWR",
]))

# ── Benchmark tickers (not in universe, used as overlays) ────────────────────
BENCHMARKS: list[str] = ["SOXX", "SPY", "QQQ", "XLK", "SMH"]


def get_universe(name: str = "semi_core") -> list[str]:
    """
    Return a ticker list by name.

    Parameters
    ----------
    name : str
        One of "semi_core", "tech_core", "sp_tech_semi", "r1000_tech".

    Returns
    -------
    list[str]
        Sorted list of ticker symbols.
    """
    mapping = {
        "semi_core":    SEMI_CORE,
        "tech_core":    TECH_CORE,
        "sp_tech_semi": SP_TECH_SEMI,
        "r1000_tech":   R1000_TECH,
    }
    key = name.lower().strip()
    if key not in mapping:
        raise ValueError(
            f"Unknown universe '{name}'.  "
            f"Valid options: {list(mapping.keys())}"
        )
    return list(mapping[key])


if __name__ == "__main__":
    for uname in ["semi_core", "sp_tech_semi", "r1000_tech"]:
        u = get_universe(uname)
        print(f"{uname:<16}: {len(u):>3} tickers  "
              f"(first 5: {u[:5]})")
