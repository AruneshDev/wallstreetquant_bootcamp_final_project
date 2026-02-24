import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SEMI  = ['NVDA','AMD','AVGO','TSM','QCOM','AMAT','LRCX','MU','KLAC','TXN','ASML','MRVL']
TECH  = ['AAPL','MSFT','GOOGL','META','AMZN']
ALL   = sorted(set(SEMI + TECH))

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

START = '2020-01-01'
END   = '2026-02-22'


def download(start: str = START, end: str = END):
    print(f"Downloading {len(ALL)} tickers | {start} → {end}")
    raw = yf.download(
        ALL,
        start=start,
        end=end,
        auto_adjust=True,
        group_by='column',
        progress=True
    )

    # Force flat columns (no multi-index)
    close  = raw['Close'].copy()
    volume = raw['Volume'].copy()

    # Drop tickers with too many missing values
    close  = close.dropna(axis=1, thresh=int(0.9 * len(close)))
    volume = volume.dropna(axis=1, thresh=int(0.9 * len(volume)))
    close  = close.ffill().dropna()
    volume = volume.ffill()

    close.index  = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)

    ret = np.log(close / close.shift(1)).dropna()

    close.to_parquet(DATA_DIR  / "prices.parquet")
    volume.to_parquet(DATA_DIR / "volume.parquet")
    ret.to_parquet(DATA_DIR    / "returns.parquet")

    print(f"\n✓ prices.parquet  : {close.shape}")
    print(f"✓ volume.parquet  : {volume.shape}")
    print(f"✓ returns.parquet : {ret.shape}")
    print(f"  Period : {close.index[0].date()} → {close.index[-1].date()}")
    print(f"  Tickers: {sorted(close.columns.tolist())}")
    return close, volume, ret


def load():
    close  = pd.read_parquet(DATA_DIR / "prices.parquet")
    volume = pd.read_parquet(DATA_DIR / "volume.parquet")
    ret    = pd.read_parquet(DATA_DIR / "returns.parquet")
    return close, volume, ret


if __name__ == "__main__":
    download()
