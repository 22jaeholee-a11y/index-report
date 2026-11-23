"""yfinance 일별 종가 수집."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

PRICE_TICKERS = {
    "ndx": "^NDX", "spy": "SPY", "rsp": "RSP", "vix": "^VIX", "vix3m": "^VIX3M",
    "oil": "CL=F", "gold": "GC=F", "dxy": "DX-Y.NYB",
    "hyg": "HYG", "ief": "IEF",
}


def fetch_prices(start: str) -> pd.DataFrame:
    raw = yf.download(list(PRICE_TICKERS.values()), start=start, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    inverse = {v: k for k, v in PRICE_TICKERS.items()}
    out = close.rename(columns=inverse).reindex(columns=list(PRICE_TICKERS)).astype(float)
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out.sort_index()
