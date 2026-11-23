"""yfinance 일별 종가 수집."""
from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

PRICE_TICKERS = {
    "ndx": "^NDX", "spy": "SPY", "rsp": "RSP", "vix": "^VIX", "vix3m": "^VIX3M",
    "oil": "CL=F", "gold": "GC=F", "dxy": "DX-Y.NYB",
    "hyg": "HYG", "ief": "IEF",
}
RETRIES = 3
RETRY_SLEEP = 3.0   # 초 (시도마다 늘어남)


def _normalize(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _fetch_one(ticker: str, start: str) -> pd.Series | None:
    """일괄 다운로드에서 빠진 티커를 하나씩 재시도 (클라우드에서 야후가 간헐적으로 일부를 비워 보냄)."""
    for attempt in range(1, RETRIES + 1):
        try:
            h = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            s = h["Close"].dropna() if "Close" in h else pd.Series(dtype=float)
            if not s.empty:
                s.index = _normalize(s.index)
                return s[~s.index.duplicated(keep="last")].astype(float)
        except Exception as e:
            log.warning("retry %s #%d failed: %s", ticker, attempt, type(e).__name__)
        if attempt < RETRIES:
            time.sleep(RETRY_SLEEP * attempt)
    return None


def fetch_prices(start: str) -> pd.DataFrame:
    raw = yf.download(list(PRICE_TICKERS.values()), start=start, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    inverse = {v: k for k, v in PRICE_TICKERS.items()}
    out = close.rename(columns=inverse).reindex(columns=list(PRICE_TICKERS)).astype(float)
    out.index = _normalize(out.index)
    out = out.sort_index()
    ndx_last = out["ndx"].last_valid_index()
    for name, ticker in PRICE_TICKERS.items():
        last = out[name].last_valid_index()
        # 통째로 비었거나, 마지막 값이 나스닥 최신일보다 오래된 경우 → 개별 재시도
        if last is None or (name != "ndx" and ndx_last is not None and last < ndx_last):
            log.warning("%s %s in batch download, retrying individually", ticker, "empty" if last is None else f"stale ({last.date()})")
            s = _fetch_one(ticker, start)
            if s is not None and (last is None or s.index[-1] > last):
                out = out.reindex(out.index.union(s.index))
                out[name] = s.reindex(out.index)
    return out.sort_index()
