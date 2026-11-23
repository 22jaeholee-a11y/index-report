"""부가 소스: GPR(점수 사용), Fear & Greed·NQ 선물(참고값)."""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests
import yfinance as yf
from pandas.tseries.offsets import BDay

log = logging.getLogger(__name__)

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
GPR_LAG_BDAYS = 5  # 파일이 주 단위로 갱신됨 → 백테스트 누출 방지용 지연
EXTRA_STALE_DAYS = {"gpr": GPR_LAG_BDAYS}
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://edition.cnn.com/",
    "Origin": "https://edition.cnn.com",
}


def _read_gpr_raw() -> pd.DataFrame:
    r = requests.get(GPR_URL, headers={"User-Agent": BROWSER_HEADERS["User-Agent"]}, timeout=60)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), engine="calamine")


def fetch_gpr() -> pd.Series:
    raw = _read_gpr_raw()
    raw = raw[pd.to_numeric(raw["DAY"], errors="coerce").notna()]
    idx = pd.to_datetime(raw["DAY"].astype(int).astype(str), format="%Y%m%d") + BDay(GPR_LAG_BDAYS)
    s = pd.Series(pd.to_numeric(raw["GPRD"], errors="coerce").to_numpy(), index=idx, name="gpr")
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_fear_greed() -> float | None:
    try:
        r = requests.get(FNG_URL, headers=BROWSER_HEADERS, timeout=30)
        r.raise_for_status()
        return float(r.json()["fear_and_greed"]["score"])
    except Exception as e:
        log.warning("Fear&Greed failed: %s", e)
        return None


def fetch_nq_futures_change() -> float | None:
    try:
        t = yf.Ticker("NQ=F")
        daily = t.history(period="10d", interval="1d")["Close"].dropna()
        hourly = t.history(period="5d", interval="1h")["Close"].dropna()
        if len(daily) < 2 or hourly.empty:
            return None
        last_day = hourly.index[-1].date()
        prev = daily[[d.date() < last_day for d in daily.index]]
        if prev.empty:
            return None
        return float(hourly.iloc[-1] / prev.iloc[-1] - 1)
    except Exception as e:
        log.warning("NQ futures failed: %s", e)
        return None


def collect_extras(cfg: dict, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict, list[str]]:
    """반환: (거래일 인덱스에 정렬된 gpr 프레임, 참고값 dict, 누락 목록)."""
    from . import _align

    missing: list[str] = []
    try:
        gpr = fetch_gpr().to_frame()
    except Exception as e:
        log.warning("GPR failed: %s", e)
        gpr = pd.DataFrame(columns=["gpr"], dtype=float)
    frame = _align(gpr, index)
    vals = {"fear_greed": fetch_fear_greed(), "nq_change": fetch_nq_futures_change()}
    if vals["fear_greed"] is None:
        missing.append("fear_greed")
    return frame, vals, missing
