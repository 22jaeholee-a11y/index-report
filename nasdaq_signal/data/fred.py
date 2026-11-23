"""FRED 시리즈 수집 + 발표 지연 시프트."""
from __future__ import annotations

import logging

import pandas as pd
import requests
from pandas.tseries.offsets import BDay

log = logging.getLogger(__name__)
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred(series: dict[str, int], start: str, api_key: str | None) -> pd.DataFrame:
    if not api_key:
        log.warning("FRED_API_KEY not set: macro indicators skipped")
        return pd.DataFrame()
    cols = {}
    for sid, lag in series.items():
        try:
            r = requests.get(FRED_URL, params={
                "series_id": sid, "api_key": api_key, "file_type": "json", "observation_start": start,
            }, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            obs = r.json()["observations"]
        except Exception as e:
            # requests 예외 메시지에는 api_key가 든 URL이 포함되므로 직접 만든 HTTP 상태만 그대로 기록
            reason = str(e) if str(e).startswith("HTTP ") else type(e).__name__
            log.warning("FRED %s failed: %s", sid, reason)
            continue
        s = pd.Series(
            pd.to_numeric([o["value"] for o in obs], errors="coerce"),
            index=pd.DatetimeIndex([o["date"] for o in obs]) + BDay(lag),
            dtype=float,
        )
        cols[sid.lower()] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()
