"""FRED 시리즈 수집 + 발표 지연 시프트."""
from __future__ import annotations

import pandas as pd
import requests
from pandas.tseries.offsets import BDay

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred(series: dict[str, int], start: str, api_key: str | None) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    cols = {}
    for sid, lag in series.items():
        r = requests.get(FRED_URL, params={
            "series_id": sid, "api_key": api_key, "file_type": "json", "observation_start": start,
        }, timeout=30)
        r.raise_for_status()
        obs = r.json()["observations"]
        s = pd.Series(
            pd.to_numeric([o["value"] for o in obs], errors="coerce"),
            index=pd.DatetimeIndex([o["date"] for o in obs]) + BDay(lag),
            dtype=float,
        )
        cols[sid.lower()] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()
