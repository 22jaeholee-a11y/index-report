"""데이터 수집 통합: 가격 + FRED (+ 부가 소스)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .fred import fetch_fred
from .prices import PRICE_TICKERS, fetch_prices

log = logging.getLogger(__name__)


class DataSourceError(RuntimeError):
    pass


@dataclass
class MarketData:
    frame: pd.DataFrame
    as_of: date
    missing: list[str] = field(default_factory=list)
    extras: dict[str, float | None] = field(default_factory=dict)


def collect_all(cfg: dict, start: str, include_extras: bool = True) -> MarketData:
    missing: list[str] = []
    stale_limit: dict[str, int] = {}

    try:
        px = fetch_prices(start)
    except Exception as e:  # 가격 전체 실패 = 핵심 데이터 실패
        raise DataSourceError(f"price download failed: {e}") from e
    if "ndx" not in px or px["ndx"].dropna().empty:
        raise DataSourceError("NDX price data is empty")
    index = px["ndx"].dropna().index
    parts = [px.reindex(index)]
    for c in PRICE_TICKERS:
        stale_limit[c] = cfg["stale_days"]

    fred_cols = [s.lower() for s in cfg["fred_series"]]
    try:
        fr = fetch_fred(cfg["fred_series"], start, os.environ.get("FRED_API_KEY"))
    except Exception as e:
        log.warning("FRED failed: %s", e)
        fr = pd.DataFrame()
    for sid, lag in cfg["fred_series"].items():
        stale_limit[sid.lower()] = cfg["stale_days"] + lag
    parts.append(_align(fr.reindex(columns=fred_cols), index))

    if include_extras:
        from . import extras  # Task 3
        extra_frame, extra_vals, extra_missing = extras.collect_extras(cfg, index)
        parts.append(extra_frame)
        for c in extra_frame.columns:
            stale_limit[c] = cfg["stale_days"] + extras.EXTRA_STALE_DAYS.get(c, 0)
        missing += extra_missing
    else:
        extra_vals = {}

    frame = pd.concat(parts, axis=1)
    frame = _apply_staleness(frame, stale_limit, missing)
    return MarketData(frame=frame, as_of=index[-1].date(), missing=missing, extras=extra_vals)


def _align(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """거래일 인덱스에 맞춤. 비거래일 관측값은 다음 거래일로 옮기고, 기준일 이후 값은 버린다(미래 누출 방지)."""
    if df.empty:
        return pd.DataFrame(index=index, columns=df.columns, dtype=float)
    df = df[df.index <= index[-1]]
    pos = index.searchsorted(df.index, side="left")
    moved = df.groupby(pos).last()
    moved.index = index[moved.index]
    return moved.reindex(index)


def _apply_staleness(frame: pd.DataFrame, limits: dict[str, int], missing: list[str]) -> pd.DataFrame:
    """열별 마지막 유효일이 기준일보다 limit 영업일 넘게 오래되면 마지막 행을 NaN으로 두고 누락 처리. 그 외엔 ffill."""
    as_of = frame.index[-1]
    out = frame.ffill()
    for c in frame.columns:
        last = frame[c].last_valid_index()
        if last is None:
            if c not in missing:
                missing.append(c)
            continue
        age = len(pd.bdate_range(last, as_of)) - 1
        if age > limits.get(c, 5):
            out.loc[as_of, c] = np.nan
            if c not in missing:
                missing.append(c)
    return out
