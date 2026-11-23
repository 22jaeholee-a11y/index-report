"""원지표 계산과 기간별 신호 변환 [-1, +1]."""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = ("short", "mid", "long")


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return frame[name].astype(float)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def compute_raw(frame: pd.DataFrame) -> pd.DataFrame:
    c = lambda n: _col(frame, n)  # noqa: E731
    ndx = c("ndx")
    raw = {
        "trend_50": ndx / ndx.rolling(50).mean() - 1,
        "trend_200": ndx / ndx.rolling(200).mean() - 1,
        "ret_20": ndx.pct_change(20, fill_method=None),
        "ret_60": ndx.pct_change(60, fill_method=None),
        "drawdown_52w": ndx / ndx.rolling(252, min_periods=1).max() - 1,
        "breadth": (c("rsp") / c("spy")).pct_change(60, fill_method=None),
        "vix": c("vix"),
        "vix_term": c("vix") / c("vix3m"),
        "gpr": c("gpr").rolling(30, min_periods=20).mean(),
        "oil_ret_20": c("oil").pct_change(20, fill_method=None),
        "gold_ret_20": c("gold").pct_change(20, fill_method=None),
        "us10y_chg_20": c("dgs10").diff(20),
        "curve": c("t10y2y"),
        "hy_spread": c("bamlh0a0hym2"),
        "dxy_ret_60": c("dxy").pct_change(60, fill_method=None),
        "nfci": c("nfci"),
        "sahm": c("sahmrealtime"),
    }
    return pd.DataFrame(raw, index=frame.index).replace([np.inf, -np.inf], np.nan)


def _pct_of_last(w: np.ndarray) -> float:
    """창의 마지막 값이 이전 값들 중 차지하는 백분위 (0~1). 현재까지의 데이터만 사용."""
    cur = w[-1]
    if np.isnan(cur):
        return np.nan
    prior = w[:-1][~np.isnan(w[:-1])]
    if prior.size == 0:
        return np.nan
    return float(((prior < cur).sum() + 0.5 * (prior == cur).sum()) / prior.size)


def rolling_percentile(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return s.rolling(window, min_periods=min_periods).apply(_pct_of_last, raw=True)


def _transform(p: pd.Series, transform: str, direction: int, band: float) -> pd.Series:
    if transform == "percentile":
        return direction * (2 * p - 1)
    # contrarian: 극단 구간에서만 percentile 신호의 반대 방향, 0→1 선형
    out = pd.Series(0.0, index=p.index)
    hi, lo = p > 1 - band, p < band
    out[hi] = -direction * (p[hi] - (1 - band)) / band
    out[lo] = direction * (band - p[lo]) / band
    out[p.isna()] = np.nan
    return out


def to_signals(raw: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    window, minp, band = cfg["percentile_window"], cfg["percentile_min_periods"], cfg["contrarian_band"]
    pct = {name: rolling_percentile(raw[name], window, minp) for name in cfg["indicators"]}
    out = {}
    for h in cfg["horizons"]:
        cols = {}
        for name, ind in cfg["indicators"].items():
            override = (ind.get("horizons") or {}).get(h, {})
            transform = override.get("transform", ind["transform"])
            direction = override.get("direction", ind["direction"])
            cols[name] = _transform(pct[name], transform, direction, band).clip(-1, 1)
        out[h] = pd.DataFrame(cols, index=raw.index)
    return out
