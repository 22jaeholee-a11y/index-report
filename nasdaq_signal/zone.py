"""매수 구간 ('우리만의 골든 크로스'): 중기 점수가 entry 이상이면 진입, exit 미만이면 청산.

신호는 그날 종가 기준 점수로 판단하고, 체결은 다음 날 종가로 가정한다 (백테스트와 동일).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class ZoneInfo:
    status: str              # entry(오늘 진입 신호) | hold(보유 중) | exit(오늘 청산 신호) | wait(대기)
    since: date | None       # 진입 신호일
    days: int                # 진입 신호일 이후 경과 거래일
    ret: float | None        # 체결가 대비 오늘 종가 수익률
    score: float             # 오늘 기준 점수


def zone_states(scores: pd.Series, entry: float, exit_: float) -> pd.Series:
    """각 날짜 종가 판단 후의 보유 상태. 점수가 없는 날은 직전 상태 유지."""
    holding, out = False, []
    for v in scores:
        if not math.isnan(v):
            if not holding and v >= entry:
                holding = True
            elif holding and v < exit_:
                holding = False
        out.append(holding)
    return pd.Series(out, index=scores.index)


def zone_info(scores: pd.Series, prices: pd.Series, entry: float, exit_: float) -> ZoneInfo:
    st = zone_states(scores, entry, exit_)
    today, prev = bool(st.iloc[-1]), bool(st.iloc[-2]) if len(st) > 1 else False
    score = float(scores.iloc[-1])
    if today and not prev:
        return ZoneInfo("entry", scores.index[-1].date(), 0, None, score)
    if not today and not prev:
        return ZoneInfo("wait", None, 0, None, score)
    # 보유 중이거나 오늘 청산: 가장 최근 진입 신호일 찾기
    held = st.iloc[:-1] if not today else st
    starts = held.index[(held & ~held.shift(1, fill_value=False))]
    sig = starts[-1]
    pos = scores.index.get_loc(sig)
    fill = prices.index[min(pos + 1, len(prices) - 1)]
    ret = float(prices.iloc[-1] / prices.loc[fill] - 1)
    days = len(scores.loc[sig:]) - 1   # 신호일 이후 경과 거래일
    return ZoneInfo("hold" if today else "exit", sig.date(), days, ret, score)
