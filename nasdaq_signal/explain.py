"""점수 근거: 지표별 기여 점수, 5년 중 위치, 학습된 경향을 사람이 읽을 문장으로."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import active_indicators
from .indicators import _pct_of_last


@dataclass
class Driver:
    name: str
    value: float        # 원지표 값
    pct: float          # 5년 창 내 백분위 (0~1)
    points: float       # 기간 점수에 대한 기여 (점)
    eff_sign: int       # +1: 값이 높을수록 이후 상승 경향, -1: 높을수록 하락 경향
    contrarian: bool    # 극단 구간에서만 작동하는 역발상 신호


def _pct(v: float) -> str:
    return f"{v:+.1%}"


# 이름: (값 표시, 5년 중 높은 쪽일 때 상태, 낮은 쪽일 때 상태)
# 문장 = "값(5년 중 위치) 현재 상태 → 과거엔 이후 상승/하락 기여점" — 방향은 학습된 기여 부호를 그대로 따름
INDICATOR_TEXT: dict[str, tuple] = {
    "trend_50": (lambda v: f"50일선 대비 {_pct(v)}", "50일선 위 과열", "50일선 아래 이탈"),
    "trend_200": (lambda v: f"200일선 대비 {_pct(v)}", "200일선 위 과열", "200일선 아래 이탈"),
    "ret_20": (lambda v: f"20일 수익률 {_pct(v)}", "단기 급등", "단기 급락"),
    "ret_60": (lambda v: f"60일 수익률 {_pct(v)}", "3개월 급등", "3개월 급락"),
    "drawdown_52w": (lambda v: f"52주 고점 대비 {_pct(v)}", "고점 근처", "고점 대비 큰 낙폭"),
    "breadth": (lambda v: f"동일/시총가중 60일 {_pct(v)}", "상승 종목 확산", "대형주 쏠림"),
    "vix": (lambda v: f"VIX {v:.1f}", "공포 높음", "공포 낮음"),
    "vix_term": (lambda v: f"VIX/VIX3M {v:.2f}", "단기 공포 급등", "단기 공포 낮음"),
    "gpr": (lambda v: f"지정학위험지수 {v:.0f}", "지정학 긴장 높음", "지정학 긴장 낮음"),
    "oil_ret_20": (lambda v: f"유가 20일 {_pct(v)}", "유가 급등", "유가 급락"),
    "gold_ret_20": (lambda v: f"금 20일 {_pct(v)}", "금 급등", "금 급락"),
    "us10y_chg_20": (lambda v: f"10년물 20일 {v * 100:+.0f}bp", "금리 급등", "금리 급락"),
    "curve": (lambda v: f"장단기 금리차 {v:+.2f}%p", "금리차 확대", "금리차 축소"),
    "credit": (lambda v: f"하이일드/국채 60일 {_pct(v)}", "신용 위험선호", "신용 위험회피"),
    "dxy_ret_60": (lambda v: f"달러 60일 {_pct(v)}", "달러 강세", "달러 약세"),
    "nfci": (lambda v: f"금융여건지수 {v:+.2f}", "금융여건 긴축", "금융여건 완화"),
    "sahm": (lambda v: f"삼 법칙 {v:.2f}", "침체 신호", "침체 신호 약함"),
}


def _setting(cfg: dict, name: str, h: str, key: str):
    ind = cfg["indicators"][name]
    return (ind.get("horizons") or {}).get(h, {}).get(key, ind[key])


def effective_sign(cfg: dict, name: str, h: str) -> int:
    d = _setting(cfg, name, h, "direction")
    return d if _setting(cfg, name, h, "transform") == "percentile" else -d


def contributions(sig_row: pd.Series, cfg: dict, h: str) -> dict[str, float]:
    """기간 점수를 지표별 기여 점수로 분해 (합 = 점수, ai_adjustment 제외)."""
    by_cat: dict[str, list[str]] = {}
    for name in active_indicators(cfg, h):
        if name in sig_row and not math.isnan(sig_row[name]):
            by_cat.setdefault(cfg["indicators"][name]["category"], []).append(name)
    wsum = sum(cfg["category_weights"][c][h] for c in by_cat)
    if wsum == 0:
        return {}
    return {n: cfg["category_weights"][c][h] / wsum * float(sig_row[n]) / len(names) * 100
            for c, names in by_cat.items() for n in names}


def last_percentiles(raw: pd.DataFrame, cfg: dict) -> pd.Series:
    window = cfg["percentile_window"]
    return pd.Series({n: _pct_of_last(raw[n].iloc[-window:].to_numpy(dtype=float)) for n in raw.columns})


def top_drivers(sig_row: pd.Series, raw_row: pd.Series, pct_row: pd.Series, cfg: dict, h: str,
                n: int = 4, min_points: float = 1.0) -> list[Driver]:
    contrib = contributions(sig_row, cfg, h)
    ranked = sorted((kv for kv in contrib.items() if abs(kv[1]) >= min_points), key=lambda kv: -abs(kv[1]))[:n]
    return [Driver(name, float(raw_row.get(name, np.nan)), float(pct_row.get(name, np.nan)), pts,
                   effective_sign(cfg, name, h), _setting(cfg, name, h, "transform") == "contrarian")
            for name, pts in ranked]


def describe(d: Driver) -> str:
    fmt, high_state, low_state = INDICATOR_TEXT.get(d.name, (lambda v: f"{d.name} {v:.2f}", "높음", "낮음"))
    value = "-" if math.isnan(d.value) else fmt(d.value)
    high = math.isnan(d.pct) or d.pct >= 0.5
    if math.isnan(d.pct):
        pos = ""
    elif high:
        pos = f"(5년 상위 {max(1, round((1 - d.pct) * 100))}%)"
    else:
        pos = f"(5년 하위 {max(1, round(d.pct * 100))}%)"
    if not math.isnan(d.pct) and 0.2 < d.pct < 0.8:  # 상·하위 20% 밖이 아니면 '급등/과열' 같은 표현은 과장
        state = "평균보다 다소 높음" if high else "평균보다 다소 낮음"
    else:
        state = high_state if high else low_state
    if d.contrarian:
        state = "극단적 " + state
    outcome = "상승" if d.points > 0 else "하락"
    return f"• {value}{pos} {state} → 과거엔 이후 {outcome} {d.points:+.0f}점"
