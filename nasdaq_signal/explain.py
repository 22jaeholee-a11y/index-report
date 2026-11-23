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
    max_points: float = math.nan  # 이 지표가 낼 수 있는 최대 기여 (± 대칭)


def _pct(v: float) -> str:
    return f"{v:+.1%}"


# 지표 이름 → 값 표시 형식
INDICATOR_TEXT: dict[str, object] = {
    "trend_50": lambda v: f"50일선 대비 {_pct(v)}",
    "trend_200": lambda v: f"200일선 대비 {_pct(v)}",
    "ret_20": lambda v: f"20일 수익률 {_pct(v)}",
    "ret_60": lambda v: f"60일 수익률 {_pct(v)}",
    "drawdown_52w": lambda v: f"52주 고점 대비 {_pct(v)}",
    "breadth": lambda v: f"동일/시총가중 60일 {_pct(v)}",
    "vix": lambda v: f"VIX {v:.1f}",
    "vix_term": lambda v: f"VIX/VIX3M {v:.2f}",
    "gpr": lambda v: f"지정학위험지수 {v:.0f}",
    "oil_ret_20": lambda v: f"유가 20일 {_pct(v)}",
    "gold_ret_20": lambda v: f"금 20일 {_pct(v)}",
    "us10y_chg_20": lambda v: f"10년물 20일 {v * 100:+.0f}bp",
    "us2y_chg_20": lambda v: f"2년물 20일 {v * 100:+.0f}bp",
    "breakeven_chg_20": lambda v: f"기대인플레 20일 {v * 100:+.0f}bp",
    "curve": lambda v: f"장단기 금리차 {v:+.2f}%p",
    "credit": lambda v: f"하이일드/국채 60일 {_pct(v)}",
    "dxy_ret_60": lambda v: f"달러 60일 {_pct(v)}",
    "nfci": lambda v: f"금융여건지수 {v:+.2f}",
    "sahm": lambda v: f"삼 법칙 {v:.2f}",
}


def _setting(cfg: dict, name: str, h: str, key: str):
    ind = cfg["indicators"][name]
    return (ind.get("horizons") or {}).get(h, {}).get(key, ind[key])


def effective_sign(cfg: dict, name: str, h: str) -> int:
    d = _setting(cfg, name, h, "direction")
    return d if _setting(cfg, name, h, "transform") == "percentile" else -d


def max_points(sig_row: pd.Series, cfg: dict, h: str) -> dict[str, float]:
    """지표별 최대 기여 = 카테고리 비중(재정규화) ÷ 카테고리 내 유효 지표 수 × 100. 합 = 100."""
    by_cat: dict[str, list[str]] = {}
    for name in active_indicators(cfg, h):
        if name in sig_row and not math.isnan(sig_row[name]):
            by_cat.setdefault(cfg["indicators"][name]["category"], []).append(name)
    wsum = sum(cfg["category_weights"][c][h] for c in by_cat)
    if wsum == 0:
        return {}
    return {n: cfg["category_weights"][c][h] / wsum / len(names) * 100
            for c, names in by_cat.items() for n in names}


def contributions(sig_row: pd.Series, cfg: dict, h: str) -> dict[str, float]:
    """기간 점수를 지표별 기여 점수로 분해 (합 = 점수, ai_adjustment 제외)."""
    return {n: m * float(sig_row[n]) for n, m in max_points(sig_row, cfg, h).items()}


def last_percentiles(raw: pd.DataFrame, cfg: dict) -> pd.Series:
    window = cfg["percentile_window"]
    return pd.Series({n: _pct_of_last(raw[n].iloc[-window:].to_numpy(dtype=float)) for n in raw.columns})


def top_drivers(sig_row: pd.Series, raw_row: pd.Series, pct_row: pd.Series, cfg: dict, h: str,
                n: int = 4, min_points: float = 1.0) -> list[Driver]:
    caps = max_points(sig_row, cfg, h)
    contrib = {k: caps[k] * float(sig_row[k]) for k in caps}
    ranked = sorted((kv for kv in contrib.items() if abs(kv[1]) >= min_points), key=lambda kv: -abs(kv[1]))[:n]
    return [Driver(name, float(raw_row.get(name, np.nan)), float(pct_row.get(name, np.nan)), pts,
                   effective_sign(cfg, name, h), _setting(cfg, name, h, "transform") == "contrarian",
                   caps[name])
            for name, pts in ranked]


def describe(d: Driver) -> str:
    """'• 값(5년 위치) → 기여/최대점' 형식."""
    fmt = INDICATOR_TEXT.get(d.name, lambda v: f"{d.name} {v:.2f}")
    value = "-" if math.isnan(d.value) else fmt(d.value)
    if math.isnan(d.pct):
        pos = ""
    elif d.pct >= 0.5:
        pos = f"(5년 상위 {max(1, round((1 - d.pct) * 100))}%)"
    else:
        pos = f"(5년 하위 {max(1, round(d.pct * 100))}%)"
    cap = "" if math.isnan(d.max_points) else f"/{d.max_points:.0f}"
    return f"• {value}{pos} → {d.points:+.0f}{cap}점"


# 지표 이름 → (표시 이름, 무엇을 재는지)
INDICATOR_HELP: dict[str, tuple[str, str]] = {
    "trend_50": ("50일선 대비", "나스닥100이 최근 50거래일 평균보다 얼마나 위·아래에 있는지 (단기 추세·과열)"),
    "trend_200": ("200일선 대비", "나스닥100이 최근 200거래일 평균보다 얼마나 위·아래에 있는지 (장기 추세)"),
    "ret_20": ("20일 수익률", "최근 한 달간 나스닥100 등락률"),
    "ret_60": ("60일 수익률", "최근 석 달간 나스닥100 등락률"),
    "drawdown_52w": ("52주 고점 대비", "최근 1년 최고점에서 얼마나 떨어져 있는지 (0%면 신고가)"),
    "breadth": ("시장 폭", "동일가중 S&P500(RSP)이 시총가중(SPY)보다 60일간 얼마나 강했는지 — 상승이 넓게 퍼졌는지, 대형주에 쏠렸는지"),
    "vix": ("VIX", "S&P500 옵션 가격에 반영된 향후 30일 예상 변동성 — '공포 지수'"),
    "vix_term": ("VIX/VIX3M", "1개월 변동성 ÷ 3개월 변동성 — 1보다 크면 당장의 공포가 더 큰 투매 국면"),
    "gpr": ("지정학위험지수", "주요 신문의 전쟁·테러·군사 긴장 기사 빈도 (30일 평균)"),
    "oil_ret_20": ("유가 20일", "WTI 원유 선물의 한 달 등락률"),
    "gold_ret_20": ("금 20일", "금 선물의 한 달 등락률 (안전자산 선호)"),
    "us10y_chg_20": ("10년물 금리 변화", "미국 10년 국채 금리의 한 달 변화 — 오르면 미래 이익의 현재가치가 줄어 기술주에 부담"),
    "us2y_chg_20": ("2년물 금리 변화", "미국 2년 국채 금리의 한 달 변화 — 연준 기준금리 기대를 가장 잘 반영"),
    "breakeven_chg_20": ("기대인플레이션 변화", "시장이 예상하는 향후 10년 물가상승률의 한 달 변화"),
    "curve": ("장단기 금리차", "10년물 − 2년물 금리 — 마이너스(역전)면 전통적인 경기침체 신호"),
    "credit": ("신용 위험선호", "하이일드 회사채 ETF(HYG) ÷ 국채 ETF(IEF)의 60일 변화 — 위험한 회사채에 돈이 몰리는지"),
    "dxy_ret_60": ("달러 지수", "달러 가치의 석 달 등락률 — 강달러는 해외 매출 큰 빅테크에 부담"),
    "nfci": ("금융여건지수", "시카고 연준 금융여건지수 — 플러스면 평균보다 긴축적"),
    "sahm": ("삼 법칙", "실업률 3개월 평균이 지난 1년 최저보다 얼마나 올랐는지 — 0.5 이상이면 침체 신호"),
}


def driver_view(d: Driver) -> dict:
    """웹 표시용: 지표 설명 + 학습된 방향 + 오늘 값."""
    label, what = INDICATOR_HELP.get(d.name, (d.name, ""))
    fmt = INDICATOR_TEXT.get(d.name)
    full = fmt(d.value) if fmt and not math.isnan(d.value) else "-"
    value = full[len(label):].strip() if full.startswith(label) else full.split(" ")[-1]
    if math.isnan(d.pct):
        position = "-"
    elif d.pct >= 0.5:
        position = f"5년 상위 {max(1, round((1 - d.pct) * 100))}%"
    else:
        position = f"5년 하위 {max(1, round(d.pct * 100))}%"
    learned = f"과거 데이터상 값이 높을수록 이후 {'상승' if d.eff_sign > 0 else '하락'}"
    if d.contrarian:
        learned += " (5년 상·하위 10% 극단 구간에서만 작동)"
    return {"name": d.name, "label": label, "what": what, "learned": learned, "value": value,
            "position": position, "points": round(d.points, 1),
            "max": None if math.isnan(d.max_points) else round(d.max_points, 1)}
