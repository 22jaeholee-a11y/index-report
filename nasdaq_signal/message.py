"""카카오 메시지 2건 포맷 (각 200자 이하)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .scoring import arrow

MAX_LEN = 200
WEEKDAYS = "월화수목금토일"
HORIZON_KO = {"short": "단기", "mid": "중기", "long": "장기"}
CATEGORY_KO = {"market": "시장", "sentiment": "심리", "geo": "지정학", "macro": "매크로"}
MISSING_KO = {
    "fear_greed": "공포탐욕", "dgs10": "10년물", "t10y2y": "장단기차", "bamlh0a0hym2": "HY스프레드",
    "nfci": "금융여건", "sahmrealtime": "삼법칙", "gpr": "GPR", "ndx": "NDX", "spy": "SPY", "rsp": "RSP",
    "vix": "VIX", "vix3m": "VIX3M", "oil": "유가", "gold": "금", "dxy": "달러",
}
MISSING_LINE_MAX = 60


@dataclass
class Report:
    as_of: date
    run_date: date
    market_closed: bool
    scores: dict[str, float]
    probs: dict[str, tuple[int, int, int] | None]
    categories: dict[str, float]
    key_values: dict[str, float]
    fear_greed: float | None
    nq_change: float | None
    missing: list[str]
    low_confidence: bool
    reauth_days: int | None


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _md(d: date) -> str:
    return f"{d.month}/{d.day}"


def _score(v: float) -> str:
    r = round(v)
    return "0" if r == 0 else f"{r:+d}"


def _num(v, fmt: str) -> str:
    return "-" if _isnan(v) else format(v, fmt)


def _message_one(r: Report, threshold: float) -> str:
    day = f"{_md(r.run_date)}({WEEKDAYS[r.run_date.weekday()]})"
    tail = f"휴장, {_md(r.as_of)} 기준" if r.market_closed else f"{_md(r.as_of)} 마감"
    lines = [f"[나스닥 방향성] {day} · {tail}"]
    for h, ko in HORIZON_KO.items():
        s = r.scores.get(h)
        if _isnan(s):
            lines.append(f"{ko} 계산 불가")
            continue
        line = f"{ko} {arrow(s, threshold)} {_score(s)}"
        p = r.probs.get(h)
        if p:
            line += f" 상{p[0]} 횡{p[1]} 하{p[2]}"
        lines.append(line)
    if r.low_confidence:
        lines.append("⚠ 신뢰도 낮음(데이터 누락)")
    return "\n".join(lines)


def _missing_line(missing: list[str]) -> str:
    names = [MISSING_KO.get(m, m) for m in missing]
    line = "⚠ 누락: " + ", ".join(names)
    k = len(names)
    while len(line) > MISSING_LINE_MAX and k > 1:
        k -= 1
        line = "⚠ 누락: " + ", ".join(names[:k]) + f" 외 {len(names) - k}개"
    return line[:MISSING_LINE_MAX]


def _message_two(r: Report) -> str:
    kv = r.key_values
    # (우선순위, 줄) — 숫자가 작을수록 중요. 200자 초과 시 큰 숫자부터 제거.
    lines: list[tuple[int, str]] = [
        (1, " ".join(f"{ko} {_num(r.categories.get(c), '+.1f')}" for c, ko in CATEGORY_KO.items())),
    ]
    if r.reauth_days is not None and r.reauth_days <= 14:
        lines.append((2, f"⚠ 재인증 필요: D-{max(r.reauth_days, 0)}"))
    if r.missing:
        lines.append((3, _missing_line(r.missing)))
    lines.append((4, f"VIX {_num(kv.get('vix'), '.1f')} · 10년물 {_num(kv.get('us10y'), '.2f')}"
                     f"{'' if _isnan(kv.get('us10y')) else '%'} · HY {_num(kv.get('hy_spread'), '.2f')}"
                     f"{'' if _isnan(kv.get('hy_spread')) else '%'}"))
    if r.nq_change is not None and r.run_date.weekday() == 0:
        lines.append((5, f"월요일 NQ선물 {r.nq_change * 100:+.1f}%"))
    if r.fear_greed is not None:
        lines.append((6, f"공포탐욕 {round(r.fear_greed)}(참고)"))
    while len("\n".join(l for _, l in lines)) > MAX_LEN and len(lines) > 1:
        lines.remove(max(lines, key=lambda x: x[0]))
    return "\n".join(l for _, l in lines)


def format_messages(report: Report, arrow_threshold: float = 20) -> list[str]:
    return [_message_one(report, arrow_threshold), _message_two(report)]
