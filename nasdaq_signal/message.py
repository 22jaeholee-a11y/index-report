"""카카오 메시지 2건 포맷 (각 200자 이하)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from .explain import Driver, describe
from .scoring import arrow

MAX_LEN = 200
WEEKDAYS = "월화수목금토일"
HORIZON_KO = {"short": "단기", "mid": "중기", "long": "장기"}
CATEGORY_KO = {"market": "시장", "sentiment": "심리", "geo": "지정학", "macro": "매크로"}
MISSING_KO = {
    "fear_greed": "Fear & Greed Index","dgs10": "10년물", "dgs2": "2년물", "t10yie": "기대인플레", "t10y2y": "장단기차", "bamlh0a0hym2": "HY스프레드",
    "nfci": "금융여건", "sahmrealtime": "삼법칙", "gpr": "GPR", "ndx": "NDX", "spy": "SPY", "rsp": "RSP",
    "vix": "VIX", "vix3m": "VIX3M", "hyg": "HYG", "ief": "IEF", "oil": "유가", "gold": "금", "dxy": "달러",
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
    calibration_stale: bool = False
    drivers: dict[str, list[Driver]] = field(default_factory=dict)
    unpredicted: set[str] = field(default_factory=set)  # 예측하지 않고 역사적 평균만 보여주는 기간
    score_pct: dict[str, float] = field(default_factory=dict)  # 오늘 점수의 과거 분포 내 위치 (0~1)


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _md(d: date) -> str:
    return f"{d.month}/{d.day}"


def _score(v: float) -> str:
    r = round(v)
    return "0" if r == 0 else f"{r:+d}"


def _num(v, fmt: str) -> str:
    return "-" if _isnan(v) else format(v, fmt)


def _cat(v) -> str:
    """카테고리 점수: 반올림 결과가 0이면 '-0.0' 대신 '+0.0'."""
    return "-" if _isnan(v) else format(0.0 if abs(v) < 0.05 else v, "+.1f")


def _prob_arrow(p: tuple[int, int, int]) -> str:
    """검증된 확률이 있으면 가장 가능성 높은 결과로 화살표 (동률은 횡보)."""
    up, flat, down = p
    if up > max(flat, down):
        return "▲"
    if down > max(up, flat):
        return "▼"
    return "─"


def _horizon_arrow(r: "Report", h: str, threshold: float) -> str:
    p = r.probs.get(h)
    return _prob_arrow(p) if p else arrow(r.scores[h], threshold)


def _relative(score: float, threshold: float) -> str:
    """점수 = 5년 기준 '평소 대비' 우호도. 화살표(가장 가능성 높은 결과)와 별개라 말로 표시."""
    if score >= threshold:
        return "평소 대비 강함"
    if score <= -threshold:
        return "평소 대비 약함"
    return "평소 대비 보통"


def _position(r: "Report", h: str, threshold: float) -> str:
    """과거 점수 분포가 있으면 '과거 상위/하위 n%', 없으면 ±threshold 기준 말."""
    p = r.score_pct.get(h)
    s = r.scores.get(h)
    if p is None or _isnan(p):
        return _relative(s, threshold)
    if p >= 0.5:
        return f"과거 상위 {max(1, round((1 - p) * 100))}%"
    return f"과거 하위 {max(1, round(p * 100))}%"


def _reason_messages(r: "Report", h: str, threshold: float) -> list[str]:
    """모든 근거 지표를 표시. 200자를 넘으면 '[중기적 1/n]'처럼 나눠 보냄."""
    s = r.scores.get(h)
    summary = "" if _isnan(s) else f" {_score(s)} ({_position(r, h, threshold)})"
    items = [describe(d) for d in r.drivers.get(h, [])] or ["• 뚜렷한 근거 지표 없음 (신호 약함)"]
    head_room = len(f"[{HORIZON_KO[h]}적 9/9]") + len(summary) + 1  # 첫 메시지 제목 최대 길이
    chunks: list[list[str]] = [[]]
    for it in items:
        budget = MAX_LEN - (head_room if len(chunks) == 1 else len(f"[{HORIZON_KO[h]}적 9/9]") + 1)
        if chunks[-1] and len("\n".join(chunks[-1] + [it])) > budget:
            chunks.append([])
        chunks[-1].append(it)
    n = len(chunks)
    out = []
    for i, chunk in enumerate(chunks, 1):
        tag = f"[{HORIZON_KO[h]}적]" if n == 1 else f"[{HORIZON_KO[h]}적 {i}/{n}]"
        head = tag + (summary if i == 1 else "")
        out.append("\n".join([head] + chunk))
    return out


def _message_one(r: Report, threshold: float) -> str:
    day = f"{_md(r.run_date)}({WEEKDAYS[r.run_date.weekday()]})"
    tail = f"휴장, {_md(r.as_of)} 기준" if r.market_closed else f"{_md(r.as_of)} 마감"
    lines = [f"[나스닥 방향성] {day} · {tail}"]
    for h, ko in HORIZON_KO.items():
        s = r.scores.get(h)
        p = r.probs.get(h)
        if h in r.unpredicted:
            if p:
                lines.append(f"{ko} {_prob_arrow(p)} 상승 {p[0]} · 횡보 {p[1]} · 하락 {p[2]} (역사적 평균, 예측 신호 없음)")
            else:
                lines.append(f"{ko} 예측 신호 없음")
            continue
        if _isnan(s):
            lines.append(f"{ko} 계산 불가")
            continue
        p = r.probs.get(h)
        if p:  # 화살표는 확률을 따르므로 부호 있는 점수 대신 '평소 대비'를 말로 표시
            line = (f"{ko} {_prob_arrow(p)} 상승 {p[0]} · 횡보 {p[1]} · 하락 {p[2]}"
                    f" ({_position(r, h, threshold)})")
        else:
            line = f"{ko} {arrow(s, threshold)} {_score(s)}"
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
    # 지표는 한 줄에 하나씩
    lines: list[tuple[float, str]] = [(1, f"{ko} {_cat(r.categories.get(c))}") for c, ko in CATEGORY_KO.items()]
    if r.reauth_days is not None and r.reauth_days <= 14:
        lines.append((2, f"⚠ 재인증 필요: D-{max(r.reauth_days, 0)}"))
    if r.missing:
        lines.append((3, _missing_line(r.missing)))
    if r.calibration_stale:
        lines.append((3.5, "⚠ 확률표 갱신 필요(백테스트 재실행)"))
    pct = lambda v: "-" if _isnan(v) else f"{v:.2f}%"  # noqa: E731
    lines += [(4, f"VIX {_num(kv.get('vix'), '.1f')}"),
              (4, f"10년물 {pct(kv.get('us10y'))}"),
              (4, f"HY {pct(kv.get('hy_spread'))}")]
    if r.nq_change is not None and r.run_date.weekday() == 0:
        lines.append((5, f"월요일 NQ선물 {r.nq_change * 100:+.1f}%"))
    if r.fear_greed is not None:
        lines.append((6, f"Fear & Greed Index {round(r.fear_greed)}"))
    while len("\n".join(l for _, l in lines)) > MAX_LEN and len(lines) > 1:
        lines.remove(max(lines, key=lambda x: x[0]))
    return "\n".join(l for _, l in lines)


def format_messages(report: Report, arrow_threshold: float = 20) -> list[str]:
    msgs = [_message_one(report, arrow_threshold)]
    if report.drivers:
        for h in HORIZON_KO:
            if h not in report.unpredicted:
                msgs += _reason_messages(report, h, arrow_threshold)
    return msgs + [_message_two(report)]
