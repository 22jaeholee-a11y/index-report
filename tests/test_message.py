from datetime import date

from nasdaq_signal.message import Report, format_messages

CATS = {"market": 0.4, "sentiment": 0.1, "geo": -0.2, "macro": 0.3}


def _report(**kw):
    base = dict(
        as_of=date(2026, 11, 23), run_date=date(2026, 11, 24), market_closed=False,
        scores={"short": 32.4, "mid": 5.0, "long": 41.0},
        probs={"short": (55, 30, 15), "mid": (38, 40, 22), "long": (62, 23, 15)},
        categories=CATS, key_values={"vix": 14.2, "us10y": 4.21, "hy_spread": 3.1},
        fear_greed=72.0, nq_change=None, missing=[], low_confidence=False, reauth_days=None,
    )
    base.update(kw)
    return Report(**base)


def test_message_one_layout():
    m1, _ = format_messages(_report())
    assert m1 == (
        "[나스닥 방향성] 11/24(화) · 11/23 마감\n"
        "단기 ▲ +32 상55 횡30 하15\n"
        "중기 ─ +5 상38 횡40 하22\n"
        "장기 ▲ +41 상62 횡23 하15"
    )


def test_message_one_without_probabilities():
    m1, _ = format_messages(_report(probs={"short": None, "mid": None, "long": None}))
    assert m1.splitlines()[1] == "단기 ▲ +32"


def test_negative_score_format():
    m1, _ = format_messages(_report(scores={"short": -45.6, "mid": 0.0, "long": 20.0}))
    assert m1.splitlines()[1].startswith("단기 ▼ -46")
    assert m1.splitlines()[2].startswith("중기 ─ 0")


def test_closed_market_header():
    m1, _ = format_messages(_report(as_of=date(2026, 11, 20), run_date=date(2026, 11, 22), market_closed=True))
    assert m1.splitlines()[0] == "[나스닥 방향성] 11/22(일) · 휴장, 11/20 기준"


def test_low_confidence_line():
    m1, _ = format_messages(_report(low_confidence=True))
    assert m1.splitlines()[-1] == "⚠ 신뢰도 낮음(데이터 누락)"


def test_message_two_layout():
    _, m2 = format_messages(_report(missing=["fear_greed"]))
    assert m2 == (
        "시장 +0.4 심리 +0.1 지정학 -0.2 매크로 +0.3\n"
        "⚠ 누락: 공포탐욕\n"
        "VIX 14.2 · 10년물 4.21% · HY 3.10%\n"
        "공포탐욕 72(참고)"
    )


def test_all_warnings_fit_200_chars():
    missing = ["dgs10", "t10y2y", "bamlh0a0hym2", "nfci", "sahmrealtime", "gpr", "rsp", "vix3m", "oil", "fear_greed"]
    r = _report(missing=missing, reauth_days=3, nq_change=0.0123, run_date=date(2026, 11, 23),
                low_confidence=True, market_closed=True, as_of=date(2026, 11, 20))
    m1, m2 = format_messages(r)
    assert len(m1) <= 200 and len(m2) <= 200
    assert m2.splitlines()[0].startswith("시장 ")
    assert "⚠ 재인증 필요: D-3" in m2


def test_missing_list_abbreviated_when_long():
    missing = [f"x{i}" * 6 for i in range(30)]
    _, m2 = format_messages(_report(missing=missing))
    assert len(m2) <= 200
    assert "외 " in m2


def test_nq_line_only_on_monday():
    _, mon = format_messages(_report(run_date=date(2026, 11, 23), nq_change=0.008))
    _, tue = format_messages(_report(run_date=date(2026, 11, 24), nq_change=0.008))
    assert "월요일 NQ선물 +0.8%" in mon
    assert "NQ선물" not in tue


def test_reauth_warning_only_within_14_days():
    _, far = format_messages(_report(reauth_days=30))
    _, near = format_messages(_report(reauth_days=14))
    assert "재인증" not in far and "D-14" in near


def test_nan_category_and_key_values_shown_as_dash():
    nan = float("nan")
    _, m2 = format_messages(_report(categories={**CATS, "macro": nan},
                                    key_values={"vix": 14.2, "us10y": nan, "hy_spread": nan}))
    assert "매크로 -" in m2.splitlines()[0]
    assert "10년물 -" in m2


def test_tiny_negative_category_shows_zero_without_sign_noise():
    _, m2 = format_messages(_report(categories={**CATS, "sentiment": -0.04}))
    assert "심리 +0.0" in m2.splitlines()[0]
