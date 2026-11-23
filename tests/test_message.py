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
        "단기 ▲ 상승 55 · 횡보 30 · 하락 15 (평소 대비 강함)\n"
        "중기 ─ 상승 38 · 횡보 40 · 하락 22 (평소 대비 보통)\n"
        "장기 ▲ 상승 62 · 횡보 23 · 하락 15 (평소 대비 강함)"
    )


def test_message_one_without_probabilities():
    m1, _ = format_messages(_report(probs={"short": None, "mid": None, "long": None}))
    assert m1.splitlines()[1] == "단기 ▲ +32"


def test_negative_score_format():
    m1, _ = format_messages(_report(scores={"short": -45.6, "mid": 0.0, "long": 20.0},
                                    probs={"short": None, "mid": None, "long": None}))
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


def test_stale_calibration_warning_line():
    _, m2 = format_messages(_report(calibration_stale=True))
    assert "⚠ 확률표 갱신 필요(백테스트 재실행)" in m2


def test_arrow_follows_most_likely_outcome_when_probabilities_exist():
    m1, _ = format_messages(_report(scores={"short": -40.0, "mid": 30.0, "long": 5.0},
                                    probs={"short": (55, 30, 15), "mid": (20, 30, 50), "long": (40, 40, 20)}))
    lines = m1.splitlines()
    assert lines[1] == "단기 ▲ 상승 55 · 횡보 30 · 하락 15 (평소 대비 약함)"
    assert lines[2].startswith("중기 ▼ 상승 20")
    assert lines[3].startswith("장기 ─ ")   # 동률이면 횡보
    assert "-40" not in m1 and "+30" not in m1   # 화살표와 반대 부호로 보이는 숫자 없음


def test_arrow_uses_score_without_probabilities():
    m1, _ = format_messages(_report(scores={"short": -40.0, "mid": 30.0, "long": 5.0},
                                    probs={"short": None, "mid": None, "long": None}))
    assert m1.splitlines()[1] == "단기 ▼ -40"


def test_probability_words_are_spelled_out():
    m1, *_ = format_messages(_report())
    assert m1.splitlines()[1] == "단기 ▲ 상승 55 · 횡보 30 · 하락 15 (평소 대비 강함)"
    assert "상55" not in m1 and "횡30" not in m1 and "하15" not in m1


def _drivers():
    from nasdaq_signal.explain import Driver
    return {
        "short": [Driver("vix", 35.9, 0.97, 12.4, 1, False), Driver("drawdown_52w", -0.083, 0.10, 9.1, -1, False),
                  Driver("us10y_chg_20", 0.31, 0.92, -6.2, -1, False)],
        "mid": [Driver("trend_50", 0.052, 0.90, -8.0, -1, False)],
        "long": [],
    }


def test_reason_messages_layout_and_length():
    msgs = format_messages(_report(drivers=_drivers()))
    assert len(msgs) == 5
    short = msgs[1].splitlines()
    assert short[0] == "[단기 근거] 우호도 +32 (평소 대비 강함)"
    assert short[1] == "• VIX 35.9(5년 상위 3%) 공포 높음 → 과거엔 이후 상승 +12점"
    assert short[2] == "• 52주 고점 대비 -8.3%(5년 하위 10%) 고점 대비 큰 낙폭 → 과거엔 이후 상승 +9점"
    assert short[3] == "• 10년물 20일 +31bp(5년 상위 8%) 금리 급등 → 과거엔 이후 하락 -6점"
    assert all(len(m) <= 200 for m in msgs)
    assert msgs[3].splitlines()[1] == "• 뚜렷한 근거 지표 없음 (신호 약함)"


def test_reason_message_drops_drivers_to_fit():
    from nasdaq_signal.explain import Driver
    many = {"short": [Driver("vix_term", 1.234, 0.99, 10 - i, 1, True) for i in range(8)], "mid": [], "long": []}
    msgs = format_messages(_report(drivers=many))
    assert len(msgs[1]) <= 200 and msgs[1].count("•") >= 2
    assert "극단적 단기 공포 급등" in msgs[1]


def test_reason_line_describes_current_side():
    from nasdaq_signal.explain import Driver, describe
    low = describe(Driver("trend_50", -0.144, 0.01, 10.0, -1, False))
    high = describe(Driver("trend_50", 0.08, 0.95, -7.0, -1, False))
    assert "50일선 아래 이탈 → 과거엔 이후 상승" in low
    assert "50일선 위 과열 → 과거엔 이후 하락" in high
