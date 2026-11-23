import numpy as np
import pandas as pd
import pytest

from nasdaq_signal.zone import zone_info, zone_states


def _s(vals):
    return pd.Series(vals, index=pd.bdate_range("2026-01-01", periods=len(vals)), dtype=float)


def test_zone_states_enter_at_entry_and_exit_below_exit():
    s = _s([0, 25, 10, -10, -25, 0, 30])
    assert zone_states(s, 20, -20).tolist() == [False, True, True, True, False, False, True]


def test_zone_states_nan_keeps_previous_state():
    s = _s([25, np.nan, 10])
    assert zone_states(s, 20, -20).tolist() == [True, True, True]


def test_zone_info_holding_measures_return_from_next_day_close():
    s = _s([0, 25, 30, 28, 26])
    px = _s([100, 100, 110, 99, 88])
    z = zone_info(s, px, 20, -20)
    assert z.status == "hold"
    assert z.since == s.index[1].date()          # 신호일
    assert z.days == 3                            # 신호일 이후 거래일 수
    assert z.ret == pytest.approx(88 / 110 - 1)   # 신호 다음 날 종가에 매수한 것으로 계산


def test_zone_info_entry_exit_and_wait():
    assert zone_info(_s([0, 10, 25]), _s([1, 1, 1]), 20, -20).status == "entry"
    z = zone_info(_s([25, 30, -25]), _s([100, 110, 121]), 20, -20)
    assert z.status == "exit" and z.ret == pytest.approx(121 / 110 - 1)
    assert zone_info(_s([0, 10, 5]), _s([1, 1, 1]), 20, -20).status == "wait"
