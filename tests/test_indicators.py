import numpy as np
import pandas as pd
import pytest

from nasdaq_signal.config import load_config
from nasdaq_signal.indicators import compute_raw, to_signals

SMALL = {"percentile_window": 100, "percentile_min_periods": 20}


def _cfg():
    cfg = load_config()
    cfg.update(SMALL)
    return cfg


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = ["ndx", "spy", "rsp", "vix", "vix3m", "oil", "gold", "dxy", "gpr",
            "dgs10", "t10y2y", "bamlh0a0hym2", "nfci", "sahmrealtime"]
    return pd.DataFrame({c: 100 + rng.normal(0, 1, n).cumsum() for c in cols}, index=idx)


def test_trend_200_definition():
    f = _frame()
    f["ndx"] = np.linspace(100, 200, len(f))
    raw = compute_raw(f)
    assert raw["trend_200"].iloc[-1] > 0
    assert raw["trend_200"].iloc[-1] == pytest.approx(f["ndx"].iloc[-1] / f["ndx"].rolling(200).mean().iloc[-1] - 1)


def test_missing_source_column_gives_nan_indicator():
    raw = compute_raw(_frame().drop(columns=["vix3m"]))
    assert raw["vix_term"].isna().all()
    assert set(raw.columns) == set(load_config()["indicators"])


def test_percentile_signal_bounds():
    sig = to_signals(compute_raw(_frame()), _cfg())
    for h, df in sig.items():
        v = df.to_numpy().ravel()
        v = v[~np.isnan(v)]
        assert v.size and v.min() >= -1 and v.max() <= 1


def test_percentile_signal_direction():
    f = _frame()
    f["bamlh0a0hym2"] = np.linspace(1, 10, len(f))   # hy_spread direction -1
    f["t10y2y"] = np.linspace(1, 10, len(f))         # curve direction +1
    sig = to_signals(compute_raw(f), _cfg())
    assert sig["short"]["curve"].iloc[-1] == pytest.approx(1.0, abs=0.01)
    assert sig["short"]["hy_spread"].iloc[-1] == pytest.approx(-1.0, abs=0.01)


def test_contrarian_vix_extreme_high_is_bullish():
    f = _frame()
    f["vix"] = np.linspace(10, 40, len(f))
    sig = to_signals(compute_raw(f), _cfg())
    assert sig["mid"]["vix"].iloc[-1] > 0.9
    assert sig["long"]["vix"].iloc[-1] > 0.9
    assert sig["short"]["vix"].iloc[-1] < -0.9


def test_contrarian_vix_extreme_low_is_bearish():
    f = _frame()
    f["vix"] = np.linspace(40, 10, len(f))
    sig = to_signals(compute_raw(f), _cfg())
    assert sig["mid"]["vix"].iloc[-1] < -0.9


def test_contrarian_neutral_inside_band():
    f = _frame()
    v = np.tile([10.0, 30.0], len(f) // 2)
    v[-1] = 20.0  # 중간 백분위
    f["vix"] = v
    sig = to_signals(compute_raw(f), _cfg())
    assert sig["mid"]["vix"].iloc[-1] == 0


def test_no_lookahead():
    f = _frame(400)
    cfg = _cfg()
    full = to_signals(compute_raw(f), cfg)["mid"]
    part = to_signals(compute_raw(f.iloc[:300]), cfg)["mid"]
    pd.testing.assert_frame_equal(full.iloc[:300], part)
