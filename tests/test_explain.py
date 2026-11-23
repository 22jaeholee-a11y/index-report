import numpy as np
import pandas as pd
import pytest

from nasdaq_signal.config import load_config
from nasdaq_signal.explain import Driver, contributions, effective_sign, top_drivers
from nasdaq_signal.scoring import score_frame


def _sig(values: dict, cfg):
    idx = pd.bdate_range("2024-01-01", periods=2)
    return {h: pd.DataFrame({k: values.get(k, 0.0) for k in cfg["indicators"]}, index=idx) for h in cfg["horizons"]}


def test_contributions_sum_to_score():
    cfg = load_config(apply_tuning=False)
    rng = np.random.default_rng(3)
    sig = _sig({k: rng.uniform(-1, 1) for k in cfg["indicators"]}, cfg)
    sig["long"]["nfci"] = np.nan
    sf = score_frame(sig, cfg)
    for h in cfg["horizons"]:
        c = contributions(sig[h].iloc[-1], cfg, h)
        assert sum(c.values()) == pytest.approx(sf[f"{h}_score"].iloc[-1])
        assert "nfci" not in c or h != "long"


def test_top_drivers_sorted_by_absolute_points():
    cfg = load_config(apply_tuning=False)
    sig = _sig({"vix": 1.0, "trend_50": -0.2, "dxy_ret_60": -0.9}, cfg)
    raw = pd.Series({k: 1.0 for k in cfg["indicators"]})
    pct = pd.Series({k: 0.5 for k in cfg["indicators"]})
    d = top_drivers(sig["short"].iloc[-1], raw, pct, cfg, "short", n=2)
    assert [x.name for x in d] == ["vix", "dxy_ret_60"]
    assert d[0].points > 0 > d[1].points
    assert isinstance(d[0], Driver)


def test_effective_sign_handles_contrarian_and_flip():
    cfg = load_config(apply_tuning=False)
    assert effective_sign(cfg, "vix", "short") == -1          # percentile, direction -1 → 높을수록 하락
    assert effective_sign(cfg, "vix", "mid") == 1             # contrarian, direction -1 → 극단 고점은 상승
    cfg["indicators"]["trend_50"]["horizons"] = {"mid": {"direction": -1}}
    assert effective_sign(cfg, "trend_50", "mid") == -1


def test_top_drivers_carry_max_points():
    cfg = load_config(apply_tuning=False)
    for name in cfg["indicators"]:  # 단기: VIX(심리)와 금리 2개(매크로)만 활성
        if name not in ("vix", "us10y_chg_20", "us2y_chg_20"):
            cfg["indicators"][name].setdefault("horizons", {})["short"] = {"exclude": True}
    sig = _sig({"vix": 0.99, "us10y_chg_20": -0.11, "us2y_chg_20": -0.18}, cfg)
    raw = pd.Series({k: 1.0 for k in cfg["indicators"]})
    pct = pd.Series({k: 0.5 for k in cfg["indicators"]})
    d = {x.name: x for x in top_drivers(sig["short"].iloc[-1], raw, pct, cfg, "short", n=3)}
    assert d["vix"].max_points == pytest.approx(0.30 / 0.45 * 100)
    assert d["us2y_chg_20"].max_points == pytest.approx(0.15 / 0.45 / 2 * 100)


def test_describe_shows_points_over_max():
    from nasdaq_signal.explain import describe
    line = describe(Driver("vix", 35.9, 0.99, 66.2, 1, False, 66.7))
    assert line == "• VIX 35.9(5년 상위 1%) → +66/67점"
    assert describe(Driver("us2y_chg_20", 0.11, 0.59, -3.0, -1, False, 16.7)) == "• 2년물 20일 +11bp(5년 상위 41%) → -3/17점"


def test_every_indicator_has_help_text():
    from nasdaq_signal.explain import INDICATOR_HELP
    cfg = load_config(apply_tuning=False)
    assert set(cfg["indicators"]) <= set(INDICATOR_HELP)


def test_driver_view_explains_learned_direction():
    from nasdaq_signal.explain import driver_view
    v = driver_view(Driver("vix", 35.9, 0.99, 22.0, 1, True, 24.0))
    assert v["label"] == "VIX" and v["value"] == "35.9" and v["position"] == "5년 상위 1%"
    assert v["points"] == 22.0 and v["max"] == 24.0
    assert "변동성" in v["what"]
    assert "높을수록 이후 상승" in v["learned"] and "극단" in v["learned"]
    v2 = driver_view(Driver("us2y_chg_20", 0.11, 0.59, -2.0, -1, False, 14.0))
    assert "높을수록 이후 하락" in v2["learned"] and "극단" not in v2["learned"]


def test_format_delta_uses_indicator_units():
    from nasdaq_signal.explain import format_delta
    assert format_delta("vix", 35.9, 34.7) == ("+1.2", "up")
    assert format_delta("us2y_chg_20", 0.11, 0.14) == ("-3bp", "down")
    assert format_delta("trend_200", -0.108, -0.112) == ("+0.4%p", "up")
    assert format_delta("curve", 0.28, 0.26) == ("+0.02%p", "up")
    assert format_delta("vix", 35.9, 35.9) == ("0", "flat")
    assert format_delta("vix", 35.9, float("nan")) == (None, None)


def test_driver_view_includes_day_changes():
    from nasdaq_signal.explain import driver_view
    v = driver_view(Driver("vix", 35.9, 0.99, 66.2, 1, False, 67.0, prev_value=28.4, prev_points=60.1))
    assert v["value_change"] == "+7.5" and v["value_dir"] == "up"
    assert v["points_change"] == "+6" and v["points_dir"] == "up"
    v2 = driver_view(Driver("vix", 35.9, 0.99, 66.2, 1, False, 67.0))
    assert v2["value_change"] is None and v2["points_change"] is None


def test_top_drivers_attach_previous_day():
    cfg = load_config(apply_tuning=False)
    sig = _sig({"vix": 1.0}, cfg)
    sig_prev = _sig({"vix": 0.5}, cfg)
    raw = pd.Series({k: 2.0 for k in cfg["indicators"]}); raw_prev = pd.Series({k: 1.0 for k in cfg["indicators"]})
    pct = pd.Series({k: 0.5 for k in cfg["indicators"]})
    d = top_drivers(sig["short"].iloc[-1], raw, pct, cfg, "short", n=1,
                    prev_sig_row=sig_prev["short"].iloc[-1], prev_raw_row=raw_prev)[0]
    assert d.name == "vix" and d.prev_value == 1.0
    assert d.prev_points == pytest.approx(d.points / 2)
