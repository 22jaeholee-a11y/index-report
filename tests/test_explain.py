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
    sig = _sig({"vix": 1.0, "trend_50": -0.2, "gpr": -0.9}, cfg)
    raw = pd.Series({k: 1.0 for k in cfg["indicators"]})
    pct = pd.Series({k: 0.5 for k in cfg["indicators"]})
    d = top_drivers(sig["short"].iloc[-1], raw, pct, cfg, "short", n=2)
    assert [x.name for x in d] == ["vix", "gpr"]
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
