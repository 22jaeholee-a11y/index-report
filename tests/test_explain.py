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
