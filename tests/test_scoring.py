import numpy as np
import pandas as pd
import pytest

from nasdaq_signal.config import active_indicators, load_config
from nasdaq_signal.scoring import Calibration, arrow, score_frame


def _signals(value=1.0, n=3, nan_category=None):
    cfg = load_config(apply_tuning=False)
    idx = pd.bdate_range("2024-01-01", periods=n)
    out = {}
    for h in cfg["horizons"]:
        df = pd.DataFrame({k: value for k in cfg["indicators"]}, index=idx)
        if nan_category:
            for k, ind in cfg["indicators"].items():
                if ind["category"] == nan_category:
                    df[k] = np.nan
        out[h] = df
    return cfg, out


def test_all_positive_signals_score_100():
    cfg, sig = _signals(1.0)
    sf = score_frame(sig, cfg)
    for h in ("short", "mid", "long"):
        assert sf[f"{h}_score"].iloc[-1] == pytest.approx(100)
        assert sf[f"{h}_missing"].iloc[-1] == 0
        assert sf[f"{h}_market"].iloc[-1] == pytest.approx(1.0)


def test_missing_category_renormalizes():
    cfg, sig = _signals(0.5, nan_category="macro")
    sf = score_frame(sig, cfg)
    assert sf["long_score"].iloc[-1] == pytest.approx(50)
    assert sf["long_missing"].iloc[-1] == pytest.approx(0.60)
    assert np.isnan(sf["long_macro"].iloc[-1])


def test_partial_missing_inside_category_counts_fraction():
    cfg, sig = _signals(0.5)
    for h in sig:
        sig[h]["dxy_ret_60"] = np.nan  # 활성 macro 지표 중 1개
    n_macro = len([k for k in active_indicators(cfg, "long") if cfg["indicators"][k]["category"] == "macro"])
    sf = score_frame(sig, cfg)
    assert sf["long_missing"].iloc[-1] == pytest.approx(0.60 / n_macro)
    assert sf["long_score"].iloc[-1] == pytest.approx(50)


def test_ai_adjustment_capped():
    cfg, sig = _signals(0.2)
    base = score_frame(sig, cfg)["short_score"].iloc[-1]
    adj = score_frame(sig, cfg, ai_adjustment={"short": 50})["short_score"].iloc[-1]
    assert adj == pytest.approx(base + 10)


def test_score_clipped_to_100():
    cfg, sig = _signals(1.0)
    assert score_frame(sig, cfg, ai_adjustment={"short": 10})["short_score"].iloc[-1] == 100


def test_arrow_thresholds():
    assert arrow(20) == "▲"
    assert arrow(19.9) == "─"
    assert arrow(-20) == "▼"


def _synthetic(n=300, seed=1):
    rng = np.random.default_rng(seed)
    scores = pd.Series(rng.uniform(-100, 100, n))
    labels = pd.Series(np.where(scores > 0, "up", "down"))
    return scores, labels


def test_calibration_roundtrip_and_lookup(tmp_path):
    s, l = _synthetic(2000)
    cal = {"short": Calibration.from_backtest(s, l, bucket=20)}
    p = tmp_path / "cal.json"
    Calibration.save_all(cal, p)
    loaded = Calibration.load_all(p)
    probs = loaded["short"].probabilities(35)
    assert probs is not None and sum(probs) == 100
    assert probs[0] == 100  # 양수 점수 → 모두 up
    assert loaded["short"].probabilities(100) == loaded["short"].probabilities(85)


def test_calibration_sparse_bucket_returns_none():
    s = pd.Series([50.0] * 10)
    l = pd.Series(["up"] * 10)
    assert Calibration.from_backtest(s, l, bucket=20).probabilities(50) is None


def test_calibration_missing_file_returns_none(tmp_path):
    assert Calibration.load_all(tmp_path / "nope.json") is None


def test_excluded_indicator_is_not_missing_and_not_scored():
    cfg, sig = _signals(0.5)
    cfg["indicators"]["curve"]["horizons"] = {"long": {"exclude": True}}
    for h in sig:
        sig[h] = sig[h].drop(columns=["curve"]) if h == "long" else sig[h]
    sig["long"]["dxy_ret_60"] = -1.0
    sf = score_frame(sig, cfg)
    assert sf["long_missing"].iloc[-1] == 0
    # 평균은 long에서 활성인 macro 지표만 (curve 및 설정상 제외 지표 미포함)
    macro_active = [k for k in active_indicators(cfg, "long") if cfg["indicators"][k]["category"] == "macro"]
    assert "curve" not in macro_active
    expected = (0.5 * (len(macro_active) - 1) - 1.0) / len(macro_active)
    assert sf["long_macro"].iloc[-1] == pytest.approx(expected)
