import json

import numpy as np
import pandas as pd

from nasdaq_signal.backtest import forward_labels, run_backtest
from nasdaq_signal.config import load_config


def test_forward_labels():
    ndx = pd.Series(np.r_[np.full(10, 100.0), np.full(15, 102.0)], index=pd.bdate_range("2024-01-01", periods=25))
    lab = forward_labels(ndx, days=10, up=0.015, down=-0.015)
    assert lab.iloc[0] == "up"          # 10일 후 +2%
    assert lab.iloc[10] == "flat"
    assert lab.iloc[-10:].isna().all()


def test_calibration_uses_only_test_period(tmp_path):
    cfg = load_config(apply_tuning=False)
    cfg.update(percentile_window=100, percentile_min_periods=20)
    cfg["paths"]["calibration"] = str(tmp_path / "cal.json")
    idx = pd.bdate_range("2014-01-01", periods=1600)
    split = idx[800]
    cfg["backtest"] = {"start": str(idx[0].date()), "split": str(split.date()), "bucket": 20}
    ndx = np.where(idx < split, np.linspace(100, 300, len(idx)), 0.0)
    ndx[idx >= split] = np.linspace(300, 50, (idx >= split).sum())
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"ndx": ndx, "vix": 20 + rng.normal(0, 1, len(idx))}, index=idx)
    result = run_backtest(cfg, frame)
    stored = json.loads(open(cfg["paths"]["calibration"], encoding="utf-8").read())
    for h in ("short", "mid", "long"):
        counts = np.array(list(stored[h].values()))
        assert counts[:, 0].sum() == 0          # 검증 구간은 전부 하락 → up 0건
        assert counts[:, 3].sum() > 0
        assert result["test"][h]["n"] == counts[:, 3].sum()
        assert result["train"][h]["n"] > 0


def test_backtest_stores_config_fingerprint_and_missing(tmp_path):
    from nasdaq_signal.scoring import Calibration, config_fingerprint
    cfg = load_config(apply_tuning=False)
    cfg.update(percentile_window=100, percentile_min_periods=20)
    cfg["paths"]["calibration"] = str(tmp_path / "cal.json")
    idx = pd.bdate_range("2014-01-01", periods=900)
    cfg["backtest"] = {"start": str(idx[0].date()), "split": str(idx[400].date()), "bucket": 20}
    frame = pd.DataFrame({"ndx": np.linspace(100, 200, len(idx))}, index=idx)
    run_backtest(cfg, frame, missing=["dgs10"])
    meta = Calibration.load_meta(cfg["paths"]["calibration"])
    assert meta == {"config": config_fingerprint(cfg), "missing": ["dgs10"]}
