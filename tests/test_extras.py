import numpy as np
import pandas as pd
import pytest

import nasdaq_signal.data as data
from nasdaq_signal.config import load_config
from nasdaq_signal.data import collect_all, extras


class FakeResp:
    def __init__(self, payload=None, status=200):
        self._p = payload
        self.status_code = status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_fetch_gpr_parses_daily_series_with_publication_lag(monkeypatch):
    raw = pd.DataFrame({"DAY": [20240102, 20240103], "GPRD": [100.0, 120.0], "var_name": ["x", "y"]})
    monkeypatch.setattr(extras, "_read_gpr_raw", lambda: raw)
    s = extras.fetch_gpr()
    assert s.name == "gpr"
    assert isinstance(s.index, pd.DatetimeIndex)
    # 5영업일 발표 지연: 2024-01-02 관측 → 2024-01-09부터 사용 가능
    assert s.loc[pd.Timestamp("2024-01-09")] == 100.0


def test_fear_greed_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(extras.requests, "get", lambda *a, **k: FakeResp(status=418))
    assert extras.fetch_fear_greed() is None


def test_fear_greed_parses_score(monkeypatch):
    monkeypatch.setattr(extras.requests, "get", lambda *a, **k: FakeResp({"fear_and_greed": {"score": 72.4}}))
    assert extras.fetch_fear_greed() == pytest.approx(72.4)


def test_collect_all_puts_gpr_in_frame_and_fg_in_extras(monkeypatch):
    idx = pd.bdate_range(end="2024-03-01", periods=30)
    pf = pd.DataFrame({k: 100.0 for k in data.PRICE_TICKERS}, index=idx)
    monkeypatch.setattr(data, "fetch_prices", lambda start: pf)
    monkeypatch.setattr(data, "fetch_fred", lambda *a: pd.DataFrame())
    monkeypatch.setattr(extras, "fetch_gpr", lambda: pd.Series(50.0, index=idx, name="gpr"))
    monkeypatch.setattr(extras, "fetch_fear_greed", lambda: 72.0)
    monkeypatch.setattr(extras, "fetch_nq_futures_change", lambda: 0.008)
    md = collect_all(load_config(), "2024-01-01")
    assert md.frame["gpr"].iloc[-1] == 50.0
    assert md.extras == {"fear_greed": 72.0, "nq_change": 0.008}
    assert "gpr" not in md.missing


def test_collect_all_failed_gpr_and_fg_marked_missing(monkeypatch):
    idx = pd.bdate_range(end="2024-03-01", periods=30)
    pf = pd.DataFrame({k: 100.0 for k in data.PRICE_TICKERS}, index=idx)
    monkeypatch.setattr(data, "fetch_prices", lambda start: pf)
    monkeypatch.setattr(data, "fetch_fred", lambda *a: pd.DataFrame())

    def boom():
        raise RuntimeError("down")
    monkeypatch.setattr(extras, "fetch_gpr", boom)
    monkeypatch.setattr(extras, "fetch_fear_greed", lambda: None)
    monkeypatch.setattr(extras, "fetch_nq_futures_change", lambda: None)
    md = collect_all(load_config(), "2024-01-01")
    assert "gpr" in md.missing and "fear_greed" in md.missing
    assert np.isnan(md.frame["gpr"].iloc[-1])
