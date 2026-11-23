import numpy as np
import pandas as pd
import pytest

import nasdaq_signal.data as data
from nasdaq_signal.data import fred, prices
from nasdaq_signal.data import DataSourceError, collect_all
from nasdaq_signal.config import load_config


class FakeResp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")


def _fake_get(obs_by_series):
    def get(url, params=None, timeout=None, **kw):
        return FakeResp({"observations": obs_by_series[params["series_id"]]})
    return get


def test_fred_shifts_by_publication_lag(monkeypatch):
    monkeypatch.setattr(fred.requests, "get", _fake_get({"NFCI": [{"date": "2024-01-05", "value": "-0.5"}]}))
    df = fred.fetch_fred({"NFCI": 5}, "2024-01-01", "key")
    assert df.loc[pd.Timestamp("2024-01-12"), "nfci"] == -0.5
    assert pd.Timestamp("2024-01-05") not in df.index


def test_fred_dot_values_become_nan(monkeypatch):
    obs = [{"date": "2024-01-02", "value": "."}, {"date": "2024-01-03", "value": "4.0"}]
    monkeypatch.setattr(fred.requests, "get", _fake_get({"DGS10": obs}))
    df = fred.fetch_fred({"DGS10": 1}, "2024-01-01", "key")
    assert np.isnan(df["dgs10"].iloc[0])
    assert df["dgs10"].iloc[1] == 4.0


def test_fred_without_key_returns_empty():
    assert fred.fetch_fred({"DGS10": 1}, "2024-01-01", None).empty


def test_fetch_prices_renames_and_fills_missing_tickers(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=3)
    close = pd.DataFrame({"^NDX": [1.0, 2.0, 3.0], "SPY": [1.0, 1.0, 1.0]}, index=idx)
    monkeypatch.setattr(prices.yf, "download", lambda *a, **k: pd.concat({"Close": close}, axis=1))
    df = prices.fetch_prices("2024-01-01")
    assert set(df.columns) == set(prices.PRICE_TICKERS)
    assert df["ndx"].iloc[-1] == 3.0
    assert df["rsp"].isna().all()


def _price_frame(n=30, end="2024-03-01"):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({k: np.linspace(100, 110, n) for k in prices.PRICE_TICKERS}, index=idx)


def _fred_frame(index):
    return pd.DataFrame({s.lower(): 1.0 for s in ["DGS10", "T10Y2Y", "BAMLH0A0HYM2", "NFCI", "SAHMREALTIME"]}, index=index)


def test_collect_all_raises_when_ndx_empty(monkeypatch):
    pf = _price_frame()
    pf["ndx"] = np.nan
    monkeypatch.setattr(data, "fetch_prices", lambda start: pf)
    monkeypatch.setattr(data, "fetch_fred", lambda *a: _fred_frame(pf.index))
    with pytest.raises(DataSourceError):
        collect_all(load_config(), "2024-01-01", include_extras=False)


def test_collect_all_marks_failed_source_missing(monkeypatch):
    pf = _price_frame()
    monkeypatch.setattr(data, "fetch_prices", lambda start: pf)

    def boom(*a):
        raise RuntimeError("fred down")
    monkeypatch.setattr(data, "fetch_fred", boom)
    md = collect_all(load_config(), "2024-01-01", include_extras=False)
    assert "dgs10" in md.missing and "nfci" in md.missing
    assert md.frame["ndx"].iloc[-1] == pytest.approx(110)
    assert md.as_of == pf.index[-1].date()


def test_collect_all_marks_stale_column_missing(monkeypatch):
    pf = _price_frame()
    pf.loc[pf.index[-7]:, "rsp"] = np.nan  # last valid value 7 business days before as_of
    monkeypatch.setattr(data, "fetch_prices", lambda start: pf)
    monkeypatch.setattr(data, "fetch_fred", lambda *a: _fred_frame(pf.index))
    md = collect_all(load_config(), "2024-01-01", include_extras=False)
    assert "rsp" in md.missing
    assert np.isnan(md.frame["rsp"].iloc[-1])
    assert "spy" not in md.missing


def test_collect_all_tolerates_weekly_fred_gap(monkeypatch):
    pf = _price_frame()
    ff = _fred_frame(pf.index)
    ff.loc[ff.index[-8]:, "nfci"] = np.nan  # weekly series: 8 bdays old is normal (lag 5 + stale 5)
    monkeypatch.setattr(data, "fetch_prices", lambda start: pf)
    monkeypatch.setattr(data, "fetch_fred", lambda *a: ff)
    md = collect_all(load_config(), "2024-01-01", include_extras=False)
    assert "nfci" not in md.missing
    assert md.frame["nfci"].iloc[-1] == 1.0
