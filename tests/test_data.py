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

    class NoData:   # 개별 재시도도 실패하는 상황 (네트워크 없이)
        def __init__(self, t):
            pass

        def history(self, **k):
            return pd.DataFrame()
    monkeypatch.setattr(prices.yf, "Ticker", NoData)
    monkeypatch.setattr(prices, "RETRY_SLEEP", 0)
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
        collect_all(load_config(apply_tuning=False), "2024-01-01", include_extras=False)


def test_collect_all_marks_failed_source_missing(monkeypatch):
    pf = _price_frame()
    monkeypatch.setattr(data, "fetch_prices", lambda start: pf)

    def boom(*a):
        raise RuntimeError("fred down")
    monkeypatch.setattr(data, "fetch_fred", boom)
    md = collect_all(load_config(apply_tuning=False), "2024-01-01", include_extras=False)
    assert "dgs10" in md.missing and "nfci" in md.missing
    assert md.frame["ndx"].iloc[-1] == pytest.approx(110)
    assert md.as_of == pf.index[-1].date()


def test_collect_all_marks_stale_column_missing(monkeypatch):
    pf = _price_frame()
    pf.loc[pf.index[-7]:, "rsp"] = np.nan  # last valid value 7 business days before as_of
    monkeypatch.setattr(data, "fetch_prices", lambda start: pf)
    monkeypatch.setattr(data, "fetch_fred", lambda *a: _fred_frame(pf.index))
    md = collect_all(load_config(apply_tuning=False), "2024-01-01", include_extras=False)
    assert "rsp" in md.missing
    assert np.isnan(md.frame["rsp"].iloc[-1])
    assert "spy" not in md.missing


def test_collect_all_tolerates_weekly_fred_gap(monkeypatch):
    pf = _price_frame()
    ff = _fred_frame(pf.index)
    ff.loc[ff.index[-8]:, "nfci"] = np.nan  # weekly series: 8 bdays old is normal (lag 5 + stale 5)
    monkeypatch.setattr(data, "fetch_prices", lambda start: pf)
    monkeypatch.setattr(data, "fetch_fred", lambda *a: ff)
    md = collect_all(load_config(apply_tuning=False), "2024-01-01", include_extras=False)
    assert "nfci" not in md.missing
    assert md.frame["nfci"].iloc[-1] == 1.0


def test_fred_failure_is_per_series_and_does_not_log_key(monkeypatch, caplog):
    def get(url, params=None, timeout=None, **kw):
        if params["series_id"] == "NFCI":
            r = FakeResp({}, status=500)
            def rfs():
                raise RuntimeError(f"500 Server Error for url: {url}?api_key={params['api_key']}")
            r.raise_for_status = rfs
            return r
        return FakeResp({"observations": [{"date": "2024-01-02", "value": "1.0"}]})
    monkeypatch.setattr(fred.requests, "get", get)
    with caplog.at_level("WARNING"):
        df = fred.fetch_fred({"DGS10": 1, "NFCI": 5}, "2024-01-01", "SECRETKEY123")
    assert "dgs10" in df.columns and "nfci" not in df.columns
    assert "SECRETKEY123" not in caplog.text
    assert "NFCI" in caplog.text


def test_fetch_prices_retries_empty_ticker_individually(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=3)
    batch = pd.DataFrame({t: [1.0, 2.0, 3.0] for t in prices.PRICE_TICKERS.values()}, index=idx)
    batch["^VIX"] = np.nan                                   # 일괄 다운로드에서 VIX만 비어서 옴
    monkeypatch.setattr(prices.yf, "download", lambda *a, **k: pd.concat({"Close": batch}, axis=1))
    calls = []

    class FakeTicker:
        def __init__(self, t):
            self.t = t

        def history(self, start=None, auto_adjust=True):
            calls.append(self.t)
            if len(calls) == 1:
                raise RuntimeError("rate limited")             # 첫 재시도 실패 → 다시 시도
            return pd.DataFrame({"Close": [20.0, 21.0, 22.0]}, index=idx.tz_localize("America/Chicago"))
    monkeypatch.setattr(prices.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(prices, "RETRY_SLEEP", 0)
    df = prices.fetch_prices("2024-01-01")
    assert calls == ["^VIX", "^VIX"]
    assert df["vix"].iloc[-1] == 22.0


def test_fetch_prices_retries_when_latest_value_missing(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=3)
    batch = pd.DataFrame({t: [1.0, 2.0, 3.0] for t in prices.PRICE_TICKERS.values()}, index=idx)
    batch.loc[idx[-1], "^VIX"] = np.nan                      # 마지막 날만 비어서 옴
    monkeypatch.setattr(prices.yf, "download", lambda *a, **k: pd.concat({"Close": batch}, axis=1))

    class T:
        def __init__(self, t):
            self.t = t

        def history(self, **k):
            return pd.DataFrame({"Close": [20.0, 21.0, 22.0]}, index=idx)
    monkeypatch.setattr(prices.yf, "Ticker", T)
    monkeypatch.setattr(prices, "RETRY_SLEEP", 0)
    assert prices.fetch_prices("2024-01-01")["vix"].iloc[-1] == 22.0
