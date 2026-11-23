from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

import nasdaq_signal.main as m
from nasdaq_signal.data import DataSourceError, MarketData
from nasdaq_signal.kakao import KakaoAuthError


class FakeClient:
    def __init__(self, auth_error=False):
        self.sent, self.auth_error = [], auth_error

    def refresh(self):
        if self.auth_error:
            raise KakaoAuthError("카카오 재인증 필요: python -m nasdaq_signal.kakao_auth")

    def days_until_reauth(self):
        return 40

    def send_text(self, text, link_url):
        assert len(text) <= 200
        self.sent.append(text)


def _cfg_path(tmp_path):
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cfg.update(percentile_window=100, percentile_min_periods=20)
    cfg["paths"] = {k: str(tmp_path / v.replace("state/", "")) for k, v in cfg["paths"].items()}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(p)


def _md(with_fred=True, n=400):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(end="2026-11-20", periods=n)
    cols = ["ndx", "spy", "rsp", "vix", "vix3m", "oil", "gold", "dxy", "gpr"]
    if with_fred:
        cols += ["dgs10", "t10y2y", "bamlh0a0hym2", "nfci", "sahmrealtime"]
    frame = pd.DataFrame({c: 100 + rng.normal(0, 1, n).cumsum() for c in cols}, index=idx)
    missing = [] if with_fred else ["dgs10", "t10y2y", "bamlh0a0hym2", "nfci", "sahmrealtime"]
    return MarketData(frame=frame, as_of=date(2026, 11, 20), missing=missing,
                      extras={"fear_greed": 40.0, "nq_change": None})


def _run(tmp_path, monkeypatch, md=None, client=None, extra=()):
    client = client or FakeClient()
    if isinstance(md, Exception):
        def collect(*a, **k):
            raise md
    else:
        collect = lambda *a, **k: md or _md()  # noqa: E731
    monkeypatch.setattr(m, "collect_all", collect)
    monkeypatch.setattr(m, "make_client", lambda cfg: client)
    rc = m.main(["--config", _cfg_path(tmp_path), "--run-date", "2026-11-21", *extra])
    return rc, client


def test_is_market_closed():
    fri = date(2026, 11, 20)
    assert m.is_market_closed(fri, date(2026, 11, 21)) is False
    assert m.is_market_closed(fri, date(2026, 11, 22)) is True
    assert m.is_market_closed(fri, date(2026, 11, 23)) is True


def test_append_history_upserts_same_date(tmp_path):
    p = tmp_path / "h.csv"
    m.append_history(str(p), {"date": "2026-11-21", "short_score": 1.0})
    m.append_history(str(p), {"date": "2026-11-22", "short_score": 5.0})
    m.append_history(str(p), {"date": "2026-11-21", "short_score": 2.0})
    df = pd.read_csv(p, encoding="utf-8-sig")
    assert list(df["date"]) == ["2026-11-21", "2026-11-22"]
    assert df.loc[df["date"] == "2026-11-21", "short_score"].item() == 2.0


def test_dry_run_sends_nothing(tmp_path, monkeypatch, capsys):
    rc, client = _run(tmp_path, monkeypatch, extra=["--dry-run"])
    assert rc == 0 and client.sent == []
    assert "[나스닥 방향성]" in capsys.readouterr().out
    assert (tmp_path / "history.csv").exists()


def test_normal_run_sends_two_messages(tmp_path, monkeypatch):
    rc, client = _run(tmp_path, monkeypatch)
    assert rc == 0 and len(client.sent) == 2
    assert client.sent[0].startswith("[나스닥 방향성] 11/21(토) · 11/20 마감")


def test_core_data_failure_sends_error_message(tmp_path, monkeypatch):
    rc, client = _run(tmp_path, monkeypatch, md=DataSourceError("NDX empty"))
    assert rc == 1 and len(client.sent) == 1 and "오류" in client.sent[0]


def test_fred_missing_still_sends_with_warning(tmp_path, monkeypatch):
    rc, client = _run(tmp_path, monkeypatch, md=_md(with_fred=False))
    assert rc == 0 and len(client.sent) == 2
    assert "신뢰도 낮음" in client.sent[0]
    assert "누락" in client.sent[1]


def test_auth_error_returns_2(tmp_path, monkeypatch):
    rc, client = _run(tmp_path, monkeypatch, client=FakeClient(auth_error=True))
    assert rc == 2 and client.sent == []
