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


def test_normal_run_sends_two_messages(tmp_path, monkeypatch):
    rc, client = _run(tmp_path, monkeypatch)
    assert rc == 0 and _structure_ok(client.sent)
    assert client.sent[0].startswith("[나스닥 방향성] 11/21(토) · 11/20 마감")
    assert any(m.startswith("[단기적") for m in client.sent) and any(m.startswith("[중기적") for m in client.sent)
    assert "•" in client.sent[1]


def test_core_data_failure_sends_error_message(tmp_path, monkeypatch):
    rc, client = _run(tmp_path, monkeypatch, md=DataSourceError("NDX empty"))
    assert rc == 1 and len(client.sent) == 1 and "오류" in client.sent[0]


def test_fred_missing_still_sends_with_warning(tmp_path, monkeypatch):
    rc, client = _run(tmp_path, monkeypatch, md=_md(with_fred=False))
    assert rc == 0 and _structure_ok(client.sent)
    # 현재 설정에서 FRED는 단기 7.5%·중기 17.5% 비중 → '신뢰도 낮음'(30%) 기준 미만, 누락 표시는 유지
    assert "누락: 10년물" in client.sent[-1]


def test_auth_error_returns_2(tmp_path, monkeypatch):
    rc, client = _run(tmp_path, monkeypatch, client=FakeClient(auth_error=True))
    assert rc == 2 and client.sent == []


def test_unexpected_error_sends_error_message_and_logs(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("corrupt calibration")
    monkeypatch.setattr(m, "build_report", boom)
    rc, client = _run(tmp_path, monkeypatch)
    assert rc == 1 and len(client.sent) == 1 and "오류" in client.sent[0]
    log_text = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "logs").glob("*.log"))
    assert "corrupt calibration" in log_text


def test_history_write_failure_does_not_block_messages(tmp_path, monkeypatch):
    def locked(*a, **k):
        raise PermissionError("history.csv is open in Excel")
    monkeypatch.setattr(m, "append_history", locked)
    rc, client = _run(tmp_path, monkeypatch)
    assert rc == 0 and _structure_ok(client.sent)


def test_dry_run_does_not_write_history(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, extra=["--dry-run"])
    assert not (tmp_path / "history.csv").exists()


def test_missing_source_is_not_scored(tmp_path):
    from nasdaq_signal.config import load_config
    cfg = yaml.safe_load(open(_cfg_path(tmp_path), encoding="utf-8"))
    md = _md()
    md.missing = ["rsp"]
    _, row = m.build_report(cfg, md, date(2026, 11, 21), None)
    assert np.isnan(row["raw_breadth"])            # RSP 누락 → 시장 폭 지표 점수 제외
    assert not np.isnan(row["mid_market"])         # 나머지 시장 지표로 계산
    assert row["mid_missing"] > 0


def test_mismatched_calibration_hides_probabilities(tmp_path):
    from nasdaq_signal.scoring import Calibration, config_fingerprint
    cfg = yaml.safe_load(open(_cfg_path(tmp_path), encoding="utf-8"))
    s = pd.Series(np.linspace(-100, 99, 4000))
    l = pd.Series(["up"] * 4000)
    cals = {h: Calibration.from_backtest(s, l, 20) for h in ("short", "mid", "long")}
    Calibration.save_all(cals, cfg["paths"]["calibration"], meta={"config": "other", "missing": []})
    report, _ = m.build_report(cfg, _md(), date(2026, 11, 21), None)
    assert all(p is None for p in report.probs.values())
    assert report.calibration_stale
    Calibration.save_all(cals, cfg["paths"]["calibration"], meta={"config": config_fingerprint(cfg), "missing": ["dgs10"]})
    report, _ = m.build_report(cfg, _md(), date(2026, 11, 21), None)
    assert report.calibration_stale
    Calibration.save_all(cals, cfg["paths"]["calibration"], meta={"config": config_fingerprint(cfg), "missing": ["fear_greed"]})
    report, _ = m.build_report(cfg, _md(), date(2026, 11, 21), None)
    assert not report.calibration_stale and report.probs["short"] is not None


def test_unpredicted_long_uses_base_rates_and_ignores_long_missing(tmp_path, monkeypatch):
    from nasdaq_signal.scoring import Calibration, config_fingerprint
    cfg = yaml.safe_load(open(_cfg_path(tmp_path), encoding="utf-8"))
    cfg["horizons"]["long"]["predict"] = False
    s = pd.Series(np.linspace(-100, 99, 4000))
    l = pd.Series(["up"] * 3000 + ["down"] * 1000)
    cals = {h: Calibration.from_backtest(s, l, 20) for h in ("short", "mid", "long")}
    Calibration.save_all(cals, cfg["paths"]["calibration"], meta={"config": config_fingerprint(cfg), "missing": []},
                         base={"short": [50, 30, 20, 100], "mid": [60, 30, 10, 100], "long": [76, 13, 11, 100]})
    report, _ = m.build_report(cfg, _md(), date(2026, 11, 21), None)
    assert report.probs["long"] == (76, 13, 11)
    assert report.unpredicted == {"long"}


def _structure_ok(sent):
    """요약 → 단기적·중기적 근거(나눠질 수 있음) → 지표 순서, 모두 200자 이하."""
    if len(sent) < 4 or not sent[0].startswith("[나스닥 방향성]") or not sent[-1].startswith("시장 "):
        return False
    middle = [m.splitlines()[0] for m in sent[1:-1]]
    return all(h.startswith(("[단기적", "[중기적")) for h in middle) and all(len(m) <= 200 for m in sent)


def test_build_report_includes_all_active_drivers_and_score_position(tmp_path):
    from nasdaq_signal.config import active_indicators
    from nasdaq_signal.scoring import Calibration, config_fingerprint
    cfg = yaml.safe_load(open(_cfg_path(tmp_path), encoding="utf-8"))
    s = pd.Series(np.linspace(-100, 99, 4000))
    cals = {h: Calibration.from_backtest(s, pd.Series(["up"] * 4000), 20) for h in ("short", "mid", "long")}
    Calibration.save_all(cals, cfg["paths"]["calibration"], meta={"config": config_fingerprint(cfg), "missing": []},
                         score_dists={h: list(np.linspace(-100, 100, 101)) for h in ("short", "mid", "long")})
    md = _md()
    report, row = m.build_report(cfg, md, date(2026, 11, 21), None)
    # 활성 지표 중 오늘 값이 있는 것 전부 (합성 데이터엔 HYG/IEF·2년물·기대인플레가 없어 해당 지표는 값 없음)
    with_value = [n for n in active_indicators(cfg, "mid") if not np.isnan(row[f"raw_{n}"])]
    assert len(with_value) >= 8
    assert sorted(d.name for d in report.drivers["mid"]) == sorted(with_value)
    assert 0.0 <= report.score_pct["short"] <= 1.0


def test_run_writes_dashboard(tmp_path, monkeypatch):
    rc, _ = _run(tmp_path, monkeypatch, extra=["--dry-run"])
    assert rc == 0 and (tmp_path / "dashboard.html").exists()


def test_dashboard_failure_does_not_block_messages(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("chart broke")
    monkeypatch.setattr(m, "write_dashboard", boom)
    rc, client = _run(tmp_path, monkeypatch)
    assert rc == 0 and _structure_ok(client.sent)


def test_inactive_category_not_shown(tmp_path):
    cfg = yaml.safe_load(open(_cfg_path(tmp_path), encoding="utf-8"))
    for n in ("gpr", "oil_ret_20", "gold_ret_20"):
        cfg["indicators"][n]["horizons"] = {h: {"exclude": True} for h in ("short", "mid", "long")}
    report, _ = m.build_report(cfg, _md(), date(2026, 11, 21), None)
    assert "geo" not in report.categories and "market" in report.categories
