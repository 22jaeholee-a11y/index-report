import json
import re
from datetime import date

import numpy as np
import pandas as pd
import yaml

from nasdaq_signal.dashboard import build_history, render_html
from nasdaq_signal.data import MarketData
from nasdaq_signal.scoring import Calibration, config_fingerprint


def _cfg(tmp_path):
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cfg.update(percentile_window=100, percentile_min_periods=20)
    cfg["paths"] = {k: str(tmp_path / v.replace("state/", "")) for k, v in cfg["paths"].items()}
    return cfg


def _md(n=400):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(end="2026-11-20", periods=n)
    cols = ["ndx", "spy", "rsp", "vix", "vix3m", "oil", "gold", "dxy", "gpr", "hyg", "ief",
            "dgs10", "dgs2", "t10yie", "t10y2y", "bamlh0a0hym2", "nfci", "sahmrealtime"]
    return MarketData(frame=pd.DataFrame({c: 100 + rng.normal(0, 1, n).cumsum() for c in cols}, index=idx),
                      as_of=date(2026, 11, 20))


def _save_cal(cfg):
    s = pd.Series(np.linspace(-100, 99, 4000))
    cals = {h: Calibration.from_backtest(s, pd.Series(["up"] * 3000 + ["down"] * 1000), 20) for h in cfg["horizons"]}
    Calibration.save_all(cals, cfg["paths"]["calibration"], meta={"config": config_fingerprint(cfg), "missing": []})


def test_build_history_has_scores_probabilities_and_live_flag(tmp_path):
    cfg = _cfg(tmp_path)
    _save_cal(cfg)
    pd.DataFrame({"date": ["2026-11-21"], "as_of": ["2026-11-20"]}).to_csv(cfg["paths"]["history"], index=False, encoding="utf-8-sig")
    h = build_history(cfg, _md(), days=120)
    assert len(h) == 120
    for col in ("ndx", "short_score", "mid_score", "short_up", "mid_up", "mid_market", "mid_sentiment", "mid_geo", "mid_macro", "live"):
        assert col in h.columns
    assert h["live"].iloc[-1] and not h["live"].iloc[0]
    assert h["short_up"].dropna().between(0, 100).all()


def test_build_history_without_calibration_has_no_probabilities(tmp_path):
    h = build_history(_cfg(tmp_path), _md(), days=50)
    assert h["short_up"].isna().all()


def test_render_html_embeds_data_and_is_self_contained(tmp_path):
    cfg = _cfg(tmp_path)
    _save_cal(cfg)
    h = build_history(cfg, _md(), days=60)
    html = render_html(h, generated="2026-11-23 09:00")
    assert html.startswith("<!doctype html>")
    assert "<title>나스닥 방향성 추이</title>" in html
    payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert len(payload["rows"]) == 60 and payload["rows"][-1]["date"] == "2026-11-20"
    assert "<script src=" not in html          # 외부 스크립트 없음 (로컬 파일로도 동작)
    assert "prefers-color-scheme: dark" in html


def test_history_marks_buy_zone(tmp_path):
    cfg = _cfg(tmp_path)
    h = build_history(cfg, _md(), days=200)
    assert "zone" in h.columns and h["zone"].dtype == bool
    html = render_html(h, generated="x")
    assert '"zone":' in html and "매수 구간" in html


def test_render_includes_today_drivers_section(tmp_path):
    from nasdaq_signal.explain import Driver
    cfg = _cfg(tmp_path)
    h = build_history(cfg, _md(), days=30)
    today = {"short": {"title": "단기적", "score": 61.0, "position": "상위 11%",
                       "drivers": [Driver("vix", 35.9, 0.99, 66.0, 1, False, 67.0)]}}
    html = render_html(h, generated="x", today=today)
    payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    d = payload["today"]["short"]
    assert d["score"] == 61.0 and d["drivers"][0]["label"] == "VIX" and d["drivers"][0]["max"] == 67.0
    assert "오늘의 근거" in html


def test_tabs_and_no_bar_legend_text(tmp_path):
    cfg = _cfg(tmp_path)
    html = render_html(build_history(cfg, _md(), days=30), generated="x")
    assert 'role="tablist"' in html
    assert 'id="tab-today"' in html and 'id="tab-chart"' in html
    assert 'id="panel-today"' in html and 'id="panel-chart"' in html
    assert "막대 = 기여 점수" not in html


def test_today_card_title_is_short():
    src = open("nasdaq_signal/dashboard.py", encoding="utf-8").read()
    assert "오늘의 근거 · ${" not in src


def test_mobile_and_synced_selection_elements(tmp_path):
    cfg = _cfg(tmp_path)
    html = render_html(build_history(cfg, _md(), days=30), generated="x")
    assert 'id="selbar"' in html and "최신으로" not in html   # 선택 날짜 막대 (해제 버튼 없음: 차트 밖 탭으로 해제)
    assert "touch-action" in html                            # 모바일 가로 스크럽
    assert "position: sticky" in html                        # 탭·선택 막대 고정
    assert 'class="tip"' not in html                         # 떠다니는 말풍선 대신 고정 표시


def test_write_dashboard_also_publishes_copy(tmp_path):
    from nasdaq_signal.dashboard import write_dashboard
    cfg = _cfg(tmp_path)
    cfg["paths"]["publish"] = str(tmp_path / "public" / "index.html")
    write_dashboard(cfg, _md(), "x")
    pub = tmp_path / "public" / "index.html"
    assert pub.exists()
    assert pub.read_text(encoding="utf-8") == (tmp_path / "dashboard.html").read_text(encoding="utf-8")


def test_vercel_serves_only_static_public_folder():
    import json as _json
    v = _json.loads(open("vercel.json", encoding="utf-8").read())
    assert v["outputDirectory"] == "public" and v["framework"] is None
    ignore = open(".vercelignore", encoding="utf-8").read().split()
    assert "/*" in ignore and "!/public" in ignore and "!/vercel.json" in ignore


def test_chart_default_range_is_one_month(tmp_path):
    html = render_html(build_history(_cfg(tmp_path), _md(), days=30), generated="x")
    assert '<button data-n="21" aria-pressed="true">1개월</button>' in html
    assert 'data-n="126" aria-pressed="true"' not in html
    assert "let N = 21," in html


def test_today_payload_has_changes_and_korean_colors(tmp_path):
    from nasdaq_signal.explain import Driver
    h = build_history(_cfg(tmp_path), _md(), days=30)
    today = {"short": {"title": "단기적", "score": 61.0, "score_prev": 58.2, "position": "상위 11%",
                       "drivers": [Driver("vix", 35.9, 0.99, 66.0, 1, False, 67.0, 28.4, 60.0)]}}
    html = render_html(h, generated="x", today=today)
    payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    t = payload["today"]["short"]
    assert t["score_change"] == "+3" and t["score_dir"] == "up"
    assert t["drivers"][0]["value_change"] == "+7.5"
    assert ".chg.up" in html and "--rise:" in html and "--fall:" in html


def test_long_press_scrub_support(tmp_path):
    html = render_html(build_history(_cfg(tmp_path), _md(), days=30), generated="x")
    assert "LONG_PRESS" in html                       # 꾹 누르기 시간
    assert "-webkit-touch-callout: none" in html      # 꾹 누를 때 시스템 메뉴 방지
    assert "passive: false" in html                   # 고르기 중 스크롤 막기 (touchmove preventDefault)


def test_buy_zone_shaded_on_all_four_charts():
    src = open("nasdaq_signal/dashboard.py", encoding="utf-8").read()
    for cid in ("c-ndx", "c-score", "c-cat"):
        assert f'{{ id: "{cid}", zone: true' in src


def test_probability_chart_removed_and_shown_in_score_readout(tmp_path):
    html = render_html(build_history(_cfg(tmp_path), _md(), days=30), generated="x")
    assert 'id="c-up"' not in html and "<h2>상승 확률 (%)</h2>" not in html
    assert 'extra: "short_up"' in html and 'extra: "mid_up"' in html   # 점수 값 옆에 (상승 n%)


def test_history_starts_at_dashboard_start(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["dashboard_start"] = "2025-09-01"
    h = build_history(cfg, _md(), days=None)
    assert h.index[0] >= pd.Timestamp("2025-09-01") and h.index[0] < pd.Timestamp("2025-09-10")


def test_long_range_buttons_and_thinning(tmp_path):
    html = render_html(build_history(_cfg(tmp_path), _md(), days=30), generated="x")
    assert '<button data-n="1260">5년</button>' in html
    assert "function thin(" in html


def test_index_chart_uses_log_scale_for_long_ranges():
    src = open("nasdaq_signal/dashboard.py", encoding="utf-8").read()
    assert '{ id: "c-ndx", zone: true, logIfWide: true' in src
    assert "function logTicks(" in src


def test_no_sent_day_shading_or_long_note(tmp_path):
    html = render_html(build_history(_cfg(tmp_path), _md(), days=30), generated="x")
    assert "실제 발송된 날 음영" not in html and 'css("--live")' not in html
    assert "회색 음영은" not in html and "차트를 누르거나 좌우로 밀면" not in html


def test_10y_20y_buttons_and_pinch_zoom_code(tmp_path):
    html = render_html(build_history(_cfg(tmp_path), _md(), days=30), generated="x")
    for n, label in ((2520, "10년"), (5040, "20년")):
        assert f'<button data-n="{n}">{label}</button>' in html
    assert "function startPinch(" in html          # 두 손가락 확대·축소
    assert "ctrlKey" in html                         # PC: 트랙패드 핀치 / Ctrl+휠
