"""방향성 추이 차트 페이지 (state/dashboard.html).

과거 점수는 현재 설정으로 다시 계산한 값이고, 실제로 발송된 날(history.csv)은 따로 표시한다.
실행: python -m nasdaq_signal.dashboard  (일일 실행 시에도 자동 생성)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .data import MarketData
from .indicators import compute_raw, mask_missing, to_signals
from .scoring import Calibration, score_frame
from .explain import driver_view, points_delta
from .zone import zone_states

CATEGORIES = ("market", "sentiment", "geo", "macro")


def build_history(cfg: dict, md: MarketData, days: int | None = None) -> pd.DataFrame:
    """days가 None이면 cfg['dashboard_start'](기본 1996-01-01)부터."""
    raw = mask_missing(compute_raw(md.frame), md.missing)
    sf = score_frame(to_signals(raw, cfg), cfg)
    cal_path = cfg["paths"]["calibration"]
    valid = Calibration.is_valid_for(Calibration.load_meta(cal_path), cfg)
    cals = (Calibration.load_all(cal_path) or {}) if valid else {}

    out = pd.DataFrame(index=sf.index)
    out["ndx"] = md.frame["ndx"]
    for h in ("short", "mid"):
        out[f"{h}_score"] = sf[f"{h}_score"]
        cal = cals.get(h)

        def up(s, cal=cal):
            if cal is None or s is None or math.isnan(s):
                return math.nan
            p = cal.probabilities(s)
            return math.nan if p is None else p[0]
        out[f"{h}_up"] = out[f"{h}_score"].map(up)
    for c in CATEGORIES:
        out[f"mid_{c}"] = sf[f"mid_{c}"]

    live_dates: set[str] = set()
    hist = Path(cfg["paths"]["history"])
    if hist.exists():
        live_dates = set(pd.read_csv(hist, encoding="utf-8-sig", dtype=str).get("as_of", pd.Series(dtype=str)).dropna())
    out["live"] = [d.date().isoformat() in live_dates for d in out.index]
    z = cfg.get("zone", {"entry": 20, "exit": -20})
    out["zone"] = zone_states(sf["mid_score"], z["entry"], z["exit"]).astype(bool)  # 전체 기간으로 계산 후 자름
    if days is not None:
        return out.iloc[-days:]
    return out[out.index >= pd.Timestamp(cfg.get("dashboard_start", "1996-01-01"))]


def _rows(h: pd.DataFrame) -> list[dict]:
    rows = []
    for d, r in h.iterrows():
        row = {"date": d.date().isoformat(), "live": bool(r["live"]), "zone": bool(r["zone"])}
        for k, v in r.items():
            if k in ("live", "zone"):
                continue
            row[k] = None if pd.isna(v) else round(float(v), 4)
        rows.append(row)
    return rows


def _today_payload(today: dict | None) -> dict:
    """{h: {title, score, position, drivers: [Driver]}} → JSON용 (지표 설명 포함)."""
    out = {}
    for h, t in (today or {}).items():
        prev = t.get("score_prev")
        chg, dirn = points_delta(t["score"], prev if prev is not None else float("nan"))
        out[h] = {"title": t["title"], "score": t["score"], "position": t.get("position", ""),
                  "score_change": chg, "score_dir": dirn,
                  "drivers": [driver_view(d) for d in t["drivers"]]}
    return out


def render_html(h: pd.DataFrame, generated: str, zone_rule: tuple[float, float] = (20, -20),
                today: dict | None = None) -> str:
    payload = json.dumps({"generated": generated, "rows": _rows(h), "today": _today_payload(today)},
                         ensure_ascii=False)
    entry, exit_ = zone_rule
    return (_TEMPLATE.replace("__ENTRY__", f"{entry:+.0f}").replace("__EXIT__", f"{exit_:+.0f}")
            .replace("__DATA__", payload.replace("</", "<\\/")))


def write_dashboard(cfg: dict, md: MarketData, generated: str, today: dict | None = None) -> Path:
    path = Path(cfg["paths"].get("dashboard", Path(cfg["paths"]["state"]) / "dashboard.html"))
    path.parent.mkdir(parents=True, exist_ok=True)
    z = cfg.get("zone", {"entry": 20, "exit": -20})
    html = render_html(build_history(cfg, md), generated, (z["entry"], z["exit"]), today)
    path.write_text(html, encoding="utf-8")
    publish = cfg["paths"].get("publish")
    if publish:  # 웹 배포용 사본 (git에 포함 → Vercel 정적 배포)
        Path(publish).parent.mkdir(parents=True, exist_ok=True)
        Path(publish).write_text(html, encoding="utf-8")
    return path


def main() -> None:
    import sys
    import webbrowser
    from datetime import datetime

    from dotenv import load_dotenv

    from .config import load_config
    from .data import collect_all

    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    cfg = load_config()
    md = collect_all(cfg, cfg.get("data_start", "1990-01-01"))
    from .main import build_report, today_drivers  # 순환 import 방지

    report, _ = build_report(cfg, md, datetime.now().date(), None)
    path = write_dashboard(cfg, md, datetime.now().strftime("%Y-%m-%d %H:%M"), today_drivers(report, cfg))
    print(f"저장: {path.resolve()}")
    if "--open" in sys.argv:
        webbrowser.open(path.resolve().as_uri())


_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>나스닥 방향성 추이</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f6f6f4; --surface-1: #fcfcfb; --border: #e4e3df;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --grid: #ecebe7; --axis: #c9c8c2; --ref: #9a9993; --live: #e4e3df; --zone: #e3f1e6;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --rise: #e34948; --fall: #2a78d6; --rise-text: #c4302f; --fall-text: #1c5cab;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2e2c;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8e8d86;
    --grid: #262624; --axis: #45443f; --ref: #6f6e68; --live: #34342f; --zone: #1d2e22;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --rise: #e66767; --fall: #3987e5; --rise-text: #f08585; --fall-text: #6da7ec;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2e2c;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8e8d86;
  --grid: #262624; --axis: #45443f; --ref: #6f6e68; --live: #262624;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --rise: #e66767; --fall: #3987e5; --rise-text: #f08585; --fall-text: #6da7ec;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.5 -apple-system, "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 12px; font-size: 13px; }
.tabs { position: sticky; top: 0; z-index: 6; display: flex; gap: 4px; background: var(--surface-0);
  border-bottom: 1px solid var(--border); margin: 0 -16px 14px; padding: 0 16px; }
.tabs button { flex: 0 0 auto; border: 0; background: none; color: var(--text-secondary); font: inherit; font-size: 15px;
  padding: 11px 14px; min-height: 44px; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px; }
.tabs button[aria-selected="true"] { color: var(--text-primary); font-weight: 600; border-bottom-color: var(--text-primary); }
.tabs button:focus-visible, .bar button:focus-visible, .selbar button:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; border-radius: 6px; }
.selbar { position: sticky; top: 45px; z-index: 5; display: flex; align-items: center; justify-content: space-between; gap: 8px;
  background: var(--surface-0); margin: -14px -16px 10px; padding: 8px 16px; border-bottom: 1px solid var(--border); font-size: 13.5px; }
.selbar b { font-variant-numeric: tabular-nums; }
.selbar .flags { color: var(--text-secondary); font-size: 12.5px; }
.selbar button { border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary); border-radius: 8px;
  padding: 6px 12px; min-height: 34px; font: inherit; font-size: 13px; cursor: pointer; }
.selbar button[hidden] { display: none; }
.bar { display: flex; gap: 6px; margin-bottom: 12px; overflow-x: auto; scrollbar-width: none; }
.bar::-webkit-scrollbar { display: none; }
.bar button { flex: 0 0 auto; border: 1px solid var(--border); background: var(--surface-1); color: var(--text-secondary);
  border-radius: 8px; padding: 7px 14px; min-height: 36px; font: inherit; font-size: 13.5px; cursor: pointer; }
.bar button[aria-pressed="true"] { color: var(--text-primary); border-color: var(--text-secondary); font-weight: 600; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 14px 14px 8px; margin-bottom: 12px; }
.card h2 { font-size: 15px; margin: 0; }
.card p { margin: 2px 0 6px; color: var(--text-secondary); font-size: 12.5px; }
.readout { display: flex; gap: 4px 14px; flex-wrap: wrap; font-size: 13px; color: var(--text-secondary); margin-bottom: 2px; min-height: 20px; }
.readout i { display: inline-block; width: 12px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 5px; }
.readout b { color: var(--text-primary); font-variant-numeric: tabular-nums; margin-left: 3px; }
.readout .ex { color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.chart { position: relative; touch-action: pan-y; }
svg { display: block; width: 100%; height: auto; overflow: visible; -webkit-tap-highlight-color: transparent;
  -webkit-touch-callout: none; -webkit-user-select: none; user-select: none; }
details { margin-top: 8px; }
summary { cursor: pointer; color: var(--text-secondary); font-size: 13.5px; padding: 6px 0; }
.tbl { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 5px 6px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
.note { color: var(--text-muted); font-size: 12px; margin-top: 18px; }
.drv { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 4px; }
.drv td { padding: 9px 6px; border-bottom: 1px solid var(--border); vertical-align: top; text-align: left; white-space: normal; }
.drv tr:last-child td { border-bottom: 0; }
.drv .nm { font-weight: 600; }
.drv .what { color: var(--text-secondary); font-size: 12px; margin-top: 2px; }
.drv .learned { color: var(--text-muted); font-size: 12px; }
.drv .val { white-space: nowrap; font-variant-numeric: tabular-nums; }
.drv .pos { color: var(--text-secondary); font-size: 12px; }
.drv .pts { width: 170px; }
.pbar { position: relative; height: 10px; background: var(--grid); border-radius: 4px; margin-top: 5px; }
.pbar::after { content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: var(--axis); }
.pbar span { position: absolute; top: 0; bottom: 0; border-radius: 4px; }
.pnum { font-variant-numeric: tabular-nums; font-weight: 600; }
.chg { font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; margin-left: 4px; white-space: nowrap; }
.chg.up { color: var(--rise-text); } .chg.down { color: var(--fall-text); } .chg.flat { color: var(--text-muted); }
.head2 { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.head2 .sc { font-size: 15px; font-weight: 700; font-variant-numeric: tabular-nums; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: -1px; margin-right: 4px; }
.sw.zone { background: var(--zone); border: 1px solid var(--axis); }
@media (max-width: 640px) {
  main { padding: 14px 12px 40px; }
  h1 { font-size: 18px; }
  .tabs { margin: 0 -12px 12px; padding: 0 12px; }
  .selbar { margin: -12px -12px 10px; padding: 8px 12px; }
  .card { padding: 12px 10px 6px; border-radius: 10px; }
  .bar { gap: 4px; }
  .bar button { flex: 1 1 0; min-width: 0; padding: 7px 0; font-size: 12.5px; }   /* 기간 버튼 8개가 한 줄에 */
  /* 오늘의 근거: 표 → 카드형 (이름·설명 한 줄, 아래에 값과 막대) */
  .drv, .drv tbody, .drv tr, .drv td { display: block; width: auto; }
  .drv tr { display: grid; grid-template-columns: 1fr 1.3fr; gap: 2px 10px; padding: 10px 2px; border-bottom: 1px solid var(--border); }
  .drv tr:last-child { border-bottom: 0; }
  .drv td { padding: 0; border: 0; }
  .drv td:first-child { grid-column: 1 / -1; margin-bottom: 4px; }
  .drv .pts { width: auto; }
}
</style>
</head>
<body>
<main>
  <h1>나스닥 방향성 추이</h1>
  <p class="sub" id="sub"></p>
  <div class="tabs" role="tablist" aria-label="보기">
    <button role="tab" id="tab-today" aria-controls="panel-today" aria-selected="true">오늘의 근거</button>
    <button role="tab" id="tab-chart" aria-controls="panel-chart" aria-selected="false">차트</button>
  </div>
  <div role="tabpanel" id="panel-today" aria-labelledby="tab-today"><div id="today"></div></div>
  <div role="tabpanel" id="panel-chart" aria-labelledby="tab-chart" hidden>
  <div class="selbar" id="selbar" aria-live="polite">
    <span><b id="sel-date"></b> <span class="flags" id="sel-flags"></span></span>
  </div>
  <div class="bar" role="group" aria-label="기간">
    <button data-n="21" aria-pressed="true">1개월</button><button data-n="63">3개월</button><button data-n="126">6개월</button>
    <button data-n="252">1년</button><button data-n="1260">5년</button><button data-n="2520">10년</button><button data-n="5040">20년</button><button data-n="0">전체</button>
  </div>
  <section class="card"><h2>나스닥100 지수</h2><p>일별 종가 · <span class="sw zone"></span>매수 구간 (중기 점수 __ENTRY__ 이상 진입, __EXIT__ 미만 청산)</p><div class="chart" id="c-ndx"></div></section>
  <section class="card"><h2>방향성 점수</h2><p>-100 ~ +100 · 0 = 5년 기준 평소 수준 · (상승 n%) = 그 점수대의 2005년 이후 실제 상승 비율 · <span class="sw zone"></span>매수 구간</p><div class="chart" id="c-score"></div></section>
  <section class="card"><h2>카테고리 점수 (중기)</h2><p>-1 ~ +1 · 카테고리 안 지표 신호의 평균 · <span class="sw zone"></span>매수 구간</p><div class="chart" id="c-cat"></div></section>
  <details><summary>표로 보기 (최근 30일)</summary><div class="tbl" id="table"></div></details>
  </div>
  <p class="note">투자 판단의 참고 자료일 뿐 예측을 보장하지 않습니다.</p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const ALL = DATA.rows;
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmtDate = s => s.slice(2).replace(/-/g, ".");
  const WD = ["일", "월", "화", "수", "목", "금", "토"];
  const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const last = ALL[ALL.length - 1];
  document.getElementById("sub").textContent = `${last.date} 마감 기준 · 생성 ${DATA.generated}`;

  const CHARTS = [
    { id: "c-ndx", zone: true, logIfWide: true, series: [{ key: "ndx", name: "나스닥100", color: "--s1" }],
      fmt: v => (Math.abs(v) < 0.5 ? 0 : v).toLocaleString("ko-KR", { maximumFractionDigits: 0 }) },
    { id: "c-score", zone: true, domain: [-100, 100], ticks: [-100, -50, 0, 50, 100], ref: 0,
      series: [{ key: "short_score", name: "단기", color: "--s1", extra: "short_up" }, { key: "mid_score", name: "중기", color: "--s2", extra: "mid_up" }],
      fmt: v => { const r = Math.round(v); return r > 0 ? "+" + r : String(r === 0 ? 0 : r); } },
    { id: "c-cat", zone: true, domain: [-1, 1], ticks: [-1, -0.5, 0, 0.5, 1], ref: 0,
      series: [{ key: "mid_market", name: "시장", color: "--s1" }, { key: "mid_sentiment", name: "심리", color: "--s2" },
               { key: "mid_geo", name: "지정학", color: "--s3" }, { key: "mid_macro", name: "매크로", color: "--s4" }],
      fmt: v => Math.abs(v) < 0.005 ? "0.00" : (v > 0 ? "+" : "") + v.toFixed(2) },
  ];
  // 값이 하나도 없는 계열(예: 점수에서 뺀 카테고리)은 그리지 않음
  const ACTIVE = CHARTS.map(c => ({ ...c, series: c.series.filter(s => ALL.some(r => r[s.key] != null)) }));

  // 로그 눈금: 1·2·5 × 10^k 중 범위 안의 값
  function logTicks(lo, hi, n) {
    const t = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
      for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (v >= lo && v <= hi) t.push(v); }
    while (t.length > n + 1) { const k = t.filter((_, i) => i % 2 === 0); if (k.length === t.length) break; t.splice(0, t.length, ...k); }
    return t;
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
    const t = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(v); return t;
  }

  // ── 선택 상태: 4개 차트가 같은 날짜를 공유 ──
  let N = 21, ROWS = [], SEL = null, LOCK = false;   // SEL = 선택 인덱스(ROWS 기준), null = 최신
  // 보이는 구간 (ALL 기준 시작·끝 인덱스). 기간 버튼이 정하고, 두 손가락 확대·축소로 바뀜
  const viewFor = n => ({ s: n ? Math.max(0, ALL.length - n) : 0, e: ALL.length - 1 });
  let VIEW = viewFor(N), PINCH = null;
  const TOUCHING = new Set();        // 지금 화면에 닿아 있는 손가락(pointerId)
  const SINGLE_CANCEL = new Set();   // 차트별 한 손가락 동작 취소 함수 (두 번째 손가락이 닿으면 모두 호출)
  document.addEventListener("pointerdown", e => { if (e.pointerType !== "touch") return;
    TOUCHING.add(e.pointerId); if (TOUCHING.size >= 2) SINGLE_CANCEL.forEach(f => f()); }, true);
  const untouch = e => TOUCHING.delete(e.pointerId);
  document.addEventListener("pointerup", untouch, true); document.addEventListener("pointercancel", untouch, true);
  const DRAWN = [];

  function setSel(i, lock) {
    SEL = i == null ? null : Math.max(0, Math.min(ROWS.length - 1, i));
    if (lock != null) LOCK = lock;
    const idx = SEL == null ? ROWS.length - 1 : SEL, r = ROWS[idx];
    DRAWN.forEach(d => d.show(idx, SEL != null));
    const dt = new Date(r.date + "T00:00:00");
    document.getElementById("sel-date").textContent = `${r.date} (${WD[dt.getDay()]})`;
    const isLatest = r.date === ALL[ALL.length - 1].date;   // 확대해서 과거만 볼 때는 "최신" 아님
    const flags = [isLatest ? "최신" : "", r.zone ? "매수 구간" : "", r.live ? "발송" : ""].filter(Boolean);
    document.getElementById("sel-flags").textContent = flags.join(" · ");
  }

  function draw(cfg, rows) {
    const host = document.getElementById(cfg.id); host.innerHTML = "";
    const readout = document.createElement("div"); readout.className = "readout"; host.appendChild(readout);
    const W = host.clientWidth || 900, narrow = W < 520;
    const H = cfg.id === "c-ndx" ? (narrow ? 170 : 200) : (narrow ? 190 : 230);
    const fs = narrow ? 10.5 : 11;
    const endLabels = !narrow && cfg.series.length > 1;
    const m = { l: narrow ? 40 : 52, r: endLabels ? 58 : (narrow ? 8 : 16), t: 8, b: 24 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.id }, host);
    let lo, hi, ticks;
    if (cfg.domain) { [lo, hi] = cfg.domain; ticks = narrow ? cfg.ticks.filter((_, k) => k % 2 === 0) : cfg.ticks; }
    else {
      const vals = rows.flatMap(r => cfg.series.map(s => r[s.key])).filter(v => v != null);
      lo = Math.min(...vals); hi = Math.max(...vals);
      var log = cfg.logIfWide && lo > 0 && hi / lo > 4;   // 30년처럼 수십 배 오른 구간은 로그 눈금 (같은 % 변화 = 같은 높이)
      if (log) { lo /= 1.08; hi *= 1.08; ticks = logTicks(lo, hi, narrow ? 3 : 5); }
      else { const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad; ticks = niceTicks(lo, hi, narrow ? 3 : 4); }
    }
    const x = i => m.l + (rows.length <= 1 ? iw / 2 : (i / (rows.length - 1)) * iw);
    const y = log ? (v => m.t + (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * ih)
                  : (v => m.t + (1 - (v - lo) / (hi - lo)) * ih);
    const step = iw / Math.max(rows.length - 1, 1);

    if (cfg.zone) {   // 매수 구간 배경 (연속 구간을 하나의 띠로)
      let st = -1;
      rows.forEach((r, i) => {
        if (r.zone && st < 0) st = i;
        if ((!r.zone || i === rows.length - 1) && st >= 0) {
          const end = r.zone ? i : i - 1;
          el("rect", { x: x(st) - step / 2, y: m.t, width: Math.max(x(end) - x(st) + step, 2), height: ih, fill: css("--zone") }, svg);
          st = -1;
        }
      });
    }
    ticks.forEach(t => {
      el("line", { x1: m.l, x2: m.l + iw, y1: y(t), y2: y(t), stroke: css("--grid"), "stroke-width": 1 }, svg);
      el("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end", "font-size": fs, fill: css("--text-muted") }, svg).textContent = cfg.fmt(t);
    });
    if (cfg.ref != null) el("line", { x1: m.l, x2: m.l + iw, y1: y(cfg.ref), y2: y(cfg.ref), stroke: css("--ref"), "stroke-width": 1, "stroke-dasharray": "4 3" }, svg);
    el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: css("--axis"), "stroke-width": 1 }, svg);
    const nlab = Math.min(narrow ? 3 : 6, rows.length);
    for (let k = 0; k < nlab; k++) {
      const i = Math.round(k * (rows.length - 1) / Math.max(nlab - 1, 1));
      el("text", { x: x(i), y: H - 6, "text-anchor": k === 0 ? "start" : k === nlab - 1 ? "end" : "middle", "font-size": fs, fill: css("--text-muted") }, svg).textContent = fmtDate(rows[i].date);
    }
    const ends = [];
    cfg.series.forEach(s => {   // 선 (2px)
      let d = "", pen = false;
      rows.forEach((r, i) => { const v = r[s.key];
        if (v == null) { pen = false; return; }
        d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); pen = true; });
      el("path", { d, fill: "none", stroke: css(s.color), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      for (let i = rows.length - 1; i >= 0; i--) if (rows[i][s.key] != null) { ends.push({ s, v: rows[i][s.key] }); break; }
    });
    if (endLabels) {   // 넓은 화면: 선 끝 이름표
      ends.sort((a, b) => y(a.v) - y(b.v));
      let prev = -1e9;
      ends.forEach(e => { const ty = Math.max(y(e.v) + 4, prev + 13); prev = ty;
        el("text", { x: m.l + iw + 6, y: ty, "font-size": 11.5, fill: css("--text-secondary") }, svg).textContent = e.s.name; });
    }
    // 선택 표시: 세로선 + 점 (4개 차트 공통 날짜)
    const vline = el("line", { y1: m.t, y2: m.t + ih, stroke: css("--text-secondary"), "stroke-width": 1, visibility: "hidden" }, svg);
    const dots = cfg.series.map(s => el("circle", { r: narrow ? 5 : 4.5, fill: css(s.color), stroke: css("--surface-1"), "stroke-width": 2, visibility: "hidden" }, svg));
    const hit = el("rect", { x: m.l - 6, y: 0, width: iw + 12, height: H, fill: "transparent", "data-hit": "1", style: "touch-action: pan-y; cursor: crosshair" }, svg);

    const idxAt = evt => { const box = svg.getBoundingClientRect(), px = (evt.clientX - box.left) * (W / box.width);
      return Math.round((px - m.l) / iw * (rows.length - 1)); };
    // 마우스: 누르면 바로 선택. 터치: 손가락을 대는 것만으로는 선택하지 않음 (스크롤 중 오선택 방지)
    //  - 움직이지 않고 떼면(탭) 선택, 옆으로 8px 이상 밀면(가로가 세로보다 클 때) 날짜 이동
    //  - 세로로 움직이면 페이지 스크롤로 두고 아무것도 하지 않음 (브라우저가 pointercancel 을 보냄)
    //  - 꾹 누르기(LONG_PRESS ms, 움직임 없이) → 그 자리 선택 + 고르기 모드: 손을 뗄 때까지 좌우로 움직이면 날짜 이동,
    //    이 동안은 손가락이 위아래로 흔들려도 페이지가 스크롤되지 않음 (touchmove preventDefault)
    const SLOP = 8, LONG_PRESS = 350;
    let dragging = false, touch = null;   // touch = { x, y, id, scrub, timer }
    const endTouch = () => { if (touch) clearTimeout(touch.timer); touch = null; };
    SINGLE_CANCEL.add(endTouch);
    const startScrub = e => { touch.scrub = true; try { hit.setPointerCapture(e.pointerId); } catch (_) {} };
    hit.addEventListener("pointerdown", e => {
      if (e.pointerType === "mouse") { dragging = true; try { hit.setPointerCapture(e.pointerId); } catch (_) {} setSel(idxAt(e), true); return; }
      endTouch();
      if (PINCH || TOUCHING.size >= 2) return;            // 두 번째 손가락 → 확대·축소 (날짜 선택 안 함)
      const t = touch = { x: e.clientX, y: e.clientY, id: e.pointerId, scrub: false };
      t.timer = setTimeout(() => {                       // 꾹 누르기
        if (touch !== t || t.scrub) return;
        startScrub(e); setSel(idxAt(e), true);
        if (navigator.vibrate) navigator.vibrate(10);
      }, LONG_PRESS);
    });
    hit.addEventListener("pointermove", e => {
      if (PINCH || TOUCHING.size >= 2) { endTouch(); return; }
      if (touch && e.pointerId === touch.id) {
        const dx = Math.abs(e.clientX - touch.x), dy = Math.abs(e.clientY - touch.y);
        if (!touch.scrub && dx > SLOP && dx > dy) { clearTimeout(touch.timer); startScrub(e); }   // 옆으로 밀기
        else if (!touch.scrub && dy > SLOP) { endTouch(); return; }                               // 세로 = 스크롤
        if (touch && touch.scrub) setSel(idxAt(e), true);
        return;
      }
      if (dragging) setSel(idxAt(e), true);                                  // 마우스로 누른 채 끌기
      else if (e.pointerType === "mouse" && !LOCK) setSel(idxAt(e), false);  // 마우스 호버 미리보기
    });
    hit.addEventListener("pointerup", e => {
      if (!PINCH && touch && e.pointerId === touch.id && !touch.scrub) setSel(idxAt(e), true);   // 탭
      endTouch(); dragging = false;
    });
    hit.addEventListener("pointercancel", () => { endTouch(); dragging = false; });
    // 고르기 중에는 스크롤 막기 (pointer 이벤트만으로는 브라우저 스크롤을 못 막음)
    hit.addEventListener("touchmove", e => { if (touch && touch.scrub && e.cancelable) e.preventDefault(); }, { passive: false });
    svg.addEventListener("contextmenu", e => e.preventDefault());   // 꾹 누를 때 시스템 메뉴 방지
    hit.addEventListener("pointerleave", e => { if (e.pointerType === "mouse" && !LOCK && !dragging) setSel(null, false); });

    return { show(i, mark) {
      const r = rows[i];
      readout.innerHTML = cfg.series.map(s =>
        `<span><i style="background:${css(s.color)}"></i>${esc(s.name)}<b>${r[s.key] == null ? "-" : esc(cfg.fmt(r[s.key]))}</b>` +
        (s.extra && r[s.extra] != null ? ` <span class="ex">(상승 ${Math.round(r[s.extra])}%)</span>` : "") + `</span>`).join("");
      vline.setAttribute("x1", x(i)); vline.setAttribute("x2", x(i)); vline.setAttribute("visibility", mark ? "visible" : "hidden");
      cfg.series.forEach((s, k) => { const v = r[s.key];
        if (!mark || v == null) { dots[k].setAttribute("visibility", "hidden"); return; }
        dots[k].setAttribute("cx", x(i)); dots[k].setAttribute("cy", y(v)); dots[k].setAttribute("visibility", "visible"); });
    } };
  }

  // 긴 기간(5년·전체)은 약 1500개 지점으로 줄여 그림 — 휴대폰에서도 가볍게. 매수 구간·발송일은 줄인 구간 안에 하나라도 있으면 표시
  function thin(rows, max) {
    if (rows.length <= max) return rows;
    const k = Math.ceil(rows.length / max), out = [];
    for (let i = rows.length - 1; i >= 0; i -= k) {   // 최신 날짜가 반드시 포함되도록 끝에서부터
      const grp = rows.slice(Math.max(0, i - k + 1), i + 1);
      out.push({ ...rows[i], zone: grp.some(r => r.zone), live: grp.some(r => r.live) });
    }
    return out.reverse();
  }
  function render() {
    const keep = SEL == null ? null : ROWS[SEL] && ROWS[SEL].date;   // 기간을 바꿔도 같은 날짜 유지
    ROWS = thin(ALL.slice(VIEW.s, VIEW.e + 1), 1500);
    DRAWN.length = 0; SINGLE_CANCEL.clear();
    ACTIVE.forEach(c => DRAWN.push(draw(c, ROWS)));
    let j = keep ? ROWS.findIndex(r => r.date === keep) : -1;
    if (keep && j < 0) j = ROWS.findIndex(r => r.date >= keep);   // 줄인 구간에서는 가장 가까운 다음 날짜
    setSel(j >= 0 ? j : null, j >= 0 ? LOCK : false);
  }
  // 차트(그래프 영역) 밖을 '탭'하면 선택 해제 → 최신 날짜로. 스크롤(손가락 이동)과 기간 버튼은 해제하지 않음
  let outside = null;
  document.addEventListener("pointerdown", e => {
    outside = (e.target.closest && (e.target.closest("[data-hit]") || e.target.closest(".bar"))) ? null : { x: e.clientX, y: e.clientY };
  });
  document.addEventListener("pointerup", e => {
    if (outside && Math.abs(e.clientX - outside.x) < 8 && Math.abs(e.clientY - outside.y) < 8 && SEL != null &&
        !document.getElementById("panel-chart").hidden) setSel(null, false);
    outside = null;
  });
  document.addEventListener("pointercancel", () => { outside = null; });
  document.querySelectorAll(".bar button").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(".bar button").forEach(o => o.setAttribute("aria-pressed", "false"));
    b.setAttribute("aria-pressed", "true"); N = +b.dataset.n; VIEW = viewFor(N); render(); }));

  // ── 확대·축소: 두 손가락 핀치(휴대폰), 트랙패드 핀치·Ctrl+휠(PC). 세 차트가 같은 구간으로 함께 바뀜 ──
  const MIN_SPAN = 10;                       // 최소 10거래일까지 확대
  const plotFrac = (x, host) => { const h = host.querySelector("[data-hit]"); if (!h) return 0.5;
    const b = h.getBoundingClientRect(); return Math.min(1, Math.max(0, (x - b.left) / b.width)); };
  let raf = 0;
  const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; render(); }); };
  function setView(anchor, frac, span) {     // anchor 날짜(ALL 인덱스)가 화면의 frac 위치에 오도록
    span = Math.max(MIN_SPAN, Math.min(ALL.length - 1, Math.round(span)));
    const s = Math.max(0, Math.min(ALL.length - 1 - span, Math.round(anchor - frac * span)));
    if (s === VIEW.s && s + span === VIEW.e) return;
    VIEW = { s, e: s + span };
    document.querySelectorAll(".bar button").forEach(o => o.setAttribute("aria-pressed", "false"));   // 기간 버튼 누르면 원래대로
    schedule();
  }
  // 휴대폰: 터치 이벤트로 화면의 두 손가락을 한꺼번에 읽음 (차트 상자 안 어디든). 페이지 확대·스크롤은 막음
  const two = e => [e.touches[0].clientX, e.touches[1].clientX];
  function startPinch(e, host) {
    const [x1, x2] = two(e), frac = plotFrac((x1 + x2) / 2, host);
    SINGLE_CANCEL.forEach(f => f());
    PINCH = { d0: Math.max(Math.abs(x1 - x2), 40), span0: VIEW.e - VIEW.s, anchor: VIEW.s + frac * (VIEW.e - VIEW.s), host };
  }
  function movePinch(e) {
    const [x1, x2] = two(e), d = Math.max(Math.abs(x1 - x2), 40);
    setView(PINCH.anchor, plotFrac((x1 + x2) / 2, PINCH.host), PINCH.span0 * PINCH.d0 / d);   // 벌리면 확대, 모으면 축소, 같이 밀면 이동
  }
  document.querySelectorAll("#panel-chart .chart").forEach(host => {
    host.addEventListener("touchstart", e => {
      if (e.touches.length >= 2) { if (e.cancelable) e.preventDefault(); startPinch(e, host); }
    }, { passive: false });
    host.addEventListener("touchmove", e => {
      if (e.touches.length >= 2) { if (e.cancelable) e.preventDefault(); if (!PINCH) startPinch(e, host); movePinch(e); }
    }, { passive: false });
    const end = e => { if (e.touches.length < 2) PINCH = null; };
    host.addEventListener("touchend", end); host.addEventListener("touchcancel", end);
    // 아이폰 사파리: 두 손가락 제스처로 페이지 전체가 확대되는 것 방지
    ["gesturestart", "gesturechange", "gestureend"].forEach(t => host.addEventListener(t, e => e.preventDefault(), { passive: false }));
  });
  document.addEventListener("wheel", e => {   // PC: 트랙패드 핀치(=ctrlKey 휠) 또는 Ctrl+휠
    if (!e.ctrlKey) return;
    const h = e.target.closest && e.target.closest("[data-hit]"); if (!h) return;
    e.preventDefault();
    const host = h.closest(".chart"), frac = plotFrac(e.clientX, host), span = VIEW.e - VIEW.s;
    setView(VIEW.s + frac * span, frac, span * Math.exp(e.deltaY * 0.01));
  }, { passive: false });
  let rt, lastW = window.innerWidth;
  window.addEventListener("resize", () => { if (window.innerWidth === lastW) return; lastW = window.innerWidth;   // 모바일 주소창 높이 변화는 무시
    clearTimeout(rt); rt = setTimeout(() => { if (!document.getElementById("panel-chart").hidden) render(); }, 150); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!document.getElementById("panel-chart").hidden) render(); });

  function table() {
    const rows = ALL.slice(-30).reverse();
    const f = (v, d) => v == null ? "-" : (d ? v.toFixed(d) : Math.round(v));
    const hasGeo = ALL.some(r => r.mid_geo != null);
    let h = "<table><thead><tr><th>날짜</th><th>나스닥100</th><th>단기 점수</th><th>중기 점수</th><th>단기 상승%</th><th>중기 상승%</th><th>시장</th><th>심리</th>" + (hasGeo ? "<th>지정학</th>" : "") + "<th>매크로</th></tr></thead><tbody>";
    rows.forEach(r => { h += `<tr><td>${r.date}${r.live ? " ●" : ""}</td><td>${f(r.ndx)}</td><td>${f(r.short_score)}</td><td>${f(r.mid_score)}</td><td>${f(r.short_up)}</td><td>${f(r.mid_up)}</td><td>${f(r.mid_market, 2)}</td><td>${f(r.mid_sentiment, 2)}</td>` + (hasGeo ? `<td>${f(r.mid_geo, 2)}</td>` : "") + `<td>${f(r.mid_macro, 2)}</td></tr>`; });
    document.getElementById("table").innerHTML = h + "</tbody></table><p class='note'>● = 실제 발송된 날</p>";
  }

  // 전일 대비 변화: +는 빨강, -는 파랑 (한국 주식 화면과 같은 색)
  const chg = (v, dir) => v == null ? "" : `<span class="chg ${dir}" title="전일 대비">${esc(v)}</span>`;
  function today() {
    const T = DATA.today || {}, host = document.getElementById("today");
    Object.keys(T).forEach(h => {
      const t = T[h];
      let html = `<section class="card"><div class="head2"><h2>${esc(t.title)}</h2>` +
        `<span class="sc">${t.score > 0 ? "+" : ""}${Math.round(t.score)}${chg(t.score_change, t.score_dir)} <span class="pos">${esc(t.position)}</span></span></div>` +
        `<table class="drv"><tbody>`;
      t.drivers.forEach(d => {
        const frac = d.max ? Math.max(-1, Math.min(1, d.points / d.max)) : 0;
        const w = Math.abs(frac) * 50, left = frac >= 0 ? 50 : 50 - w;
        const col = frac >= 0 ? "var(--rise)" : "var(--fall)";   // 상승 쪽 빨강, 하락 쪽 파랑
        html += `<tr><td><div class="nm">${esc(d.label)}</div><div class="what">${esc(d.what)}</div><div class="learned">${esc(d.learned)}</div></td>` +
          `<td class="val">${esc(d.value)}${chg(d.value_change, d.value_dir)}<div class="pos">${esc(d.position)}</div></td>` +
          `<td class="pts"><span class="pnum">${d.points > 0 ? "+" : ""}${Math.round(d.points)}</span> / ${d.max == null ? "-" : Math.round(d.max)}점${chg(d.points_change, d.points_dir)}` +
          `<div class="pbar" role="img" aria-label="${Math.round(d.points)}점"><span style="left:${left}%;width:${w}%;background:${col}"></span></div></td></tr>`;
      });
      host.insertAdjacentHTML("beforeend", html + "</tbody></table></section>");
    });
  }

  // 탭: 오늘의 근거 / 차트 (주소 #chart 로 차트 탭 바로 열기). 숨겨진 상태에선 폭이 0이라 탭을 열 때 다시 그림
  const TABS = { today: ["tab-today", "panel-today"], chart: ["tab-chart", "panel-chart"] };
  function show(name, push) {
    Object.entries(TABS).forEach(([k, [t, p]]) => {
      const on = k === name;
      document.getElementById(t).setAttribute("aria-selected", on);
      document.getElementById(t).tabIndex = on ? 0 : -1;
      document.getElementById(p).hidden = !on;
    });
    if (name === "chart") render();
    if (push) history.replaceState(null, "", name === "chart" ? "#chart" : "#");
  }
  Object.entries(TABS).forEach(([k, [t]]) => document.getElementById(t).addEventListener("click", () => show(k, true)));
  document.querySelector(".tabs").addEventListener("keydown", e => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const next = document.getElementById("tab-today").getAttribute("aria-selected") === "true" ? "chart" : "today";
    show(next, true); document.getElementById(TABS[next][0]).focus();
  });
  table(); today(); show(location.hash === "#chart" ? "chart" : "today", false);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
