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

CATEGORIES = ("market", "sentiment", "geo", "macro")


def build_history(cfg: dict, md: MarketData, days: int = 750) -> pd.DataFrame:
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
    return out.iloc[-days:]


def _rows(h: pd.DataFrame) -> list[dict]:
    rows = []
    for d, r in h.iterrows():
        row = {"date": d.date().isoformat(), "live": bool(r["live"])}
        for k, v in r.items():
            if k == "live":
                continue
            row[k] = None if pd.isna(v) else round(float(v), 4)
        rows.append(row)
    return rows


def render_html(h: pd.DataFrame, generated: str) -> str:
    payload = json.dumps({"generated": generated, "rows": _rows(h)}, ensure_ascii=False)
    return _TEMPLATE.replace("__DATA__", payload.replace("</", "<\\/"))


def write_dashboard(cfg: dict, md: MarketData, generated: str) -> Path:
    path = Path(cfg["paths"].get("dashboard", Path(cfg["paths"]["state"]) / "dashboard.html"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(build_history(cfg, md), generated), encoding="utf-8")
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
    md = collect_all(cfg, (pd.Timestamp.today() - pd.DateOffset(years=7)).date().isoformat())
    path = write_dashboard(cfg, md, datetime.now().strftime("%Y-%m-%d %H:%M"))
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
  --grid: #ecebe7; --axis: #c9c8c2; --ref: #9a9993; --live: #f0efec;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2e2c;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8e8d86;
    --grid: #262624; --axis: #45443f; --ref: #6f6e68; --live: #262624;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2e2c;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8e8d86;
  --grid: #262624; --axis: #45443f; --ref: #6f6e68; --live: #262624;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.5 -apple-system, "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; font-size: 13px; }
.bar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px; }
.bar button { border: 1px solid var(--border); background: var(--surface-1); color: var(--text-secondary);
  border-radius: 8px; padding: 6px 12px; font: inherit; font-size: 13px; cursor: pointer; }
.bar button[aria-pressed="true"] { color: var(--text-primary); border-color: var(--text-secondary); font-weight: 600; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 14px 14px 8px; margin-bottom: 14px; }
.card h2 { font-size: 15px; margin: 0; }
.card p { margin: 2px 0 8px; color: var(--text-secondary); font-size: 12.5px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12.5px; color: var(--text-secondary); margin-bottom: 4px; }
.legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 5px; }
.chart { position: relative; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 9px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.12); white-space: nowrap; display: none; z-index: 2; }
.tip b { display: block; margin-bottom: 2px; }
.tip span { display: block; color: var(--text-secondary); }
.tip em { font-style: normal; color: var(--text-primary); font-variant-numeric: tabular-nums; }
.tip i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
details { margin-top: 8px; }
summary { cursor: pointer; color: var(--text-secondary); font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 4px 6px; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
.note { color: var(--text-muted); font-size: 12px; margin-top: 18px; }
</style>
</head>
<body>
<main>
  <h1>나스닥 방향성 추이</h1>
  <p class="sub" id="sub"></p>
  <div class="bar" role="group" aria-label="기간">
    <button data-n="21">1개월</button><button data-n="63">3개월</button><button data-n="126" aria-pressed="true">6개월</button>
    <button data-n="252">1년</button><button data-n="0">전체</button>
  </div>
  <section class="card"><h2>나스닥100 지수</h2><p>일별 종가</p><div class="chart" id="c-ndx"></div></section>
  <section class="card"><h2>방향성 점수</h2><p>-100 ~ +100 · 0 = 5년 기준 평소 수준</p><div class="chart" id="c-score"></div></section>
  <section class="card"><h2>상승 확률 (%)</h2><p>점수 구간별 과거(2018~) 실제 결과로 계산 · 50% 기준선</p><div class="chart" id="c-up"></div></section>
  <section class="card"><h2>카테고리 점수 (중기)</h2><p>-1 ~ +1 · 카테고리 안 지표 신호의 평균</p><div class="chart" id="c-cat"></div></section>
  <details><summary>표로 보기 (최근 30일)</summary><div id="table"></div></details>
  <p class="note">음영 구간은 실제로 카카오톡이 발송된 날입니다. 그 이전 값은 현재 설정으로 과거 데이터를 다시 계산한 것이라,
  당시 실제로 받았을 메시지보다 잘 맞아 보일 수 있습니다. 투자 판단의 참고 자료일 뿐 예측을 보장하지 않습니다.</p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const ALL = DATA.rows;
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmtDate = s => s.slice(2).replace(/-/g, ".");
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const last = ALL[ALL.length - 1];
  document.getElementById("sub").textContent =
    `${last.date} 마감 기준 · 생성 ${DATA.generated}`;

  const CHARTS = [
    { id: "c-ndx", series: [{ key: "ndx", name: "나스닥100", color: "--s1" }], fmt: v => v.toLocaleString("ko-KR", { maximumFractionDigits: 0 }) },
    { id: "c-score", domain: [-100, 100], ticks: [-100, -50, 0, 50, 100], ref: 0,
      series: [{ key: "short_score", name: "단기", color: "--s1" }, { key: "mid_score", name: "중기", color: "--s2" }],
      fmt: v => (v > 0 ? "+" : "") + Math.round(v) },
    { id: "c-up", domain: [0, 100], ticks: [0, 25, 50, 75, 100], ref: 50,
      series: [{ key: "short_up", name: "단기 상승", color: "--s1" }, { key: "mid_up", name: "중기 상승", color: "--s2" }],
      fmt: v => Math.round(v) + "%" },
    { id: "c-cat", domain: [-1, 1], ticks: [-1, -0.5, 0, 0.5, 1], ref: 0,
      series: [{ key: "mid_market", name: "시장", color: "--s1" }, { key: "mid_sentiment", name: "심리", color: "--s2" },
               { key: "mid_geo", name: "지정학", color: "--s3" }, { key: "mid_macro", name: "매크로", color: "--s4" }],
      fmt: v => (v > 0 ? "+" : "") + v.toFixed(2) },
  ];

  function niceTicks(lo, hi) {
    const span = hi - lo || 1, step0 = span / 4, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
    const t = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(v); return t;
  }

  function draw(cfg, rows) {
    const host = document.getElementById(cfg.id); host.innerHTML = "";
    if (cfg.series.length > 1) {
      const lg = document.createElement("div"); lg.className = "legend";
      cfg.series.forEach(s => { const it = document.createElement("span");
        it.innerHTML = `<i style="background:${css(s.color)}"></i>${s.name}`; lg.appendChild(it); });
      host.appendChild(lg);
    }
    const W = host.clientWidth || 900, H = cfg.id === "c-ndx" ? 200 : 230;
    const m = { l: 52, r: cfg.series.length > 1 ? 58 : 16, t: 10, b: 26 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.id }, host);
    let lo, hi, ticks;
    if (cfg.domain) { [lo, hi] = cfg.domain; ticks = cfg.ticks; }
    else {
      const vals = rows.flatMap(r => cfg.series.map(s => r[s.key])).filter(v => v != null);
      lo = Math.min(...vals); hi = Math.max(...vals); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
      ticks = niceTicks(lo, hi);
    }
    const x = i => m.l + (rows.length <= 1 ? iw / 2 : (i / (rows.length - 1)) * iw);
    const y = v => m.t + (1 - (v - lo) / (hi - lo)) * ih;

    // 실제 발송된 날 음영
    rows.forEach((r, i) => { if (r.live) {
      const w = Math.max(iw / Math.max(rows.length - 1, 1), 2);
      el("rect", { x: x(i) - w / 2, y: m.t, width: w, height: ih, fill: css("--live") }, svg); } });
    ticks.forEach(t => {
      el("line", { x1: m.l, x2: m.l + iw, y1: y(t), y2: y(t), stroke: css("--grid"), "stroke-width": 1 }, svg);
      const lab = el("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", "font-size": 11, fill: css("--text-muted") }, svg);
      lab.textContent = cfg.fmt(t);
    });
    if (cfg.ref != null) el("line", { x1: m.l, x2: m.l + iw, y1: y(cfg.ref), y2: y(cfg.ref), stroke: css("--ref"), "stroke-width": 1, "stroke-dasharray": "4 3" }, svg);
    el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: css("--axis"), "stroke-width": 1 }, svg);
    const nlab = Math.min(6, rows.length);
    for (let k = 0; k < nlab; k++) {
      const i = Math.round(k * (rows.length - 1) / Math.max(nlab - 1, 1));
      const t = el("text", { x: x(i), y: H - 6, "text-anchor": k === 0 ? "start" : k === nlab - 1 ? "end" : "middle", "font-size": 11, fill: css("--text-muted") }, svg);
      t.textContent = fmtDate(rows[i].date);
    }
    // 선 (2px) + 끝 이름표
    const ends = [];
    cfg.series.forEach(s => {
      let d = "", pen = false;
      rows.forEach((r, i) => { const v = r[s.key];
        if (v == null) { pen = false; return; }
        d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); pen = true; });
      el("path", { d, fill: "none", stroke: css(s.color), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      for (let i = rows.length - 1; i >= 0; i--) if (rows[i][s.key] != null) { ends.push({ s, i, v: rows[i][s.key] }); break; }
    });
    if (cfg.series.length > 1) {
      ends.sort((a, b) => y(a.v) - y(b.v));
      let prev = -1e9;
      ends.forEach(e => { let ty = Math.max(y(e.v) + 4, prev + 13); prev = ty;
        const t = el("text", { x: m.l + iw + 6, y: ty, "font-size": 11.5, fill: css("--text-secondary") }, svg);
        t.textContent = e.s.name; });
    }
    // 호버: 세로선 + 점 + 툴팁
    const vline = el("line", { y1: m.t, y2: m.t + ih, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" }, svg);
    const dots = cfg.series.map(s => el("circle", { r: 4.5, fill: css(s.color), stroke: css("--surface-1"), "stroke-width": 2, visibility: "hidden" }, svg));
    const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
    const tip = document.createElement("div"); tip.className = "tip"; host.appendChild(tip);
    function move(evt) {
      const box = svg.getBoundingClientRect(), px = (evt.clientX - box.left) * (W / box.width);
      const i = Math.max(0, Math.min(rows.length - 1, Math.round((px - m.l) / iw * (rows.length - 1))));
      const r = rows[i];
      vline.setAttribute("x1", x(i)); vline.setAttribute("x2", x(i)); vline.setAttribute("visibility", "visible");
      cfg.series.forEach((s, k) => { const v = r[s.key];
        if (v == null) { dots[k].setAttribute("visibility", "hidden"); return; }
        dots[k].setAttribute("cx", x(i)); dots[k].setAttribute("cy", y(v)); dots[k].setAttribute("visibility", "visible"); });
      tip.innerHTML = `<b>${r.date}${r.live ? " · 발송" : ""}</b>` + cfg.series.map(s =>
        `<span><i style="background:${css(s.color)}"></i>${s.name} <em>${r[s.key] == null ? "-" : cfg.fmt(r[s.key])}</em></span>`).join("");
      tip.style.display = "block";
      const hx = x(i) * (box.width / W), tw = tip.offsetWidth;
      tip.style.left = (hx + 12 + tw > box.width ? hx - tw - 12 : hx + 12) + "px";
      tip.style.top = (host.querySelector(".legend") ? 24 : 0) + 8 + "px";
    }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", () => { tip.style.display = "none"; vline.setAttribute("visibility", "hidden");
      dots.forEach(d => d.setAttribute("visibility", "hidden")); });
  }

  function table() {
    const rows = ALL.slice(-30).reverse();
    const f = (v, d) => v == null ? "-" : (d ? v.toFixed(d) : Math.round(v));
    let h = "<table><thead><tr><th>날짜</th><th>나스닥100</th><th>단기 점수</th><th>중기 점수</th><th>단기 상승%</th><th>중기 상승%</th><th>시장</th><th>심리</th><th>지정학</th><th>매크로</th></tr></thead><tbody>";
    rows.forEach(r => { h += `<tr><td>${r.date}${r.live ? " ●" : ""}</td><td>${f(r.ndx)}</td><td>${f(r.short_score)}</td><td>${f(r.mid_score)}</td><td>${f(r.short_up)}</td><td>${f(r.mid_up)}</td><td>${f(r.mid_market, 2)}</td><td>${f(r.mid_sentiment, 2)}</td><td>${f(r.mid_geo, 2)}</td><td>${f(r.mid_macro, 2)}</td></tr>`; });
    document.getElementById("table").innerHTML = h + "</tbody></table><p class='note'>● = 실제 발송된 날</p>";
  }

  let N = 126;
  function render() { const rows = N ? ALL.slice(-N) : ALL; CHARTS.forEach(c => draw(c, rows)); }
  document.querySelectorAll(".bar button").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(".bar button").forEach(o => o.setAttribute("aria-pressed", "false"));
    b.setAttribute("aria-pressed", "true"); N = +b.dataset.n; render(); }));
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(render, 120); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  render(); table();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
