"""일일 실행: python -m nasdaq_signal.main [--dry-run] [--config PATH]"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .config import load_config
from .data import DataSourceError, MarketData, collect_all
from .explain import last_percentiles, top_drivers
from .indicators import compute_raw, mask_missing, to_signals
from .kakao import FileTokenStore, KakaoAuthError, KakaoClient
from .message import Report, format_messages
from .scoring import Calibration, score_frame

log = logging.getLogger("nasdaq_signal")
KST = timezone(timedelta(hours=9))  # 한국은 서머타임 없음
ERROR_TEXT = "[나스닥 방향성] 오류: 핵심 데이터 수집 실패 — 로그 확인"
UNEXPECTED_TEXT = "[나스닥 방향성] 오류: 실행 중 예외 발생 — 로그 확인"
HISTORY_YEARS = 7


def is_market_closed(as_of: date, run_date_kst: date) -> bool:
    return as_of != run_date_kst - timedelta(days=1)


def _last(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].iloc[-1]) if col in frame else math.nan


def build_report(cfg: dict, md: MarketData, run_date: date, reauth_days: int | None) -> tuple[Report, dict]:
    raw = mask_missing(compute_raw(md.frame), md.missing)
    signals = to_signals(raw, cfg)
    last = score_frame(signals, cfg).iloc[-1]
    cal_path = cfg["paths"]["calibration"]
    stale = not Calibration.is_valid_for(Calibration.load_meta(cal_path), cfg)
    cals = {} if stale else (Calibration.load_all(cal_path) or {})
    scores, probs = {}, {}
    for h in cfg["horizons"]:
        s = float(last[f"{h}_score"])
        scores[h] = s
        probs[h] = cals[h].probabilities(s) if h in cals and not math.isnan(s) else None
    cats = sorted({ind["category"] for ind in cfg["indicators"].values()})
    pct = last_percentiles(raw, cfg)
    drivers = {h: top_drivers(signals[h].iloc[-1], raw.iloc[-1], pct, cfg, h) for h in cfg["horizons"]}
    report = Report(
        as_of=md.as_of, run_date=run_date, market_closed=is_market_closed(md.as_of, run_date),
        scores=scores, probs=probs,
        categories={c: float(last[f"mid_{c}"]) for c in cats},
        key_values={"vix": _last(md.frame, "vix"), "us10y": _last(md.frame, "dgs10"),
                    "hy_spread": _last(md.frame, "bamlh0a0hym2")},
        fear_greed=md.extras.get("fear_greed"), nq_change=md.extras.get("nq_change"),
        missing=list(md.missing),
        low_confidence=any(float(last[f"{h}_missing"]) > cfg["low_confidence_missing"] for h in cfg["horizons"]),
        reauth_days=reauth_days,
        calibration_stale=stale,
        drivers=drivers,
    )
    row = {"date": run_date.isoformat(), "as_of": md.as_of.isoformat()}
    row.update({f"raw_{k}": v for k, v in raw.iloc[-1].items()})
    row.update({k: v for k, v in last.items()})
    row.update({"fear_greed": report.fear_greed, "nq_change": report.nq_change,
                "missing": ";".join(md.missing)})
    return report, row


def append_history(path: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if p.exists():
        old = pd.read_csv(p, encoding="utf-8-sig", dtype={"date": str})
        old = old[old["date"] != row["date"]]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values("date").to_csv(p, index=False, encoding="utf-8-sig")


def make_client(cfg: dict) -> KakaoClient:
    key = os.environ.get("KAKAO_REST_API_KEY")
    if not key:
        raise KakaoAuthError("KAKAO_REST_API_KEY 가 설정되지 않았습니다 (.env)")
    return KakaoClient(key, FileTokenStore(cfg["paths"]["tokens"]), os.environ.get("KAKAO_CLIENT_SECRET") or None)


def _setup_logging(log_dir: str, run_date: date) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(log_dir) / f"{run_date.isoformat()}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="나스닥 방향성 일일 알림")
    ap.add_argument("--dry-run", action="store_true", help="발송 없이 콘솔 출력")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-date", help="KST 실행일 지정(YYYY-MM-DD, 테스트용)")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    cfg = load_config(args.config)
    run_date = date.fromisoformat(args.run_date) if args.run_date else datetime.now(KST).date()
    _setup_logging(cfg["paths"]["logs"], run_date)
    link = cfg["kakao"]["link_url"]

    client = None
    if not args.dry_run:
        try:
            client = make_client(cfg)
            client.refresh()
        except KakaoAuthError as e:
            log.error("%s", e)
            print(e, file=sys.stderr)
            return 2

    def deliver(texts: list[str]) -> None:
        for t in texts:
            if client is None:
                print(t, end="\n\n")
            else:
                client.send_text(t, link)

    try:
        start = (run_date - timedelta(days=365 * HISTORY_YEARS)).isoformat()
        md = collect_all(cfg, start)
        reauth = client.days_until_reauth() if client else None
        report, row = build_report(cfg, md, run_date, reauth)
        deliver(format_messages(report, cfg["arrow_threshold"]))
        log.info("%s: scores=%s missing=%s", "dry-run" if args.dry_run else "sent",
                 report.scores, report.missing)
    except Exception as e:  # 어떤 실패든 로그 + 오류 알림 (조용한 실패 금지)
        log.exception("run failed: %s", e)
        try:
            deliver([ERROR_TEXT if isinstance(e, DataSourceError) else UNEXPECTED_TEXT])
        except Exception:
            log.exception("error notification also failed")
        return 1

    if not args.dry_run:
        try:  # 기록 실패(예: 엑셀이 파일을 잡고 있음)는 발송을 막지 않음
            append_history(cfg["paths"]["history"], row)
        except Exception:
            log.exception("history write failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
