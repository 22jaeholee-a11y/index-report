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

from .config import active_indicators, load_config
from .data import DataSourceError, MarketData, collect_all
from .dashboard import write_dashboard
from .explain import last_percentiles, top_drivers
from .indicators import compute_raw, mask_missing, to_signals
from .kakao import EncryptedFileTokenStore, FileTokenStore, KakaoAuthError, KakaoClient
from .message import HORIZON_KO, Report, format_messages, position_text
from .zone import zone_info
from .scoring import (Calibration, base_probabilities, load_score_distributions, score_frame,
                      score_percentile)

log = logging.getLogger("nasdaq_signal")
KST = timezone(timedelta(hours=9))  # 한국은 서머타임 없음
ERROR_TEXT = "[NASDAQ] 오류: 핵심 데이터 수집 실패 — 로그 확인"
UNEXPECTED_TEXT = "[NASDAQ] 오류: 실행 중 예외 발생 — 로그 확인"
HISTORY_YEARS = 7


def is_market_closed(as_of: date, run_date_kst: date) -> bool:
    return as_of != run_date_kst - timedelta(days=1)


def _last(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].iloc[-1]) if col in frame else math.nan


def build_report(cfg: dict, md: MarketData, run_date: date, reauth_days: int | None) -> tuple[Report, dict]:
    raw = mask_missing(compute_raw(md.frame), md.missing)
    signals = to_signals(raw, cfg)
    sf = score_frame(signals, cfg)
    last = sf.iloc[-1]
    zone_rule = (cfg["zone"]["entry"], cfg["zone"]["exit"])
    zone = zone_info(sf["mid_score"], md.frame["ndx"], *zone_rule)
    cal_path = cfg["paths"]["calibration"]
    stale = not Calibration.is_valid_for(Calibration.load_meta(cal_path), cfg)
    cals = {} if stale else (Calibration.load_all(cal_path) or {})
    unpredicted = {h for h, spec in cfg["horizons"].items() if spec.get("predict", True) is False}
    base = {} if stale else base_probabilities(cal_path)
    scores, probs = {}, {}
    for h in cfg["horizons"]:
        s = float(last[f"{h}_score"])
        scores[h] = s
        if h in unpredicted:  # 검증된 예측 신호가 없는 기간: 점수 대신 역사적 기본 비율
            probs[h] = base.get(h)
        else:
            probs[h] = cals[h].probabilities(s) if h in cals and not math.isnan(s) else None
    predicted = [h for h in cfg["horizons"] if h not in unpredicted]
    # 중기에서 쓰이는 지표가 있는 카테고리만 표시
    cats = [c for c in cfg["category_weights"]
            if any(cfg["indicators"][n]["category"] == c for n in active_indicators(cfg, "mid"))]
    pct = last_percentiles(raw, cfg)
    # 해당 기간에 쓰이는 지표 전부 (기여 점수 큰 순)
    has_prev = len(raw) > 1   # 전 거래일 값 (웹 '오늘의 근거'의 전일 대비 표시용)
    drivers = {h: top_drivers(signals[h].iloc[-1], raw.iloc[-1], pct, cfg, h, n=len(cfg["indicators"]), min_points=0,
                              prev_sig_row=signals[h].iloc[-2] if has_prev else None,
                              prev_raw_row=raw.iloc[-2] if has_prev else None)
               for h in predicted}
    prev_scores = {h: float(sf[f"{h}_score"].iloc[-2]) for h in predicted} if has_prev else {}
    dists = {} if stale else load_score_distributions(cal_path)
    score_pct = {h: score_percentile(dists[h], scores[h]) for h in predicted
                 if h in dists and not math.isnan(scores[h])}
    report = Report(
        as_of=md.as_of, run_date=run_date, market_closed=is_market_closed(md.as_of, run_date),
        scores=scores, probs=probs,
        categories={c: float(last[f"mid_{c}"]) for c in cats},
        key_values={"vix": _last(md.frame, "vix"), "us10y": _last(md.frame, "dgs10"),
                    "hy_spread": _last(md.frame, "bamlh0a0hym2")},
        fear_greed=md.extras.get("fear_greed"), nq_change=md.extras.get("nq_change"),
        missing=list(md.missing),
        low_confidence=any(float(last[f"{h}_missing"]) > cfg["low_confidence_missing"] for h in predicted),
        reauth_days=reauth_days,
        calibration_stale=stale,
        drivers=drivers,
        unpredicted=unpredicted,
        score_pct=score_pct,
        zone=zone,
        zone_rule=zone_rule,
        page_url=cfg["kakao"].get("page_url"),
        prev_scores=prev_scores,
    )
    row = {"date": run_date.isoformat(), "as_of": md.as_of.isoformat()}
    row.update({f"raw_{k}": v for k, v in raw.iloc[-1].items()})
    row.update({k: v for k, v in last.items()})
    row.update({"fear_greed": report.fear_greed, "nq_change": report.nq_change,
                "missing": ";".join(md.missing)})
    return report, row


def today_drivers(report: Report, cfg: dict) -> dict:
    """웹 '오늘의 근거' 섹션용: 예측하는 기간별 점수·위치·근거 지표."""
    return {h: {"title": f"{HORIZON_KO[h]}적", "score": report.scores[h], "score_prev": report.prev_scores.get(h),
                "position": position_text(report, h, cfg["arrow_threshold"]), "drivers": report.drivers[h]}
            for h in report.drivers}


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
    token_key = os.environ.get("KAKAO_TOKEN_KEY")
    # 키가 있으면(클라우드 포함) 저장소에 커밋되는 암호화 토큰 파일, 없으면 로컬 평문 파일
    store = (EncryptedFileTokenStore(cfg["paths"]["tokens_enc"], token_key) if token_key
             else FileTokenStore(cfg["paths"]["tokens"]))
    return KakaoClient(key, store, os.environ.get("KAKAO_CLIENT_SECRET") or None)


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
        # 차트 30년 표시를 위해 오래전부터 받음 (오늘 점수는 최근 5년 창만 쓰므로 결과 동일)
        start = cfg.get("data_start") or (run_date - timedelta(days=365 * HISTORY_YEARS)).isoformat()
        md = collect_all(cfg, start)
        reauth = client.days_until_reauth() if client else None
        report, row = build_report(cfg, md, run_date, reauth)
        deliver(format_messages(report, cfg["arrow_threshold"], reasons=cfg["kakao"].get("send_reasons", False)))
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
    try:  # 차트 페이지 생성 실패도 발송을 막지 않음 (발송 기록 반영 위해 history 저장 뒤)
        write_dashboard(cfg, md, datetime.now(KST).strftime("%Y-%m-%d %H:%M"), today_drivers(report, cfg))
    except Exception:
        log.exception("dashboard write failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
