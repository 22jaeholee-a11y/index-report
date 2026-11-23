"""백테스트: 과거 점수 재계산, 학습/검증 성과, 검증 구간 보정표 저장.

실행: python -m nasdaq_signal.backtest
"""
from __future__ import annotations

import pandas as pd

from .indicators import compute_raw, to_signals
from .scoring import Calibration, config_fingerprint, score_frame


def forward_labels(ndx: pd.Series, days: int, up: float, down: float) -> pd.Series:
    fwd = ndx.shift(-days) / ndx - 1
    lab = pd.Series(pd.NA, index=ndx.index, dtype=object)
    lab[fwd > up] = "up"
    lab[fwd < down] = "down"
    lab[(fwd >= down) & (fwd <= up)] = "flat"
    lab[fwd.isna()] = None
    return lab


def _stats(score: pd.Series, label: pd.Series, fwd: pd.Series, threshold: float) -> dict:
    df = pd.DataFrame({"s": score, "l": label, "r": fwd}).dropna()
    if df.empty:
        return {"n": 0, "hit_rate": None, "corr": None}
    pred = pd.Series("flat", index=df.index)
    pred[df["s"] >= threshold] = "up"
    pred[df["s"] <= -threshold] = "down"
    return {
        "n": int(len(df)),
        "hit_rate": float((pred == df["l"]).mean()),
        "corr": float(df["s"].rank().corr(df["r"].rank())) if df["s"].nunique() > 1 else None,
    }


def run_backtest(cfg: dict, frame: pd.DataFrame, missing: list[str] | tuple = ()) -> dict:
    bt = cfg["backtest"]
    scores = score_frame(to_signals(compute_raw(frame), cfg), cfg)
    start, split = pd.Timestamp(bt["start"]), pd.Timestamp(bt["split"])
    ndx = frame["ndx"]
    result: dict = {"train": {}, "test": {}}
    cals = {}
    for h, spec in cfg["horizons"].items():
        label = forward_labels(ndx, spec["days"], spec["up"], spec["down"])
        fwd = ndx.shift(-spec["days"]) / ndx - 1
        s = scores[f"{h}_score"]
        train = (s.index >= start) & (s.index < split)
        test = s.index >= split
        result["train"][h] = _stats(s[train], label[train], fwd[train], cfg["arrow_threshold"])
        result["test"][h] = _stats(s[test], label[test], fwd[test], cfg["arrow_threshold"])
        cals[h] = Calibration.from_backtest(s[test], label[test], bt["bucket"])
    Calibration.save_all(cals, cfg["paths"]["calibration"],
                         meta={"config": config_fingerprint(cfg), "missing": list(missing)})
    return result


def main() -> None:
    from dotenv import load_dotenv

    from .config import load_config
    from .data import collect_all

    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    cfg = load_config()
    md = collect_all(cfg, "2003-01-01")
    print(f"데이터 기준일 {md.as_of}, 누락: {md.missing or '없음'}")
    res = run_backtest(cfg, md.frame, md.missing)
    print(f"{'구간':<6}{'기간':<7}{'표본':>7}{'적중률':>9}{'상관':>8}")
    for part in ("train", "test"):
        for h, st in res[part].items():
            hr = f"{st['hit_rate']:.1%}" if st["hit_rate"] is not None else "-"
            co = f"{st['corr']:+.3f}" if st["corr"] is not None else "-"
            print(f"{part:<8}{h:<7}{st['n']:>7}{hr:>9}{co:>8}")
    print(f"보정표 저장: {cfg['paths']['calibration']}")
    if not Calibration.is_valid_for(Calibration.load_meta(cfg["paths"]["calibration"]), cfg):
        print("⚠ 점수 지표가 누락된 상태로 만든 보정표입니다 — 일일 메시지에 확률이 표시되지 않습니다.")


if __name__ == "__main__":
    main()
