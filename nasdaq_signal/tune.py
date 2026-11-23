"""학습 구간만으로 지표별·기간별 방향을 정해 tuning.yaml에 저장.

규칙: 학습 구간(backtest.start ~ split)을 앞/뒤 절반으로 나눠, 신호와 미래수익률의 순위상관이
두 절반 모두 |IC| >= min_ic 이고 부호가 같을 때만 사용한다. 양수면 유지, 음수면 방향을 뒤집고,
그 외에는 해당 기간에서 제외한다. 검증 구간(split~)은 보지 않는다.

실행: python -m nasdaq_signal.tune  →  이후 python -m nasdaq_signal.backtest 로 검증
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

MIN_IC = 0.03


def select_directions(ic_a: dict, ic_b: dict, min_ic: float = MIN_IC) -> dict:
    out = {}
    for key, a in ic_a.items():
        b = ic_b.get(key, math.nan)
        if any(math.isnan(x) for x in (a, b)) or min(abs(a), abs(b)) < min_ic or (a > 0) != (b > 0):
            out[key] = "exclude"
        else:
            out[key] = "keep" if a > 0 else "flip"
    return out


def _base_direction(cfg: dict, name: str, h: str) -> int:
    ind = cfg["indicators"][name]
    return (ind.get("horizons") or {}).get(h, {}).get("direction", ind["direction"])


def build_overrides(cfg: dict, decisions: dict) -> dict:
    ov: dict = {}
    for (h, name), d in sorted(decisions.items()):
        if d == "flip":
            ov.setdefault(name, {})[h] = {"direction": -_base_direction(cfg, name, h)}
        elif d == "exclude":
            ov.setdefault(name, {})[h] = {"exclude": True}
    return ov


def _ic(sig: pd.Series, fwd: pd.Series, mask) -> float:
    d = pd.DataFrame({"s": sig[mask], "r": fwd[mask]}).dropna()
    if len(d) < 250 or d["s"].nunique() < 2:
        return math.nan
    return float(d["s"].rank().corr(d["r"].rank()))


def compute_ics(cfg: dict, frame: pd.DataFrame) -> tuple[dict, dict]:
    from .indicators import compute_raw, to_signals

    sig = to_signals(compute_raw(frame), cfg)
    start, split = pd.Timestamp(cfg["backtest"]["start"]), pd.Timestamp(cfg["backtest"]["split"])
    mid = start + (split - start) / 2
    ndx = frame["ndx"]
    ic_a, ic_b = {}, {}
    for h, spec in cfg["horizons"].items():
        days = spec["days"]
        fwd = ndx.shift(-days) / ndx - 1
        idx = sig[h].index
        # 각 절반의 미래수익률 창이 다음 구간(특히 검증 구간)을 넘지 않도록 끝을 days만큼 당김
        cut_a = idx[max(idx.searchsorted(mid) - days, 0)]
        cut_b = idx[max(idx.searchsorted(split) - days, 0)]
        first = (idx >= start) & (idx < cut_a)
        second = (idx >= mid) & (idx < cut_b)
        for name in sig[h].columns:
            ic_a[(h, name)] = _ic(sig[h][name], fwd, first)
            ic_b[(h, name)] = _ic(sig[h][name], fwd, second)
    return ic_a, ic_b


def main() -> None:
    import sys

    from dotenv import load_dotenv

    from .config import load_config, tuning_path
    from .data import collect_all

    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    cfg = load_config(apply_tuning=False)
    md = collect_all(cfg, "2003-01-01")
    ic_a, ic_b = compute_ics(cfg, md.frame)
    decisions = select_directions(ic_a, ic_b)
    print(f"{'기간':<6}{'지표':<14}{'IC 앞':>8}{'IC 뒤':>8}  결정")
    for key in sorted(decisions):
        print(f"{key[0]:<6}{key[1]:<14}{ic_a[key]:>+8.3f}{ic_b[key]:>+8.3f}  {decisions[key]}")
    out = {
        "generated": date.today().isoformat(),
        "rule": f"train halves {cfg['backtest']['start']}..{cfg['backtest']['split']}, |IC|>={MIN_IC}, same sign",
        "overrides": build_overrides(cfg, decisions),
    }
    path = tuning_path(cfg, "config.yaml")
    path.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=True), encoding="utf-8")
    print(f"\n저장: {path} — 이제 python -m nasdaq_signal.backtest 로 검증 구간 성과를 확인하세요.")


if __name__ == "__main__":
    main()
