"""카테고리·기간 가중 합산, 화살표, 확률 보정표."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("up", "flat", "down")
MIN_SAMPLES = 30


def score_frame(signals: dict[str, pd.DataFrame], cfg: dict,
                ai_adjustment: dict[str, float] | None = None) -> pd.DataFrame:
    cap = cfg["ai_adjustment_cap"]
    ai_adjustment = ai_adjustment or {}
    cats: dict[str, list[str]] = {}
    for name, ind in cfg["indicators"].items():
        cats.setdefault(ind["category"], []).append(name)

    cols = {}
    for h, sig in signals.items():
        weighted = pd.Series(0.0, index=sig.index)
        wsum = pd.Series(0.0, index=sig.index)
        missing = pd.Series(0.0, index=sig.index)
        for cat, names in cats.items():
            w = cfg["category_weights"][cat][h]
            block = sig.reindex(columns=names)
            cat_score = block.mean(axis=1, skipna=True)
            cols[f"{h}_{cat}"] = cat_score
            valid = cat_score.notna()
            weighted += (cat_score.fillna(0) * w)
            wsum += valid * w
            missing += block.isna().mean(axis=1) * w
        score = (weighted / wsum.replace(0, np.nan)) * 100
        adj = float(np.clip(ai_adjustment.get(h, 0.0), -cap, cap))
        cols[f"{h}_score"] = (score + adj).clip(-100, 100)
        cols[f"{h}_missing"] = missing
    return pd.DataFrame(cols)


def arrow(score: float, threshold: float = 20) -> str:
    if score >= threshold:
        return "▲"
    if score <= -threshold:
        return "▼"
    return "─"


def _bucket_key(score: float, bucket: int) -> int:
    k = math.floor(score / bucket) * bucket
    return int(min(max(k, -100), 100 - bucket))


class Calibration:
    """점수 구간별 실제 결과 분포. counts[bucket] = [up, flat, down, n]."""

    def __init__(self, counts: dict[int, list[int]], bucket: int):
        self.counts = counts
        self.bucket = bucket

    @classmethod
    def from_backtest(cls, scores: pd.Series, labels: pd.Series, bucket: int) -> "Calibration":
        df = pd.DataFrame({"s": scores.to_numpy(), "l": labels.to_numpy()}).dropna()
        counts: dict[int, list[int]] = {}
        for s, l in zip(df["s"], df["l"]):
            c = counts.setdefault(_bucket_key(s, bucket), [0, 0, 0, 0])
            c[LABELS.index(l)] += 1
            c[3] += 1
        return cls(counts, bucket)

    def probabilities(self, score: float) -> tuple[int, int, int] | None:
        c = self.counts.get(_bucket_key(score, self.bucket))
        if not c or c[3] < MIN_SAMPLES:
            return None
        raw = [x * 100 / c[3] for x in c[:3]]
        floors = [math.floor(x) for x in raw]
        # 최대 잉여 방식으로 합을 100에 맞춤
        for i in sorted(range(3), key=lambda i: raw[i] - floors[i], reverse=True)[: 100 - sum(floors)]:
            floors[i] += 1
        return tuple(floors)  # type: ignore[return-value]

    @staticmethod
    def save_all(cals: dict[str, "Calibration"], path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {h: {str(k): v for k, v in sorted(c.counts.items())} for h, c in cals.items()}
        payload["_bucket"] = next(iter(cals.values())).bucket if cals else 20
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def load_all(path: str | Path) -> dict[str, "Calibration"] | None:
        p = Path(path)
        if not p.exists():
            return None
        payload = json.loads(p.read_text(encoding="utf-8"))
        bucket = payload.pop("_bucket", 20)
        return {h: Calibration({int(k): v for k, v in d.items()}, bucket) for h, d in payload.items()}
