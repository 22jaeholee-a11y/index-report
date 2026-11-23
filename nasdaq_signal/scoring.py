"""카테고리·기간 가중 합산, 화살표, 확률 보정표."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import active_indicators

FINGERPRINT_KEYS = ("horizons", "category_weights", "indicators", "percentile_window",
                    "percentile_min_periods", "contrarian_band", "fred_series")
REFERENCE_ONLY = {"fear_greed", "bamlh0a0hym2"}  # 표시 전용  # 점수에 쓰이지 않는 참고값 — 보정표 유효성에 영향 없음


def config_fingerprint(cfg: dict) -> str:
    sub = {k: cfg.get(k) for k in FINGERPRINT_KEYS}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


LABELS = ("up", "flat", "down")
MIN_SAMPLES = 30


def score_frame(signals: dict[str, pd.DataFrame], cfg: dict,
                ai_adjustment: dict[str, float] | None = None) -> pd.DataFrame:
    cap = cfg["ai_adjustment_cap"]
    ai_adjustment = ai_adjustment or {}
    cols = {}
    for h, sig in signals.items():
        cats: dict[str, list[str]] = {c: [] for c in cfg["category_weights"]}
        for name in active_indicators(cfg, h):
            cats[cfg["indicators"][name]["category"]].append(name)
        weighted = pd.Series(0.0, index=sig.index)
        wsum = pd.Series(0.0, index=sig.index)
        missing = pd.Series(0.0, index=sig.index)
        used_w = 0.0   # 이 기간에 실제로 쓰는 카테고리 비중 합 (누락 비율의 분모)
        for cat, names in cats.items():
            w = cfg["category_weights"][cat][h]
            if not names:  # 이 기간에 쓰는 지표가 없는 카테고리 → 가중치 재분배, 누락 아님
                cols[f"{h}_{cat}"] = pd.Series(np.nan, index=sig.index)
                continue
            block = sig.reindex(columns=names)
            cat_score = block.mean(axis=1, skipna=True)
            cols[f"{h}_{cat}"] = cat_score
            valid = cat_score.notna()
            weighted += (cat_score.fillna(0) * w)
            wsum += valid * w
            missing += block.isna().mean(axis=1) * w
            used_w += w
        if used_w > 0:   # 쓰이지 않는 카테고리 비중까지 분모에 넣으면 누락이 과소평가됨 (예: 단기 VIX 누락)
            missing = missing / used_w
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


def _percentages(c: list[int]) -> tuple[int, int, int]:
    """[up, flat, down, n] → 합이 100인 정수 백분율 (최대 잉여 방식)."""
    raw = [x * 100 / c[3] for x in c[:3]]
    floors = [math.floor(x) for x in raw]
    for i in sorted(range(3), key=lambda i: raw[i] - floors[i], reverse=True)[: 100 - sum(floors)]:
        floors[i] += 1
    return tuple(floors)  # type: ignore[return-value]


def score_distribution(scores: pd.Series) -> list[float]:
    """과거 점수의 0~100 백분위 값 101개 (오름차순)."""
    return [float(x) for x in np.nanpercentile(scores.dropna().to_numpy(), np.arange(101))]


def score_percentile(dist: list[float], score: float) -> float:
    """오늘 점수가 과거 점수 분포에서 차지하는 위치 (0~1)."""
    return float(np.interp(score, dist, np.linspace(0, 1, len(dist)), left=0.0, right=1.0))


def load_score_distributions(path: str | Path) -> dict[str, list[float]]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("_score_dist") or {}


def base_probabilities(path: str | Path) -> dict[str, tuple[int, int, int]]:
    p = Path(path)
    if not p.exists():
        return {}
    base = json.loads(p.read_text(encoding="utf-8")).get("_base") or {}
    return {h: _percentages(c) for h, c in base.items() if c and c[3] > 0}


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
        return _percentages(c)

    @staticmethod
    def save_all(cals: dict[str, "Calibration"], path: str | Path, meta: dict | None = None,
                 base: dict[str, list[int]] | None = None,
                 score_dists: dict[str, list[float]] | None = None) -> None:
        """base: 기간별 역사적 기본 비율 [up, flat, down, n] (점수와 무관한 전체 표본)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {h: {str(k): v for k, v in sorted(c.counts.items())} for h, c in cals.items()}
        payload["_bucket"] = next(iter(cals.values())).bucket if cals else 20
        payload["_meta"] = meta
        payload["_base"] = base
        payload["_score_dist"] = score_dists
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def load_all(path: str | Path) -> dict[str, "Calibration"] | None:
        p = Path(path)
        if not p.exists():
            return None
        payload = json.loads(p.read_text(encoding="utf-8"))
        bucket = payload.pop("_bucket", 20)
        return {h: Calibration({int(k): v for k, v in d.items()}, bucket)
                for h, d in payload.items() if not h.startswith("_")}

    @staticmethod
    def load_meta(path: str | Path) -> dict | None:
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8")).get("_meta")

    @staticmethod
    def is_valid_for(meta: dict | None, cfg: dict) -> bool:
        """보정표가 현재 설정으로, 점수 지표 누락 없이 만들어졌는지."""
        if not meta or meta.get("config") != config_fingerprint(cfg):
            return False
        return not (set(meta.get("missing") or []) - REFERENCE_ONLY)
