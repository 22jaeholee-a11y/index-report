"""설정 파일(config.yaml) 로드와 검증."""
from __future__ import annotations

from pathlib import Path

import yaml

TRANSFORMS = {"percentile", "contrarian"}


class ConfigError(ValueError):
    pass


def load_config(path: str | Path = "config.yaml", apply_tuning: bool = True) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if apply_tuning:
        _merge_tuning(cfg, tuning_path(cfg, path))
    _validate(cfg)
    return cfg


def tuning_path(cfg: dict, config_path: str | Path) -> Path:
    """tuning.yaml 위치: 설정 파일과 같은 폴더 기준 (절대경로면 그대로)."""
    p = Path(cfg.get("tuning_file", "tuning.yaml"))
    return p if p.is_absolute() else Path(config_path).parent / p


def _merge_tuning(cfg: dict, path: Path) -> None:
    """tuning.yaml(학습 구간 데이터로 정한 기간별 방향/제외)을 지표 설정에 덮어쓴다."""
    if not path.exists():
        return
    tuning = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for name, per_h in (tuning.get("overrides") or {}).items():
        if name not in cfg["indicators"]:
            raise ConfigError(f"tuning: unknown indicator '{name}'")
        hs = cfg["indicators"][name].setdefault("horizons", {}) or {}
        cfg["indicators"][name]["horizons"] = hs
        for h, o in per_h.items():
            hs.setdefault(h, {}).update(o)


def _validate(cfg: dict) -> None:
    horizons = cfg["horizons"]
    weights = cfg["category_weights"]
    for h, spec in horizons.items():
        if "predict" in spec and not isinstance(spec["predict"], bool):
            raise ConfigError(f"horizon '{h}': predict must be true/false")
    for h in horizons:
        total = sum(w.get(h, 0.0) for w in weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"category_weights for '{h}' sum to {total}, expected 1")
    for name, ind in cfg["indicators"].items():
        if ind["category"] not in weights:
            raise ConfigError(f"indicator '{name}': unknown category '{ind['category']}'")
        transforms = [ind["transform"]] + [
            o.get("transform", ind["transform"]) for o in (ind.get("horizons") or {}).values()
        ]
        for t in transforms:
            if t not in TRANSFORMS:
                raise ConfigError(f"indicator '{name}': unknown transform '{t}'")
        for h, o in (ind.get("horizons") or {}).items():
            if h not in horizons:
                raise ConfigError(f"indicator '{name}': unknown horizon '{h}'")
            if "exclude" in o and not isinstance(o["exclude"], bool):
                raise ConfigError(f"indicator '{name}': exclude must be true/false")


def active_indicators(cfg: dict, horizon: str) -> list[str]:
    """해당 기간에 제외(exclude)되지 않은 지표 이름."""
    return [n for n, ind in cfg["indicators"].items()
            if not ((ind.get("horizons") or {}).get(horizon, {}).get("exclude", False))]
