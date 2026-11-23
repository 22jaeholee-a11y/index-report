"""설정 파일(config.yaml) 로드와 검증."""
from __future__ import annotations

from pathlib import Path

import yaml

TRANSFORMS = {"percentile", "contrarian"}


class ConfigError(ValueError):
    pass


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    horizons = cfg["horizons"]
    weights = cfg["category_weights"]
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
        for h in ind.get("horizons") or {}:
            if h not in horizons:
                raise ConfigError(f"indicator '{name}': unknown horizon '{h}'")
