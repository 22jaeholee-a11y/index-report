import yaml

from nasdaq_signal.config import load_config
from nasdaq_signal.tune import build_overrides, select_directions


def test_select_keeps_flips_and_excludes():
    a = {("short", "x"): 0.10, ("short", "y"): -0.08, ("short", "z"): 0.10, ("short", "w"): 0.01}
    b = {("short", "x"): 0.05, ("short", "y"): -0.04, ("short", "z"): -0.06, ("short", "w"): 0.20}
    d = select_directions(a, b, min_ic=0.03)
    assert d[("short", "x")] == "keep"
    assert d[("short", "y")] == "flip"
    assert d[("short", "z")] == "exclude"     # 두 기간 부호 불일치
    assert d[("short", "w")] == "exclude"     # 한 기간 신호 약함


def test_select_treats_nan_as_exclude():
    d = select_directions({("long", "x"): float("nan")}, {("long", "x"): 0.2}, min_ic=0.03)
    assert d[("long", "x")] == "exclude"


def test_build_overrides_flips_horizon_direction():
    cfg = load_config(apply_tuning=False)
    ov = build_overrides(cfg, {("short", "vix"): "flip", ("mid", "vix"): "keep", ("long", "curve"): "exclude"})
    assert ov["vix"]["short"] == {"direction": 1}        # 기본 -1 → 뒤집음
    assert "mid" not in ov.get("vix", {})
    assert ov["curve"]["long"] == {"exclude": True}


def test_load_config_merges_tuning_file(tmp_path):
    base = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    base["tuning_file"] = str(tmp_path / "tuning.yaml")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    (tmp_path / "tuning.yaml").write_text(yaml.safe_dump(
        {"overrides": {"vix": {"mid": {"direction": 1}}, "curve": {"short": {"exclude": True}}}}), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg["indicators"]["vix"]["horizons"]["mid"] == {"transform": "contrarian", "direction": 1}
    assert cfg["indicators"]["curve"]["horizons"]["short"] == {"exclude": True}
    raw = load_config(tmp_path / "c.yaml", apply_tuning=False)
    assert "short" not in (raw["indicators"]["curve"].get("horizons") or {})
